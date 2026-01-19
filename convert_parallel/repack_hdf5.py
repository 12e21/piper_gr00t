"""
将目录中的多个 HDF5 文件重新划分成包含指定数量 episodes 的 HDF5 文件

例如：
- 输入目录包含 file1.hdf5 (10 episodes), file2.hdf5 (15 episodes)
- 指定每文件 5 个 episodes
- 输出：output_0.hdf5 (5 episodes), output_1.hdf5 (5 episodes), ..., output_4.hdf5 (5 episodes)
"""

import h5py
import numpy as np
from pathlib import Path
import typer
from typing import Optional
from tqdm import tqdm
from collections import defaultdict


def copy_group(src_group: h5py.Group, dst_group: h5py.Group):
    """
    递归复制 HDF5 group 及其所有 datasets 和子 groups

    Args:
        src_group: 源 group
        dst_group: 目标 group
    """
    # 复制属性
    for attr_name, attr_value in src_group.attrs.items():
        dst_group.attrs[attr_name] = attr_value

    # 复制所有内容
    for name, obj in src_group.items():
        if isinstance(obj, h5py.Dataset):
            # 复制 dataset
            dst_group.create_dataset(
                name,
                data=obj[()],
                dtype=obj.dtype,
                compression=obj.compression,
                compression_opts=obj.compression_opts,
                shuffle=obj.shuffle,
            )
            # 复制 dataset 属性
            for attr_name, attr_value in obj.attrs.items():
                dst_group[name].attrs[attr_name] = attr_value
        elif isinstance(obj, h5py.Group):
            # 递归复制子 group
            new_group = dst_group.create_group(name)
            copy_group(obj, new_group)


def collect_episodes_from_directory(
    input_dir: Path,
    pattern: str = "*.hdf5"
) -> dict[str, tuple[Path, str]]:
    """
    从目录中的所有 HDF5 文件收集 episodes

    Args:
        input_dir: 输入目录
        pattern: 文件匹配模式

    Returns:
        字典：{episode_name: (file_path, group_name)}
    """
    episodes = {}

    hdf5_files = sorted(input_dir.glob(pattern))

    if not hdf5_files:
        typer.echo(f"❌ 在 {input_dir} 中没有找到匹配 '{pattern}' 的文件", err=True)
        raise typer.Exit(1)

    typer.echo(f"📂 扫描目录: {input_dir}")
    typer.echo(f"📁 找到 {len(hdf5_files)} 个 HDF5 文件\n")

    for hdf5_file in tqdm(hdf5_files, desc="扫描 episodes"):
        with h5py.File(hdf5_file, "r") as f:
            for group_name in f.keys():
                # 使用唯一的 episode 名称
                episode_key = f"{hdf5_file.stem}/{group_name}"
                episodes[episode_key] = (hdf5_file, group_name)

    return episodes


def repack_hdf5_files(
    input_dir: str = typer.Option(..., "--input", "-i", help="输入 HDF5 文件目录"),
    output_dir: str = typer.Option("./repack_output", "--output", "-o", help="输出目录"),
    episodes_per_file: int = typer.Option(..., "--episodes-per-file", "-e", help="每个输出文件包含的 episodes 数量"),
    prefix: str = typer.Option("repack_", "--prefix", help="输出文件名前缀"),
    pattern: str = typer.Option("*.hdf5", "--pattern", help="输入文件匹配模式"),
    overwrite: bool = typer.Option(False, "--overwrite", help="覆盖已存在的文件"),
    dry_run: bool = typer.Option(False, "--dry-run", help="预览模式，不实际写入文件"),
) -> None:
    """
    将目录中的多个 HDF5 文件重新划分成包含指定数量 episodes 的 HDF5 文件
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 检查输入目录
    if not input_path.exists():
        typer.echo(f"❌ 输入目录不存在: {input_dir}", err=True)
        raise typer.Exit(1)

    if not input_path.is_dir():
        typer.echo(f"❌ 输入路径不是目录: {input_dir}", err=True)
        raise typer.Exit(1)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 收集所有 episodes
    episodes = collect_episodes_from_directory(input_path, pattern)

    if not episodes:
        typer.echo("❌ 没有找到任何 episodes", err=True)
        raise typer.Exit(1)

    total_episodes = len(episodes)
    num_output_files = (total_episodes + episodes_per_file - 1) // episodes_per_file

    typer.echo(f"📊 统计信息:")
    typer.echo(f"   总 episodes 数: {total_episodes}")
    typer.echo(f"   每文件 episodes 数: {episodes_per_file}")
    typer.echo(f"   将生成 {num_output_files} 个文件")
    if total_episodes % episodes_per_file != 0:
        typer.echo(f"   最后一个文件将包含 {total_episodes % episodes_per_file} 个 episodes")
    typer.echo()

    if dry_run:
        typer.echo("🔍 预览模式 - 将生成以下文件:\n")
        for file_idx in range(num_output_files):
            start_idx = file_idx * episodes_per_file
            end_idx = min(start_idx + episodes_per_file, total_episodes)
            episode_names = list(episodes.keys())[start_idx:end_idx]

            typer.echo(f"📄 {prefix}{file_idx}.hdf5 ({len(episode_names)} episodes):")
            for name in episode_names[:3]:  # 只显示前3个
                typer.echo(f"     - {name}")
            if len(episode_names) > 3:
                typer.echo(f"     ... 还有 {len(episode_names) - 3} 个")
            typer.echo()
        typer.echo("✨ 预览完成（使用 --dry-run=false 实际执行）")
        return

    # 开始重新打包
    typer.echo(f"💾 输出目录: {output_path}\n")

    episode_names = sorted(episodes.keys())

    for file_idx in tqdm(range(num_output_files), desc="重新打包"):
        start_idx = file_idx * episodes_per_file
        end_idx = min(start_idx + episodes_per_file, total_episodes)
        batch_episodes = episode_names[start_idx:end_idx]

        output_filename = f"{prefix}{file_idx}.hdf5"
        output_file = output_path / output_filename

        # 检查文件是否已存在
        if output_file.exists() and not overwrite:
            typer.echo(f"⚠️  跳过 {output_filename}（文件已存在，使用 --overwrite 覆盖）")
            continue

        # 创建新文件并复制 episodes
        with h5py.File(output_file, "w") as out_f:
            for episode_key in batch_episodes:
                src_file, group_name = episodes[episode_key]

                # 打开源文件并复制 group
                with h5py.File(src_file, "r") as in_f:
                    src_group = in_f[group_name]
                    dst_group = out_f.create_group(group_name)
                    copy_group(src_group, dst_group)

        typer.echo(f"✅ 已保存: {output_filename} ({len(batch_episodes)} episodes)")

    typer.echo(f"\n✨ 完成！共生成 {num_output_files} 个文件到 {output_path}")


def analyze_hdf5_directory(
    input_dir: str = typer.Option(..., "--input", "-i", help="输入 HDF5 文件目录"),
    pattern: str = typer.Option("*.hdf5", "--pattern", help="输入文件匹配模式"),
    episodes_per_file: int = typer.Option(50, "--episodes-per-file", "-e", help="目标每文件 episodes 数量"),
) -> None:
    """
    分析目录中的 HDF5 文件，显示 episodes 分布和重新打包建议
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        typer.echo(f"❌ 输入目录不存在: {input_dir}", err=True)
        raise typer.Exit(1)

    # 收集所有 episodes
    episodes = collect_episodes_from_directory(input_path, pattern)

    if not episodes:
        typer.echo("❌ 没有找到任何 episodes", err=True)
        raise typer.Exit(1)

    # 按 file 分组统计
    file_episodes = defaultdict(list)
    for episode_key, (file_path, group_name) in episodes.items():
        file_episodes[file_path].append(group_name)

    total_episodes = len(episodes)
    total_files = len(file_episodes)

    typer.echo(f"📊 分析结果:")
    typer.echo(f"\n总文件数: {total_files}")
    typer.echo(f"总 episodes 数: {total_episodes}")
    typer.echo(f"平均每文件 episodes 数: {total_episodes / total_files:.1f}")

    # 显示每个文件的 episodes 数量
    typer.echo(f"\n各文件 episodes 分布:")
    for file_path, group_names in sorted(file_episodes.items()):
        typer.echo(f"  {file_path.name}: {len(group_names)} episodes")

    # 重新打包建议
    num_output_files = (total_episodes + episodes_per_file - 1) // episodes_per_file
    typer.echo(f"\n💡 重新打包建议 (每文件 {episodes_per_file} episodes):")
    typer.echo(f"  将生成 {num_output_files} 个文件")
    if total_episodes % episodes_per_file != 0:
        typer.echo(f"  最后一个文件将包含 {total_episodes % episodes_per_file} 个 episodes")


# 创建主 app 和子命令
app = typer.Typer(help="HDF5 文件重新打包工具")
app.command(name="repack")(repack_hdf5_files)
app.command(name="analyze")(analyze_hdf5_directory)


if __name__ == "__main__":
    app()
