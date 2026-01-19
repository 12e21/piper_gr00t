#!/usr/bin/env python3
"""
将包含多个 group 的 HDF5 文件拆分成多个单独的 HDF5 文件
每个 group 会被保存为一个独立的 HDF5 文件
"""

import h5py
import numpy as np
from pathlib import Path
import typer
from typing import Optional, List
from tqdm import tqdm


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


def split_hdf5_file(
    input_file: str = typer.Option(..., "--input", "-i", help="输入的 HDF5 文件路径"),
    output_dir: str = typer.Option("./split_output", "--output", "-o", help="输出目录"),
    prefix: str = typer.Option("", "--prefix", help="输出文件名前缀"),
    groups: Optional[List[str]] = typer.Option(None, help="指定要拆分的 group 名称（可多次使用，未指定则拆分所有）"),
    overwrite: bool = typer.Option(False, "--overwrite", help="覆盖已存在的文件"),
) -> None:
    """
    将包含多个 group 的 HDF5 文件拆分成多个单独的 HDF5 文件

    每个 group 会被保存为一个独立的 HDF5 文件，文件名为 <prefix><group_name>.hdf5
    """
    input_path = Path(input_file)

    # 检查输入文件
    if not input_path.exists():
        typer.echo(f"❌ 输入文件不存在: {input_file}", err=True)
        raise typer.Exit(1)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    typer.echo(f"📂 读取文件: {input_path}")
    typer.echo(f"💾 输出目录: {output_path}")

    # 打开输入文件并获取 groups
    with h5py.File(input_path, "r") as f:
        # 获取顶层 groups（通常是 episodes）
        all_groups = list(f.keys())

        if not all_groups:
            typer.echo("❌ 文件中没有找到任何 group", err=True)
            raise typer.Exit(1)

        # 过滤指定的 groups
        if groups:
            invalid_groups = [g for g in groups if g not in all_groups]
            if invalid_groups:
                typer.echo(f"❌ 指定的 group 不存在: {', '.join(invalid_groups)}", err=True)
                raise typer.Exit(1)
            groups_to_split = groups
        else:
            groups_to_split = all_groups

        typer.echo(f"\n找到 {len(all_groups)} 个 groups，将拆分 {len(groups_to_split)} 个\n")

        # 拆分每个 group
        for group_name in tqdm(groups_to_split, desc="拆分 groups"):
            src_group = f[group_name]

            # 构造输出文件名
            output_filename = f"{prefix}{group_name}.hdf5"
            output_file = output_path / output_filename

            # 检查文件是否已存在
            if output_file.exists() and not overwrite:
                typer.echo(f"⚠️  跳过 {output_filename}（文件已存在，使用 --overwrite 覆盖）")
                continue

            # 创建新文件并复制 group
            with h5py.File(output_file, "w") as out_f:
                # 创建根 group（使用原 group 名称）
                dst_group = out_f.create_group(group_name)
                copy_group(src_group, dst_group)

            typer.echo(f"✅ 已保存: {output_filename}")

    typer.echo(f"\n✨ 完成！共拆分 {len(groups_to_split)} 个 groups 到 {output_path}")


def list_hdf5_groups(
    input_file: str = typer.Option(..., "--input", "-i", help="输入的 HDF5 文件路径"),
) -> None:
    """
    列出 HDF5 文件中的所有 groups
    """
    input_path = Path(input_file)

    if not input_path.exists():
        typer.echo(f"❌ 输入文件不存在: {input_file}", err=True)
        raise typer.Exit(1)

    typer.echo(f"📂 文件: {input_path}")
    typer.echo(f"📏 大小: {input_path.stat().st_size / 1024 / 1024:.2f} MB\n")

    with h5py.File(input_path, "r") as f:
        groups = list(f.keys())
        typer.echo(f"Groups ({len(groups)} 个):")
        for i, name in enumerate(groups, 1):
            group = f[name]
            typer.echo(f"  [{i}] {name}")
            # 显示属性
            if group.attrs:
                for attr_name, attr_value in group.attrs.items():
                    typer.echo(f"      {attr_name}: {attr_value}")


# 创建主 app 和子命令
app = typer.Typer(help="HDF5 文件拆分工具")
app.command()(split_hdf5_file)
app.command()(list_hdf5_groups)


if __name__ == "__main__":
    app()
