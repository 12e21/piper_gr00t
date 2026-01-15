#!/usr/bin/env python3
"""
快速查看 HDF5 文件数据结构的工具脚本
"""

import h5py
import numpy as np
import argparse
from pathlib import Path


def print_structure(name, obj, level=0, show_attrs=False, preview_data=False):
    """递归打印 HDF5 文件结构"""
    indent = "  " * level

    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 Group: {name}")
        if show_attrs and obj.attrs:
            print(f"{indent}   Attributes:")
            for attr_name, attr_value in obj.attrs.items():
                print(f"{indent}     - {attr_name}: {attr_value}")

    elif isinstance(obj, h5py.Dataset):
        dtype_str = str(obj.dtype)
        shape_str = str(obj.shape)
        print(f"{indent}📊 Dataset: {name}")
        print(f"{indent}   Shape: {shape_str}")
        print(f"{indent}   Dtype: {dtype_str}")

        # 显示属性
        if show_attrs and obj.attrs:
            print(f"{indent}   Attributes:")
            for attr_name, attr_value in obj.attrs.items():
                print(f"{indent}     - {attr_name}: {attr_value}")

        # 预览数据
        if preview_data:
            print(f"{indent}   Preview:", end=" ")
            try:
                if obj.size == 0:
                    print("Empty dataset")
                elif obj.ndim == 0:
                    # 标量
                    print(f"{obj[()]}")
                elif obj.ndim == 1:
                    # 1D 数组
                    n_show = min(5, len(obj))
                    print(f"[{', '.join(map(str, obj[:n_show]))}]{'...' if len(obj) > n_show else ''}")
                else:
                    # 多维数组，显示第一个维度
                    print(f"First element: {obj[0]}")
            except Exception as e:
                print(f"(Unable to preview: {e})")


def explore_hdf5(filepath, show_attrs=False, preview_data=False, max_level=None):
    """
    探索 HDF5 文件结构

    Args:
        filepath: HDF5 文件路径
        show_attrs: 是否显示属性
        preview_data: 是否预览数据
        max_level: 最大显示层级（None 表示全部显示）
    """
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"❌ 文件不存在: {filepath}")
        return

    print(f"\n{'='*60}")
    print(f"HDF5 文件: {filepath}")
    print(f"文件大小: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*60}\n")

    try:
        with h5py.File(filepath, 'r') as f:
            print(f"根对象数量: {len(f.keys())} 个")
            print(f"文件模式: {f.mode}")
            print(f"驱动器: {f.driver}\n")

            print("文件结构:")
            print("-" * 60)

            if max_level is not None:
                # 自定义层级遍历
                def traverse_with_level(name, obj, current_level=0):
                    if current_level <= max_level:
                        print_structure(name, obj, current_level, show_attrs, preview_data)
                    if isinstance(obj, h5py.Group) and current_level < max_level:
                        obj.visititems(lambda n, o: traverse_with_level(n, o, current_level + 1))

                f.visititems(lambda n, o: traverse_with_level(n, o, 0))
            else:
                f.visititems(lambda n, o: print_structure(n, o, 0, show_attrs, preview_data))

            print("-" * 60)

            # 统计信息
            groups = []
            datasets = []

            def collect_info(name, obj):
                if isinstance(obj, h5py.Group):
                    groups.append(name)
                elif isinstance(obj, h5py.Dataset):
                    datasets.append((name, obj.shape, obj.dtype, obj.size * obj.dtype.itemsize))

            f.visititems(collect_info)

            print(f"\n📈 统计信息:")
            print(f"  Groups: {len(groups)}")
            print(f"  Datasets: {len(datasets)}")

            if datasets:
                total_size = sum(d[3] for d in datasets)
                print(f"  总数据大小: {total_size / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")


def interactive_explore(filepath):
    """交互式探索 HDF5 文件"""
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"❌ 文件不存在: {filepath}")
        return

    try:
        with h5py.File(filepath, 'r') as f:
            print(f"\n✅ 已打开: {filepath}")
            print(f"输入 'help' 查看可用命令\n")

            current_path = []

            while True:
                # 显示当前位置
                if current_path:
                    print(f"\n📍 当前位置: {'/'.join(current_path)}")
                else:
                    print(f"\n📍 当前位置: / (根)")

                # 获取当前组
                current_group = f
                for part in current_path:
                    current_group = current_group[part]

                # 显示内容
                if isinstance(current_group, h5py.Group):
                    keys = list(current_group.keys())
                    if keys:
                        print("内容:")
                        for i, key in enumerate(keys, 1):
                            obj = current_group[key]
                            if isinstance(obj, h5py.Group):
                                print(f"  [{i}] 📁 {key}/")
                            elif isinstance(obj, h5py.Dataset):
                                print(f"  [{i}] 📊 {key} {obj.shape} {obj.dtype}")
                    else:
                        print("(空)")

                # 获取命令
                cmd = input("\n> ").strip().split()
                if not cmd:
                    continue

                if cmd[0] == 'help':
                    print("""
可用命令:
  help              - 显示帮助
  cd <name>         - 进入组（使用 '..' 返回上级）
  ls                - 列出当前组内容
  info <name>       - 显示数据集详细信息
  preview <name>    - 预览数据集数据
  pwd               - 显示当前位置
  exit 或 quit      - 退出
                    """)
                elif cmd[0] in ['exit', 'quit']:
                    break
                elif cmd[0] == 'pwd':
                    print(f"{'/' + '/'.join(current_path) if current_path else '/'}")
                elif cmd[0] == 'ls':
                    keys = list(current_group.keys())
                    for key in keys:
                        obj = current_group[key]
                        if isinstance(obj, h5py.Group):
                            print(f"📁 {key}/")
                        elif isinstance(obj, h5py.Dataset):
                            print(f"📊 {key} {obj.shape} {obj.dtype}")
                elif cmd[0] == 'cd':
                    if len(cmd) < 2:
                        print("❌ 请指定组名")
                        continue
                    if cmd[1] == '..':
                        if current_path:
                            current_path.pop()
                    elif cmd[1] in current_group and isinstance(current_group[cmd[1]], h5py.Group):
                        current_path.append(cmd[1])
                    else:
                        print(f"❌ 组 '{cmd[1]}' 不存在")
                elif cmd[0] == 'info':
                    if len(cmd) < 2:
                        print("❌ 请指定对象名")
                        continue
                    if cmd[1] in current_group:
                        obj = current_group[cmd[1]]
                        if isinstance(obj, h5py.Dataset):
                            print(f"\n📊 数据集: {cmd[1]}")
                            print(f"   形状: {obj.shape}")
                            print(f"   数据类型: {obj.dtype}")
                            print(f"   大小: {obj.size} 元素")
                            print(f"   字节大小: {obj.nbytes} bytes")
                        elif isinstance(obj, h5py.Group):
                            print(f"\n📁 组: {cmd[1]}")
                            print(f"   成员数量: {len(obj.keys())} 个")
                            print(f"   成员列表: {list(obj.keys())}")

                        # 显示属性（Group 和 Dataset 都有）
                        if obj.attrs:
                            print(f"   属性:")
                            for attr_name, attr_value in obj.attrs.items():
                                print(f"     - {attr_name}: {attr_value}")
                        else:
                            print(f"   属性: (无)")
                    else:
                        print(f"❌ '{cmd[1]}' 不存在")
                elif cmd[0] == 'preview':
                    if len(cmd) < 2:
                        print("❌ 请指定数据集名")
                        continue
                    if cmd[1] in current_group:
                        obj = current_group[cmd[1]]
                        if isinstance(obj, h5py.Dataset):
                            print(f"\n📊 预览: {cmd[1]}")
                            try:
                                data = obj[()]
                                if isinstance(data, np.ndarray):
                                    if data.ndim <= 2 and data.size <= 100:
                                        print(data)
                                    else:
                                        print(f"形状: {data.shape}")
                                        print(f"数据类型: {data.dtype}")
                                        print(f"值范围: [{np.min(data)}, {np.max(data)}]")
                                        print(f"第一个元素: {data.flat[0]}")
                                else:
                                    print(data)
                            except Exception as e:
                                print(f"❌ 无法读取数据: {e}")
                        else:
                            print(f"❌ '{cmd[1]}' 是一个组，不是数据集")
                    else:
                        print(f"❌ '{cmd[1]}' 不存在")
                else:
                    print(f"❌ 未知命令: {cmd[0]} (输入 'help' 查看帮助)")

    except Exception as e:
        print(f"❌ 错误: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='快速查看 HDF5 文件数据结构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本查看
  python read_hdf5.py data.h5

  # 显示属性和数据预览
  python read_hdf5.py data.h5 --attrs --preview

  # 限制显示层级
  python read_hdf5.py data.h5 --max-level 2

  # 交互式模式
  python read_hdf5.py data.h5 --interactive
        """
    )

    parser.add_argument('filepath', help='HDF5 文件路径')
    parser.add_argument('-a', '--attrs', action='store_true', help='显示属性')
    parser.add_argument('-p', '--preview', action='store_true', help='预览数据')
    parser.add_argument('-l', '--max-level', type=int, default=None, help='最大显示层级')
    parser.add_argument('-i', '--interactive', action='store_true', help='交互式模式')

    args = parser.parse_args()

    if args.interactive:
        interactive_explore(args.filepath)
    else:
        explore_hdf5(args.filepath, args.attrs, args.preview, args.max_level)


if __name__ == '__main__':
    main()
