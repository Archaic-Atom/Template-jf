# -*- coding: utf-8 -*-
"""Generate the training/testing list CSV consumed by --trainListPath.

生成 --trainListPath 所需的训练/测试清单 CSV。

The dataloader reads this CSV, so its columns are entirely up to you —
just keep them in sync with ``get_train_dataset``. This script ships as a
working example over a directory of images.

dataloader 读取这个 CSV，所以列名完全由你决定 —— 只要和
``get_train_dataset`` 对得上即可。本脚本是一个可运行的示例，
遍历一个图片目录生成清单。

Usage / 用法:
    python Source/Tools/generate_train_list.py \
        --root_path /path/to/dataset --output ./Datasets/my_training_list.csv
"""

import argparse
import csv
import os
from typing import List

_IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. 解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', type=str, required=True,
                        help='dataset root to scan / 待扫描的数据集根目录')
    parser.add_argument('--output', type=str,
                        default='./Datasets/dataset_example_training_list.csv',
                        help='output csv path / 输出 csv 路径')
    return parser.parse_args()


def collect(root_path: str) -> List[str]:
    """Collect image paths under ``root_path``.

    收集 ``root_path`` 下的图片路径。

    Returns:
        Sorted absolute paths. 排序后的绝对路径列表。
    """
    found = []
    for cur_dir, _, files in os.walk(root_path):
        for name in files:
            if name.lower().endswith(_IMG_EXT):
                found.append(os.path.abspath(os.path.join(cur_dir, name)))
    return sorted(found)


def main() -> None:
    """Write the list CSV. 写出清单 CSV。"""
    args = parse_args()
    if not os.path.isdir(args.root_path):
        raise SystemExit(f'root_path is not a directory: {args.root_path}')

    samples = collect(args.root_path)
    if not samples:
        raise SystemExit(f'no images found under: {args.root_path}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        # REPLACE the header to match your dataloader.
        # REPLACE 这一行表头，使其与你的 dataloader 对应。
        writer.writerow(['img'])
        for path in samples:
            writer.writerow([path])

    print(f'wrote {len(samples)} rows -> {args.output}')


if __name__ == '__main__':
    main()
