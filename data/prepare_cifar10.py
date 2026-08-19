#!/usr/bin/env python3
"""
下载 CIFAR-10 并转换为 ImageFolder 格式。

输出目录结构：
    data/cifar10_imagefolder/
    ├── train/
    │   ├── airplane/
    │   ├── automobile/
    │   └── ...（共 10 类，每类 5000 张）
    └── val/
        ├── airplane/
        └── ...（共 10 类，每类 1000 张）

用法：
    python data/prepare_cifar10.py
    python data/prepare_cifar10.py --out-dir data/cifar10_imagefolder
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10
from tqdm import tqdm

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "cifar10_imagefolder",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "_cifar10_raw",
        help="CIFAR-10 原始文件下载目录",
    )
    return parser.parse_args()


def dump_split(dataset: CIFAR10, out_dir: Path, split: str) -> None:
    split_dir = out_dir / split
    for cls in CLASSES:
        (split_dir / cls).mkdir(parents=True, exist_ok=True)

    counter = {cls: 0 for cls in CLASSES}
    for img, label in tqdm(zip(dataset.data, dataset.targets), total=len(dataset.targets), desc=split):
        cls = CLASSES[label]
        idx = counter[cls]
        Image.fromarray(img).save(split_dir / cls / f"{idx:05d}.png")
        counter[cls] += 1


def main() -> None:
    args = parse_args()

    print("下载 CIFAR-10（如已下载则跳过）...")
    # 替换为清华镜像
    import torchvision.datasets.cifar as _cifar_mod
    _cifar_mod.CIFAR10.url = (
        "https://mirror.tuna.tsinghua.edu.cn/help/dataset/"
        "cifar-10-python.tar.gz"
    )
    train_ds = CIFAR10(root=str(args.raw_dir), train=True, download=True)
    val_ds = CIFAR10(root=str(args.raw_dir), train=False, download=True)

    print(f"写出 ImageFolder 到: {args.out_dir}")
    dump_split(train_ds, args.out_dir, "train")
    dump_split(val_ds, args.out_dir, "val")

    print("完成。目录结构示例：")
    for split in ("train", "val"):
        counts = {cls: len(list((args.out_dir / split / cls).glob("*.png"))) for cls in CLASSES}
        total = sum(counts.values())
        print(f"  {split}/  {total} 张")


if __name__ == "__main__":
    main()
