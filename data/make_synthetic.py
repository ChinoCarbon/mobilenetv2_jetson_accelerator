#!/usr/bin/env python3
"""
生成合成分类数据集（随机图像），用于验证训练流程正确性。
不需要网络，秒级生成。

用法：
    python data/make_synthetic.py
    python data/make_synthetic.py --num-classes 10 --train 500 --val 100 --size 224
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

CLASSES = [f"class_{i:02d}" for i in range(10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--train", type=int, default=200, help="每类训练集数量")
    parser.add_argument("--val", type=int, default=50, help="每类验证集数量")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_split(out_dir: Path, classes: list[str], n: int, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for i, cls in enumerate(classes):
        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        # 每类用不同基底颜色，让模型能学到区分
        base = np.array([i * 25 % 255, (i * 40 + 50) % 255, (i * 60 + 100) % 255], dtype=np.int16)
        for j in range(n):
            noise = rng.integers(-30, 31, size=(size, size, 3), dtype=np.int16)
            img_arr = np.clip(base + noise, 0, 255).astype(np.uint8)
            Image.fromarray(img_arr).save(cls_dir / f"{j:05d}.png")


def main() -> None:
    args = parse_args()
    classes = CLASSES[: args.num_classes]

    print(f"生成合成数据集 → {args.out_dir}")
    print(f"  类别: {classes}")
    print(f"  train: {args.train} 张/类，val: {args.val} 张/类，尺寸: {args.size}x{args.size}")

    make_split(args.out_dir / "train", classes, args.train, args.size, args.seed)
    make_split(args.out_dir / "val", classes, args.val, args.size, args.seed + 1)

    total_train = args.num_classes * args.train
    total_val = args.num_classes * args.val
    print(f"完成：train {total_train} 张，val {total_val} 张")
    print(f"\n训练命令：")
    print(f"  python train/train_classifier.py --data-dir {args.out_dir}")


if __name__ == "__main__":
    main()
