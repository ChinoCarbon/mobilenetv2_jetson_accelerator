#!/usr/bin/env python3
"""
整理 NEU-DET 数据集为 ImageFolder 分类格式。

NEU-DET 原始结构（需手动下载后放入 --raw-dir）：
    NEU-DET/
    └── IMAGES/
        ├── crazing_1.jpg ... crazing_300.jpg
        ├── inclusion_1.jpg ... inclusion_300.jpg
        ├── patches_1.jpg ... patches_300.jpg
        ├── pitted_1.jpg ... pitted_300.jpg
        ├── rolled-in_scale_1.jpg ... rolled-in_scale_300.jpg
        └── scratches_1.jpg ... scratches_300.jpg

下载地址（需注册或直接搜索）：
    http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

输出结构（8:2 train/val 划分）：
    data/neu_det_cls/
    ├── train/
    │   ├── crazing/     (240 张)
    │   ├── inclusion/   (240 张)
    │   └── ...
    └── val/
        ├── crazing/     (60 张)
        └── ...

用法：
    python data/prepare_neu_det.py --raw-dir /path/to/NEU-DET
    python data/prepare_neu_det.py --raw-dir /path/to/NEU-DET --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CLASSES = ["crazing", "inclusion", "patches", "pitted", "rolled-in_scale", "scratches"]

# 文件名前缀 → 统一类名（输出目录名）
PREFIX_TO_CLASS = {
    "crazing":       "crazing",
    "inclusion":     "inclusion",
    "patches":       "patches",
    "pitted":        "pitted",
    "rolled-in_scale": "rolled",   # 简化目录名
    "scratches":     "scratches",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="NEU-DET 根目录（含 IMAGES/ 子目录）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "neu_det_cls",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_images_dir(raw_dir: Path) -> Path:
    """兼容多种解压结构。"""
    for candidate in [raw_dir, raw_dir / "IMAGES", raw_dir / "NEU-DET" / "IMAGES"]:
        if candidate.is_dir() and any(candidate.glob("*.jpg")):
            return candidate
    raise FileNotFoundError(
        f"在 {raw_dir} 下未找到包含 jpg 的图像目录。\n"
        f"请将 NEU-DET 解压后，用 --raw-dir 指向含 IMAGES/ 的根目录。"
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    img_dir = find_images_dir(args.raw_dir)
    print(f"图像源目录: {img_dir}")

    # 按前缀分组
    groups: dict[str, list[Path]] = {cls: [] for cls in PREFIX_TO_CLASS.values()}
    for img in sorted(img_dir.glob("*.jpg")):
        for prefix, cls in PREFIX_TO_CLASS.items():
            if img.name.startswith(prefix):
                groups[cls].append(img)
                break

    print("\n各类图像数量:")
    for cls, imgs in groups.items():
        print(f"  {cls}: {len(imgs)}")

    # 划分并复制
    print(f"\n输出至: {args.out_dir}  (val_ratio={args.val_ratio})")
    stats: dict[str, dict[str, int]] = {"train": {}, "val": {}}

    for cls, imgs in groups.items():
        if not imgs:
            print(f"  警告: {cls} 无图像，跳过")
            continue
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * args.val_ratio))
        splits = {"val": imgs[:n_val], "train": imgs[n_val:]}

        for split, split_imgs in splits.items():
            dst_dir = args.out_dir / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in split_imgs:
                shutil.copy2(src, dst_dir / src.name)
            stats[split][cls] = len(split_imgs)

    print("\n划分结果:")
    for split in ("train", "val"):
        total = sum(stats[split].values())
        print(f"  {split}: {total} 张  " + ", ".join(f"{c}={n}" for c, n in stats[split].items()))

    print(f"\n训练命令:")
    print(f"  python train/train_classifier.py --data-dir {args.out_dir}")


if __name__ == "__main__":
    main()
