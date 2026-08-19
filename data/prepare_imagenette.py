#!/usr/bin/env python3
"""
下载 ImageNette（ImageNet 10 类子集）并整理为 ImageFolder 格式。

输出目录结构：
    data/imagenette/
    ├── train/
    │   ├── tench/
    │   ├── English_springer/
    │   └── ...（共 10 类，约 9469 张）
    └── val/
        ├── tench/
        └── ...（共 10 类，约 3925 张）

用法：
    python data/prepare_imagenette.py
    python data/prepare_imagenette.py --size 320   # 下载 320px 版本（默认 full）
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

# ImageNette 官方下载地址（fastai）
URLS = {
    "full": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
    "320":  "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    "160":  "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
}

# WordNet ID → 可读类名
WNID_TO_NAME = {
    "n01440764": "tench",
    "n02102040": "English_springer",
    "n02979186": "cassette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "French_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        choices=["full", "320", "160"],
        default="320",
        help="下载尺寸版本（320 推荐，下载约 600 MB）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "imagenette",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "_imagenette_raw",
    )
    return parser.parse_args()


class DownloadProgress(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载: {url}")
    with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)


def extract(tar_path: Path, out_dir: Path) -> Path:
    print(f"解压: {tar_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(out_dir)
    # 解压后目录名如 imagenette2-320
    extracted = [p for p in out_dir.iterdir() if p.is_dir()]
    assert extracted, "解压失败，未找到子目录"
    return extracted[0]


def reorganize(src_dir: Path, out_dir: Path) -> None:
    """将 wnid 子目录重命名为可读类名，并输出 ImageFolder 结构。"""
    for split in ("train", "val"):
        split_src = src_dir / split
        if not split_src.is_dir():
            raise FileNotFoundError(f"未找到 {split_src}")
        for wnid_dir in sorted(split_src.iterdir()):
            if not wnid_dir.is_dir():
                continue
            cls_name = WNID_TO_NAME.get(wnid_dir.name, wnid_dir.name)
            dst = out_dir / split / cls_name
            if dst.exists():
                print(f"  跳过（已存在）: {dst}")
                continue
            print(f"  {split}/{wnid_dir.name} → {split}/{cls_name}  ", end="")
            shutil.copytree(wnid_dir, dst)
            count = len(list(dst.rglob("*.JPEG"))) + len(list(dst.rglob("*.jpg")))
            print(f"({count} 张)")


def main() -> None:
    args = parse_args()
    url = URLS[args.size]
    tar_path = args.raw_dir / Path(url).name

    if not tar_path.exists():
        download(url, tar_path)
    else:
        print(f"已有压缩包，跳过下载: {tar_path}")

    extracted_dir = args.raw_dir / tar_path.stem.replace(".tgz", "").replace(".tar", "")
    # 兼容 imagenette2-320.tgz → imagenette2-320
    candidates = list(args.raw_dir.glob("imagenette2*"))
    if candidates:
        extracted_dir = candidates[0]
    else:
        extracted_dir = extract(tar_path, args.raw_dir)

    if not extracted_dir.is_dir():
        extracted_dir = extract(tar_path, args.raw_dir)

    print(f"\n整理到: {args.out_dir}")
    reorganize(extracted_dir, args.out_dir)

    print("\n完成。各 split 统计：")
    for split in ("train", "val"):
        imgs = list((args.out_dir / split).rglob("*.JPEG")) + \
               list((args.out_dir / split).rglob("*.jpg"))
        print(f"  {split}: {len(imgs)} 张")

    print(f"\n训练命令：")
    print(f"  python train/train_classifier.py --data-dir {args.out_dir}")


if __name__ == "__main__":
    main()
