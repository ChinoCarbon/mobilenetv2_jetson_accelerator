#!/usr/bin/env python3
"""MobileNetV2 缺陷分类训练脚本。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 classifier")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="配置文件路径",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="数据集根目录")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_transforms(input_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.14)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_loaders(
    data_dir: Path,
    input_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"训练集目录不存在: {train_dir}\n"
            f"请按 data/README.md 准备数据集后重试。"
        )
    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"验证集目录不存在: {val_dir}\n"
            f"请按 data/README.md 准备数据集后重试。"
        )

    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(input_size, True))
    val_ds = datasets.ImageFolder(val_dir, transform=build_transforms(input_size, False))

    if train_ds.classes != val_ds.classes:
        raise ValueError("train 与 val 的类别不一致")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = correct / max(total, 1)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    avg_loss = total_loss / max(total, 1)
    return avg_loss, acc, f1


def save_checkpoint(
    path: Path,
    model: nn.Module,
    classes: list[str],
    args: dict,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "args": args,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_dir = Path(args.data_dir or cfg["data"]["dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    input_size = args.input_size or cfg["data"]["input_size"]
    epochs = args.epochs or cfg["train"]["epochs"]
    batch_size = args.batch_size or cfg["train"]["batch_size"]
    lr = args.lr or cfg["train"]["lr"]
    pretrained = cfg["train"]["pretrained"] and not args.no_pretrained
    checkpoint_dir = ROOT / cfg["train"]["checkpoint_dir"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")

    train_loader, val_loader, classes = build_loaders(
        data_dir,
        input_size,
        batch_size,
        cfg["data"]["num_workers"],
    )
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    model = build_classifier(
        num_classes=num_classes,
        width_mult=cfg["model"]["width_mult"],
        dropout=cfg["model"]["dropout"],
        pretrained=pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0.0
    run_args = {
        "data_dir": str(data_dir),
        "input_size": input_size,
        "num_classes": num_classes,
        "pretrained": pretrained,
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, criterion)
        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f}"
        )

        metrics = {"val_acc": val_acc, "val_f1": val_f1, "val_loss": val_loss, "epoch": epoch}
        save_checkpoint(checkpoint_dir / "last.pt", model, classes, run_args, metrics)

        if val_f1 >= best_f1:
            best_f1 = val_f1
            save_checkpoint(checkpoint_dir / "best.pt", model, classes, run_args, metrics)
            print(f"  -> saved best checkpoint (F1={best_f1:.4f})")

    meta = {
        "classes": classes,
        "best_f1": best_f1,
        "input_size": input_size,
        "num_classes": num_classes,
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Training done. Best F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
