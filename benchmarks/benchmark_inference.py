#!/usr/bin/env python3
"""PyTorch Eager 推理 benchmark（后续可扩展自定义 CUDA runtime）。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MobileNetV2 inference")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: Path, device: torch.device, cfg: dict) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = ckpt["args"]["num_classes"]
    model = build_classifier(
        num_classes=num_classes,
        width_mult=cfg["model"]["width_mult"],
        dropout=0.0,
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


@torch.no_grad()
def benchmark(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    latencies: list[float] = []
    for _ in range(iterations):
        start.record()
        model(x)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

    latencies.sort()
    n = len(latencies)
    return {
        "mean_ms": sum(latencies) / n,
        "p50_ms": latencies[int(n * 0.50)],
        "p95_ms": latencies[int(n * 0.95)],
        "p99_ms": latencies[int(n * 0.99)],
        "throughput_fps": 1000.0 * x.size(0) / (sum(latencies) / n),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if not torch.cuda.is_available():
        print("CUDA 不可用，将使用 CPU（计时仅供参考）")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    input_size = args.input_size or cfg["data"]["input_size"]
    warmup = args.warmup or cfg["benchmark"]["warmup"]
    iterations = args.iterations or cfg["benchmark"]["iterations"]

    model = load_model(args.checkpoint, device, cfg)
    x = torch.randn(args.batch_size, 3, input_size, input_size, device=device)

    if device.type == "cuda":
        stats = benchmark(model, x, warmup, iterations)
    else:
        t0 = time.perf_counter()
        for _ in range(warmup + iterations):
            model(x)
        elapsed = time.perf_counter() - t0
        avg = elapsed / iterations * 1000
        stats = {"mean_ms": avg, "p50_ms": avg, "p95_ms": avg, "p99_ms": avg, "throughput_fps": 1000 / avg}

    print(f"Backend: PyTorch Eager")
    print(f"Device: {device}")
    print(f"Input: [{args.batch_size}, 3, {input_size}, {input_size}]")
    print(f"Warmup: {warmup}, Iterations: {iterations}")
    print("---")
    for k, v in stats.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()
