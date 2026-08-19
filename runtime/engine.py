"""
推理运行时（规划中）。

职责：
- 解析 MobileNetV2 层配置
- 调度 CUDA 算子
- 静态内存池与 buffer 复用
- 与 PyTorch 输出数值对齐验证
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RuntimeConfig:
    input_size: int = 224
    batch_size: int = 1
    use_fp16: bool = False
    use_memory_pool: bool = True


class MobileNetV2Runtime:
    """轻量推理运行时占位实现。"""

    def __init__(self, checkpoint: str, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig()
        self.checkpoint = checkpoint
        self._model: Optional[torch.nn.Module] = None

    def load(self) -> None:
        raise NotImplementedError("CUDA runtime 尚未实现，请先用 PyTorch Eager benchmark")

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("CUDA runtime 尚未实现")
