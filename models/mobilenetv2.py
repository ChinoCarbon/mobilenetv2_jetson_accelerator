"""
MobileNetV2 分类器（ImageNet 预训练 backbone + 自定义分类头）。

结构参考: MobileNetV2: Inverted Residuals and Linear Bottlenecks
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class MobileNetV2Classifier(nn.Module):
    """MobileNetV2 + 线性分类头。"""

    def __init__(
        self,
        num_classes: int,
        width_mult: float = 1.0,
        dropout: float = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights, width_mult=width_mult)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        self._init_classifier()

    def _init_classifier(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_classifier(
    num_classes: int,
    width_mult: float = 1.0,
    dropout: float = 0.2,
    pretrained: bool = False,
) -> MobileNetV2Classifier:
    return MobileNetV2Classifier(
        num_classes=num_classes,
        width_mult=width_mult,
        dropout=dropout,
        pretrained=pretrained,
    )
