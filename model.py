from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(pretrained: bool = True) -> nn.Module:
    """
    Construct a ResNet18-based regressor for readability prediction.

    Args:
        pretrained: Whether to start from ImageNet-pretrained weights.

    Returns:
        A torch.nn.Module ready for regression training.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


__all__ = ["build_model"]
