from __future__ import annotations

from functools import lru_cache
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_vision_encoder(device: str = "cpu") -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)


def encode_image(model: nn.Module, image_path: str, device: str = "cpu") -> List[float]:
    image = Image.open(image_path).convert("RGB")
    tensor = _TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor)
    return features.squeeze(0).cpu().tolist()


__all__ = ["load_vision_encoder", "encode_image"]
