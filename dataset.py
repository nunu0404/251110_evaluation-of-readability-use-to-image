from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CodeReadabilityImageDataset(Dataset):
    """
    Dataset that pairs rendered code images with readability scores.

    Expects the CSV file to contain columns: `id` and `readability`.
    The image corresponding to each row is resolved as `<image_root>/<id>.png`
    unless the identifier already carries an extension.
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.transform = transform

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.frame = pd.read_csv(self.csv_path)
        required_columns = {"id", "readability"}
        missing = required_columns - set(self.frame.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image_path = self._resolve_image_path(str(row["id"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for id={row['id']}: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        readability = torch.tensor(float(row["readability"]), dtype=torch.float32)
        return image, readability

    def _resolve_image_path(self, identifier: str) -> Path:
        name = identifier
        if not identifier.lower().endswith(".png"):
            name = f"{identifier}.png"
        return self.image_root / name


__all__ = ["CodeReadabilityImageDataset"]
