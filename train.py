from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CodeReadabilityImageDataset
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train readability regressors on code images.")
    parser.add_argument("--csv-train", required=True, help="Path to training CSV file.")
    parser.add_argument("--csv-val", required=True, help="Path to validation CSV file.")
    parser.add_argument("--csv-test", required=True, help="Path to test CSV file.")
    parser.add_argument("--image-root", required=True, help="Root directory that stores rendered images.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW.")
    parser.add_argument("--save-path", required=True, help="Destination to save the best model checkpoint.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes for data loading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = CodeReadabilityImageDataset(args.csv_train, args.image_root, transform=common_transform)
    val_dataset = CodeReadabilityImageDataset(args.csv_val, args.image_root, transform=common_transform)
    test_dataset = CodeReadabilityImageDataset(args.csv_test, args.image_root, transform=common_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_mse = math.inf
    os.makedirs(Path(args.save_path).parent, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, len(train_dataset))
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MSE: {val_metrics['mse']:.4f}, "
            f"MAE: {val_metrics['mae']:.4f}, "
            f"Pearson: {val_metrics['pearson']:.4f}, "
            f"Spearman: {val_metrics['spearman']:.4f}"
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                args.save_path,
            )

    if Path(args.save_path).is_file():
        checkpoint = torch.load(args.save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    test_metrics = evaluate(model, test_loader, device)
    print(
        "Test metrics - "
        f"MSE: {test_metrics['mse']:.4f}, "
        f"MAE: {test_metrics['mae']:.4f}, "
        f"Pearson: {test_metrics['pearson']:.4f}, "
        f"Spearman: {test_metrics['spearman']:.4f}"
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dataset_size: int,
) -> float:
    model.train()
    running_loss = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / max(1, dataset_size)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).squeeze(1)
            predictions.extend(outputs.cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return compute_metrics(np.array(targets, dtype=np.float64), np.array(predictions, dtype=np.float64))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    pearson = _pearsonr(y_true, y_pred)
    spearman = _spearmanr(y_true, y_pred)
    return {"mse": mse, "mae": mae, "pearson": pearson, "spearman": spearman}


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean
    numerator = float(np.sum(x_centered * y_centered))
    denominator = math.sqrt(float(np.sum(x_centered**2)) * float(np.sum(y_centered**2)))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    x_ranked = _rankdata(x)
    y_ranked = _rankdata(y)
    return _pearsonr(x_ranked, y_ranked)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    unique_vals, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            mask = inverse == idx
            ranks[mask] = ranks[mask].mean()
    return ranks


if __name__ == "__main__":
    main()
