from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from datasets import load_from_disk

from renderer import render_code_to_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HF dataset to CSV splits and render images.")
    parser.add_argument(
        "--dataset-path",
        default="code_readability_merged_dataset",
        help="Path to the HuggingFace dataset saved via save_to_disk.",
    )
    parser.add_argument("--output-dir", default="data", help="Directory to store generated CSV files.")
    parser.add_argument("--image-root", default="images", help="Directory to store rendered PNG images.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of data for training split.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of data for validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling before splitting.")
    parser.add_argument("--id-prefix", default="sample_", help="Prefix for generated sample identifiers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.dataset_path)["train"]
    df = dataset.to_pandas()

    if "code_snippet" not in df.columns or "score" not in df.columns:
        raise ValueError("Dataset must contain 'code_snippet' and 'score' columns.")

    df = df.rename(columns={"code_snippet": "code", "score": "readability"})
    df["id"] = [f"{args.id_prefix}{i:05d}" for i in range(len(df))]
    df = df[["id", "code", "readability"]]
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    train_df, val_df, test_df = _split_dataframe(df, args.train_ratio, args.val_ratio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    image_root = Path(args.image_root)
    image_root.mkdir(parents=True, exist_ok=True)

    for row in df.itertuples(index=False):
        output_path = image_root / f"{row.id}.png"
        if output_path.is_file():
            continue
        render_code_to_image(str(row.code), str(output_path))


def _split_dataframe(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


if __name__ == "__main__":
    main()
