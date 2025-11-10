from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from datasets import load_dataset


def load_code_readability_merged(
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    train_ratio, val_ratio, test_ratio = split_ratio
    if any(r <= 0 for r in split_ratio) or abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError("split_ratio must be positive and sum to 1.")

    dataset = load_dataset("se2p/code-readability-merged", split="train")
    samples = []
    for idx, record in enumerate(dataset):
        samples.append(
            {
                "id": idx,
                "code": record["code_snippet"],
                "score": float(record["score"]),
            }
        )

    rng = random.Random(seed)
    rng.shuffle(samples)

    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_split = samples[:train_end]
    val_split = samples[train_end:val_end]
    test_split = samples[val_end:]
    return train_split, val_split, test_split


__all__ = ["load_code_readability_merged"]
