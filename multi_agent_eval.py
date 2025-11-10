from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from agents_readability import AggregatorAgent, TextAgent, VisualAgent
from dataset_utils import load_code_readability_merged
from renderer import render_code_to_image
from vision_encoder import load_vision_encoder

_VISION_MODEL = None
_VISUAL_AGENTS: Dict[str, VisualAgent] = {}
_TEXT_AGENT: TextAgent | None = None
_AGGREGATOR: AggregatorAgent | None = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_visual_agent(mode: str) -> VisualAgent:
    if mode not in {"multimodal", "heuristic"}:
        mode = "multimodal"
    agent = _VISUAL_AGENTS.get(mode)
    if agent is None:
        agent = VisualAgent(_get_vision_model(), device=_DEVICE, mode=mode)
        _VISUAL_AGENTS[mode] = agent
    return agent


def _get_vision_model():
    global _VISION_MODEL
    if _VISION_MODEL is None:
        _VISION_MODEL = load_vision_encoder(device=_DEVICE)
    return _VISION_MODEL


def _get_text_agent() -> TextAgent:
    global _TEXT_AGENT
    if _TEXT_AGENT is None:
        _TEXT_AGENT = TextAgent()
    return _TEXT_AGENT


def _get_aggregator() -> AggregatorAgent:
    global _AGGREGATOR
    if _AGGREGATOR is None:
        _AGGREGATOR = AggregatorAgent()
    return _AGGREGATOR


def evaluate_readability_multi_agent(code: str, image_path: str, vision_mode: str) -> Dict[str, object]:
    visual = _get_visual_agent(vision_mode)
    text = _get_text_agent()
    aggregator = _get_aggregator()
    layout_result = visual.evaluate(image_path)
    text_result = text.evaluate(code)
    final_result = aggregator.aggregate(layout_result, text_result)
    return {
        "layout_score": layout_result["layout_score"],
        "layout_reason": layout_result["reason"],
        "text_score": text_result["text_score"],
        "text_reason": text_result["reason"],
        "final_score": final_result["final_score"],
        "final_reason": final_result["reason"],
    }


def _ensure_image(code: str, image_path: Path) -> None:
    if image_path.is_file():
        return
    render_code_to_image(code, str(image_path))


def _correlations(y_true: List[float], y_pred: List[float]) -> Tuple[float, float]:
    if len(y_true) < 2:
        return 0.0, 0.0
    pear = pearsonr(y_true, y_pred)[0]
    spear = spearmanr(y_true, y_pred)[0]
    return float(pear), float(spear)


def _save_results(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = [
        "id",
        "human_score",
        "layout_score",
        "text_score",
        "final_score",
        "layout_reason",
        "text_reason",
        "final_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate readability with multi-agent LLM pipeline.")
    parser.add_argument("--max-samples", type=int, default=50, help="Number of test samples to evaluate.")
    parser.add_argument("--image-dir", default="rendered_eval_images", help="Directory for cached code images.")
    parser.add_argument("--results-path", default=None, help="Optional CSV path to save per-sample results.")
    parser.add_argument("--vision-mode", choices=["multimodal", "heuristic"], default="multimodal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, _, test_split = load_code_readability_merged()
    subset = test_split[: args.max_samples]
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    human_scores: List[float] = []
    text_scores: List[float] = []
    image_scores: List[float] = []
    multi_scores: List[float] = []

    total = len(subset)
    results_rows: List[Dict[str, object]] = []
    for idx, sample in enumerate(subset, start=1):
        print(f"[{idx}/{total}] Evaluating sample id={sample['id']}", flush=True)
        image_path = image_dir / f"{int(sample['id']):05d}.png"
        _ensure_image(sample["code"], image_path)
        result = evaluate_readability_multi_agent(sample["code"], str(image_path), vision_mode=args.vision_mode)
        print(
            "    -> layout={:.3f}, text={:.3f}, final={:.3f}".format(
                float(result["layout_score"]), float(result["text_score"]), float(result["final_score"])
            ),
            flush=True,
        )
        human_scores.append(float(sample["score"]))
        text_scores.append(float(result["text_score"]))
        image_scores.append(float(result["layout_score"]))
        multi_scores.append(float(result["final_score"]))
        results_rows.append(
            {
                "id": sample["id"],
                "human_score": float(sample["score"]),
                "layout_score": float(result["layout_score"]),
                "text_score": float(result["text_score"]),
                "final_score": float(result["final_score"]),
                "layout_reason": result["layout_reason"],
                "text_reason": result["text_reason"],
                "final_reason": result["final_reason"],
            }
        )

    pairs = [
        ("Text-only LLM", text_scores),
        ("Image-only ResNet proxy", image_scores),
        ("Multi-Agent LLM Judge", multi_scores),
    ]

    for name, preds in pairs:
        pear, spear = _correlations(human_scores, preds)
        print(f"{name}: Pearson={pear:.4f}, Spearman={spear:.4f}")

    if args.results_path:
        _save_results(args.results_path, results_rows)
        print(f"Saved per-sample results to {args.results_path}")


if __name__ == "__main__":
    main()
