from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import random

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, kendalltau

from agents_readability import HybridAgent, TextAgent, VisualAgent, VisualOCRTextAgent
from dataset_utils import load_code_readability_merged
from renderer import render_code_to_image
from vision_encoder import load_vision_encoder

_VISION_MODEL = None
_VISUAL_AGENTS: Dict[str, VisualAgent] = {}
_OCR_AGENT: VisualOCRTextAgent | None = None
_TEXT_AGENT: TextAgent | None = None
_HYBRID_AGENT: HybridAgent | None = None
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


def _get_hybrid_agent() -> HybridAgent:
    global _HYBRID_AGENT
    if _HYBRID_AGENT is None:
        _HYBRID_AGENT = HybridAgent(_get_vision_model(), device=_DEVICE)
    return _HYBRID_AGENT


def _get_ocr_agent() -> VisualOCRTextAgent | None:
    global _OCR_AGENT
    if _OCR_AGENT is None:
        try:
            _OCR_AGENT = VisualOCRTextAgent()
        except ImportError:
            _OCR_AGENT = None
    return _OCR_AGENT


def evaluate_readability_multi_agent(code: str, image_path: str, vision_mode: str) -> Dict[str, object]:
    visual = _get_visual_agent(vision_mode)
    text = _get_text_agent()
    hybrid = _get_hybrid_agent()
    layout_result = visual.evaluate(image_path)
    text_result = text.evaluate(code)
    ocr_agent = _get_ocr_agent()
    ocr_result = ocr_agent.evaluate(image_path) if ocr_agent else None
    hybrid_result = hybrid.evaluate(code, image_path)
    return {
        "layout_score": layout_result["layout_score"],
        "layout_reason": layout_result.get("reason"),
        "text_score": text_result["text_score"],
        "text_reason": text_result.get("reason"),
        "ocr_score": ocr_result["text_score"] if ocr_result else None,
        "ocr_reason": ocr_result.get("reason") if ocr_result else None,
        "hybrid_score": hybrid_result["hybrid_score"],
        "hybrid_reason": hybrid_result.get("reason"),
    }


def _ensure_image(code: str, image_path: Path) -> None:
    if image_path.is_file():
        return
    render_code_to_image(code, str(image_path))


def _correlations(y_true: List[float], y_pred: List[float]) -> Tuple[float, float, float]:
    if len(y_true) < 2:
        return 0.0, 0.0, 0.0
    pear = pearsonr(y_true, y_pred)[0]
    spear = spearmanr(y_true, y_pred)[0]
    kend = kendalltau(y_true, y_pred)[0]
    return float(pear), float(spear), float(kend)


def _pairwise_accuracy(y_true: List[float], y_pred: List[float]) -> float:
    n = len(y_true)
    if n < 2:
        return 0.0
    agreements = 0
    comparisons = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff_true = y_true[i] - y_true[j]
            diff_pred = y_pred[i] - y_pred[j]
            if abs(diff_true) < 1e-8:
                continue
            if abs(diff_pred) < 1e-8:
                continue
            comparisons += 1
            if diff_true * diff_pred > 0:
                agreements += 1
    if comparisons == 0:
        return 0.0
    return agreements / comparisons


def _error_stats(y_true: List[float], y_pred: List[float]) -> Tuple[float, float]:
    if not y_true:
        return 0.0, 0.0
    arr_true = np.asarray(y_true, dtype=np.float64)
    arr_pred = np.asarray(y_pred, dtype=np.float64)
    mae = np.mean(np.abs(arr_true - arr_pred))
    rmse = np.sqrt(np.mean((arr_true - arr_pred) ** 2))
    return float(mae), float(rmse)


def _calibrate_scores(reference: List[float], predictions: List[float]) -> List[float]:
    ref = np.asarray(reference, dtype=np.float64)
    preds = np.asarray(predictions, dtype=np.float64)
    if ref.size < 2 or preds.size < 2:
        return preds.tolist()
    std_pred = preds.std()
    if std_pred < 1e-6:
        return preds.tolist()
    std_ref = ref.std() if ref.std() > 0 else std_pred
    calibrated = (preds - preds.mean()) * (std_ref / std_pred) + ref.mean()
    calibrated = np.clip(calibrated, 1.0, 5.0)
    return calibrated.tolist()


def _save_results(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = [
        "id",
        "human_score",
        "layout_score",
        "layout_score_calibrated",
        "text_score",
        "text_score_calibrated",
        "ocr_score",
        "ocr_score_calibrated",
        "hybrid_score",
        "hybrid_score_calibrated",
        "hybrid_reason",
        "layout_reason",
        "text_reason",
        "ocr_reason",
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
    parser.add_argument("--seed", type=int, default=None, help="Seed for random sampling (default: random).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randrange(1_000_000_000)
    full_dataset, _, _ = load_code_readability_merged(split_ratio=(1.0, 0.0, 0.0), seed=seed)
    subset = full_dataset[: min(args.max_samples, len(full_dataset))]
    print(f"Sampling {len(subset)} snippets with seed={seed}")
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    human_scores: List[float] = []
    text_scores: List[float] = []
    image_scores: List[float] = []
    hybrid_scores: List[float] = []
    ocr_scores: List[float] = []
    ocr_truth: List[float] = []

    total = len(subset)
    results_rows: List[Dict[str, object]] = []
    for idx, sample in enumerate(subset, start=1):
        print(f"[{idx}/{total}] Evaluating sample id={sample['id']}", flush=True)
        image_path = image_dir / f"{int(sample['id']):05d}.png"
        _ensure_image(sample["code"], image_path)
        try:
            result = evaluate_readability_multi_agent(sample["code"], str(image_path), vision_mode=args.vision_mode)
        except RuntimeError as err:
            print(f"    !! Evaluation failed: {err}", flush=True)
            continue
        log_line = "    -> layout={:.3f}, text={:.3f}, hybrid={:.3f}".format(
            float(result["layout_score"]),
            float(result["text_score"]),
            float(result["hybrid_score"]),
        )
        if result.get("ocr_score") is not None:
            log_line += ", ocr={:.3f}".format(float(result["ocr_score"]))
        print(log_line, flush=True)
        ocr_score = result.get("ocr_score")
        human_scores.append(float(sample["score"]))
        text_scores.append(float(result["text_score"]))
        image_scores.append(float(result["layout_score"]))
        hybrid_scores.append(float(result["hybrid_score"]))
        if ocr_score is not None:
            ocr_value = float(ocr_score)
            ocr_scores.append(ocr_value)
            ocr_truth.append(float(sample["score"]))
        results_rows.append(
            {
                "id": sample["id"],
                "human_score": float(sample["score"]),
                "layout_score": float(result["layout_score"]),
                "layout_reason": result.get("layout_reason"),
                "text_score": float(result["text_score"]),
                "text_reason": result.get("text_reason"),
                "ocr_score": ocr_score,
                "ocr_reason": result.get("ocr_reason"),
                "hybrid_score": float(result["hybrid_score"]),
                "hybrid_reason": result.get("hybrid_reason"),
            }
        )

    calibrated_text = _calibrate_scores(human_scores, text_scores)
    calibrated_layout = _calibrate_scores(human_scores, image_scores)
    calibrated_hybrid = _calibrate_scores(human_scores, hybrid_scores)
    calibrated_ocr = _calibrate_scores(ocr_truth, ocr_scores) if ocr_scores else None

    ocr_idx = 0
    for idx, row in enumerate(results_rows):
        row["layout_score_calibrated"] = calibrated_layout[idx]
        row["text_score_calibrated"] = calibrated_text[idx]
        row["hybrid_score_calibrated"] = calibrated_hybrid[idx]
        if calibrated_ocr and row["ocr_score"] is not None:
            row["ocr_score_calibrated"] = calibrated_ocr[ocr_idx]
            ocr_idx += 1

    pairs = [
        ("Text-only LLM", calibrated_text, human_scores),
        ("Image-only Vision Judge", calibrated_layout, human_scores),
        ("Hybrid Multimodal Judge", calibrated_hybrid, human_scores),
    ]
    if calibrated_ocr and ocr_truth:
        pairs.append(("OCR + TextAgent Judge", calibrated_ocr, ocr_truth))

    for name, preds, truths in pairs:
        pear, spear, kend = _correlations(truths, preds)
        pair_acc = _pairwise_accuracy(truths, preds)
        mae, rmse = _error_stats(truths, preds)
        print(
            f"{name}: Pearson={pear:.4f}, Spearman={spear:.4f}, KendallTau={kend:.4f}, "
            f"PairAcc={pair_acc:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}"
        )

    if args.results_path:
        _save_results(args.results_path, results_rows)
        print(f"Saved per-sample results to {args.results_path}")


if __name__ == "__main__":
    main()
def _get_ocr_agent() -> VisualOCRTextAgent | None:
    global _OCR_AGENT
    if _OCR_AGENT is None:
        try:
            _OCR_AGENT = VisualOCRTextAgent()
        except ImportError:
            _OCR_AGENT = None
    return _OCR_AGENT
