from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Tuple, Optional
from statistics import mean

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from llm_client import generate_chat_completion, generate_multimodal_completion
from vision_encoder import encode_image
from code_metrics import analyze_code

try:
    import pytesseract
except ImportError:
    pytesseract = None


# ---------------------------------------------------------------------------#
# Utility helpers
# ---------------------------------------------------------------------------#
def _safe_json_parse(payload: str) -> Dict[str, object]:
    if not payload or payload.startswith("ERROR"):
        return {}
    start, end = payload.find("{"), payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    snippet = payload[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return {}


def _clamp_score(value: float, default: float = 3.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return float(np.clip(round(score, 2), 1.0, 5.0))


def _summarize_features(values: Dict[str, float]) -> str:
    return "\n".join(f"- {k}: {v:.3f}" for k, v in values.items())


# ---------------------------------------------------------------------------#
# Visual feature extraction
# ---------------------------------------------------------------------------#
def _load_gray_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = ImageOps.equalize(img.filter(ImageFilter.SMOOTH))
    return np.array(img, dtype=np.float32) / 255.0


def _text_mask(arr: np.ndarray) -> np.ndarray:
    threshold = np.percentile(arr, 40)
    return (arr < threshold).astype(np.uint8)


def _indentation_stats(mask: np.ndarray) -> Tuple[float, float]:
    leading_cols = []
    for row in mask:
        idx = np.where(row > 0)[0]
        if len(idx):
            leading_cols.append(idx[0])
    if not leading_cols:
        return 0.0, 0.0
    return float(np.mean(leading_cols)), float(np.std(leading_cols))


def _whitespace_clusters(mask: np.ndarray) -> float:
    col_profile = mask.sum(axis=0) / (mask.shape[0] + 1e-6)
    diff = np.diff(col_profile)
    return float(np.mean(np.abs(diff)))


def _block_spacing(mask: np.ndarray) -> float:
    row_profile = mask.sum(axis=1)
    blank_runs, run = [], 0
    for count in row_profile:
        if count == 0:
            run += 1
        elif run:
            blank_runs.append(run)
            run = 0
    if run:
        blank_runs.append(run)
    if not blank_runs:
        return 0.0
    return float(np.mean(blank_runs))


def _line_width_variance(mask: np.ndarray) -> float:
    widths = []
    for row in mask:
        cols = np.where(row > 0)[0]
        widths.append(len(cols))
    return float(np.std(widths))


def _staircase_signal(mask: np.ndarray) -> float:
    positions = []
    step = max(1, mask.shape[0] // 200)
    for row in mask[::step]:
        idx = np.where(row > 0)[0]
        if len(idx):
            positions.append(idx[0])
    if len(positions) < 5:
        return 0.0
    diffs = np.diff(positions)
    return float(np.mean(diffs > 2) * np.std(diffs))


def _structure_attention(path: str, encoder=None, device: str = "cpu") -> Dict[str, float]:
    if encoder is None:
        return {
            "embedding_mean": 0.0,
            "embedding_std": 0.0,
            "embedding_max": 0.0,
            "embedding_min": 0.0,
            "high_energy_ratio": 0.0,
        }
    vec = np.asarray(encode_image(encoder, path, device=device), dtype=np.float32)
    return {
        "embedding_mean": float(vec.mean()),
        "embedding_std": float(vec.std()),
        "embedding_max": float(vec.max()),
        "embedding_min": float(vec.min()),
        "high_energy_ratio": float(np.mean(vec > vec.mean() + vec.std())),
    }


def _visual_descriptors(path: str, encoder=None, device: str = "cpu") -> Dict[str, float]:
    arr = _load_gray_image(path)
    mask = _text_mask(arr)
    indent_mean, indent_std = _indentation_stats(mask)
    descriptors = {
        "indent_mean": indent_mean,
        "indent_std": indent_std,
        "whitespace_cluster_score": _whitespace_clusters(mask),
        "block_spacing_avg": _block_spacing(mask),
        "line_width_std": _line_width_variance(mask),
        "staircase_signal": _staircase_signal(mask),
        "ink_ratio": float(mask.mean()),
    }
    descriptors.update(_structure_attention(path, encoder=encoder, device=device))
    return descriptors


def _qualitative_bucket(value: float, bands: Tuple[float, float, float]) -> str:
    low, mid, high = bands
    if value < low:
        return "low"
    if value < mid:
        return "moderate"
    if value < high:
        return "high"
    return "very high"


def _visual_feature_summary(features: Dict[str, float]) -> str:
    return "\n".join(
        [
            f"- indent spread {features.get('indent_std', 0.0):.1f}px ({_qualitative_bucket(features.get('indent_std', 0.0), (4.0, 12.0, 25.0))})",
            f"- whitespace change {features.get('whitespace_cluster_score', 0.0):.3f}",
            f"- block spacing {features.get('block_spacing_avg', 0.0):.1f} rows",
            f"- staircase {features.get('staircase_signal', 0.0):.3f}",
            f"- ink ratio {features.get('ink_ratio', 0.0):.3f}",
        ]
    )


def _text_metric_summary(metrics: Dict[str, float]) -> str:
    keys = [
        "avg_line_length",
        "indentation_std",
        "comment_ratio",
        "cyclomatic_proxy",
        "nesting_mean",
        "snake_case_ratio",
    ]
    return "\n".join(f"- {k}: {metrics.get(k, 0.0):.3f}" for k in keys)


def _text_highlights(metrics: Dict[str, float]) -> List[str]:
    highlights: List[str] = []
    if metrics.get("avg_line_length", 0.0) > 90:
        highlights.append("Lines exceed 90 characters on average, hurting readability.")
    elif metrics.get("avg_line_length", 0.0) < 45:
        highlights.append("Lines stay compact (<45 chars), aiding quick scanning.")
    if metrics.get("indentation_std", 0.0) > 6.0:
        highlights.append("Indentation varies widely, indicating uneven nesting.")
    if metrics.get("comment_ratio", 0.0) < 0.03:
        highlights.append("Almost no comments are present.")
    elif metrics.get("comment_ratio", 0.0) > 0.20:
        highlights.append("Comments are frequent and can aid comprehension.")
    if metrics.get("cyclomatic_proxy", 1.0) > 12:
        highlights.append("Control flow is complex with many branches/loops.")
    if metrics.get("nesting_mean", 0.0) > 2.5 or metrics.get("deep_ratio", 0.0) > 0.25:
        highlights.append("Deep nesting occurs often; consider flattening logic.")
    if metrics.get("uppercase_id_ratio", 0.0) > 0.3:
        highlights.append("Identifiers rely heavily on uppercase naming.")
    if metrics.get("snake_case_ratio", 0.0) < 0.15:
        highlights.append("Snake_case naming is rare, reducing consistency.")
    if not highlights:
        highlights.append("No extreme metric values detected; readability depends on naming and logic clarity.")
    return highlights


def _summarize_checklist(label: str, scores: Dict[str, float]) -> str:
    if not scores:
        return f"{label} 평가 근거 없음."
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ", ".join(f"{k}={v:.1f}" for k, v in ordered[:2])
    bottom = ", ".join(f"{k}={v:.1f}" for k, v in ordered[-2:])
    return f"{label} 강점({top}), 취약({bottom})."


# ---------------------------------------------------------------------------#
# OCR helpers
# ---------------------------------------------------------------------------#
def _llm_ocr(path: str) -> Tuple[Optional[str], float]:
    prompt = (
        "Transcribe the attached code screenshot. "
        "Preserve indentation, spaces, and blank lines exactly. "
        "Return ONLY the code inside triple backticks."
    )
    response = generate_multimodal_completion(
        system_prompt="You are a meticulous code transcription agent.",
        text_prompt=prompt,
        image_path=path,
        temperature=0.1,
    )
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).rstrip(), 0.95
    if response.strip():
        return response.strip(), 0.6
    return None, 0.0


def _tesseract_fallback(path: str) -> Optional[str]:
    if pytesseract is None:
        return None
    img = ImageOps.autocontrast(Image.open(path).convert("L"))
    text = pytesseract.image_to_string(img, config="--psm 6 preserve_interword_spaces=1")
    return text if text.strip() else None


# ---------------------------------------------------------------------------#
# Textual feature extraction
# ---------------------------------------------------------------------------#
def _extra_code_metrics(code: str) -> Dict[str, float]:
    lines = code.splitlines()
    stripped = [ln for ln in lines if ln.strip()]
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|&&|\|\||[+\-*/%=<>]", code)
    numbers = re.findall(r"\b\d+(\.\d+)?\b", code)
    depth_trace, depth = [], 0
    for ch in code:
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth = max(0, depth - 1)
        depth_trace.append(depth)
    indentation = [len(ln) - len(ln.lstrip(" \t")) for ln in stripped]
    comment_lines = sum(ln.strip().startswith(("//", "#")) for ln in lines)
    return {
        "line_count": len(lines),
        "non_empty_ratio": len(stripped) / (len(lines) + 1e-6),
        "avg_line_length": mean(map(len, stripped)) if stripped else 0.0,
        "line_length_variance": float(np.var(list(map(len, stripped))) if stripped else 0.0),
        "indentation_std": float(np.std(indentation) if indentation else 0.0),
        "comment_ratio": comment_lines / (len(lines) + 1e-6),
        "token_density": len(tokens) / (len(stripped) + 1e-6),
        "numeric_ratio": len(numbers) / (len(tokens) + 1e-6),
        "branching_ratio": sum(tok in {"if", "else", "for", "while", "case"} for tok in tokens)
        / (len(tokens) + 1e-6),
        "operator_density": sum(tok in {"+", "-", "*", "/", "==", "!=", "<=", ">="} for tok in tokens)
        / (len(tokens) + 1e-6),
        "nesting_mean": float(np.mean(depth_trace)),
        "nesting_std": float(np.std(depth_trace)),
        "deep_ratio": sum(d > 3 for d in depth_trace) / (len(depth_trace) + 1e-6),
        "cyclomatic_proxy": 1
        + sum(tok in {"if", "else if", "for", "while", "catch", "case"} for tok in tokens),
        "identifier_len_mean": mean(len(tok) for tok in tokens if tok.isidentifier()) if tokens else 0.0,
        "uppercase_id_ratio": sum(tok.isupper() for tok in tokens if tok.isidentifier())
        / (len(tokens) + 1e-6),
        "snake_case_ratio": sum("_" in tok for tok in tokens if tok.isidentifier())
        / (len(tokens) + 1e-6),
    }


class VisualAgent:
    def __init__(self, vision_model, device: str = "cpu", mode: str = "multimodal") -> None:
        self.model = vision_model
        self.device = device
        self.mode = mode

    def evaluate(self, image_path: str) -> Dict[str, object]:
        features = _visual_descriptors(image_path, encoder=self.model, device=self.device)
        feature_summary = _visual_feature_summary(features)
        prompt = (
            "Judge the layout readability of the attached code screenshot. "
            "Rely only on visual cues from the PNG: indentation staircase, whitespace spacing, block grouping, alignment, "
            "and overall visual nesting hints. Provide 1.00-5.00 scores for indentation clarity, block grouping, "
            "whitespace balance, line-width moderation, and visual nesting cues, then summarize with a concise layout_score "
            "and confidence. Do NOT reuse example numbers; infer them from the actual image plus measurements. "
            "Respond ONLY with JSON structured as "
            "{\"layout_score\": VALUE, \"reason\": \"...\", \"checklist\": {\"indentation\": VALUE, \"block_grouping\": VALUE, "
            "\"whitespace\": VALUE, \"line_width\": VALUE, \"nesting_visual\": VALUE}, \"confidence\": VALUE}. "
            "\nMeasurements derived from the image:\n"
            f"{feature_summary}"
        )
        response = generate_multimodal_completion(
            system_prompt="You are a meticulous visual readability critic.",
            text_prompt=prompt,
            image_path=image_path,
            temperature=0.3,
        )
        data = _safe_json_parse(response)
        if not data or "layout_score" not in data:
            raise RuntimeError("VisualAgent LLM response missing layout_score")
        data["layout_score"] = _clamp_score(data.get("layout_score"), 3.0)
        data["confidence"] = float(np.clip(data.get("confidence", 0.65), 0.15, 0.95))
        checklist = data.get("checklist", {}) or {}
        data["checklist"] = checklist
        data.setdefault("reason", "Visual assessment derived solely from the screenshot and measurements.")
        return data

class TextAgent:
    def evaluate(self, code: str) -> Dict[str, object]:
        base = analyze_code(code)
        extra = _extra_code_metrics(code)
        merged = {**base, **extra}
        metric_summary = _text_metric_summary(merged)
        highlights = _text_highlights(merged)
        highlight_text = "\n".join(f"- {msg}" for msg in highlights)
        prompt = (
            "You are an expert readability reviewer. "
            "Consider naming, indentation discipline, nesting transitions, block length variation, comment density, "
            "and overall cognitive load. Use the numeric metrics to justify a broad use of the 1-5 scale. "
            "Scores must reflect the provided measurements (e.g., long lines -> lower scores, rich comments -> higher scores); "
            "do not repeat the same numeric pattern across different snippets, and use at least two decimal places for each score. "
            "Return ONLY JSON such as "
            "{\"text_score\": SCORE, \"reason\": \"...\", "
            "\"checklist\": {\"naming\": SCORE, \"indentation\": SCORE, "
            "\"cognitive_load\": SCORE, \"structure\": SCORE, \"documentation\": SCORE}}."
        )
        response = generate_chat_completion(
            system_prompt="You score code readability with nuanced judgement.",
            user_prompt=(
                prompt
                + "\n\nMetric highlights:\n"
                + highlight_text
                + "\n\nMetrics:\n"
                + metric_summary
                + "\n\nCode:\n"
                + code
            ),
            temperature=0.45,
        )
        data = _safe_json_parse(response)
        if not data or "text_score" not in data:
            raise RuntimeError("TextAgent LLM response missing text_score")
        data["text_score"] = _clamp_score(data.get("text_score"), 3.0)
        checklist = data.get("checklist", {}) or {}
        checklist.setdefault("naming", float(np.clip(extra.get("snake_case_ratio", 0.0) * 10 + 2.5, 1.0, 5.0)))
        checklist.setdefault("indentation", float(np.clip(5.0 - extra.get("indentation_std", 0.0) / 2.0, 1.0, 5.0)))
        checklist.setdefault(
            "cognitive_load", float(np.clip(5.0 - extra.get("cyclomatic_proxy", 1.0) / 4.0, 1.0, 5.0))
        )
        checklist.setdefault(
            "structure", float(np.clip(5.0 - (extra.get("nesting_std", 0.0) + extra.get("deep_ratio", 0.0) * 5), 1.0, 5.0))
        )
        checklist.setdefault(
            "documentation", float(np.clip(extra.get("comment_ratio", 0.0) * 20 + 1.0, 1.0, 5.0))
        )
        data["checklist"] = checklist
        data.setdefault("reason", _summarize_checklist("text", checklist))
        return data

class VisualOCRTextAgent:
    def evaluate(self, image_path: str) -> Dict[str, object]:
        text, conf = _llm_ocr(image_path)
        if (not text or conf < 0.65) and pytesseract is not None:
            fallback = _tesseract_fallback(image_path)
            if fallback:
                text, conf = fallback, 0.45
        if not text:
            raise RuntimeError("OCR failed")
        text_agent = TextAgent()
        result = text_agent.evaluate(text)
        blended = conf * result["text_score"] + (1 - conf) * 3.0
        result["text_score"] = _clamp_score(blended, 3.0)
        result["ocr_confidence"] = conf
        base_reason = result.get("reason", "").strip()
        ocr_note = f"OCR confidence {conf:.2f}"
        result["reason"] = (base_reason + " " + ocr_note).strip() if base_reason else ocr_note
        return result



class HybridAgent:
    def __init__(self, vision_model, device: str = "cpu") -> None:
        self.model = vision_model
        self.device = device

    def evaluate(self, code: str, image_path: str) -> Dict[str, object]:
        visual_features = _visual_descriptors(image_path, encoder=self.model, device=self.device)
        visual_summary = _visual_feature_summary(visual_features)
        layout_hint = float(
            np.clip(
                4.2
                - visual_features.get("indent_std", 0.0) / 18.0
                - visual_features.get("whitespace_cluster_score", 0.0) * 35.0
                + visual_features.get("staircase_signal", 0.0) * 1.2,
                1.0,
                5.0,
            )
        )

        base = analyze_code(code)
        extra = _extra_code_metrics(code)
        merged = {**base, **extra}
        metric_summary = _text_metric_summary(merged)
        highlights = "\n".join(_text_highlights(merged)[:3])
        text_hint = float(
            np.clip(
                3.9
                - extra.get("cyclomatic_proxy", 1.0) / 6.0
                - extra.get("nesting_mean", 0.0) / 2.0
                + extra.get("comment_ratio", 0.0) * 9.0,
                1.0,
                5.0,
            )
        )

        code_excerpt = "\n".join(code.splitlines()[:80])
        if len(code_excerpt) > 1500:
            code_excerpt = code_excerpt[:1500] + "\n..."

        prompt = (
            "You are a multimodal readability judge. "
            "Use BOTH the PNG screenshot (spacing, indentation staircase, alignment) and the raw code excerpt/metrics "
            "to output a single readability score in JSON {\"hybrid_score\": SCORE, \"reason\": \"...\"}. "
            "Auto-estimated anchors: "
            f"layout≈{layout_hint:.2f}, text≈{text_hint:.2f}; use them only as guidance and adjust freely if the evidence disagrees."
            "\nVisual cues:\n"
            f"{visual_summary}"
            + "\n\nMetric highlights:\n"
            + highlights
            + "\n\nMetrics:\n"
            + metric_summary
            + "\n\nCode excerpt:\n```code\n"
            + code_excerpt
            + "\n```"
        )
        response = generate_multimodal_completion(
            system_prompt="You combine visual and textual cues to rate readability.",
            text_prompt=prompt,
            image_path=image_path,
            temperature=0.35,
        )
        data = _safe_json_parse(response)
        if not data or "hybrid_score" not in data:
            raise RuntimeError("HybridAgent LLM response missing hybrid_score")
        data["hybrid_score"] = _clamp_score(data.get("hybrid_score"), 3.0)
        data.setdefault("reason", "Combined visual+text evaluation of layout and structure.")
        return data
