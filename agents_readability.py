from __future__ import annotations

import json
import math
from typing import Dict, List

import numpy as np
from PIL import Image

from llm_client import generate_chat_completion, generate_multimodal_completion
from vision_encoder import encode_image
from code_metrics import analyze_code


def _safe_json_parse(payload: str) -> Dict[str, str]:
    if not payload or payload.startswith("ERROR:"):
        return {}
    start = payload.find("{")
    end = payload.rfind("}")
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
        return default
    return float(max(1.0, min(5.0, score)))


class VisualAgent:
    def __init__(self, vision_model, device: str = "cpu", mode: str = "multimodal") -> None:
        self.model = vision_model
        self.device = device
        self.mode = mode

    def evaluate(self, image_path: str) -> Dict[str, object]:
        layout_metrics = _analyze_layout(image_path)
        if self.mode == "multimodal":
            multimodal_result = self._evaluate_multimodal(image_path, layout_metrics)
            if multimodal_result is not None:
                return multimodal_result
            print("[VisualAgent] Falling back to heuristic pipeline for image:", image_path, flush=True)
        return self._evaluate_with_features(image_path, layout_metrics)

    def _evaluate_multimodal(self, image_path: str, layout_metrics: Dict[str, float]) -> Dict[str, object] | None:
        heuristic = _heuristic_layout_score(layout_metrics)
        qualitative = _describe_layout(layout_metrics)
        summary_text = (
            "참고용 수치:\n"
            f"- line_count: {layout_metrics['line_count']}\n"
            f"- empty_line_ratio: {layout_metrics['empty_line_ratio']:.2f}\n"
            f"- indent_std: {layout_metrics['indent_std']:.2f}\n"
            f"- ink_ratio: {layout_metrics['ink_ratio']:.2f}\n"
            f"- line_density: {layout_metrics['line_density']:.2f}\n"
            f"- block_transitions: {layout_metrics['block_transitions']}\n"
            f"- brightness mean/std: {layout_metrics['mean_intensity']:.2f}/{layout_metrics['std_intensity']:.2f}\n"
            f"- heuristic_layout_score: {heuristic:.2f}\n"
            f"- qualitative_notes: {qualitative}\n"
        )
        user_prompt = (
            "첨부된 코드 이미지를 평가하는 SE-Jury 스타일 시각 심사관입니다.\n"
            "평가 단계:\n"
            "1) 들여쓰기 일관성, 빈 줄/공백 비율, 블록 구분, 주석 정렬, 행 길이/밀도를 각각 살펴 긍정/부정 신호를 정리합니다.\n"
            "2) 이미지에서 직접 관찰한 내용이 수치 메모와 다르면, 이미지 관찰을 우선합니다.\n"
            "3) 전체 인상을 1.0(매우 나쁨)~5.0(매우 우수) 범위로 결정하고 한 문장 이유를 작성합니다.\n"
            "4) 최종 출력은 JSON 한 줄만 사용합니다. 예시: {\"layout_score\": 4.0, \"reason\": \"...\"}\n"
            "참고 수치:\n"
            f"{summary_text}\n"
            "점수 분포가 한곳에 몰리지 않도록 1.0~5.0 전체 범위를 적극적으로 활용하세요."
        )
        response = generate_multimodal_completion(
            system_prompt="당신은 코드의 시각적 레이아웃을 평가하는 전문가입니다.",
            text_prompt=user_prompt,
            image_path=image_path,
            temperature=0.2,
        )
        print(f"[VisualAgent-multimodal] image={image_path} response={response}", flush=True)
        data = _safe_json_parse(response)
        if not data:
            return None
        score = _clamp_score(data.get("layout_score"), heuristic)
        reason = data.get("reason") or "image-based assessment"
        return {"layout_score": score, "reason": reason}

    def _evaluate_with_features(self, image_path: str, layout_metrics: Dict[str, float]) -> Dict[str, object]:
        features = encode_image(self.model, image_path, device=self.device)
        stats = np.array(features, dtype=np.float32)
        proxy_score = _heuristic_layout_score(layout_metrics)
        summary = (
            "feature_stats: "
            f"mean={stats.mean():.4f}, std={stats.std():.4f}, "
            f"min={stats.min():.4f}, max={stats.max():.4f}, "
            f"abs_sum={np.abs(stats).sum():.2f}; "
            "layout_metrics: "
            f"lines={layout_metrics['line_count']}, empty_line_ratio={layout_metrics['empty_line_ratio']:.2f}, "
            f"indent_std={layout_metrics['indent_std']:.2f}, ink_ratio={layout_metrics['ink_ratio']:.2f}, "
            f"density={layout_metrics['line_density']:.2f}, "
            f"block_transitions={layout_metrics['block_transitions']}, "
            f"brightness_mean={layout_metrics['mean_intensity']:.2f}, brightness_std={layout_metrics['std_intensity']:.2f}; "
            f"heuristic_layout_score={proxy_score:.2f}"
        )
        qualitative = _describe_layout(layout_metrics)
        system_prompt = "당신은 코드의 시각적 레이아웃을 기반으로 가독성을 평가하는 전문가입니다."
        user_prompt = (
            "다음은 코드 이미지에서 추출한 시각적 특징입니다:\n"
            f"{summary}\n"
            f"관찰 메모: {qualitative}\n"
            f"수치 기반 예비 가독성 점수(참고용): {proxy_score:.2f}\n"
            "SE-Jury 심사 방식으로, 들여쓰기/공백/블록/주석/행 밀도 기준을 각각 검토한 뒤 "
            "1.0~5.0 범위에서 최종 layout_score를 정하세요. 이미지를 직접 본 판단을 우선하고, "
            "점수가 단일 값에 치우치지 않도록 전체 범위를 적극적으로 사용합니다.\n"
            '최종 출력은 JSON 한 줄: {"layout_score": float, "reason": "..."}'
        )
        response = generate_chat_completion(system_prompt, user_prompt, temperature=0.35)
        print(f"[VisualAgent-heuristic] image={image_path} response={response}", flush=True)
        data = _safe_json_parse(response)
        score = _clamp_score(data.get("layout_score"), proxy_score)
        reason = data.get("reason") or "heuristic fallback"
        return {"layout_score": score, "reason": reason}


class TextAgent:
    def evaluate(self, code: str) -> Dict[str, object]:
        system_prompt = "당신은 코드 가독성 평가 전문가입니다."
        metrics = analyze_code(code)
        checklist = (
            f"- line_count: {metrics['line_count']}\n"
            f"- avg_line_length: {metrics['avg_line_len']:.1f}, max_line_length: {metrics['max_line_len']}\n"
            f"- avg_indent: {metrics['avg_indent']:.1f}, indent_std: {metrics['indent_std']:.1f}, estimated_depth: {metrics['estimated_depth']:.1f}\n"
            f"- blank_ratio: {metrics['blank_ratio']:.2f}, comment_ratio: {metrics['comment_ratio']:.2f}\n"
        )
        user_prompt = (
            "다음 코드를 분석하세요:\n"
            f"{code}\n\n"
            "참고용 코드 통계:\n"
            f"{checklist}\n"
            "네이밍, 들여쓰기 일관성, 중첩 깊이, 함수/블록 길이, 직관성, 주석 품질 등을 기준으로 "
            "1.0~5.0의 text_score와 한 문장 reason을 JSON으로 출력하세요. "
            '형식: {"text_score": float, "reason": "..."}'
        )
        response = generate_chat_completion(system_prompt, user_prompt, temperature=0.2)
        print(f"[TextAgent] response={response}", flush=True)
        data = _safe_json_parse(response)
        score = _clamp_score(data.get("text_score"), 3.0)
        reason = data.get("reason") or "fallback"
        return {"text_score": score, "reason": reason}


class AggregatorAgent:
    def evaluate_inputs(self, layout: Dict[str, object], text: Dict[str, object]):
        return self.aggregate(layout, text)

    def aggregate(self, layout: Dict[str, object], text: Dict[str, object]) -> Dict[str, object]:
        system_prompt = "당신은 두 명의 심사위원 평가를 통합하는 판정자입니다."
        layout_score = float(layout.get("layout_score", 3.0))
        text_score = float(text.get("text_score", 3.0))
        layout_reason = layout.get("reason", "")
        text_reason = text.get("reason", "")
        user_prompt = (
            f"레이아웃 평가: 점수={layout_score:.2f}, 이유={layout_reason}\n"
            f"텍스트 평가: 점수={text_score:.2f}, 이유={text_reason}\n"
            "두 평가를 종합하여 최종 가독성 점수 final_score(1.0~5.0)를 정하고, 한두 문장으로 근거를 설명하세요. "
            'JSON으로만 출력하세요. 형식: {"final_score": float, "reason": "..."}'
        )
        response = generate_chat_completion(system_prompt, user_prompt, temperature=0.05)
        print(f"[AggregatorAgent] response={response}", flush=True)
        data = _safe_json_parse(response)
        blended_mean = (layout_score + text_score) / 2.0
        if not data:
            final_score = blended_mean
            return {"final_score": _clamp_score(final_score), "reason": "average of two agents"}
        llm_score = _clamp_score(data.get("final_score"), blended_mean)
        reason = data.get("reason") or "average of two agents"
        final_score = _clamp_score(0.6 * llm_score + 0.4 * blended_mean)
        return {"final_score": final_score, "reason": reason}


__all__ = ["VisualAgent", "TextAgent", "AggregatorAgent"]


def _analyze_layout(image_path: str) -> Dict[str, float]:
    image = Image.open(image_path).convert("L")
    arr = np.array(image, dtype=np.float32) / 255.0
    background = float(np.percentile(arr, 5))
    signal_threshold = background + 0.05
    signal_mask = arr > signal_threshold

    ink_ratio = float(signal_mask.mean())
    row_signal = signal_mask.sum(axis=1)
    non_empty_rows = row_signal > 0
    line_count = int(non_empty_rows.sum())
    empty_line_ratio = float((row_signal == 0).mean())

    indent_positions: List[float] = []
    for row_idx in np.where(non_empty_rows)[0]:
        cols = np.where(signal_mask[row_idx])[0]
        if cols.size:
            indent_positions.append(cols[0] / arr.shape[1])
    indent_std = float(np.std(indent_positions)) if indent_positions else 0.0

    line_density = float(row_signal[non_empty_rows].mean() / arr.shape[1]) if line_count else 0.0
    block_transitions = int(np.sum(np.abs(np.diff(non_empty_rows.astype(np.int8)))))

    return {
        "ink_ratio": ink_ratio,
        "line_count": line_count,
        "empty_line_ratio": empty_line_ratio,
        "indent_std": indent_std,
        "line_density": line_density,
        "block_transitions": block_transitions,
        "mean_intensity": float(arr.mean()),
        "std_intensity": float(arr.std()),
    }


def _describe_layout(metrics: Dict[str, float]) -> str:
    comments: List[str] = []
    if metrics["empty_line_ratio"] > 0.35:
        comments.append("빈 줄 비율이 높아 블록들이 많이 분리되어 있음")
    elif metrics["empty_line_ratio"] < 0.1:
        comments.append("빈 줄이 거의 없어 코드가 빽빽하게 배치됨")

    if metrics["indent_std"] < 0.03:
        comments.append("들여쓰기 시작 위치가 매우 일정함")
    elif metrics["indent_std"] > 0.12:
        comments.append("들여쓰기 폭이 들쭉날쭉함")

    if metrics["line_density"] > 0.25:
        comments.append("각 행의 문자 밀도가 높은 편")
    elif metrics["line_density"] < 0.12:
        comments.append("행당 문자 밀도가 낮아 여백이 많음")

    if metrics["block_transitions"] > 40:
        comments.append("블록 전환이 잦아 구조가 자주 바뀜")
    elif metrics["block_transitions"] < 15:
        comments.append("블록 전환이 적어 단조로운 구조")

    if not comments:
        comments.append("표준적인 밀도와 들여쓰기 패턴")
    return "; ".join(comments)


def _heuristic_layout_score(metrics: Dict[str, float]) -> float:
    score = 3.0
    if metrics["indent_std"] < 0.03:
        score += 0.4
    elif metrics["indent_std"] > 0.12:
        score -= 0.4

    if metrics["empty_line_ratio"] > 0.35:
        score -= 0.2
    elif metrics["empty_line_ratio"] < 0.1:
        score += 0.1

    if metrics["line_density"] > 0.3:
        score -= 0.2
    elif metrics["line_density"] < 0.12:
        score += 0.1

    if metrics["block_transitions"] < 15:
        score += 0.1
    elif metrics["block_transitions"] > 45:
        score -= 0.2

    if metrics["ink_ratio"] < 0.12:
        score += 0.1
    elif metrics["ink_ratio"] > 0.35:
        score -= 0.2
    return float(max(1.0, min(5.0, score)))
