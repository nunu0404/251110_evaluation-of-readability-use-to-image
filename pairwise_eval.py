from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from agents_readability import (
    _extra_code_metrics,
    _llm_ocr,
    _text_highlights,
    _text_metric_summary,
    _tesseract_fallback,
)
from code_metrics import analyze_code
from dataset_utils import load_code_readability_merged
from llm_client import generate_chat_completion, generate_multimodal_completion
from renderer import render_code_to_image


Pair = Tuple[Dict[str, object], Dict[str, object]]

_CODE_CONTEXT_CACHE: Dict[str, Dict[str, str]] = {}
_OCR_TEXT_CACHE: Dict[str, str] = {}


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


def _extract_choice(data: Dict[str, object]) -> Tuple[Optional[str], Optional[float]]:
    confidence: Optional[float] = None
    if isinstance(data.get("confidence"), (int, float)):
        confidence = float(data["confidence"])
    for key in ("choice", "better", "preference", "selected"):
        value = data.get(key)
        if isinstance(value, str):
            cleaned = value.strip().upper()
            if cleaned in {"A", "B"}:
                return cleaned, confidence
            if cleaned in {"TIE", "EQUAL"}:
                return None, confidence
    return None, confidence


def _code_excerpt(code: str, max_lines: int = 40, max_chars: int = 900) -> str:
    lines = code.splitlines()[:max_lines]
    excerpt = "\n".join(lines)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars] + "\n..."
    return excerpt


def _code_context(sample: Dict[str, object]) -> Dict[str, str]:
    key = str(sample["id"])
    cached = _CODE_CONTEXT_CACHE.get(key)
    if cached:
        return cached
    code = sample["code"]
    base = analyze_code(code)
    extra = _extra_code_metrics(code)
    merged = {**base, **extra}
    summary = _text_metric_summary(merged)
    highlight_list = _text_highlights(merged)
    highlights = "\n".join(highlight_list[:2]) if highlight_list else "No special highlights."
    excerpt = _code_excerpt(code)
    ctx = {"summary": summary, "highlights": highlights, "excerpt": excerpt}
    _CODE_CONTEXT_CACHE[key] = ctx
    return ctx


def _ensure_image(sample: Dict[str, object], image_dir: Path) -> Path:
    image_path = image_dir / f"{int(sample['id']):05d}.png"
    if not image_path.is_file():
        render_code_to_image(sample["code"], str(image_path))
    return image_path


@contextmanager
def _paired_image(path_a: Path, path_b: Path):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    try:
        _compose_pair_image(path_a, path_b, Path(tmp.name))
        yield tmp.name
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def _compose_pair_image(path_a: Path, path_b: Path, target: Path) -> None:
    left = Image.open(path_a).convert("RGB")
    right = Image.open(path_b).convert("RGB")
    label_height = 26
    gap = 16
    width = left.width + right.width + gap
    height = max(left.height, right.height) + label_height
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(left, (0, label_height))
    canvas.paste(right, (left.width + gap, label_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((6, 6), "Snippet A (left)", fill=(0, 0, 0), font=font)
    draw.text((left.width + gap + 6, 6), "Snippet B (right)", fill=(0, 0, 0), font=font)
    canvas.save(target)


def _visual_pairwise(
    sample_a: Dict[str, object], sample_b: Dict[str, object], image_dir: Path
) -> Tuple[Optional[str], Optional[float]]:
    path_a = _ensure_image(sample_a, image_dir)
    path_b = _ensure_image(sample_b, image_dir)
    with _paired_image(path_a, path_b) as combo:
        def _ask(force: bool) -> Tuple[Optional[str], Optional[float]]:
            suffix = (
                " Even if they seem equally readable, you MUST still choose one (whichever appears slightly clearer)."
                if force
                else ""
            )
            prompt = (
                "You are comparing two code screenshots placed side-by-side. "
                "Snippet A is on the LEFT, snippet B is on the RIGHT. "
                "Judge readability strictly from layout cues (indentation, whitespace distribution, block grouping, line length). "
                "Return ONLY JSON like {\"choice\": \"A\"|\"B\", \"confidence\": 0-1}." + suffix
            )
            response = generate_multimodal_completion(
                system_prompt="Visual readability comparator.",
                text_prompt=prompt,
                image_path=combo,
                temperature=0.35,
            )
            return _extract_choice(_safe_json_parse(response))

        choice, conf = _ask(force=False)
        if choice not in {"A", "B"}:
            choice, conf = _ask(force=True)
    return choice, conf


def _text_pairwise(
    sample_a: Dict[str, object], sample_b: Dict[str, object], label: str = "Text"
) -> Tuple[Optional[str], Optional[float]]:
    ctx_a = _code_context(sample_a)
    ctx_b = _code_context(sample_b)
    summaries = (
        "Snippet A metrics:\n"
        f"{ctx_a['summary']}\nHighlights:\n{ctx_a['highlights']}\nExcerpt:\n```code\n{ctx_a['excerpt']}\n```\n\n"
        "Snippet B metrics:\n"
        f"{ctx_b['summary']}\nHighlights:\n{ctx_b['highlights']}\nExcerpt:\n```code\n{ctx_b['excerpt']}\n```\n"
    )

    prompts = [
        (
            f"You are a readability judge comparing two {label} snippets.\n"
            "Use the metrics, highlights, and excerpts to decide which is easier to read overall.\n"
            + summaries
            + "Respond ONLY with JSON {\"choice\": \"A\"|\"B\", \"confidence\": 0-1}."
        ),
        (
            f"You are a readability judge comparing two {label} snippets.\n"
            "You must pick whichever is even slightly easier to read.\n"
            + summaries
            + "Return JSON {\"choice\": \"A\"|\"B\", \"confidence\": 0-1}; ties are not allowed."
        ),
        (
            "Final decision required.\n"
            + summaries.replace("Excerpt:\n", "Excerpt (trimmed):\n")
            + "Output EXACT JSON in one line, e.g. {\"choice\":\"A\",\"confidence\":0.55}. No other text."
        ),
    ]

    choice: Optional[str] = None
    conf: Optional[float] = None
    for prompt in prompts:
        response = generate_chat_completion(
            system_prompt="You compare code readability using structured evidence.",
            user_prompt=prompt,
            temperature=0.2,
        )
        choice, conf = _extract_choice(_safe_json_parse(response))
        if choice in {"A", "B"}:
            break
    return choice, conf


def _ocr_pairwise(
    sample_a: Dict[str, object], sample_b: Dict[str, object], image_dir: Path
) -> Tuple[Optional[str], Optional[float]]:
    path_a = _ensure_image(sample_a, image_dir)
    path_b = _ensure_image(sample_b, image_dir)
    text_a = _get_ocr_text(path_a)
    text_b = _get_ocr_text(path_b)
    if not text_a or not text_b:
        return None, None
    pseudo_a = {"id": f"ocr_{sample_a['id']}", "code": text_a}
    pseudo_b = {"id": f"ocr_{sample_b['id']}", "code": text_b}
    return _text_pairwise(pseudo_a, pseudo_b, label="OCR-derived")


def _hybrid_pairwise(
    sample_a: Dict[str, object], sample_b: Dict[str, object], image_dir: Path
) -> Tuple[Optional[str], Optional[float]]:
    path_a = _ensure_image(sample_a, image_dir)
    path_b = _ensure_image(sample_b, image_dir)
    ctx_a = _code_context(sample_a)
    ctx_b = _code_context(sample_b)
    with _paired_image(path_a, path_b) as combo:
        base_prompt = (
            "Compare readability using BOTH the combined PNG (left=A, right=B) and the textual evidence below.\n"
            "Focus on indentation clarity, whitespace balance, naming, nesting, and documentation quality.\n"
            "Snippet A summary:\n"
            f"{ctx_a['summary']}\nHighlights:\n{ctx_a['highlights']}\nExcerpt:\n```code\n{ctx_a['excerpt']}\n```\n\n"
            "Snippet B summary:\n"
            f"{ctx_b['summary']}\nHighlights:\n{ctx_b['highlights']}\nExcerpt:\n```code\n{ctx_b['excerpt']}\n```\n"
            "Return ONLY JSON {\"choice\": \"A\"|\"B\", \"confidence\": 0-1}."
        )

        def _ask(force: bool) -> Tuple[Optional[str], Optional[float]]:
            prompt = (
                base_prompt
                if not force
                else base_prompt
                + "\nIf both appear similar, you STILL must choose whichever seems marginally clearer."
            )
            response = generate_multimodal_completion(
                system_prompt="Hybrid (visual + textual) readability judge.",
                text_prompt=prompt,
                image_path=combo,
                temperature=0.35,
            )
            return _extract_choice(_safe_json_parse(response))

        choice, conf = _ask(force=False)
        if choice not in {"A", "B"}:
            choice, conf = _ask(force=True)
    return choice, conf


def _get_ocr_text(image_path: Path) -> Optional[str]:
    key = str(image_path)
    if key in _OCR_TEXT_CACHE:
        return _OCR_TEXT_CACHE[key]
    text, conf = _llm_ocr(str(image_path))
    if (not text or conf < 0.65):
        try:
            fallback = _tesseract_fallback(str(image_path))
        except Exception:
            fallback = None
        if fallback:
            text = fallback
    if text:
        _OCR_TEXT_CACHE[key] = text
        return text
    return None


def _pair_accuracy(pairs: Iterable[Pair], image_dir: Path) -> None:
    stats = {
        "visual": {"win": 0, "total": 0},
        "text": {"win": 0, "total": 0},
        "ocr": {"win": 0, "total": 0},
        "hybrid": {"win": 0, "total": 0},
    }
    pair_index = 0

    for sample_a, sample_b in pairs:
        human_a = float(sample_a["score"])
        human_b = float(sample_b["score"])
        if abs(human_a - human_b) < 1e-6:
            continue
        human_pref = "A" if human_a > human_b else "B"
        pair_index += 1
        print(
            f"\nPair {pair_index}: A(id={int(sample_a['id'])}, score={human_a:.3f}) "
            f"vs B(id={int(sample_b['id'])}, score={human_b:.3f}) -> human prefers {human_pref}"
        )

        decisions = {
            "visual": _visual_pairwise(sample_a, sample_b, image_dir),
            "text": _text_pairwise(sample_a, sample_b),
            "ocr": _ocr_pairwise(sample_a, sample_b, image_dir),
            "hybrid": _hybrid_pairwise(sample_a, sample_b, image_dir),
        }

        for name, result in decisions.items():
            choice, conf = result
            conf_str = f"{conf:.2f}" if isinstance(conf, float) else "-"
            status = choice if choice in {"A", "B"} else "None"
            outcome = "match" if choice == human_pref else ("miss" if choice in {"A", "B"} else "-")
            print(f"  {name.title():>7}: choice={status}, conf={conf_str}, match={outcome}")
            if choice not in {"A", "B"}:
                continue
            stats[name]["total"] += 1
            if choice == human_pref:
                stats[name]["win"] += 1

    for name, result in stats.items():
        total = result["total"]
        accuracy = result["win"] / total if total else 0.0
        print(f"{name.title()} Agent Pair Accuracy: {accuracy:.3f} ({result['win']}/{total})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise readability evaluation.")
    parser.add_argument("--num-pairs", type=int, default=20, help="Number of random pairs to evaluate.")
    parser.add_argument("--image-dir", default="rendered_eval_images")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randrange(1_000_000_000)
    random.seed(seed)
    full_dataset, _, _ = load_code_readability_merged(split_ratio=(1.0, 0.0, 0.0), seed=seed)
    if len(full_dataset) < 2:
        raise RuntimeError("Dataset too small for pairwise evaluation.")

    pairs: List[Pair] = []
    while len(pairs) < args.num_pairs:
        sample_a, sample_b = random.sample(full_dataset, 2)
        pairs.append((sample_a, sample_b))

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {len(pairs)} pairs with seed={seed}")
    _pair_accuracy(pairs, image_dir)


if __name__ == "__main__":
    main()
