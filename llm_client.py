import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests


def generate_chat_completion(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    endpoint, headers, model = _prepare_chat_request()
    if endpoint is None:
        return model  # contains error string

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": max(0.0, min(2.0, temperature)),
    }
    return _post_chat_completion(endpoint, headers, payload)


def generate_multimodal_completion(
    system_prompt: str,
    text_prompt: str,
    image_path: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    endpoint, headers, model = _prepare_response_request()
    if endpoint is None:
        return model

    image_b64 = _encode_image(image_path)
    if isinstance(image_b64, str) and image_b64.startswith("ERROR"):
        return image_b64

    payload = {
        "model": model,
        "instructions": system_prompt,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text_prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/{_guess_image_ext(image_path)};base64,{image_b64}",
                    },
                ],
            }
        ],
        "max_output_tokens": max_tokens,
        "temperature": max(0.0, min(2.0, temperature)),
    }
    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        data = response.json()
        content_blocks = []
        if "output" in data:
            content_blocks = data["output"]
        elif "choices" in data:
            content_blocks = data["choices"]
        for block in content_blocks:
            block_content = block.get("content")
            if isinstance(block_content, list):
                text_parts = [
                    item.get("text", "")
                    for item in block_content
                    if item.get("type") in {"output_text", "text"}
                ]
                combined = "\n".join(filter(None, text_parts)).strip()
                if combined:
                    return combined
        return "ERROR: Unsupported response format"
    except requests.RequestException as exc:
        return f"ERROR: {exc}"
    except ValueError:
        return "ERROR: Invalid JSON response"


def _prepare_chat_request() -> tuple[str | None, Dict[str, str], str]:
    return _prepare_endpoint("/chat/completions")


def _prepare_response_request() -> tuple[str | None, Dict[str, str], str]:
    return _prepare_endpoint("/responses")


def _prepare_endpoint(path: str) -> tuple[str | None, Dict[str, str], str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, {}, "ERROR: Missing OPENAI_API_KEY"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    endpoint = f"{base_url.rstrip('/')}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    return endpoint, headers, model


def _post_chat_completion(endpoint: str, headers: Dict[str, str], payload: Dict[str, Any]) -> str:
    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices")
        if not choices:
            return "ERROR: Empty response"
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, str):
            return content.strip() or "ERROR: Blank response"
        if isinstance(content, list):
            texts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
            combined = "\n".join(filter(None, texts)).strip()
            return combined or "ERROR: Blank response"
        return "ERROR: Unsupported response format"
    except requests.RequestException as exc:
        return f"ERROR: {exc}"
    except ValueError:
        return "ERROR: Invalid JSON response"


def _encode_image(image_path: str):
    try:
        data = Path(image_path).read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except OSError as exc:
        return f"ERROR: {exc}"


def _guess_image_ext(image_path: str) -> str:
    ext = Path(image_path).suffix.lower().lstrip(".")
    return ext or "png"


__all__ = ["generate_chat_completion", "generate_multimodal_completion"]
