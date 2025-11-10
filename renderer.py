from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageColor, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import TextLexer
from pygments.styles import get_style_by_name
from pygments.token import Token

IMAGE_SIZE = (1080, 720)
PADDING = 48
LINE_SPACING = 6
BACKGROUND = "#1e1e1e"
DEFAULT_COLOR = "#f8f8f2"
FONT_PATHS = [
    "C:/Windows/Fonts/JetBrainsMonoNL-Regular.ttf",
    "C:/Windows/Fonts/JetBrainsMono-Regular.ttf",
    "C:/Windows/Fonts/consola.ttf",
]


def render_code_to_image(code: str, output_path: str) -> None:
    lexer = TextLexer(stripnl=False, ensurenl=True)
    normalized = code.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    tokens = list(lex(normalized, lexer))

    font = _load_font(26)
    char_width = max(1, _text_width(font, "M"))
    max_chars = max(1, (IMAGE_SIZE[0] - 2 * PADDING) // char_width)
    lines = _split_tokens(tokens, max_chars)
    line_height = _line_height(font)
    max_lines = max(1, (IMAGE_SIZE[1] - 2 * PADDING) // line_height)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [[(Token.Text, "…truncated…")]]

    image = Image.new("RGB", IMAGE_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    style = get_style_by_name("monokai")
    cache: Dict[Token, Tuple[int, int, int]] = {}

    y = PADDING
    for line in lines:
        x = PADDING
        for token_type, text in line:
            if not text:
                continue
            color = _token_color(token_type, style, cache)
            draw.text((x, y), text, font=font, fill=color)
            x += _text_width(font, text)
        y += line_height

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path, format="PNG")


def _split_tokens(tokens: Sequence[Tuple[Token, str]], max_chars: int) -> List[List[Tuple[Token, str]]]:
    lines: List[List[Tuple[Token, str]]] = [[]]
    count = 0
    for token_type, value in tokens:
        for char in value:
            if char == "\n":
                lines.append([])
                count = 0
                continue
            if count >= max_chars:
                lines.append([])
                count = 0
            if not lines[-1]:
                lines[-1] = []
            lines[-1].append((token_type, char))
            count += 1
    return [line if line else [(Token.Text, "")] for line in lines]


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _text_width(font: ImageFont.ImageFont, text: str) -> int:
    if hasattr(font, "getlength"):
        return int(font.getlength(text))
    return font.getbbox(text)[2] - font.getbbox(text)[0]


def _line_height(font: ImageFont.ImageFont) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent + LINE_SPACING


def _token_color(token_type: Token, style, cache: Dict[Token, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if token_type in cache:
        return cache[token_type]
    data = style.style_for_token(token_type)
    color = data.get("color")
    rgb = ImageColor.getrgb(f"#{color}") if color else ImageColor.getrgb(DEFAULT_COLOR)
    cache[token_type] = rgb
    return rgb


__all__ = ["render_code_to_image"]
