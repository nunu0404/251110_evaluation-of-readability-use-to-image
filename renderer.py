from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageColor, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.styles import get_style_by_name
from pygments.token import Token

IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 720
PADDING = 48
MAX_FONT_SIZE = 28
MIN_FONT_SIZE = 14
LINE_GAP = 6
STYLE_NAME = "monokai"
BACKGROUND_COLOR = "#1e1e1e"
DEFAULT_FOREGROUND = "#f8f8f2"

FONT_CANDIDATES = [
    "C:/Windows/Fonts/JetBrainsMonoNL-Regular.ttf",
    "C:/Windows/Fonts/JetBrainsMono-Regular.ttf",
    "C:/Windows/Fonts/JetBrainsMono-Medium.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/Consola.ttf",
]


def render_code_to_image(code: str, output_path: str) -> None:
    """
    Render a syntax highlighted representation of `code` into a fixed-size PNG image.

    Args:
        code: Code snippet to render. Tabs are expanded to four spaces before drawing.
        output_path: Destination path for the PNG image. Parent directories are created if needed.
    """
    lexer = PythonLexer(stripall=False, ensurenl=False)
    normalized_code = code.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    line_tokens = _tokenize_lines(normalized_code, lexer)

    font_size = MAX_FONT_SIZE
    font = None
    wrapped_lines: List[List[Tuple[Token, str]]] = []
    line_height = 0
    max_lines = 1

    while font_size >= MIN_FONT_SIZE:
        font = _load_font(font_size)
        char_width = max(1, int(_text_width(font, "M")))
        max_chars = max(1, (IMAGE_WIDTH - 2 * PADDING) // char_width)
        wrapped_lines = _wrap_lines(line_tokens, max_chars)
        line_height = _line_height(font)
        max_lines = max(1, (IMAGE_HEIGHT - 2 * PADDING) // line_height)
        if len(wrapped_lines) <= max_lines:
            break
        font_size -= 2

    if font is None:
        font = _load_font(MIN_FONT_SIZE)

    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[: max(1, max_lines - 1)]
        wrapped_lines.append([(Token.Text, "… code truncated …")])

    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)

    style = get_style_by_name(STYLE_NAME)
    palette_cache: Dict[Token, Tuple[int, int, int]] = {}

    y = PADDING
    for segments in wrapped_lines:
        x = PADDING
        for token_type, text in segments:
            if not text:
                continue
            color = _token_color(token_type, style, palette_cache)
            draw.text((x, y), text, font=font, fill=color)
            x += _text_width(font, text)
        y += line_height
        if y > IMAGE_HEIGHT - PADDING:
            break

    output_path = str(output_path)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    image.save(output_path, format="PNG")


def _tokenize_lines(code: str, lexer: PythonLexer) -> List[List[Tuple[Token, str]]]:
    lines: List[List[Tuple[Token, str]]] = [[]]
    for token_type, value in lex(code, lexer):
        parts = value.split("\n")
        for idx, part in enumerate(parts):
            if part:
                lines[-1].append((token_type, part))
            if idx < len(parts) - 1:
                lines.append([])
    return lines


def _wrap_lines(
    lines: Sequence[Sequence[Tuple[Token, str]]], max_chars: int
) -> List[List[Tuple[Token, str]]]:
    wrapped: List[List[Tuple[Token, str]]] = []
    for tokens in lines:
        if not tokens:
            wrapped.append([(Token.Text, "")])
            continue
        current_line: List[Tuple[Token, str]] = []
        count = 0
        for token_type, text in tokens:
            start = 0
            text_length = len(text)
            while start < text_length:
                remaining = max_chars - count
                if remaining <= 0:
                    wrapped.append(current_line)
                    current_line = []
                    count = 0
                    remaining = max_chars
                chunk = text[start : start + remaining]
                if chunk:
                    current_line.append((token_type, chunk))
                    count += len(chunk)
                start += len(chunk) if chunk else remaining
            if text == "":
                current_line.append((token_type, ""))
        if current_line:
            wrapped.append(current_line)
        else:
            wrapped.append([(Token.Text, "")])
    return wrapped


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _line_height(font: ImageFont.ImageFont) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent + LINE_GAP


def _text_width(font: ImageFont.ImageFont, text: str) -> int:
    if hasattr(font, "getlength"):
        return int(font.getlength(text))
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def _token_color(
    token_type: Token,
    style,
    cache: Dict[Token, Tuple[int, int, int]],
) -> Tuple[int, int, int]:
    if token_type in cache:
        return cache[token_type]
    style_attrs = style.style_for_token(token_type)
    hex_color = style_attrs.get("color")
    rgb = ImageColor.getrgb(f"#{hex_color}") if hex_color else ImageColor.getrgb(DEFAULT_FOREGROUND)
    cache[token_type] = rgb
    return rgb


__all__ = ["render_code_to_image"]
