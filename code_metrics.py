from __future__ import annotations

import math
import re
from typing import Dict


COMMENT_PATTERNS = [
    re.compile(r"^\s*#"),
    re.compile(r"^\s*//"),
    re.compile(r"^\s*/\*"),
    re.compile(r"^\s*\*"),
]


def analyze_code(code: str) -> Dict[str, float]:
    lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    total_lines = len(lines)
    non_empty = [line for line in lines if line.strip()]
    line_lengths = [len(line) for line in lines]

    indent_levels = []
    comment_lines = 0
    for line in lines:
        stripped = line.lstrip()
        indent_levels.append(len(line) - len(stripped))
        if any(pattern.match(line) for pattern in COMMENT_PATTERNS):
            comment_lines += 1

    avg_line_len = sum(line_lengths) / total_lines if total_lines else 0.0
    max_line_len = max(line_lengths) if line_lengths else 0

    avg_indent = sum(indent_levels) / total_lines if total_lines else 0.0
    indent_std = float(
        math.sqrt(
            sum((lvl - avg_indent) ** 2 for lvl in indent_levels) / total_lines
        )
    ) if total_lines else 0.0

    blank_ratio = 1.0 - (len(non_empty) / total_lines) if total_lines else 0.0
    comment_ratio = comment_lines / total_lines if total_lines else 0.0

    estimated_depth = max(indent_levels) / 4.0 if indent_levels else 0.0

    return {
        "line_count": total_lines,
        "avg_line_len": avg_line_len,
        "max_line_len": max_line_len,
        "avg_indent": avg_indent,
        "indent_std": indent_std,
        "blank_ratio": blank_ratio,
        "comment_ratio": comment_ratio,
        "estimated_depth": estimated_depth,
    }


__all__ = ["analyze_code"]
