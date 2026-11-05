# -*- coding: utf-8 -*-
"""型紙寸法および座標単位変換ユーティリティ (内部: メートル, UI: センチメートル)."""

import math
import re


def m_to_cm(meters: float) -> float:
    """メートルをセンチメートルに変換する."""
    return float(meters) * 100.0


def cm_to_m(cm: float) -> float:
    """センチメートルをメートルに変換する."""
    return float(cm) * 0.01


def format_cm(
    meters: float,
    *,
    decimals: int = 1,
    with_unit: bool = True,
) -> str:
    """メートル実数値をUI表示用のセンチメートル文字列に整形する."""
    cm = m_to_cm(meters)
    formatted = f"{cm:.{decimals}f}"
    return f"{formatted} cm" if with_unit else formatted


def parse_length_input(
    text: str,
    *,
    default_unit: str = "cm",
) -> float:
    """ユーザー入力文字列（例: '25.4', '25.4cm', '0.254m', '254mm'）を解釈し、メートル値(float)を返す.
    
    不正な値、空文字列、非有限値（NaN, Inf）の場合は ValueError を送出する.
    """
    if not isinstance(text, str):
        raise ValueError(f"Input must be a string, got {type(text)}")

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input string is empty")

    pattern = r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)\s*$"
    match = re.match(pattern, cleaned)
    if not match:
        raise ValueError(f"Invalid length input: '{text}'")

    num_str, unit_str = match.groups()
    try:
        val = float(num_str)
    except ValueError as e:
        raise ValueError(f"Could not parse numeric value from '{text}': {e}") from e

    if not math.isfinite(val):
        raise ValueError(f"Non-finite length value: '{text}'")

    unit = unit_str.lower() if unit_str else default_unit.lower()

    if unit in ("cm", "centimeter", "centimeters"):
        return val * 0.01
    elif unit in ("m", "meter", "meters"):
        return val
    elif unit in ("mm", "millimeter", "millimeters"):
        return val * 0.001
    elif unit in ("in", "inch", "inches"):
        return val * 0.0254
    else:
        raise ValueError(f"Unsupported unit '{unit}' in '{text}'")
