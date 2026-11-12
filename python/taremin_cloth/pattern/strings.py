# -*- coding: utf-8 -*-
"""型紙エディタのUI表示文字列定義 (Blender非依存・i18n連携)."""

from __future__ import annotations

# ツール表示名 (英語マスターキー)
TOOL_NAMES: dict[str, str] = {
    "SELECT": "Select / Move",
    "RECTANGLE": "Rectangle",
    "POLYGON": "Polygon",
    "ELLIPSE": "Ellipse",
    "INTERNAL_LINE": "Internal Line",
    "NOTCH": "Notch",
    "CUT": "Cut",
    "DART": "Dart",
    "SEW": "Sew",
}

# ツール別操作ガイド (英語マスターキー)
TOOL_GUIDES: dict[str, str] = {
    "SELECT": "Click: Select / DblClick or W: Split / Shift+D: Dup / M: Flip / T: Rotate Up / U: Unfold / X: Delete",
    "RECTANGLE": "Drag opposite corners / Esc: Cancel",
    "POLYGON": "Click points / Click start or Enter to close / Esc: Cancel",
    "ELLIPSE": "Drag opposite corners / Esc: Cancel",
    "INTERNAL_LINE": "Click points / Enter: Confirm / Esc: Cancel",
    "NOTCH": "Click edge to add notch / Alt: Free pos / Esc: Cancel",
    "CUT": "Drag cut line across boundary twice / Esc: Cancel",
    "DART": "Click edge, then click dart apex / Esc: Cancel",
    "SEW": "Click two edges to sew / Shift: Chain / Alt: Full edge / R: Flip / B: Bone Wrap",
}


def get_tool_name(tool_key: str) -> str:
    """ツール識別子からローカライズされた表示名を取得する."""
    literal = TOOL_NAMES.get(tool_key, tool_key)
    try:
        from ..i18n import trans
        return trans(literal)
    except Exception:
        return literal


def get_tool_guide(tool_key: str) -> str:
    """ツール識別子からローカライズされた操作ガイドを取得する."""
    literal = TOOL_GUIDES.get(tool_key, "")
    try:
        from ..i18n import trans
        return trans(literal)
    except Exception:
        return literal


def get_ui_label(literal: str) -> str:
    """UIテキストをローカライズして取得する."""
    try:
        from ..i18n import trans
        return trans(literal)
    except Exception:
        return literal

