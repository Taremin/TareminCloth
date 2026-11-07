# -*- coding: utf-8 -*-
"""型紙エディタのキーマップおよびショートカット解決モジュール (Blender非依存)."""

from __future__ import annotations

from typing import Optional

# ツール切替ショートカットキー定義 (数字キーおよびアルファベットショートカット)
TOOL_HOTKEYS: dict[str, str] = {
    "ONE": "SELECT",
    "Q": "SELECT",
    "V": "SELECT",
    "TWO": "RECTANGLE",
    "R": "RECTANGLE",
    "THREE": "POLYGON",
    "P": "POLYGON",
    "FOUR": "ELLIPSE",
    "E": "ELLIPSE",
    "FIVE": "INTERNAL_LINE",
    "I": "INTERNAL_LINE",
    "SIX": "NOTCH",
    "N": "NOTCH",
    "SEVEN": "SEW",
    "S": "SEW",
    "J": "SEW",
    "EIGHT": "CUT",
    "K": "CUT",
    "NINE": "DART",
    "D": "DART",
}


def resolve_tool_hotkey(key: str) -> Optional[str]:
    """入力されたキー識別子（例: 'ONE', 'R', 'q'）から対応するツール名を解決して返す.
    
    該当するツールが存在しない場合は None を返す.
    """
    if not key:
        return None
    return TOOL_HOTKEYS.get(key.upper())

