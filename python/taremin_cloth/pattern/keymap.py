# -*- coding: utf-8 -*-
"""型紙エディタのキーマップおよびショートカット解決モジュール (Blender非依存)."""

from __future__ import annotations

from typing import Optional

# ツール切替ショートカットキー定義 (数字キー 1〜9 のみ使用し、文字キーはモーダル編集と統一のため解除)
TOOL_HOTKEYS: dict[str, str] = {
    "ONE": "SELECT",
    "TWO": "RECTANGLE",
    "THREE": "POLYGON",
    "FOUR": "ELLIPSE",
    "FIVE": "INTERNAL_LINE",
    "SIX": "NOTCH",
    "SEVEN": "SEW",
    "EIGHT": "CUT",
    "NINE": "DART",
}


def resolve_tool_hotkey(key: str) -> Optional[str]:
    """入力されたキー識別子（例: 'ONE', 'TWO'）から対応するツール名を解決して返す.
    
    該当するツールが存在しない場合は None を返す.
    """
    if not key:
        return None
    return TOOL_HOTKEYS.get(key.upper())

