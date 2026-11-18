# -*- coding: utf-8 -*-
"""型紙エディタのキーマップおよびショートカット解決モジュール (Blender非依存)."""

from __future__ import annotations

from typing import Optional

# 既定のツール切替ショートカットキー定義 (数字キー 1〜9, 0 のみ使用し、文字キーはモーダル編集と統一のため解除)
DEFAULT_TOOL_HOTKEYS: dict[str, str] = {
    "ONE": "SELECT",
    "TWO": "RECTANGLE",
    "THREE": "POLYGON",
    "FOUR": "ELLIPSE",
    "FIVE": "INTERNAL_LINE",
    "SIX": "NOTCH",
    "SEVEN": "SEW",
    "EIGHT": "CUT",
    "NINE": "DART",
    "ZERO": "SPLIT_EDGE",
}
TOOL_HOTKEYS: dict[str, str] = DEFAULT_TOOL_HOTKEYS

_HOTKEY_DISPLAY_LABELS: dict[str, str] = {
    "ONE": "1",
    "TWO": "2",
    "THREE": "3",
    "FOUR": "4",
    "FIVE": "5",
    "SIX": "6",
    "SEVEN": "7",
    "EIGHT": "8",
    "NINE": "9",
    "ZERO": "0",
}


def resolve_tool_hotkey(
    key: str,
    *,
    hotkeys: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """入力されたキー識別子（例: 'ONE', 'ZERO'）から対応するツール名を解決して返す.

    該当するツールが存在しない場合は None を返す.
    """
    if not key:
        return None
    mapping = hotkeys if hotkeys is not None else TOOL_HOTKEYS
    return mapping.get(key.upper())


def get_tool_hotkey_label(
    tool_name: str,
    *,
    hotkeys: Optional[dict[str, str]] = None,
) -> str:
    """指定ツールに割り当てられた数字ラベル ('1'..'9', '0' または '') を返す."""
    mapping = hotkeys if hotkeys is not None else TOOL_HOTKEYS
    for key_id, mapped_tool in mapping.items():
        if mapped_tool == tool_name:
            return _HOTKEY_DISPLAY_LABELS.get(key_id, key_id)
    return ""

