"""
ブラシツールパッケージ (taremin_cloth.brush)
基本実装 (base / math) と個別ブラシ (single_grab / range_grab) を分離する。
新規ブラシの追加は: 新規モジュール + TOOLS registry 登録 (+ RNA Enum 項目) のみ。
"""

from .single_grab import SingleGrabTool
from .range_grab import RangeGrabTool

TOOLS = {
    'GRAB': SingleGrabTool,
    'RANGE_GRAB': RangeGrabTool,
}

__all__ = [
    "TOOLS",
    "SingleGrabTool",
    "RangeGrabTool",
    "create_tools",
    "active_tool_name",
]


def create_tools():
    """モーダル実行時にツールインスタンス群を生成する"""
    return {name: cls() for name, cls in TOOLS.items()}


def active_tool_name(brush_settings):
    """ブラシ設定から有効なツール名を取得する (未知値は GRAB にフォールバック)"""
    try:
        mode = getattr(brush_settings, "tool_mode", 'GRAB')
    except Exception:
        return 'GRAB'
    return mode if mode in TOOLS else 'GRAB'
