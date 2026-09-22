"""
ブラシツールパッケージ (taremin_cloth.brush)
基本実装 (base / math) と個別ブラシ (single_grab / range_grab) を分離する。
新規ブラシの追加は: 新規モジュール + TOOLS registry 登録 (+ RNA Enum 項目) のみ。
"""

from .single_grab import SingleGrabTool
from .range_grab import RangeGrabTool
from .smooth import SmoothTool

TOOLS = {
    'GRAB': SingleGrabTool,
    'RANGE_GRAB': RangeGrabTool,
    'SMOOTH': SmoothTool,
}

__all__ = [
    "TOOLS",
    "SingleGrabTool",
    "RangeGrabTool",
    "SmoothTool",
    "create_tools",
    "active_tool_name",
    "get_brush_hud_info",
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


def get_brush_hud_info(brush_settings, radial=None, _tools=None):
    """HUD表示用の構造化ブラシ情報を返す（docs/hud.md:5）。

    新規ブラシは TOOLS 登録のみで対応可能。呼出側での分岐直書き禁止。
    戻り値は BaseBrushTool.hud_info() と同形式のdict。
    """
    try:
        name = active_tool_name(brush_settings)
        tools = _tools if _tools is not None else {k: cls() for k, cls in TOOLS.items()}
        tool = tools.get(name)
        if tool is None:
            tool = tools.get('GRAB')
        if tool is None:
            return {"tool": 'GRAB', "label_key": 'Grab', "radius": None,
                    "strength": None, "falloff": None,
                    "show_falloff": False, "adjusting": None}
        return tool.hud_info(brush_settings, radial)
    except Exception:
        return {"tool": 'GRAB', "label_key": 'Grab', "radius": None,
                "strength": None, "falloff": None,
                "show_falloff": False, "adjusting": None}
