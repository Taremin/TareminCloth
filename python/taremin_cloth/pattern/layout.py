# -*- coding: utf-8 -*-
"""型紙エディタのHUDおよびUIパネルレイアウト計算 (Blender非依存)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rect2D:
    """2D画面矩形領域 (左下原点ピクセル座標)."""
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y + self.h

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.right and self.y <= py <= self.top


@dataclass
class EditorLayout:
    """エディタHUD各要素の画面配置領域."""
    header_bar: Rect2D
    footer_bar: Rect2D
    info_card: Rect2D


def compute_hud_layout(screen_width: int, screen_height: int) -> EditorLayout:
    """画面サイズに基づきHUDの配置矩形を計算して返す."""
    sw = float(max(100, screen_width))
    sh = float(max(100, screen_height))

    margin = 12.0
    bar_height = 32.0

    header = Rect2D(x=margin, y=sh - margin - bar_height, w=sw - margin * 2.0, h=bar_height)
    footer = Rect2D(x=margin, y=margin, w=sw - margin * 2.0, h=bar_height)
    info_card = Rect2D(x=sw - margin - 240.0, y=sh - margin - bar_height - 150.0, w=240.0, h=140.0)

    return EditorLayout(
        header_bar=header,
        footer_bar=footer,
        info_card=info_card,
    )
