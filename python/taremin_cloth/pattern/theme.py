# -*- coding: utf-8 -*-
"""型紙エディタのカラーテーマおよびスタイル定義 (Blender非依存)."""

from __future__ import annotations

from dataclasses import dataclass

ColorRGBA = tuple[float, float, float, float]


@dataclass(frozen=True)
class ThemeColors:
    """エディタ各要素のRGBAカラー定義 (0.0 - 1.0)."""
    # キャンバス背景とグリッド
    background: ColorRGBA = (0.15, 0.15, 0.16, 1.0)
    grid_major: ColorRGBA = (0.24, 0.24, 0.26, 0.8)
    grid_minor: ColorRGBA = (0.19, 0.19, 0.21, 0.5)
    origin_axis_x: ColorRGBA = (0.8, 0.3, 0.3, 0.6)
    origin_axis_y: ColorRGBA = (0.3, 0.8, 0.3, 0.6)

    # ピース
    piece_fill: ColorRGBA = (0.25, 0.27, 0.32, 0.45)
    piece_fill_selected: ColorRGBA = (0.32, 0.38, 0.48, 0.6)
    piece_outline: ColorRGBA = (0.85, 0.88, 0.92, 1.0)
    piece_outline_selected: ColorRGBA = (0.35, 0.70, 1.0, 1.0)
    edge_selected: ColorRGBA = (1.0, 0.70, 0.20, 1.0)

    # ノード
    node_corner: ColorRGBA = (0.9, 0.9, 0.9, 1.0)
    node_smooth: ColorRGBA = (0.4, 0.85, 0.5, 1.0)
    node_auto: ColorRGBA = (0.35, 0.75, 1.0, 1.0)
    node_selected: ColorRGBA = (1.0, 0.85, 0.2, 1.0)
    node_hover: ColorRGBA = (1.0, 1.0, 0.4, 1.0)

    # ハンドル
    handle_line: ColorRGBA = (0.6, 0.6, 0.6, 0.7)
    handle_point: ColorRGBA = (0.3, 0.8, 1.0, 0.9)

    # 縫合
    seam_line: ColorRGBA = (0.95, 0.55, 0.2, 0.85)
    seam_wrap: ColorRGBA = (0.75, 0.42, 1.0, 0.95)
    seam_mismatch: ColorRGBA = (0.95, 0.25, 0.25, 0.9)
    seam_candidate: ColorRGBA = (0.3, 0.9, 0.4, 0.9)

    # プレビュー
    preview_line: ColorRGBA = (0.4, 0.8, 1.0, 0.9)
    preview_fill: ColorRGBA = (0.4, 0.8, 1.0, 0.2)

    # 内部線
    internal_line: ColorRGBA = (0.28, 0.78, 0.72, 0.9)
    internal_line_selected: ColorRGBA = (0.45, 0.95, 0.88, 1.0)

    # ノッチ（合印）
    notch: ColorRGBA = (1.0, 0.85, 0.2, 0.95)
    notch_selected: ColorRGBA = (1.0, 0.95, 0.4, 1.0)

    # 寸法線テキスト
    dimension_text: ColorRGBA = (0.75, 0.78, 0.84, 0.85)

    # HUD・UI
    hud_bg: ColorRGBA = (0.1, 0.1, 0.12, 0.85)
    hud_border: ColorRGBA = (0.3, 0.3, 0.35, 0.8)
    hud_text: ColorRGBA = (0.95, 0.95, 0.95, 1.0)
    hud_text_dim: ColorRGBA = (0.65, 0.65, 0.7, 1.0)
    hud_accent: ColorRGBA = (0.35, 0.75, 1.0, 1.0)


DEFAULT_THEME: ThemeColors = ThemeColors()
