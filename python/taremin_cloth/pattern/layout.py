# -*- coding: utf-8 -*-
"""型紙エディタのHUDおよびUIパネルレイアウト計算 (Blender非依存)."""

from __future__ import annotations

from dataclasses import dataclass

TOOL_ORDER: tuple[str, ...] = (
    "SELECT",
    "RECTANGLE",
    "POLYGON",
    "ELLIPSE",
    "INTERNAL_LINE",
    "NOTCH",
    "SEW",
    "CUT",
    "DART",
)


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
    left_rail: Rect2D
    tool_buttons: dict[str, Rect2D]
    info_card: Rect2D
    card_fields: dict[str, Rect2D]
    compass_rect: Rect2D
    footer_bar: Rect2D
    build_button: Rect2D
    close_button: Rect2D


def compute_hud_layout(
    screen_width: int,
    screen_height: int,
    *,
    left_offset: float = 0.0,
    right_offset: float = 0.0,
    top_offset: float = 0.0,
    bottom_offset: float = 0.0,
) -> EditorLayout:
    """画面サイズおよび4方向パネルオフセット（ヘッダー・ツールヘッダー・Tパネル・Nパネル）に基づき5領域HUDの配置矩形を計算して返す."""
    sw = float(max(100, screen_width))
    sh = float(max(100, screen_height))
    lo = float(max(0.0, left_offset))
    ro = float(max(0.0, right_offset))
    to = float(max(0.0, top_offset))
    bo = float(max(0.0, bottom_offset))

    margin = 12.0
    bar_height = 32.0
    avail_w = max(200.0, sw - margin * 2.0 - lo - ro)

    # 1. ヘッダーバー (画面上部・ヘッダー/ツールヘッダーのオーバーレイ考慮)
    header = Rect2D(x=margin + lo, y=sh - margin - to - bar_height, w=avail_w, h=bar_height)

    # 2. フッターバー (画面下部・下部バーオフセット考慮)
    footer = Rect2D(x=margin + lo, y=margin + bo, w=avail_w, h=bar_height)

    # フッター内アクションボタン
    btn_h = bar_height - 8.0
    close_w = 90.0
    build_w = 140.0
    close_btn = Rect2D(x=footer.right - close_w - 4.0, y=footer.y + 4.0, w=close_w, h=btn_h)
    build_btn = Rect2D(x=close_btn.x - build_w - 8.0, y=footer.y + 4.0, w=build_w, h=btn_h)

    # 3. 左ツールレール (画面左側縦並び・Tパネルオフセット考慮)
    rail_w = 56.0
    btn_size_h = 36.0
    btn_gap = 4.0
    rail_h = len(TOOL_ORDER) * (btn_size_h + btn_gap) + 8.0
    rail_y = header.y - 8.0 - rail_h
    left_rail = Rect2D(x=margin + lo, y=max(footer.top + 8.0, rail_y), w=rail_w, h=rail_h)

    tool_buttons: dict[str, Rect2D] = {}
    cur_y = left_rail.top - 4.0 - btn_size_h
    for tool_name in TOOL_ORDER:
        tool_buttons[tool_name] = Rect2D(
            x=left_rail.x + 4.0,
            y=cur_y,
            w=rail_w - 8.0,
            h=btn_size_h,
        )
        cur_y -= (btn_size_h + btn_gap)

    # 4. 情報カード (画面右上・Nパネルオフセット考慮)
    card_w = 250.0
    card_h = 320.0
    info_card = Rect2D(x=sw - margin - ro - card_w, y=header.y - 8.0 - card_h, w=card_w, h=card_h)

    # 情報カード内のクリック入力可能フィールド（テキスト描画のベースライン位置に厳密同期）
    # renderer.py での行描画:
    #   y_cursor = info_card.top - 20.0 (行0: Piece)
    #   y_cursor = info_card.top - 38.0 (行1: Nodes)
    #   y_cursor = info_card.top - 56.0 (行2: Resolution)
    #   y_cursor = info_card.top - 74.0 (行3: Layer)
    #   y_cursor = info_card.top - 92.0 (行4: Bone)
    #   y_cursor = info_card.top - 110.0 (行5: Bone Pos)
    #   y_cursor = info_card.top - 128.0 (行6: Bone Angle)
    #   y_cursor = info_card.top - 146.0 (行7: Bone Offset)
    #   y_cursor = info_card.top - 164.0 (行8: Pattern Up)
    field_h = 18.0
    card_fields: dict[str, Rect2D] = {
        "target_edge_length": Rect2D(
            x=info_card.x + 6.0,
            y=(info_card.top - 56.0) - 5.0,
            w=card_w - 12.0,
            h=field_h,
        ),
        "layer_id": Rect2D(
            x=info_card.x + 6.0,
            y=(info_card.top - 74.0) - 5.0,
            w=card_w - 12.0,
            h=field_h,
        ),
        "target_bone": Rect2D(
            x=info_card.x + 6.0,
            y=(info_card.top - 92.0) - 5.0,
            w=card_w - 12.0,
            h=field_h,
        ),
        "bone_t": Rect2D(
            x=info_card.x + 6.0,
            y=(info_card.top - 110.0) - 5.0,
            w=card_w - 12.0,
            h=field_h,
        ),
        "bone_angle_deg": Rect2D(
            x=info_card.x + 6.0,
            y=(info_card.top - 128.0) - 5.0,
            w=card_w - 12.0,
            h=field_h,
        ),
        "surface_offset": Rect2D(
            x=info_card.x + 6.0,
            y=(info_card.top - 146.0) - 5.0,
            w=card_w - 12.0,
            h=field_h,
        ),
        "pattern_up_angle_deg": Rect2D(
            x=info_card.x + 6.0,
            y=(info_card.top - 164.0) - 5.0,
            w=card_w - 12.0,
            h=field_h,
        ),
        "notch_from_start": Rect2D(
            x=info_card.x + 6.0,
            y=(info_card.top - 38.0) - 5.0,
            w=card_w - 12.0,
            h=field_h,
        ),
        "notch_ratio": Rect2D(
            x=info_card.x + 6.0,
            y=(info_card.top - 74.0) - 5.0,
            w=card_w - 12.0,
            h=field_h,
        ),
        # 辺選択カード用フィールド (行2〜行10)
        "edge_subdivide": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 56.0) - 5.0, w=card_w - 12.0, h=field_h),
        "edge_unfold": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 74.0) - 5.0, w=card_w - 12.0, h=field_h),
        "edge_mirror_dup": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 92.0) - 5.0, w=card_w - 12.0, h=field_h),
        "edge_copy_shape": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 110.0) - 5.0, w=card_w - 12.0, h=field_h),
        "edge_paste_shape": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 128.0) - 5.0, w=card_w - 12.0, h=field_h),
        "shape_match_mode": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 146.0) - 5.0, w=card_w - 12.0, h=field_h),
        "shape_match_apply_endpoint_offset": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 164.0) - 5.0, w=card_w - 12.0, h=field_h),
        "shape_match_reverse": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 182.0) - 5.0, w=card_w - 12.0, h=field_h),
        "shape_match_flip_curvature": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 200.0) - 5.0, w=card_w - 12.0, h=field_h),
        "shape_match_ease": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 218.0) - 5.0, w=card_w - 12.0, h=field_h),
        "edge_match_seam": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 236.0) - 5.0, w=card_w - 12.0, h=field_h),
        # 縫合選択カード用フィールド (行6〜行12)
        "seam_match_a_to_b": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 128.0) - 5.0, w=card_w - 12.0, h=field_h),
        "seam_match_b_to_a": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 146.0) - 5.0, w=card_w - 12.0, h=field_h),
        "seam_match_mode": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 164.0) - 5.0, w=card_w - 12.0, h=field_h),
        "seam_match_apply_endpoint_offset": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 182.0) - 5.0, w=card_w - 12.0, h=field_h),
        "seam_match_flip_curvature": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 200.0) - 5.0, w=card_w - 12.0, h=field_h),
        "seam_match_ease": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 218.0) - 5.0, w=card_w - 12.0, h=field_h),
        "seam_routing": Rect2D(x=info_card.x + 6.0, y=(info_card.top - 236.0) - 5.0, w=card_w - 12.0, h=field_h),
    }

    compass_rect = Rect2D(
        x=info_card.x + 15.0,
        y=info_card.y + 12.0,
        w=card_w - 30.0,
        h=110.0,
    )

    return EditorLayout(
        header_bar=header,
        left_rail=left_rail,
        tool_buttons=tool_buttons,
        info_card=info_card,
        card_fields=card_fields,
        compass_rect=compass_rect,
        footer_bar=footer,
        build_button=build_btn,
        close_button=close_btn,
    )
