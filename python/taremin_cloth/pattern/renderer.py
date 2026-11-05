# -*- coding: utf-8 -*-
"""型紙エディタの描画コマンド生成モジュール (Blender非依存)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

from .cards import format_selection_card_lines
from .document import Document, NodeKind, PathRole
from .editor import EditorController
from .geometry import (
    Vec2,
    build_arc_table,
    compute_auto_handles,
    make_segment_controls,
)
from .layout import compute_hud_layout
from .session import ViewTransform2D
from .strings import TOOL_GUIDES, TOOL_NAMES
from .theme import DEFAULT_THEME, ColorRGBA, ThemeColors


class DrawCommandKind(str, Enum):
    """描画コマンド種別."""
    LINE = "LINE"
    POLYLINE = "POLYLINE"
    POLYGON = "POLYGON"
    CIRCLE = "CIRCLE"
    RECT = "RECT"
    TEXT = "TEXT"


@dataclass
class DrawCommand:
    """単一の2D描画コマンド (スクリーンピクセル座標系)."""
    kind: DrawCommandKind
    color: ColorRGBA
    points: list[tuple[float, float]] = field(default_factory=list)
    width: float = 1.0
    radius: float = 4.0
    filled: bool = True
    text: str = ""
    font_size: int = 12


def build_render_commands(
    doc: Document,
    controller: EditorController,
    *,
    screen_size: tuple[int, int],
    theme: Optional[ThemeColors] = None,
) -> list[DrawCommand]:
    """エディタの全可視要素（グリッド、ピース、輪郭、ノード、縫合、HUD）の描画コマンド列を構築して返す."""
    th = theme or DEFAULT_THEME
    view = controller.view
    sw, sh = screen_size
    commands: list[DrawCommand] = []

    # 1. グリッド線の生成
    # 画面四隅のワールド座標を逆算
    w_min_x, w_min_y = view.screen_to_world((0.0, 0.0))
    w_max_x, w_max_y = view.screen_to_world((float(sw), float(sh)))
    w_left, w_right = min(w_min_x, w_max_x), max(w_min_x, w_max_x)
    w_bottom, w_top = min(w_min_y, w_max_y), max(w_min_y, w_max_y)

    grid_step = 0.05  # 5cm刻み
    if view.zoom < 300.0:
        grid_step = 0.1  # ズームアウト時は10cm
    elif view.zoom > 3000.0:
        grid_step = 0.01  # ズームイン時は1cm

    start_x = math.floor(w_left / grid_step) * grid_step
    end_x = math.ceil(w_right / grid_step) * grid_step
    x_val = start_x
    while x_val <= end_x:
        s_p0 = view.world_to_screen((x_val, w_bottom))
        s_p1 = view.world_to_screen((x_val, w_top))
        is_major = abs(round(x_val / (grid_step * 2.0)) * (grid_step * 2.0) - x_val) < 1e-5
        col = th.grid_major if is_major else th.grid_minor
        commands.append(DrawCommand(
            kind=DrawCommandKind.LINE,
            color=col,
            points=[s_p0, s_p1],
            width=1.0,
        ))
        x_val += grid_step

    start_y = math.floor(w_bottom / grid_step) * grid_step
    end_y = math.ceil(w_top / grid_step) * grid_step
    y_val = start_y
    while y_val <= end_y:
        s_p0 = view.world_to_screen((w_left, y_val))
        s_p1 = view.world_to_screen((w_right, y_val))
        is_major = abs(round(y_val / (grid_step * 2.0)) * (grid_step * 2.0) - y_val) < 1e-5
        col = th.grid_major if is_major else th.grid_minor
        commands.append(DrawCommand(
            kind=DrawCommandKind.LINE,
            color=col,
            points=[s_p0, s_p1],
            width=1.0,
        ))
        y_val += grid_step

    # 2. ピースの面塗りつぶし・輪郭線・ノード
    for piece_id, piece in doc.pieces.items():
        is_piece_selected = controller.selection.is_piece_selected(piece_id)
        fill_col = th.piece_fill_selected if is_piece_selected else th.piece_fill
        outline_col = th.piece_outline_selected if is_piece_selected else th.piece_outline

        # ピース外周のポリゴン塗りつぶし
        outer_path = piece.paths.get(piece.outer_path_id)
        if outer_path and len(outer_path.nodes) >= 3:
            s_poly_pts: list[tuple[float, float]] = []
            for nd in outer_path.nodes:
                w_pt = piece.transform.transform_point(nd.position)
                s_poly_pts.append(view.world_to_screen(w_pt))
            commands.append(DrawCommand(
                kind=DrawCommandKind.POLYGON,
                color=fill_col,
                points=s_poly_pts,
                filled=True,
            ))

        # パス輪郭線（ベジェ曲線の分割サンプリング）
        for path_id, path in piece.paths.items():
            n_nodes = len(path.nodes)
            if n_nodes < 2:
                continue

            num_segs = n_nodes if path.closed else n_nodes - 1
            for i in range(num_segs):
                n0 = path.nodes[i]
                n1 = path.nodes[(i + 1) % n_nodes]

                out_h = n0.out_handle
                in_h = n1.in_handle
                if n0.kind == NodeKind.AUTO:
                    p_prev = path.nodes[(i - 1) % n_nodes].position if path.closed else path.nodes[max(0, i - 1)].position
                    _, out_h = compute_auto_handles(p_prev, n0.position, n1.position)
                if n1.kind == NodeKind.AUTO:
                    p_next = path.nodes[(i + 2) % n_nodes].position if path.closed else path.nodes[min(n_nodes - 1, i + 2)].position
                    in_h, _ = compute_auto_handles(n0.position, n1.position, p_next)

                ctrls = make_segment_controls(n0.position, out_h, in_h, n1.position)
                table = build_arc_table(*ctrls)

                s_seg_pts: list[tuple[float, float]] = []
                for pt in table.points:
                    w_pt = piece.transform.transform_point(pt)
                    s_seg_pts.append(view.world_to_screen(w_pt))

                commands.append(DrawCommand(
                    kind=DrawCommandKind.POLYLINE,
                    color=outline_col,
                    points=s_seg_pts,
                    width=2.5 if is_piece_selected else 1.5,
                ))

            # ノード描画
            for nd in path.nodes:
                w_pt = piece.transform.transform_point(nd.position)
                s_pt = view.world_to_screen(w_pt)
                is_node_sel = controller.selection.is_node_selected(nd.id)

                if is_node_sel:
                    node_col = th.node_selected
                elif nd.kind == NodeKind.SMOOTH:
                    node_col = th.node_smooth
                elif nd.kind == NodeKind.AUTO:
                    node_col = th.node_auto
                else:
                    node_col = th.node_corner

                commands.append(DrawCommand(
                    kind=DrawCommandKind.CIRCLE,
                    color=node_col,
                    points=[s_pt],
                    radius=5.0 if is_node_sel else 3.5,
                    filled=True,
                ))

                # 選択中ノードのハンドル描画
                if is_node_sel and nd.kind != NodeKind.CORNER:
                    if nd.in_handle is not None:
                        h_pos = (nd.position[0] + nd.in_handle[0], nd.position[1] + nd.in_handle[1])
                        w_h = piece.transform.transform_point(h_pos)
                        s_h = view.world_to_screen(w_h)
                        commands.append(DrawCommand(
                            kind=DrawCommandKind.LINE,
                            color=th.handle_line,
                            points=[s_pt, s_h],
                            width=1.0,
                        ))
                        commands.append(DrawCommand(
                            kind=DrawCommandKind.CIRCLE,
                            color=th.handle_point,
                            points=[s_h],
                            radius=3.0,
                            filled=True,
                        ))
                    if nd.out_handle is not None:
                        h_pos = (nd.position[0] + nd.out_handle[0], nd.position[1] + nd.out_handle[1])
                        w_h = piece.transform.transform_point(h_pos)
                        s_h = view.world_to_screen(w_h)
                        commands.append(DrawCommand(
                            kind=DrawCommandKind.LINE,
                            color=th.handle_line,
                            points=[s_pt, s_h],
                            width=1.0,
                        ))
                        commands.append(DrawCommand(
                            kind=DrawCommandKind.CIRCLE,
                            color=th.handle_point,
                            points=[s_h],
                            radius=3.0,
                            filled=True,
                        ))

    # 3. 縫合線 (Seams) の描画
    for seam in doc.seams.values():
        _, _, mismatch = doc.compute_seam_lengths(seam.id)
        seam_col = th.seam_mismatch if mismatch > 0.15 else th.seam_line

        # 辺Aと辺Bのノード位置から中点を算出
        p_a = doc.pieces.get(seam.a.piece_id)
        p_b = doc.pieces.get(seam.b.piece_id)
        if p_a and p_b:
            path_a = p_a.paths.get(seam.a.path_id)
            path_b = p_b.paths.get(seam.b.path_id)
            if path_a and path_b:
                n_a0 = next((n for n in path_a.nodes if n.id == seam.a.start_node_id), None)
                n_a1 = next((n for n in path_a.nodes if n.id == seam.a.end_node_id), None)
                n_b0 = next((n for n in path_b.nodes if n.id == seam.b.start_node_id), None)
                n_b1 = next((n for n in path_b.nodes if n.id == seam.b.end_node_id), None)

                if n_a0 and n_a1 and n_b0 and n_b1:
                    mid_a = ((n_a0.position[0] + n_a1.position[0]) * 0.5, (n_a0.position[1] + n_a1.position[1]) * 0.5)
                    mid_b = ((n_b0.position[0] + n_b1.position[0]) * 0.5, (n_b0.position[1] + n_b1.position[1]) * 0.5)
                    w_a = p_a.transform.transform_point(mid_a)
                    w_b = p_b.transform.transform_point(mid_b)
                    s_a = view.world_to_screen(w_a)
                    s_b = view.world_to_screen(w_b)

                    commands.append(DrawCommand(
                        kind=DrawCommandKind.LINE,
                        color=seam_col,
                        points=[s_a, s_b],
                        width=1.5,
                    ))

    # 4. 縫合候補ハイライト
    for er in controller.preview_seam_candidates:
        piece = doc.pieces.get(er.piece_id)
        if piece:
            path = piece.paths.get(er.path_id)
            if path:
                n0 = next((n for n in path.nodes if n.id == er.start_node_id), None)
                n1 = next((n for n in path.nodes if n.id == er.end_node_id), None)
                if n0 and n1:
                    w0 = piece.transform.transform_point(n0.position)
                    w1 = piece.transform.transform_point(n1.position)
                    s0 = view.world_to_screen(w0)
                    s1 = view.world_to_screen(w1)
                    commands.append(DrawCommand(
                        kind=DrawCommandKind.LINE,
                        color=th.seam_candidate,
                        points=[s0, s1],
                        width=3.5,
                    ))

    # 5. 作図中プレビュー線
    if len(controller.preview_points) >= 2:
        if controller.active_tool_name == "RECTANGLE":
            p0 = controller.preview_points[0]
            p1 = controller.preview_points[1]
            rect_pts = [
                view.world_to_screen((p0[0], p0[1])),
                view.world_to_screen((p1[0], p0[1])),
                view.world_to_screen((p1[0], p1[1])),
                view.world_to_screen((p0[0], p1[1])),
                view.world_to_screen((p0[0], p0[1])),
            ]
            commands.append(DrawCommand(
                kind=DrawCommandKind.POLYLINE,
                color=th.preview_line,
                points=rect_pts,
                width=1.5,
            ))
        else:
            s_pts = [view.world_to_screen(pt) for pt in controller.preview_points]
            commands.append(DrawCommand(
                kind=DrawCommandKind.POLYLINE,
                color=th.preview_line,
                points=s_pts,
                width=1.5,
            ))

    # 6. HUD UI 要素の描画
    layout = compute_hud_layout(sw, sh)

    # ヘッダーバー
    commands.append(DrawCommand(
        kind=DrawCommandKind.RECT,
        color=th.hud_bg,
        points=[(layout.header_bar.x, layout.header_bar.y), (layout.header_bar.right, layout.header_bar.top)],
        filled=True,
    ))
    tool_label = TOOL_NAMES.get(controller.active_tool_name, controller.active_tool_name)
    commands.append(DrawCommand(
        kind=DrawCommandKind.TEXT,
        color=th.hud_text,
        points=[(layout.header_bar.x + 12.0, layout.header_bar.y + 8.0)],
        text=f"{doc.name} | ツール: {tool_label}",
        font_size=13,
    ))

    # フッターバー（操作ガイド）
    commands.append(DrawCommand(
        kind=DrawCommandKind.RECT,
        color=th.hud_bg,
        points=[(layout.footer_bar.x, layout.footer_bar.y), (layout.footer_bar.right, layout.footer_bar.top)],
        filled=True,
    ))
    guide_text = TOOL_GUIDES.get(controller.active_tool_name, "")
    commands.append(DrawCommand(
        kind=DrawCommandKind.TEXT,
        color=th.hud_text_dim,
        points=[(layout.footer_bar.x + 12.0, layout.footer_bar.y + 8.0)],
        text=guide_text,
        font_size=12,
    ))

    # 選択情報カード
    card_lines = format_selection_card_lines(doc, controller.selection)
    if card_lines:
        commands.append(DrawCommand(
            kind=DrawCommandKind.RECT,
            color=th.hud_bg,
            points=[(layout.info_card.x, layout.info_card.y), (layout.info_card.right, layout.info_card.top)],
            filled=True,
        ))
        y_cursor = layout.info_card.top - 20.0
        for line in card_lines:
            commands.append(DrawCommand(
                kind=DrawCommandKind.TEXT,
                color=th.hud_text,
                points=[(layout.info_card.x + 10.0, y_cursor)],
                text=line,
                font_size=11,
            ))
            y_cursor -= 18.0

    return commands
