# -*- coding: utf-8 -*-
"""型紙エディタの描画コマンド生成モジュール (Blender非依存)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .cards import format_selection_card_lines
from .document import Document, NodeKind, NotchKind, PathRole
from .editor import EditorController
from .geometry import (
    build_arc_table,
    evaluate_cubic,
    point_at_arc_length,
    arc_length_at_t,
)
from .layout import compute_hud_layout
from .strings import get_tool_guide, get_tool_name
from .theme import DEFAULT_THEME, ColorRGBA, ThemeColors
from .units import format_cm
from ..i18n import trans


class DrawCommandKind(str, Enum):
    """描画コマンド種別."""
    LINE = "LINE"
    POLYLINE = "POLYLINE"
    POLYGON = "POLYGON"
    CIRCLE = "CIRCLE"
    RECT = "RECT"
    TEXT = "TEXT"
    IMAGE = "IMAGE"


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
    image_path: str = ""


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

    # 1.5. 参考下絵画像（Reference Image）の描画
    if doc.reference_image and doc.reference_image.visible and doc.reference_image.image_path:
        ref_img = doc.reference_image
        rx, ry = ref_img.offset
        rw = max(0.01, ref_img.width)
        rh = rw * getattr(ref_img, "aspect_ratio", 1.0)
        # 4隅のワールド座標
        c_p0 = view.world_to_screen((rx, ry))
        c_p1 = view.world_to_screen((rx + rw, ry))
        c_p2 = view.world_to_screen((rx + rw, ry + rh))
        c_p3 = view.world_to_screen((rx, ry + rh))
        img_col: ColorRGBA = (1.0, 1.0, 1.0, float(ref_img.opacity))
        commands.append(DrawCommand(
            kind=DrawCommandKind.IMAGE,
            color=img_col,
            points=[c_p0, c_p1, c_p2, c_p3],
            image_path=ref_img.image_path,
        ))
        # 下絵の枠線
        commands.append(DrawCommand(
            kind=DrawCommandKind.POLYLINE,
            color=(0.4, 0.7, 1.0, 0.4),
            points=[c_p0, c_p1, c_p2, c_p3, c_p0],
            width=1.0,
        ))

    # 2. ピースの面塗りつぶし・輪郭線・ノード
    for piece_id, piece in doc.pieces.items():
        is_piece_selected = controller.selection.is_piece_selected(piece_id)
        fill_col = th.piece_fill_selected if is_piece_selected else th.piece_fill
        outline_col = th.piece_outline_selected if is_piece_selected else th.piece_outline

        # ピース外周のポリゴン塗りつぶし (ベジェ曲線サンプリング対応)
        outer_path = piece.paths.get(piece.outer_path_id)
        if outer_path and len(outer_path.nodes) >= 3:
            sampled_outline = outer_path.sample_outline_points(steps_per_segment=16)
            s_poly_pts: list[tuple[float, float]] = []
            for loc_pt in sampled_outline:
                w_pt = piece.transform.transform_point(loc_pt)
                s_poly_pts.append(view.world_to_screen(w_pt))
            commands.append(DrawCommand(
                kind=DrawCommandKind.POLYGON,
                color=fill_col,
                points=s_poly_pts,
                filled=True,
            ))

        # パス輪郭線および内部線（ベジェ曲線の分割サンプリング）
        for path_id, path in piece.paths.items():
            n_nodes = len(path.nodes)
            if n_nodes < 2:
                continue

            is_internal = (path.role == PathRole.INTERNAL)
            if is_internal:
                line_col = th.internal_line_selected if is_piece_selected else th.internal_line
                line_w = 1.8 if is_piece_selected else 1.2
            else:
                line_col = outline_col
                line_w = 2.5 if is_piece_selected else 1.5

            num_segs = n_nodes if path.closed else n_nodes - 1
            for i in range(num_segs):
                n0 = path.nodes[i]
                ctrls = path.get_segment_controls(i)
                table = build_arc_table(*ctrls)

                s_seg_pts: list[tuple[float, float]] = []
                for pt in table.points:
                    w_pt = piece.transform.transform_point(pt)
                    s_seg_pts.append(view.world_to_screen(w_pt))

                commands.append(DrawCommand(
                    kind=DrawCommandKind.POLYLINE,
                    color=line_col,
                    points=s_seg_pts,
                    width=line_w,
                ))

                # 寸法線テキスト（Lキーでトグル、5mm以上のセグメント/部分区間に表示）
                if getattr(controller, "show_lengths", True) and table.total_length > 0.005:
                    bps = doc.get_segment_notch_breakpoints(piece.id, n0.id, path_id=path.id) if not is_internal else [0.0, 1.0]
                    if len(bps) > 2:
                        for k in range(len(bps) - 1):
                            t_start = bps[k]
                            t_end = bps[k + 1]
                            s_start = arc_length_at_t(table, t_start)
                            s_end = arc_length_at_t(table, t_end)
                            sub_len = abs(s_end - s_start)
                            if sub_len > 0.005:
                                t_mid = (t_start + t_end) * 0.5
                                pt_mid = evaluate_cubic(ctrls[0], ctrls[1], ctrls[2], ctrls[3], t_mid)
                                w_mid = piece.transform.transform_point(pt_mid)
                                s_mid = view.world_to_screen(w_mid)
                                commands.append(DrawCommand(
                                    kind=DrawCommandKind.TEXT,
                                    color=th.dimension_text,
                                    points=[(s_mid[0] + 4.0, s_mid[1] + 4.0)],
                                    text=format_cm(sub_len),
                                    font_size=10,
                                ))
                    else:
                        mid_pt, _ = point_at_arc_length(table, table.total_length * 0.5)
                        w_mid = piece.transform.transform_point(mid_pt)
                        s_mid = view.world_to_screen(w_mid)
                        len_str = format_cm(table.total_length)
                        commands.append(DrawCommand(
                            kind=DrawCommandKind.TEXT,
                            color=th.dimension_text,
                            points=[(s_mid[0] + 4.0, s_mid[1] + 4.0)],
                            text=len_str,
                            font_size=10,
                        ))

            # ノード描画
            for nd in path.nodes:
                w_pt = piece.transform.transform_point(nd.position)
                s_pt = view.world_to_screen(w_pt)
                is_node_sel = controller.selection.is_node_selected(nd.id)

                if is_internal:
                    node_col = th.node_selected if is_node_sel else th.internal_line
                    node_rad = 4.0 if is_node_sel else 2.5
                else:
                    if is_node_sel:
                        node_col = th.node_selected
                    elif nd.kind == NodeKind.SMOOTH:
                        node_col = th.node_smooth
                    elif nd.kind == NodeKind.AUTO:
                        node_col = th.node_auto
                    else:
                        node_col = th.node_corner
                    node_rad = 5.0 if is_node_sel else 3.5

                commands.append(DrawCommand(
                    kind=DrawCommandKind.CIRCLE,
                    color=node_col,
                    points=[s_pt],
                    radius=node_rad,
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

    # 2.5 ノッチ (Notches) の描画
    for notch_id, notch in doc.notches.items():
        piece = doc.pieces.get(notch.piece_id)
        if not piece:
            continue
        path = piece.paths.get(notch.path_id)
        if not path or len(path.nodes) < 2:
            continue

        n_nodes = len(path.nodes)
        num_segs = n_nodes if path.closed else n_nodes - 1
        for i in range(num_segs):
            n0 = path.nodes[i]
            if n0.id == notch.segment_start_id:
                ctrls = path.get_segment_controls(i)
                c0, c1, c2, c3 = ctrls
                t = max(0.0, min(1.0, notch.t))
                p_loc = evaluate_cubic(c0, c1, c2, c3, t)

                # 微分により接線方向と外向き法線方向を計算
                dt = 0.001
                if t + dt <= 1.0:
                    p_forward = evaluate_cubic(c0, c1, c2, c3, t + dt)
                    tx, ty = p_forward[0] - p_loc[0], p_forward[1] - p_loc[1]
                else:
                    p_backward = evaluate_cubic(c0, c1, c2, c3, t - dt)
                    tx, ty = p_loc[0] - p_backward[0], p_loc[1] - p_backward[1]

                t_len = math.hypot(tx, ty)
                if t_len > 1e-6:
                    tx /= t_len
                    ty /= t_len
                else:
                    tx, ty = 1.0, 0.0

                # 反時計回り（CCW）外周における内向き法線: (-ty, tx)
                nx, ny = -ty, tx

                d = notch.depth
                hw = notch.width * 0.5

                if notch.kind == NotchKind.SLIT:
                    shape_loc = [p_loc, (p_loc[0] + nx * d, p_loc[1] + ny * d)]
                elif notch.kind == NotchKind.U:
                    shape_loc = [
                        (p_loc[0] - tx * hw, p_loc[1] - ty * hw),
                        (p_loc[0] - tx * hw + nx * d, p_loc[1] - ty * hw + ny * d),
                        (p_loc[0] + tx * hw + nx * d, p_loc[1] + ty * hw + ny * d),
                        (p_loc[0] + tx * hw, p_loc[1] + ty * hw),
                    ]
                else:  # NotchKind.V
                    shape_loc = [
                        (p_loc[0] - tx * hw, p_loc[1] - ty * hw),
                        (p_loc[0] + nx * d, p_loc[1] + ny * d),
                        (p_loc[0] + tx * hw, p_loc[1] + ty * hw),
                    ]

                is_notch_sel = controller.selection.is_notch_selected(notch.id)
                n_col = th.notch_selected if is_notch_sel else th.notch
                s_shape_pts = [view.world_to_screen(piece.transform.transform_point(pt)) for pt in shape_loc]
                commands.append(DrawCommand(
                    kind=DrawCommandKind.POLYLINE,
                    color=n_col,
                    points=s_shape_pts,
                    width=2.5 if is_notch_sel else 1.8,
                ))
                break

    # 3. 縫合線 (Seams) の描画（中心ではなく両端を線で結び、方向性を可視化）
    for seam in doc.seams.values():
        _, _, mismatch = doc.compute_seam_lengths(seam.id)
        is_selected = (seam.id in controller.selection.seam_ids)
        if is_selected:
            seam_col = (1.0, 0.85, 0.2, 1.0)
            line_width = 2.5
        elif mismatch > 0.15:
            seam_col = th.seam_mismatch
            line_width = 1.8
        else:
            seam_col = th.seam_line
            line_width = 1.8

        p_a = doc.pieces.get(seam.a.piece_id)
        p_b = doc.pieces.get(seam.b.piece_id)
        if p_a and p_b:
            w_a0, mid_a, w_a1, match_b0, mid_b, match_b1 = doc.get_seam_geometry(seam)
            s_a0 = view.world_to_screen(w_a0)
            s_a1 = view.world_to_screen(w_a1)
            s_b0 = view.world_to_screen(match_b0)
            s_b1 = view.world_to_screen(match_b1)

            # 端点ペア0を結ぶ主縫合線 (始点側)
            commands.append(DrawCommand(
                kind=DrawCommandKind.LINE,
                color=seam_col,
                points=[s_a0, s_b0],
                width=line_width,
            ))
            # 端点ペア1を結ぶ主縫合線 (終点側)
            commands.append(DrawCommand(
                kind=DrawCommandKind.LINE,
                color=seam_col,
                points=[s_a1, s_b1],
                width=line_width,
            ))

            # 中央を結ぶ薄い補助線（面同士の接続ガイド）
            s_mid_a = view.world_to_screen(mid_a)
            s_mid_b = view.world_to_screen(mid_b)
            guide_col = (seam_col[0], seam_col[1], seam_col[2], seam_col[3] * 0.35)
            commands.append(DrawCommand(
                kind=DrawCommandKind.LINE,
                color=guide_col,
                points=[s_mid_a, s_mid_b],
                width=1.0,
            ))

            # 始点マーカー（小さなドット）でペア0を識別可能にする
            commands.append(DrawCommand(
                kind=DrawCommandKind.CIRCLE,
                color=seam_col,
                points=[s_a0],
                radius=3.0,
                filled=True,
            ))
            commands.append(DrawCommand(
                kind=DrawCommandKind.CIRCLE,
                color=seam_col,
                points=[s_b0],
                radius=3.0,
                filled=True,
            ))

    # 4. 縫合候補ハイライト (ベジェ曲線サンプリング対応)
    for er in controller.preview_seam_candidates:
        _, _, _, w_pts = doc.evaluate_edge_range_geometry(er, steps=16)
        if len(w_pts) >= 2:
            s_pts = [view.world_to_screen(pt) for pt in w_pts]
            commands.append(DrawCommand(
                kind=DrawCommandKind.POLYLINE,
                color=th.seam_candidate,
                points=s_pts,
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
    layout = compute_hud_layout(
        sw,
        sh,
        left_offset=getattr(controller, "left_offset", 0.0),
        right_offset=getattr(controller, "right_offset", 0.0),
        top_offset=getattr(controller, "top_offset", 0.0),
        bottom_offset=getattr(controller, "bottom_offset", 0.0),
    )

    # ヘッダーバー
    commands.append(DrawCommand(
        kind=DrawCommandKind.RECT,
        color=th.hud_bg,
        points=[(layout.header_bar.x, layout.header_bar.y), (layout.header_bar.right, layout.header_bar.top)],
        filled=True,
    ))
    tool_label = get_tool_name(controller.active_tool_name)
    commands.append(DrawCommand(
        kind=DrawCommandKind.TEXT,
        color=th.hud_text,
        points=[(layout.header_bar.x + 12.0, layout.header_bar.y + 8.0)],
        text=f"{doc.name} | {trans('Tool')}: {tool_label}",
        font_size=13,
    ))

    # 左ツールレール
    commands.append(DrawCommand(
        kind=DrawCommandKind.RECT,
        color=th.hud_bg,
        points=[(layout.left_rail.x, layout.left_rail.y), (layout.left_rail.right, layout.left_rail.top)],
        filled=True,
    ))

    tool_short_labels: dict[str, str] = {
        "SELECT": "Sel",
        "RECTANGLE": "Rect",
        "POLYGON": "Poly",
        "ELLIPSE": "Ellip",
        "INTERNAL_LINE": "IntLn",
        "NOTCH": "Notch",
        "SEW": "Sew",
        "CUT": "Cut",
        "DART": "Dart",
    }

    for t_name, btn in layout.tool_buttons.items():
        is_active = (controller.active_tool_name == t_name)
        btn_col: ColorRGBA = (0.28, 0.42, 0.65, 0.95) if is_active else (0.22, 0.24, 0.27, 0.85)
        commands.append(DrawCommand(
            kind=DrawCommandKind.RECT,
            color=btn_col,
            points=[(btn.x, btn.y), (btn.right, btn.top)],
            filled=True,
        ))
        if is_active:
            commands.append(DrawCommand(
                kind=DrawCommandKind.RECT,
                color=th.piece_outline_selected,
                points=[(btn.x, btn.y), (btn.right, btn.top)],
                filled=False,
                width=1.5,
            ))
        short_txt = tool_short_labels.get(t_name, t_name[:4])
        commands.append(DrawCommand(
            kind=DrawCommandKind.TEXT,
            color=th.hud_text if is_active else th.hud_text_dim,
            points=[(btn.x + 6.0, btn.y + 11.0)],
            text=short_txt,
            font_size=11,
        ))

    # フッターバー（操作ガイド）
    commands.append(DrawCommand(
        kind=DrawCommandKind.RECT,
        color=th.hud_bg,
        points=[(layout.footer_bar.x, layout.footer_bar.y), (layout.footer_bar.right, layout.footer_bar.top)],
        filled=True,
    ))
    if controller.picking_bone:
        guide_text = trans("Pick Bone Mode: Click bone in 3D View [RMB/Esc: Cancel, X: Clear]")
    elif controller.sliding_field is not None:
        guide_text = trans("Slider Drag: Move mouse left/right [Shift: Fine tuning]")
    else:
        guide_text = get_tool_guide(controller.active_tool_name)
    commands.append(DrawCommand(
        kind=DrawCommandKind.TEXT,
        color=th.hud_text_dim,
        points=[(layout.footer_bar.x + 12.0, layout.footer_bar.y + 8.0)],
        text=guide_text,
        font_size=12,
    ))

    # フッター右側アクションボタン: Build 3D
    commands.append(DrawCommand(
        kind=DrawCommandKind.RECT,
        color=(0.18, 0.48, 0.28, 0.95),
        points=[(layout.build_button.x, layout.build_button.y), (layout.build_button.right, layout.build_button.top)],
        filled=True,
    ))
    commands.append(DrawCommand(
        kind=DrawCommandKind.RECT,
        color=(0.3, 0.8, 0.45, 1.0),
        points=[(layout.build_button.x, layout.build_button.y), (layout.build_button.right, layout.build_button.top)],
        filled=False,
        width=1.5,
    ))
    commands.append(DrawCommand(
        kind=DrawCommandKind.TEXT,
        color=(1.0, 1.0, 1.0, 1.0),
        points=[(layout.build_button.x + 24.0, layout.build_button.y + 7.0)],
        text=trans("Build 3D"),
        font_size=12,
    ))

    # フッター右側アクションボタン: Close
    commands.append(DrawCommand(
        kind=DrawCommandKind.RECT,
        color=(0.38, 0.22, 0.22, 0.9),
        points=[(layout.close_button.x, layout.close_button.y), (layout.close_button.right, layout.close_button.top)],
        filled=True,
    ))
    commands.append(DrawCommand(
        kind=DrawCommandKind.RECT,
        color=(0.65, 0.35, 0.35, 1.0),
        points=[(layout.close_button.x, layout.close_button.y), (layout.close_button.right, layout.close_button.top)],
        filled=False,
        width=1.0,
    ))
    commands.append(DrawCommand(
        kind=DrawCommandKind.TEXT,
        color=(1.0, 1.0, 1.0, 1.0),
        points=[(layout.close_button.x + 22.0, layout.close_button.y + 7.0)],
        text=trans("Close"),
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

        # スライダー進行バーおよびボーン選択枠の背景描画
        if (
            not controller.selection.notch_ids
            and not controller.selection.seam_ids
            and len(controller.selection.piece_ids) == 1
        ):
            sel_pid = next(iter(controller.selection.piece_ids))
            pl = doc.placements.get(sel_pid)
            if pl:
                slider_ratios = {
                    "bone_t": max(0.0, min(1.0, pl.bone_t)),
                    "bone_angle_deg": max(0.0, min(1.0, (pl.bone_angle_deg + 180.0) / 360.0)),
                    "surface_offset": max(0.0, min(1.0, pl.surface_offset / 0.2)),
                }
                for f_name, ratio in slider_ratios.items():
                    if f_name in layout.card_fields:
                        f_rect = layout.card_fields[f_name]
                        is_active_slider = (controller.sliding_field == f_name)
                        bar_color = (0.22, 0.45, 0.55, 0.65) if is_active_slider else (0.16, 0.28, 0.35, 0.4)
                        commands.append(DrawCommand(
                            kind=DrawCommandKind.RECT,
                            color=bar_color,
                            points=[(f_rect.x, f_rect.y), (f_rect.x + f_rect.w * ratio, f_rect.top)],
                            filled=True,
                        ))
                # Pick Bone モード中の target_bone 行ハイライト枠
                if controller.picking_bone and "target_bone" in layout.card_fields:
                    tb_rect = layout.card_fields["target_bone"]
                    commands.append(DrawCommand(
                        kind=DrawCommandKind.RECT,
                        color=(0.2, 0.85, 1.0, 1.0),
                        points=[(tb_rect.x, tb_rect.y), (tb_rect.right, tb_rect.top)],
                        filled=False,
                        width=1.5,
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

        # 角度コンパス (Compass Widget) の描画
        if (
            not controller.selection.notch_ids
            and not controller.selection.seam_ids
            and len(controller.selection.piece_ids) == 1
        ):
            sel_pid = next(iter(controller.selection.piece_ids))
            pl = doc.placements.get(sel_pid)
            if pl and layout.compass_rect:
                c_rect = layout.compass_rect
                cx = c_rect.x + c_rect.w * 0.5
                cy = c_rect.y + c_rect.h * 0.5
                r = min(c_rect.w, c_rect.h) * 0.36

                # コンパス背景パネル
                commands.append(DrawCommand(
                    kind=DrawCommandKind.RECT,
                    color=(0.14, 0.16, 0.19, 0.9),
                    points=[(c_rect.x, c_rect.y), (c_rect.right, c_rect.top)],
                    filled=True,
                ))
                commands.append(DrawCommand(
                    kind=DrawCommandKind.RECT,
                    color=(0.28, 0.32, 0.38, 0.7),
                    points=[(c_rect.x, c_rect.y), (c_rect.right, c_rect.top)],
                    filled=False,
                    width=1.0,
                ))

                # ボーン断面円（外枠と中心点）
                commands.append(DrawCommand(
                    kind=DrawCommandKind.CIRCLE,
                    color=(0.3, 0.35, 0.42, 0.8),
                    points=[(cx, cy)],
                    radius=r,
                    filled=False,
                    width=1.5,
                ))
                commands.append(DrawCommand(
                    kind=DrawCommandKind.CIRCLE,
                    color=(0.5, 0.55, 0.65, 0.9),
                    points=[(cx, cy)],
                    radius=3.0,
                    filled=True,
                ))

                # 角度針 (法線ベクトル方向)
                # 上が 0° (Front), 右が 90° (Right), 下が 180° (Back), 左が -90° (Left)
                theta_rad = math.radians(pl.bone_angle_deg)
                nx = math.sin(theta_rad)
                ny = math.cos(theta_rad)
                tip = (cx + r * nx, cy + r * ny)

                commands.append(DrawCommand(
                    kind=DrawCommandKind.LINE,
                    color=(0.2, 0.85, 1.0, 1.0),
                    points=[(cx, cy), tip],
                    width=2.5,
                ))
                commands.append(DrawCommand(
                    kind=DrawCommandKind.CIRCLE,
                    color=(0.2, 0.85, 1.0, 1.0),
                    points=[tip],
                    radius=4.0,
                    filled=True,
                ))

                # 接線バー (型紙平面の向き)
                tx, ty = -ny, nx
                bar_len = r * 0.75
                p_bar_a = (tip[0] - tx * bar_len, tip[1] - ty * bar_len)
                p_bar_b = (tip[0] + tx * bar_len, tip[1] + ty * bar_len)
                commands.append(DrawCommand(
                    kind=DrawCommandKind.LINE,
                    color=(1.0, 1.0, 1.0, 0.95),
                    points=[p_bar_a, p_bar_b],
                    width=2.0,
                ))

                # 筒モード (is_tubular == True) の場合の裏側迂回ルート (Waypoints Loop)
                if pl.is_tubular:
                    back_nx = -nx
                    back_ny = -ny
                    r_back = r * 1.15
                    w1 = (cx + r_back * (back_nx * 0.866 - back_ny * 0.5), cy + r_back * (back_ny * 0.866 + back_nx * 0.5))
                    w2 = (cx + r_back * (back_nx * 0.866 + back_ny * 0.5), cy + r_back * (back_ny * 0.866 - back_nx * 0.5))
                    commands.append(DrawCommand(
                        kind=DrawCommandKind.POLYLINE,
                        color=(1.0, 0.6, 0.15, 0.95),
                        points=[p_bar_a, w1, w2, p_bar_b],
                        width=1.8,
                    ))

                # 角度テキスト表示
                deg_label = f"{int(pl.bone_angle_deg)}°"
                commands.append(DrawCommand(
                    kind=DrawCommandKind.TEXT,
                    color=(0.9, 0.9, 0.9, 0.9),
                    points=[(c_rect.x + 8.0, c_rect.y + 6.0)],
                    text=f"{trans('Angle')}: {deg_label}",
                    font_size=10,
                ))

        # インライン編集中の枠線・カーソル描画
        if controller.editing_field and controller.editing_field in layout.card_fields:
            f_rect = layout.card_fields[controller.editing_field]
            commands.append(DrawCommand(
                kind=DrawCommandKind.RECT,
                color=(0.15, 0.18, 0.22, 0.95),
                points=[(f_rect.x, f_rect.y), (f_rect.right, f_rect.top)],
                filled=True,
            ))
            commands.append(DrawCommand(
                kind=DrawCommandKind.RECT,
                color=(1.0, 0.8, 0.2, 1.0),
                points=[(f_rect.x, f_rect.y), (f_rect.right, f_rect.top)],
                filled=False,
                width=1.5,
            ))
            label_map = {
                "target_edge_length": trans("Resolution"),
                "layer_id": trans("Layer"),
                "target_bone": trans("Bone"),
                "bone_t": trans("Bone Pos"),
                "bone_angle_deg": trans("Bone Angle"),
                "surface_offset": trans("Bone Offset"),
                "is_tubular": trans("Tubular"),
                "notch_from_start": trans("From Start"),
                "notch_ratio": trans("Ratio"),
            }
            label = label_map.get(controller.editing_field, controller.editing_field)
            commands.append(DrawCommand(
                kind=DrawCommandKind.TEXT,
                color=(1.0, 0.9, 0.3, 1.0),
                points=[(layout.info_card.x + 10.0, f_rect.y + 5.0)],
                text=f"{label}: {controller.editing_buffer}|",
                font_size=11,
            ))

    return commands
