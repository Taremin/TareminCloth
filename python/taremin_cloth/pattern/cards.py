# -*- coding: utf-8 -*-
"""型紙エディタの情報表示カード (Blender非依存・i18n対応)."""

from __future__ import annotations

import os

from .document import Document, NodeKind, SeamRouting
from .session import SelectionState
from .transform_ops import summarize_multi_node_info
from .units import format_cm
from ..i18n import trans


def _node_kind_label(kind: NodeKind) -> str:
    if kind == NodeKind.SMOOTH:
        return trans("Smooth")
    if kind == NodeKind.AUTO:
        return trans("Auto")
    return trans("Corner")


def format_selection_card_lines(
    doc: Document,
    selection: SelectionState,
    *,
    shape_match_mode: str = "SIMILAR",
    shape_match_reverse: bool = True,
    shape_match_flip_curvature: bool = False,
    shape_match_apply_endpoint_offset: bool = False,
    shape_match_ease_value: float = 0.0,
    shape_match_ease_is_ratio: bool = False,
    has_edge_clipboard: bool = False,
    clipboard_length: float = 0.0,
) -> list[str]:
    """選択中要素（ピース、縫合、ノード、辺等）をHUDカード用のテキスト行リストとして生成する."""
    lines: list[str] = []

    # モードラベル
    if shape_match_mode == "SIMILAR":
        mode_label = trans("Similar")
    elif shape_match_mode == "LENGTH_MOVE_END":
        mode_label = trans("Move End")
    else:
        mode_label = trans("Bulge Len")

    # イセコミ表示
    if shape_match_ease_is_ratio:
        ease_str = f"{shape_match_ease_value * 100.0:+.1f}%"
    else:
        ease_str = f"{shape_match_ease_value * 100.0:+.2f} cm"

    if selection.notch_ids:
        nid = next(iter(selection.notch_ids))
        notch = doc.notches.get(nid)
        if notch:
            piece = doc.pieces.get(notch.piece_id)
            lines.append(f"{trans('Notch')}: {piece.name if piece else ''}")
            segs = doc.evaluate_path_segments(notch.piece_id, notch.path_id)
            target_seg = next((tbl for s_id, _, tbl in segs if s_id == notch.segment_start_id), None)
            if target_seg:
                from .geometry import arc_length_at_t
                s_from_start = arc_length_at_t(target_seg, notch.t)
                s_to_end = max(0.0, target_seg.total_length - s_from_start)
                lines.append(f"{trans('From Start')}: {format_cm(s_from_start)}")
                lines.append(f"{trans('To End')}: {format_cm(s_to_end)}")
                lines.append(f"{trans('Ratio')}: {int(round(notch.t * 100))}%")
            return lines

    if selection.seam_ids:
        sid = next(iter(selection.seam_ids))
        seam = doc.seams.get(sid)
        if seam:
            lines.append(f"{trans('Seam')}: {sid[:6]}")
            len_a, len_b, mismatch = doc.compute_seam_lengths(sid)
            lines.append(f"{trans('Length A')}: {format_cm(len_a)}")
            lines.append(f"{trans('Length B')}: {format_cm(len_b)}")
            lines.append(f"{trans('Mismatch')}: {int(round(mismatch * 100))}%")
            dir_str = trans("Reversed") if seam.reverse else trans("Normal")
            lines.append(f"{trans('Direction')}: {dir_str}")
            lines.append(trans("Press F to flip"))
            lines.append(f"[{trans('Match A -> B (Ctrl+M)')}]")
            lines.append(f"[{trans('Match B -> A (Ctrl+Shift+M)')}]")
            lines.append(f"{trans('Match Mode (O)')}: {mode_label}")
            lines.append(f"{trans('Apply End (Shift+O)')}: {trans('ON') if shape_match_apply_endpoint_offset else trans('OFF')}")
            lines.append(f"{trans('Flip Curve')}: {trans('ON') if shape_match_flip_curvature else trans('OFF')}")
            lines.append(f"{trans('Ease')}: {ease_str}")
            routing_str = trans("Bone Wrap") if seam.routing == SeamRouting.BONE_WRAP else trans("Direct")
            lines.append(f"{trans('Routing (B)')}: {routing_str}")
            return lines

    if selection.edge_ranges:
        er = selection.get_active_edge_range()
        if er:
            piece = doc.pieces.get(er.piece_id)
            path = piece.paths.get(er.path_id) if piece else None
            n_edges = len(selection.edge_ranges)
            if n_edges == 1:
                len_m = doc.compute_edge_range_length(er)
                is_curved = False
                if path:
                    res = path.resolve_edge_range(er)
                    if res:
                        seg_i = res[0]
                        n_nodes = len(path.nodes)
                        nd0 = path.nodes[seg_i]
                        nd1 = path.nodes[(seg_i + 1) % n_nodes]
                        is_curved = (
                            nd0.kind != NodeKind.CORNER
                            or nd1.kind != NodeKind.CORNER
                            or nd0.out_handle is not None
                            or nd1.in_handle is not None
                        )
                kind_str = trans("Curved") if is_curved else trans("Straight")
                lines.append(f"{trans('Edge')}: {piece.name if piece else ''}")
                lines.append(f"{trans('Length')}: {format_cm(len_m)} ({kind_str})")
            else:
                total_len_m = sum(doc.compute_edge_range_length(item) for item in selection.edge_ranges)
                lines.append(f"{trans('Edges')}: {n_edges} {trans('selected')}")
                lines.append(f"{trans('Total Length')}: {format_cm(total_len_m)}")

            lines.append(f"[{trans('Subdivide (W)')}]")
            lines.append(f"[{trans('Unfold (U)')}]")
            lines.append(f"[{trans('Mirror Dup (Shift+U)')}]")
            lines.append(f"[{trans('Copy Shape (Ctrl+C)')}]")
            if has_edge_clipboard:
                lines.append(f"[{trans('Paste Shape (Ctrl+V)')}] ({format_cm(clipboard_length)})")
            else:
                lines.append(f"[{trans('Paste Shape (Ctrl+V)')}]")
            lines.append(f"{trans('Match Mode (O)')}: {mode_label}")
            lines.append(f"{trans('Apply End (Shift+O)')}: {trans('ON') if shape_match_apply_endpoint_offset else trans('OFF')}")
            lines.append(f"{trans('Reverse Dir')}: {trans('ON') if shape_match_reverse else trans('OFF')}")
            lines.append(f"{trans('Flip Curve')}: {trans('ON') if shape_match_flip_curvature else trans('OFF')}")
            lines.append(f"{trans('Ease')}: {ease_str}")

            has_connected_seam = any(
                any(
                    s_er.piece_id == er.piece_id
                    and s_er.path_id == er.path_id
                    and {s_er.start_node_id, s_er.end_node_id} == {er.start_node_id, er.end_node_id}
                    for s_er in (seam.effective_a() + seam.effective_b())
                )
                for seam in doc.seams.values()
            )
            if has_connected_seam:
                lines.append(f"[{trans('Match Seam (Ctrl+M)')}]")
            return lines

    # ピース非選択時のノード選択（1点 / 複数点）
    if selection.node_ids and not selection.piece_ids:
        nodes_info, (bbox_w, bbox_h) = summarize_multi_node_info(doc, selection)
        if len(nodes_info) == 1:
            piece, nd, w_pos = nodes_info[0]
            lines.append(f"{trans('Node')}: {piece.name}")
            lines.append(f"{trans('Pos')}: ({format_cm(w_pos[0])}, {format_cm(w_pos[1])})")
            lines.append(f"{trans('Kind (C)')}: {_node_kind_label(nd.kind)}")
            lines.append(trans("G: Move / X: Delete"))
            return lines
        if len(nodes_info) >= 2:
            piece_names = {p.name for p, _, _ in nodes_info}
            p_label = next(iter(piece_names)) if len(piece_names) == 1 else f"{len(piece_names)} {trans('pieces')}"
            lines.append(f"{trans('Selected Nodes')}: {len(nodes_info)} ({p_label})")
            lines.append(f"{trans('Size')}: {format_cm(bbox_w)} x {format_cm(bbox_h)}")
            lines.append(trans("G: Move / R: Rotate / S: Scale"))
            lines.append(f"{trans('Align X')}:")
            lines.append("")  # 行4: Align X ボタン枠用スペース
            lines.append(f"{trans('Align Y')}:")
            lines.append("")  # 行6: Align Y ボタン枠用スペース
            return lines

    if not selection.has_any_selection():
        if doc.reference_image:
            ref_img = doc.reference_image
            img_name = os.path.basename(ref_img.image_path) if ref_img.image_path else trans("None")
            if len(img_name) > 18:
                img_name = img_name[:15] + "..."
            vis_str = trans("ON") if ref_img.visible else trans("OFF")
            opacity_pct = int(round(max(0.0, min(1.0, float(ref_img.opacity))) * 100))
            lines.append(f"{trans('Ref Image')}: {img_name}")
            lines.append(f"{trans('Visible')}: {vis_str}")
            lines.append(f"{trans('Opacity')}: {opacity_pct}%")
            lines.append(f"{trans('Width')}: {format_cm(ref_img.width)}")
            lines.append(f"{trans('Offset X')}: {format_cm(ref_img.offset[0])}")
            lines.append(f"{trans('Offset Y')}: {format_cm(ref_img.offset[1])}")
            lines.append(f"[{trans('Clear Image')}]")
            lines.append(trans("Click piece or node"))
            return lines
        lines.append(trans("No Selection"))
        lines.append(trans("Click piece or node"))
        return lines

    if len(selection.piece_ids) == 1:
        pid = next(iter(selection.piece_ids))
        piece = doc.pieces.get(pid)
        if piece:
            lines.append(f"{trans('Piece')}: {piece.name}")
            outer = piece.paths.get(piece.outer_path_id)
            if outer:
                lines.append(f"{trans('Nodes')}: {len(outer.nodes)}")
            lines.append(f"{trans('Resolution')}: {format_cm(piece.target_edge_length)}")
            lines.append(f"{trans('Layer')}: {piece.layer_id}")
            pl = doc.placements.get(pid)
            bone_str = pl.target_bone if (pl and pl.target_bone) else trans("None")
            lines.append(f"{trans('Bone')}: {bone_str}")
            bone_t_val = pl.bone_t if pl else 0.5
            lines.append(f"{trans('Bone Pos')}: {int(bone_t_val * 100)}%")
            angle_val = pl.bone_angle_deg if pl else 0.0
            lines.append(f"{trans('Bone Angle')}: {int(angle_val)}°")
            offset_val = pl.surface_offset if pl else 0.08
            lines.append(f"{trans('Bone Offset')}: {format_cm(offset_val)}")
            up_angle_val = pl.pattern_up_angle_deg if pl else 0.0
            lines.append(f"{trans('Pattern Up (T)')}: {int(round(up_angle_val))}°")
            if selection.node_ids:
                lines.append(f"{trans('Selected Nodes')}: {len(selection.node_ids)}")
    elif len(selection.piece_ids) > 1:
        selected_pieces = [doc.pieces[pid] for pid in selection.piece_ids if pid in doc.pieces]
        if selected_pieces:
            total_nodes = sum(
                len(p.paths[p.outer_path_id].nodes)
                for p in selected_pieces
                if p.outer_path_id in p.paths
            )
            mixed_lbl = trans("Mixed")

            def fmt_uniform(values: list, formatter) -> str:
                first = formatter(values[0])
                return first if all(formatter(v) == first for v in values[1:]) else mixed_lbl

            res_str = fmt_uniform([p.target_edge_length for p in selected_pieces], format_cm)
            layer_str = fmt_uniform([p.layer_id for p in selected_pieces], str)
            pls = [doc.placements.get(p.id) for p in selected_pieces]
            bone_vals = [(pl.target_bone if (pl and pl.target_bone) else "") for pl in pls]
            bone_str = fmt_uniform(bone_vals, lambda b: b if b else trans("None"))
            bone_t_str = fmt_uniform([pl.bone_t if pl else 0.5 for pl in pls], lambda v: f"{int(v * 100)}%")
            angle_str = fmt_uniform([pl.bone_angle_deg if pl else 0.0 for pl in pls], lambda v: f"{int(v)}°")
            offset_str = fmt_uniform([pl.surface_offset if pl else 0.08 for pl in pls], format_cm)
            up_str = fmt_uniform([pl.pattern_up_angle_deg if pl else 0.0 for pl in pls], lambda v: f"{int(round(v))}°")

            lines.append(f"{trans('Pieces')}: {len(selected_pieces)} {trans('selected')}")
            lines.append(f"{trans('Nodes')}: {total_nodes}")
            lines.append(f"{trans('Resolution')}: {res_str}")
            lines.append(f"{trans('Layer')}: {layer_str}")
            lines.append(f"{trans('Bone')}: {bone_str}")
            lines.append(f"{trans('Bone Pos')}: {bone_t_str}")
            lines.append(f"{trans('Bone Angle')}: {angle_str}")
            lines.append(f"{trans('Bone Offset')}: {offset_str}")
            lines.append(f"{trans('Pattern Up (T)')}: {up_str}")
            if selection.node_ids:
                lines.append(f"{trans('Selected Nodes')}: {len(selection.node_ids)}")

    return lines
