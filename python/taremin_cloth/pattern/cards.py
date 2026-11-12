# -*- coding: utf-8 -*-
"""型紙エディタの情報表示カード (Blender非依存・i18n対応)."""

from __future__ import annotations

from .document import Document, NodeKind
from .session import SelectionState
from .units import format_cm
from ..i18n import trans


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
            lines.append(trans("Press R to flip"))
            lines.append(f"[{trans('Match A -> B (Ctrl+M)')}]")
            lines.append(f"[{trans('Match B -> A (Ctrl+Shift+M)')}]")
            lines.append(f"{trans('Match Mode (O)')}: {mode_label}")
            lines.append(f"{trans('Apply End (Shift+O)')}: {trans('ON') if shape_match_apply_endpoint_offset else trans('OFF')}")
            lines.append(f"{trans('Flip Curve')}: {trans('ON') if shape_match_flip_curvature else trans('OFF')}")
            lines.append(f"{trans('Ease')}: {ease_str}")
            return lines

    if selection.edge_ranges:
        er = selection.get_active_edge_range()
        if er:
            piece = doc.pieces.get(er.piece_id)
            path = piece.paths.get(er.path_id) if piece else None
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

            # 接続された縫合線の有無をチェックして案内
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

    if not selection.piece_ids:
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
            tubular_val = pl.is_tubular if pl else False
            lines.append(f"{trans('Tubular')}: {trans('ON') if tubular_val else trans('OFF')}")
            if selection.node_ids:
                lines.append(f"{trans('Selected Nodes')}: {len(selection.node_ids)}")
    else:
        lines.append(f"{trans('Selected Pieces')}: {len(selection.piece_ids)}")
        if selection.node_ids:
            lines.append(f"{trans('Selected Nodes')}: {len(selection.node_ids)}")

    return lines
