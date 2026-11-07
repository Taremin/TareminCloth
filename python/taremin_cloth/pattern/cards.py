# -*- coding: utf-8 -*-
"""型紙エディタの情報表示カード (Blender非依存・i18n対応)."""

from __future__ import annotations

from .document import Document
from .session import SelectionState
from .units import format_cm
from ..i18n import trans


def format_selection_card_lines(
    doc: Document,
    selection: SelectionState,
) -> list[str]:
    """選択中要素（ピース、縫合、ノード等）をHUDカード用のテキスト行リストとして生成する."""
    lines: list[str] = []
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
            if abs(len_a - len_b) > 0.001:
                lines.append(f"{trans('Mismatch')}: {int(round(mismatch * 100))}%")
            dir_str = trans("Reversed") if seam.reverse else trans("Normal")
            lines.append(f"{trans('Direction')}: {dir_str}")
            lines.append(trans("Press R to flip"))
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
