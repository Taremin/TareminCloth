# -*- coding: utf-8 -*-
"""型紙エディタの情報表示カード (Blender非依存)."""

from __future__ import annotations

from typing import Optional

from .document import Document
from .session import SelectionState
from .units import format_cm


def format_selection_card_lines(
    doc: Document,
    selection: SelectionState,
) -> list[str]:
    """選択中ピースの各種プロパティ（名前、ノード数、寸法等）をHUDカード用のテキスト行リストとして生成する."""
    lines: list[str] = []
    if not selection.piece_ids:
        lines.append("未選択 (No Selection)")
        lines.append("ピースまたはノードをクリック")
        return lines

    if len(selection.piece_ids) == 1:
        pid = next(iter(selection.piece_ids))
        piece = doc.pieces.get(pid)
        if piece:
            lines.append(f"ピース: {piece.name}")
            outer = piece.paths.get(piece.outer_path_id)
            if outer:
                lines.append(f"ノード数: {len(outer.nodes)}")
            lines.append(f"解像度: {format_cm(piece.target_edge_length)}")
            lines.append(f"レイヤー: {piece.layer_id}")
            if selection.node_ids:
                lines.append(f"選択ノード: {len(selection.node_ids)} 個")
    else:
        lines.append(f"選択ピース: {len(selection.piece_ids)} 個")
        if selection.node_ids:
            lines.append(f"選択ノード: {len(selection.node_ids)} 個")

    return lines
