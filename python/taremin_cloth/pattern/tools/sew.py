# -*- coding: utf-8 -*-
"""型紙エディタの縫合定義ツール (Blender非依存)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..document import Document, EdgeRange, Seam, SeamMode
from ..ids import new_id
from ..selection import HitResult, HitTargetKind, hit_test
from .base import BaseTool

if TYPE_CHECKING:
    from ..editor import EditorController, EditorEvent


class SewTool(BaseTool):
    """2つの辺範囲をクリック選択して縫合 (Seam) を定義するツール."""

    name: str = "SEW"

    def __init__(self) -> None:
        self.first_edge: Optional[EdgeRange] = None

    def on_mouse_down(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.button != "LEFT":
            return False

        # 辺のヒットテスト
        hit = hit_test(controller.doc, event.world_pos)
        if hit is None or hit.kind != HitTargetKind.EDGE or not hit.path_id:
            return False

        if not hit.segment_start_node_id or not hit.segment_end_node_id:
            return False

        edge_range = EdgeRange(
            piece_id=hit.piece_id,
            path_id=hit.path_id,
            start_node_id=hit.segment_start_node_id,
            end_node_id=hit.segment_end_node_id,
        )

        if self.first_edge is None:
            # 1つ目の辺を選択
            self.first_edge = edge_range
            controller.preview_seam_candidates = [edge_range]
            return True
        else:
            # 2つ目の辺を選択 -> 縫合作成
            # 同一边の自己縫合は禁止
            if (
                self.first_edge.piece_id == edge_range.piece_id
                and self.first_edge.path_id == edge_range.path_id
                and self.first_edge.start_node_id == edge_range.start_node_id
                and self.first_edge.end_node_id == edge_range.end_node_id
            ):
                return True

            before_doc = Document.from_json(controller.doc.to_json())
            seam_id = new_id()
            seam = Seam(
                id=seam_id,
                a=self.first_edge,
                b=edge_range,
                mode=SeamMode.SEGMENT,
            )
            controller.doc.seams[seam_id] = seam

            controller.history.record_change(
                description="Add Seam",
                before_doc=before_doc,
                after_doc=controller.doc,
            )

            self.cancel(controller)
            return True

    def on_mouse_move(self, controller: EditorController, event: EditorEvent) -> bool:
        return False

    def on_mouse_up(self, controller: EditorController, event: EditorEvent) -> bool:
        return False

    def on_key_press(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.key == "ESCAPE":
            self.cancel(controller)
            return True
        return False

    def cancel(self, controller: EditorController) -> None:
        self.first_edge = None
        controller.preview_seam_candidates.clear()
