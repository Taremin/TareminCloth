# -*- coding: utf-8 -*-
"""合印（ノッチ）作成ツール (NotchTool).

外周の辺上をクリックして指定位置（パラメータ t）に合印（Notch）を配置する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..document import Notch, NotchKind, NotchSnapResult
from ..ids import new_id
from ..selection import HitResult, HitTargetKind, hit_test
from ..session import NotchPreviewState
from .base import BaseTool

if TYPE_CHECKING:
    from ..editor import EditorController, EditorEvent


class NotchTool(BaseTool):
    """外周の辺に合印ノッチを追加するツール."""

    def __init__(self, *, default_kind: NotchKind = NotchKind.V) -> None:
        self.default_kind = default_kind

    def _resolve_notch_candidate(
        self,
        controller: EditorController,
        event: EditorEvent,
    ) -> Optional[tuple[HitResult, NotchSnapResult]]:
        """現在のマウス位置からノッチ配置候補の辺ヒットとスナップ結果を解決する."""
        threshold = controller.view.pixel_to_world_dist(12.0)
        hit = hit_test(
            controller.doc,
            event.world_pos,
            edge_threshold=threshold,
            allowed_kinds={HitTargetKind.EDGE},
        )
        if not hit or hit.kind != HitTargetKind.EDGE:
            return None
        if not hit.segment_start_node_id or not hit.path_id:
            return None

        piece = controller.doc.pieces.get(hit.piece_id)
        if not piece or not piece.paths.get(hit.path_id):
            return None

        clip_len = controller.copied_edge_length
        snap_res = controller.doc.resolve_notch_snap(
            hit.piece_id,
            hit.segment_start_node_id,
            raw_t=hit.edge_t,
            path_id=hit.path_id,
            snap_enabled=not event.alt,
            clipboard_length=clip_len,
        )
        return (hit, snap_res)

    def on_mouse_down(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.button != "LEFT":
            return False

        cand = self._resolve_notch_candidate(controller, event)
        if cand is None:
            return False

        hit, snap_res = cand
        if snap_res.existing_notch_id:
            controller.selection.select_notch(hit.piece_id, snap_res.existing_notch_id)
            controller.notch_preview = None
            controller.is_dirty = True
            return True

        before_doc = controller.doc.clone()
        notch_id = new_id()
        notch = Notch(
            id=notch_id,
            piece_id=hit.piece_id,
            path_id=hit.path_id,
            segment_start_id=hit.segment_start_node_id,
            t=snap_res.t,
            kind=self.default_kind,
            depth=0.005,  # 5mm
            width=0.004,  # 4mm
        )
        controller.doc.notches[notch_id] = notch

        # 作成したノッチを選択状態にする
        controller.selection.select_notch(hit.piece_id, notch_id)
        controller.notch_preview = None

        controller.history.record_change(
            description="Add Notch",
            before_doc=before_doc,
            after_doc=controller.doc,
        )
        controller.is_dirty = True
        return True

    def on_mouse_move(self, controller: EditorController, event: EditorEvent) -> bool:
        prev = controller.notch_preview
        cand = self._resolve_notch_candidate(controller, event)
        if cand is None:
            controller.notch_preview = None
        else:
            hit, snap_res = cand
            kind = self.default_kind
            depth = 0.005
            width = 0.004
            if snap_res.existing_notch_id:
                existing = controller.doc.notches.get(snap_res.existing_notch_id)
                if existing is not None:
                    kind = existing.kind
                    depth = existing.depth
                    width = existing.width
            controller.notch_preview = NotchPreviewState.from_snap_result(
                snap_res,
                mode="CREATE",
                piece_id=hit.piece_id,
                path_id=hit.path_id,
                segment_start_id=hit.segment_start_node_id,
                kind=kind,
                depth=depth,
                width=width,
                snap_enabled=not event.alt,
            )

        if controller.notch_preview != prev:
            controller.is_dirty = True
            return True
        return False

    def on_mouse_up(self, controller: EditorController, event: EditorEvent) -> bool:
        return False

    def on_key_press(self, controller: EditorController, event: EditorEvent) -> bool:
        return False

    def cancel(self, controller: EditorController) -> None:
        if controller.notch_preview is not None:
            controller.notch_preview = None
            controller.is_dirty = True
