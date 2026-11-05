# -*- coding: utf-8 -*-
"""型紙エディタの選択およびドラッグ移動ツール (Blender非依存)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..document import Document, NodeKind
from ..selection import HitResult, HitTargetKind, hit_test
from ..session import compute_snap
from .base import BaseTool

if TYPE_CHECKING:
    from ..editor import EditorController, EditorEvent


class SelectTool(BaseTool):
    """要素選択およびドラッグ移動ツール."""

    name: str = "SELECT"

    def __init__(self) -> None:
        self.is_dragging: bool = False
        self.drag_start_world: tuple[float, float] = (0.0, 0.0)
        self.drag_target: Optional[HitResult] = None
        self.before_drag_doc: Optional[Document] = None

    def on_mouse_down(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.button != "LEFT":
            return False

        # ヒットテスト実行
        hit = hit_test(
            controller.doc,
            event.world_pos,
            selected_node_ids=controller.selection.node_ids,
        )

        self.before_drag_doc = Document.from_json(controller.doc.to_json())
        self.drag_start_world = event.world_pos
        self.drag_target = hit

        if hit is None:
            if not event.shift:
                controller.selection.clear()
            return True

        self.is_dragging = True

        if hit.kind == HitTargetKind.NODE and hit.node_id:
            controller.selection.select_node(hit.piece_id, hit.node_id, add=event.shift)
        elif hit.kind == HitTargetKind.PIECE_BODY:
            controller.selection.select_piece(hit.piece_id, add=event.shift)

        return True

    def on_mouse_move(self, controller: EditorController, event: EditorEvent) -> bool:
        if not self.is_dragging or self.drag_target is None:
            return False

        dx = event.world_pos[0] - self.drag_start_world[0]
        dy = event.world_pos[1] - self.drag_start_world[1]

        target = self.drag_target
        if target.kind == HitTargetKind.PIECE_BODY:
            # ピース全体の移動
            piece = controller.doc.pieces.get(target.piece_id)
            if piece and self.before_drag_doc:
                orig_piece = self.before_drag_doc.pieces.get(target.piece_id)
                if orig_piece:
                    piece.transform = piece.transform.__class__(
                        tx=orig_piece.transform.tx + dx,
                        ty=orig_piece.transform.ty + dy,
                        rotation_rad=orig_piece.transform.rotation_rad,
                        sx=orig_piece.transform.sx,
                        sy=orig_piece.transform.sy,
                    )
            return True

        elif target.kind == HitTargetKind.NODE and target.node_id:
            # 単一ノードの移動
            piece = controller.doc.pieces.get(target.piece_id)
            if piece and self.before_drag_doc:
                orig_piece = self.before_drag_doc.pieces.get(target.piece_id)
                if orig_piece and target.path_id:
                    path = piece.paths.get(target.path_id)
                    orig_path = orig_piece.paths.get(target.path_id)
                    if path and orig_path:
                        orig_node = next((n for n in orig_path.nodes if n.id == target.node_id), None)
                        node = next((n for n in path.nodes if n.id == target.node_id), None)
                        if orig_node and node:
                            # スナップ計算（自ノードは除外）
                            snap = compute_snap(
                                (orig_node.position[0] + dx, orig_node.position[1] + dy),
                                controller.doc,
                                view=controller.view,
                                settings=controller.snap_settings,
                                exclude_node_ids={target.node_id},
                            )
                            node.position = snap.world_pos
            return True

        return False

    def on_mouse_up(self, controller: EditorController, event: EditorEvent) -> bool:
        if not self.is_dragging:
            return False

        self.is_dragging = False
        if self.before_drag_doc is not None:
            controller.history.record_change(
                description="Move Element",
                before_doc=self.before_drag_doc,
                after_doc=controller.doc,
            )
            self.before_drag_doc = None

        self.drag_target = None
        return True

    def on_key_press(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.key == "ESCAPE":
            self.cancel(controller)
            return True
        return False

    def cancel(self, controller: EditorController) -> None:
        if self.is_dragging and self.before_drag_doc is not None:
            controller.doc = self.before_drag_doc
            self.before_drag_doc = None
        self.is_dragging = False
        self.drag_target = None
