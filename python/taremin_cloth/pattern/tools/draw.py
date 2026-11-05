# -*- coding: utf-8 -*-
"""型紙エディタの作図ツール (矩形・ポリゴン) (Blender非依存)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from ..document import Document
from ..session import SnapKind, compute_snap
from .base import BaseTool

if TYPE_CHECKING:
    from ..editor import EditorController, EditorEvent


class RectangleTool(BaseTool):
    """対角ドラッグにより反時計回りの矩形ピースを作成するツール."""

    name: str = "RECTANGLE"

    def __init__(self) -> None:
        self.is_drawing: bool = False
        self.start_pos: tuple[float, float] = (0.0, 0.0)

    def on_mouse_down(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.button != "LEFT":
            return False

        # スナップ適用
        snap = compute_snap(
            event.world_pos,
            controller.doc,
            view=controller.view,
            settings=controller.snap_settings,
        )
        self.start_pos = snap.world_pos
        self.is_drawing = True
        controller.preview_points = [self.start_pos, self.start_pos]
        return True

    def on_mouse_move(self, controller: EditorController, event: EditorEvent) -> bool:
        if not self.is_drawing:
            return False

        snap = compute_snap(
            event.world_pos,
            controller.doc,
            view=controller.view,
            settings=controller.snap_settings,
        )
        controller.preview_points = [self.start_pos, snap.world_pos]
        return True

    def on_mouse_up(self, controller: EditorController, event: EditorEvent) -> bool:
        if not self.is_drawing:
            return False

        self.is_drawing = False
        controller.preview_points.clear()

        snap = compute_snap(
            event.world_pos,
            controller.doc,
            view=controller.view,
            settings=controller.snap_settings,
        )
        end_pos = snap.world_pos

        # 微小ドラッグの無視
        dx = abs(end_pos[0] - self.start_pos[0])
        dy = abs(end_pos[1] - self.start_pos[1])
        if dx < 0.005 or dy < 0.005:
            return True

        before_doc = Document.from_json(controller.doc.to_json())
        piece = controller.doc.add_rectangle_piece(self.start_pos, end_pos)
        controller.selection.select_piece(piece.id)

        controller.history.record_change(
            description="Add Rectangle Piece",
            before_doc=before_doc,
            after_doc=controller.doc,
        )
        return True

    def on_key_press(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.key == "ESCAPE":
            self.cancel(controller)
            return True
        return False

    def cancel(self, controller: EditorController) -> None:
        self.is_drawing = False
        controller.preview_points.clear()


class PolygonTool(BaseTool):
    """頂点を連続クリックして多角形ピースを作成するツール."""

    name: str = "POLYGON"

    def __init__(self) -> None:
        self.points: list[tuple[float, float]] = []

    def on_mouse_down(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.button != "LEFT":
            return False

        # スナップ適用
        snap = compute_snap(
            event.world_pos,
            controller.doc,
            view=controller.view,
            settings=controller.snap_settings,
        )
        cur_pt = snap.world_pos

        # 開始点クリックによるクローズ判定
        if len(self.points) >= 3:
            first_pt = self.points[0]
            dist_to_first = math.hypot(cur_pt[0] - first_pt[0], cur_pt[1] - first_pt[1])
            threshold = controller.view.pixel_to_world_dist(controller.snap_settings.snap_threshold_pixels)
            if dist_to_first <= threshold:
                self._finish_polygon(controller)
                return True

        self.points.append(cur_pt)
        controller.preview_points = list(self.points)
        return True

    def on_mouse_move(self, controller: EditorController, event: EditorEvent) -> bool:
        if not self.points:
            return False

        snap = compute_snap(
            event.world_pos,
            controller.doc,
            view=controller.view,
            settings=controller.snap_settings,
        )
        # 既存頂点列 + 現在カーソル位置をプレビュー
        controller.preview_points = self.points + [snap.world_pos]
        return True

    def on_mouse_up(self, controller: EditorController, event: EditorEvent) -> bool:
        return bool(self.points)

    def on_key_press(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.key == "ENTER":
            if len(self.points) >= 3:
                self._finish_polygon(controller)
                return True
        elif event.key == "ESCAPE":
            self.cancel(controller)
            return True
        return False

    def _finish_polygon(self, controller: EditorController) -> None:
        if len(self.points) < 3:
            self.cancel(controller)
            return

        before_doc = Document.from_json(controller.doc.to_json())
        try:
            piece = controller.doc.add_polygon_piece(self.points)
            controller.selection.select_piece(piece.id)
            controller.history.record_change(
                description="Add Polygon Piece",
                before_doc=before_doc,
                after_doc=controller.doc,
            )
        except ValueError:
            # 自己交差などの不正幾何時は追加をスキップ
            pass

        self.cancel(controller)

    def cancel(self, controller: EditorController) -> None:
        self.points.clear()
        controller.preview_points.clear()
