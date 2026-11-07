# -*- coding: utf-8 -*-
"""内部線描画ツール (InternalLineTool).

ピース内部に対話的にクリックで頂点を配置し、Enterキーで内部線（PathRole.INTERNAL）を作成する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..document import Document, Node, NodeKind, Path, PathRole
from ..geometry import point_in_polygon
from ..ids import new_id
from ..session import compute_snap
from .base import BaseTool

if TYPE_CHECKING:
    from ..editor import EditorController, EditorEvent


def _dist_sq(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    # audit: allow-positional
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy


def _find_target_piece(doc: Document, world_pt: tuple[float, float]) -> Optional[str]:
    """ワールド座標から内部線を引く対象ピースのIDを特定する（内外判定 + 最近傍フォールバック）."""
    # 1. 外周ポリゴン内外判定
    for pid, piece in doc.pieces.items():
        loc_p = piece.transform.inverse_apply(world_pt)
        outer = piece.paths.get(piece.outer_path_id)
        if outer and len(outer.nodes) >= 3:
            poly_pts = outer.sample_outline_points(steps_per_segment=16)
            if point_in_polygon(loc_p, poly_pts):
                return pid

    # 2. 内外判定で見つからない場合、外周ノードへの最短距離で最も近いピースを探索
    best_pid: Optional[str] = None
    best_dist_sq = float("inf")

    for pid, piece in doc.pieces.items():
        outer = piece.paths.get(piece.outer_path_id)
        if not outer:
            continue
        for n in outer.nodes:
            w_pos = piece.transform.transform_point(n.position)
            d = _dist_sq(world_pt, w_pos)
            if d < best_dist_sq:
                best_dist_sq = d
                best_pid = pid

    # 最短距離が 1m (1.0m^2 = 1.0) 以内ならそのピースを採用
    if best_dist_sq < 1.0:
        return best_pid

    return None


class InternalLineTool(BaseTool):
    """ピース内部に折れ線（内部線）を作成するツール."""

    def __init__(self) -> None:
        self.active_piece_id: Optional[str] = None
        self.local_points: list[tuple[float, float]] = []

    def on_mouse_down(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.button != "LEFT":
            return False

        doc = controller.doc
        world_pt = event.world_pos

        # 1点目: 対象ピースを特定
        if self.active_piece_id is None:
            target_piece_id = _find_target_piece(doc, world_pt)
            if not target_piece_id:
                return False

            piece = doc.pieces[target_piece_id]
            self.active_piece_id = target_piece_id

            # スナップ計算
            snap = compute_snap(
                world_pt,
                doc,
                view=controller.view,
                settings=controller.snap_settings,
            )
            loc_pos = piece.transform.inverse_apply(snap.world_pos)
            self.local_points = [loc_pos]

            controller.preview_points = [snap.world_pos, snap.world_pos]
            controller.is_dirty = True
            return True

        # 2点目以降: 点を追加
        piece = doc.pieces.get(self.active_piece_id)
        if not piece:
            self.cancel(controller)
            return False

        snap = compute_snap(
            world_pt,
            doc,
            view=controller.view,
            settings=controller.snap_settings,
        )
        loc_pos = piece.transform.inverse_apply(snap.world_pos)

        # 直前の点とほぼ同一位置（ダブルクリック等）なら確定
        if len(self.local_points) >= 2:
            prev_loc = self.local_points[-1]
            if _dist_sq(loc_pos, prev_loc) < 0.0001:  # 1cm 未満
                return self._finish(controller)

        self.local_points.append(loc_pos)

        w_pts = [piece.transform.transform_point(p) for p in self.local_points]
        w_pts.append(snap.world_pos)
        controller.preview_points = w_pts
        controller.is_dirty = True
        return True

    def on_mouse_move(self, controller: EditorController, event: EditorEvent) -> bool:
        if self.active_piece_id is None or not self.local_points:
            return False

        piece = controller.doc.pieces.get(self.active_piece_id)
        if not piece:
            return False

        snap = compute_snap(
            event.world_pos,
            controller.doc,
            view=controller.view,
            settings=controller.snap_settings,
        )
        w_pts = [piece.transform.transform_point(p) for p in self.local_points]
        w_pts.append(snap.world_pos)
        controller.preview_points = w_pts
        controller.is_dirty = True
        return True

    def on_mouse_up(self, controller: EditorController, event: EditorEvent) -> bool:
        return False

    def on_key_press(self, controller: EditorController, event: EditorEvent) -> bool:
        k = event.key.upper()
        # メインEnter(RET), テンキーEnter(NUMPAD_ENTER), SPACE等に対応
        if k in ("ENTER", "RET", "RETURN", "NUMPAD_ENTER", "SPACE"):
            if self._finish(controller):
                return True

        if k == "ESCAPE":
            if self.active_piece_id:
                self.cancel(controller)
                controller.is_dirty = True
                return True

        return False

    def _finish(self, controller: EditorController) -> bool:
        """内部線の作成を確定する."""
        if not self.active_piece_id or len(self.local_points) < 2:
            return False

        piece = controller.doc.pieces.get(self.active_piece_id)
        if not piece:
            self.cancel(controller)
            return False

        before_doc = Document.from_json(controller.doc.to_json())
        path_id = new_id()
        nodes = [
            Node(id=new_id(), position=p, kind=NodeKind.CORNER)
            for p in self.local_points
        ]
        internal_path = Path(
            id=path_id,
            role=PathRole.INTERNAL,
            nodes=nodes,
            closed=False,
        )
        piece.paths[path_id] = internal_path

        controller.history.record_change(
            description="Add Internal Line",
            before_doc=before_doc,
            after_doc=controller.doc,
        )
        self.cancel(controller)
        controller.is_dirty = True
        return True

    def cancel(self, controller: EditorController) -> None:
        self.active_piece_id = None
        self.local_points.clear()
        controller.preview_points.clear()
