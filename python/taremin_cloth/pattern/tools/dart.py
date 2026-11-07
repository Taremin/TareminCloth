# -*- coding: utf-8 -*-
"""ダーツ作成ツール (DartTool).

外周上に開口部を持ち、ピース内部の頂点（Apex）へ向かう三角形ダーツを作成し、
ダーツ両辺を結ぶ自動縫合（Seam）を生成する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..document import EdgeRange, Node, NodeKind, Piece, Seam
from ..geometry import arc_length_at_t, build_arc_table, point_at_arc_length
from ..ids import new_id
from ..selection import HitTargetKind, hit_test
from .base import BaseTool

if TYPE_CHECKING:
    from ..editor import EditorController, EditorEvent


class DartTool(BaseTool):
    """外周から内部へ三角形ダーツを作成し自動縫合するツール."""

    name: str = "DART"

    def __init__(self, *, default_width: float = 0.02) -> None:
        self.default_width = default_width
        self.step: int = 0  # 0: 外周クリック待ち, 1: 先端(Apex)クリック待ち
        self.target_piece_id: Optional[str] = None
        self.target_seg_start_id: Optional[str] = None
        self.edge_pos: tuple[float, float] = (0.0, 0.0)
        self.edge_t: float = 0.5

    def on_mouse_down(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.button != "LEFT":
            return False

        doc = controller.doc
        world_pt = event.world_pos

        if self.step == 0:
            # 1点目: 外周辺のクリック
            threshold = controller.view.pixel_to_world_dist(12.0)
            hit = hit_test(
                doc,
                world_pt,
                edge_threshold=threshold,
                allowed_kinds={HitTargetKind.EDGE},
            )
            if not hit or hit.kind != HitTargetKind.EDGE or not hit.segment_start_node_id:
                return False

            piece = doc.pieces.get(hit.piece_id)
            if not piece:
                return False

            outer = piece.paths.get(piece.outer_path_id)
            if not outer:
                return False

            nodes = outer.nodes
            seg_idx = next((i for i, n in enumerate(nodes) if n.id == hit.segment_start_node_id), None)
            if seg_idx is None:
                return False

            self.target_piece_id = hit.piece_id
            self.target_seg_start_id = hit.segment_start_node_id
            self.edge_t = hit.edge_t

            # ローカル座標
            loc_p = piece.transform.inverse_apply(world_pt)
            self.edge_pos = loc_p

            self.step = 1
            controller.preview_points = [world_pt, world_pt]
            controller.is_dirty = True
            return True

        elif self.step == 1:
            # 2点目: ダーツ先端（Apex）のクリック
            piece = doc.pieces.get(self.target_piece_id or "")
            if not piece:
                self.cancel(controller)
                return False

            loc_apex = piece.transform.inverse_apply(world_pt)
            success = self._create_dart(controller, piece, loc_apex=loc_apex)
            self.cancel(controller)
            controller.is_dirty = True
            return success

        return False

    def on_mouse_move(self, controller: EditorController, event: EditorEvent) -> bool:
        if self.step == 1 and self.target_piece_id:
            piece = controller.doc.pieces.get(self.target_piece_id)
            if piece:
                w_start = piece.transform.transform_point(self.edge_pos)
                controller.preview_points = [w_start, event.world_pos]
                controller.is_dirty = True
                return True
        return False

    def on_mouse_up(self, controller: EditorController, event: EditorEvent) -> bool:
        return False

    def on_key_press(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.key.upper() == "ESCAPE":
            self.cancel(controller)
            return True
        return False

    def cancel(self, controller: EditorController) -> None:
        self.step = 0
        self.target_piece_id = None
        self.target_seg_start_id = None
        controller.preview_points.clear()

    def _create_dart(
        self,
        controller: EditorController,
        piece: Piece,
        *,
        loc_apex: tuple[float, float],
    ) -> bool:
        """ダーツスリットを外周パスに挿入し、両辺を縫合する."""
        outer = piece.paths.get(piece.outer_path_id)
        if not outer:
            return False

        nodes = outer.nodes
        n_nodes = len(nodes)
        seg_idx = next((i for i, n in enumerate(nodes) if n.id == self.target_seg_start_id), None)
        if seg_idx is None:
            return False

        before_doc = controller.doc.clone()

        orig_start_node = nodes[seg_idx]
        orig_end_node = nodes[(seg_idx + 1) % n_nodes]

        # 曲線セグメントの制御点と弧長テーブルを取得
        ctrls = outer.get_segment_controls(seg_idx)
        table = build_arc_table(*ctrls, steps=64)
        total_len = table.total_length

        # 基準弧長と開口幅に基づく曲線上の位置計算
        s_center = max(0.0, min(total_len, arc_length_at_t(table, self.edge_t)))
        w = max(0.005, self.default_width)
        s1 = max(0.001 * total_len, s_center - w * 0.5)
        s2 = min(0.999 * total_len, s_center + w * 0.5)

        p_left, t_left = point_at_arc_length(table, s1)
        p_right, t_right = point_at_arc_length(table, s2)

        # 3ノードを作成: left -> apex -> right
        n_left = Node(id=new_id(), position=p_left, kind=NodeKind.CORNER)
        n_apex = Node(id=new_id(), position=loc_apex, kind=NodeKind.CORNER)
        n_right = Node(id=new_id(), position=p_right, kind=NodeKind.CORNER)

        # 外周ノード列の seg_idx と seg_idx+1 の間に [n_left, n_apex, n_right] を挿入
        new_nodes = list(nodes[:seg_idx + 1]) + [n_left, n_apex, n_right] + list(nodes[seg_idx + 1:])
        outer.nodes = new_nodes

        # 元セグメント上のノッチを再マッピング・削除
        notches_to_delete: list[str] = []
        for nid, notch in list(controller.doc.notches.items()):
            if (
                notch.piece_id == piece.id
                and notch.path_id == outer.id
                and notch.segment_start_id == orig_start_node.id
            ):
                if notch.t < t_left - 1e-4:
                    notch.t = max(0.01, min(0.99, notch.t / max(t_left, 1e-6)))
                elif notch.t > t_right + 1e-4:
                    notch.segment_start_id = n_right.id
                    notch.t = max(
                        0.01,
                        min(0.99, (notch.t - t_right) / max(1.0 - t_right, 1e-6)),
                    )
                else:
                    notches_to_delete.append(nid)
        for nid in notches_to_delete:
            controller.doc.notches.pop(nid, None)

        # 元セグメントを参照していた既存 Seam の再マッピング・削除
        dead_seams: set[str] = set()
        for seam_id_key, seam in list(controller.doc.seams.items()):
            seam_invalid = False
            for er in seam.iter_unique_edge_ranges():
                if er.piece_id != piece.id or er.path_id != outer.id:
                    continue
                if er.start_node_id == orig_start_node.id and er.end_node_id == orig_end_node.id:
                    # 順方向参照
                    min_t = min(er.t0, er.t1)
                    max_t = max(er.t0, er.t1)
                    if max_t <= t_left + 1e-5:
                        er.end_node_id = n_left.id
                        er.t0 = er.t0 / max(t_left, 1e-6)
                        er.t1 = er.t1 / max(t_left, 1e-6)
                    elif min_t >= t_right - 1e-5:
                        er.start_node_id = n_right.id
                        er.t0 = (er.t0 - t_right) / max(1.0 - t_right, 1e-6)
                        er.t1 = (er.t1 - t_right) / max(1.0 - t_right, 1e-6)
                    else:
                        seam_invalid = True
                elif er.start_node_id == orig_end_node.id and er.end_node_id == orig_start_node.id:
                    # 逆方向参照
                    p_t0 = 1.0 - er.t0
                    p_t1 = 1.0 - er.t1
                    min_p = min(p_t0, p_t1)
                    max_p = max(p_t0, p_t1)
                    if max_p <= t_left + 1e-5:
                        er.start_node_id = n_left.id
                        er.end_node_id = orig_start_node.id
                        er.t0 = 1.0 - (p_t0 / max(t_left, 1e-6))
                        er.t1 = 1.0 - (p_t1 / max(t_left, 1e-6))
                    elif min_p >= t_right - 1e-5:
                        er.start_node_id = orig_end_node.id
                        er.end_node_id = n_right.id
                        er.t0 = 1.0 - ((p_t0 - t_right) / max(1.0 - t_right, 1e-6))
                        er.t1 = 1.0 - ((p_t1 - t_right) / max(1.0 - t_right, 1e-6))
                    else:
                        seam_invalid = True

            if seam_invalid:
                dead_seams.add(seam_id_key)
            else:
                seam.sync_primary_from_chains()

        for dead_id in dead_seams:
            controller.doc.seams.pop(dead_id, None)

        # ダーツ両辺 (left -> apex) と (apex -> right) の縫合を生成
        seam_id = new_id()
        dart_seam = Seam(
            id=seam_id,
            a=EdgeRange(piece_id=piece.id, path_id=outer.id, start_node_id=n_left.id, end_node_id=n_apex.id),
            b=EdgeRange(piece_id=piece.id, path_id=outer.id, start_node_id=n_apex.id, end_node_id=n_right.id),
            reverse=True,
        )
        controller.doc.seams[seam_id] = dart_seam

        controller.history.record_change(
            description="Add Dart",
            before_doc=before_doc,
            after_doc=controller.doc,
        )
        return True
