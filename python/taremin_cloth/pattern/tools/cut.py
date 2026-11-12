# -*- coding: utf-8 -*-
"""裁断ツール (CutTool).

ピースの外周を2回横切るカットラインを引き、ピースを2つの独立したピースに分割し、
カットライン間に自動縫合（Seam）を生成する。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from ..document import Document, EdgeRange, Node, NodeKind, Path, PathRole, Piece, Seam
from ..geometry import (
    Vec2,
    build_arc_table,
    point_in_polygon,
    segment_intersection,
    split_cubic_bezier,
)
from ..ids import new_id
from .base import BaseTool

if TYPE_CHECKING:
    from ..editor import EditorController, EditorEvent


@dataclass
class _CutHit:
    seg_idx: int
    edge_t: float
    line_t: float
    pt: Vec2
    left_ctrls: tuple[Vec2, Vec2, Vec2, Vec2]
    right_ctrls: tuple[Vec2, Vec2, Vec2, Vec2]
    is_curved: bool


class CutTool(BaseTool):
    """外周を横切るラインでピースを裁断分割するツール."""

    name: str = "CUT"

    def __init__(self) -> None:
        self.is_cutting: bool = False
        self.start_world: tuple[float, float] = (0.0, 0.0)

    def on_mouse_down(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.button != "LEFT":
            return False

        self.is_cutting = True
        self.start_world = event.world_pos
        controller.preview_points = [event.world_pos, event.world_pos]
        controller.is_dirty = True
        return True

    def on_mouse_move(self, controller: EditorController, event: EditorEvent) -> bool:
        if not self.is_cutting:
            return False

        controller.preview_points = [self.start_world, event.world_pos]
        controller.is_dirty = True
        return True

    def on_mouse_up(self, controller: EditorController, event: EditorEvent) -> bool:
        if not self.is_cutting:
            return False

        self.is_cutting = False
        p_start = self.start_world
        p_end = event.world_pos
        controller.preview_points.clear()

        # カットラインが極端に短い場合はキャンセル
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        if math.hypot(dx, dy) < 0.01:  # 1cm 未満
            controller.is_dirty = True
            return True

        # カット対象ピースの探索と分割実行
        success = self._perform_cut(controller, line_start=p_start, line_end=p_end)
        controller.is_dirty = True
        return success

    def on_key_press(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.key.upper() == "ESCAPE":
            self.cancel(controller)
            return True
        return False

    def cancel(self, controller: EditorController) -> None:
        self.is_cutting = False
        controller.preview_points.clear()

    def _perform_cut(
        self,
        controller: EditorController,
        *,
        line_start: tuple[float, float],
        line_end: tuple[float, float],
    ) -> bool:
        """指定カットラインでピースの裁断分割を実行する."""
        doc = controller.doc

        for piece_id, piece in list(doc.pieces.items()):
            outer = piece.paths.get(piece.outer_path_id)
            if not outer or len(outer.nodes) < 3:
                continue

            # ローカル座標系へ変換
            l_start = piece.transform.inverse_apply(line_start)
            l_end = piece.transform.inverse_apply(line_end)

            nodes = outer.nodes
            n_nodes = len(nodes)
            hits: list[_CutHit] = []

            for i in range(n_nodes):
                ctrls = outer.get_segment_controls(i)
                n0 = nodes[i]
                n1 = nodes[(i + 1) % n_nodes]
                is_curved = (
                    n0.kind != NodeKind.CORNER
                    or n1.kind != NodeKind.CORNER
                    or n0.out_handle is not None
                    or n1.in_handle is not None
                )

                table = build_arc_table(*ctrls, steps=64)
                pts = table.points
                num_subsegs = len(pts) - 1

                for k in range(num_subsegs):
                    res = segment_intersection(pts[k], pts[k + 1], l_start, l_end)
                    if res:
                        pt, t_sub, t_line = res
                        edge_t = (k + t_sub) / max(1.0, float(num_subsegs))
                        edge_t = max(0.001, min(0.999, edge_t))
                        left_c, right_c = split_cubic_bezier(ctrls[0], ctrls[1], ctrls[2], ctrls[3], edge_t)
                        hits.append(_CutHit(
                            seg_idx=i,
                            edge_t=edge_t,
                            line_t=t_line,
                            pt=pt,
                            left_ctrls=left_c,
                            right_ctrls=right_c,
                            is_curved=is_curved,
                        ))
                        break

            # ちょうど2つの異なるエッジと交差した場合に分割可能
            if len(hits) == 2:
                hits.sort(key=lambda item: item.line_t)
                hit0, hit1 = hits[0], hits[1]

                if hit0.seg_idx == hit1.seg_idx:
                    continue  # 同一セグメント上の重複

                if hit0.seg_idx > hit1.seg_idx:
                    hit0, hit1 = hit1, hit0

                idx0 = hit0.seg_idx
                idx1 = hit1.seg_idx

                before_doc = Document.from_json(doc.to_json())

                # 新規ノードの生成（ベジェハンドルの保持）
                out_h_cut0_a = (hit0.right_ctrls[1][0] - hit0.pt[0], hit0.right_ctrls[1][1] - hit0.pt[1]) if hit0.is_curved else None
                in_h_cut1_a = (hit1.left_ctrls[2][0] - hit1.pt[0], hit1.left_ctrls[2][1] - hit1.pt[1]) if hit1.is_curved else None

                node_cut0_a = Node(
                    id=new_id(),
                    position=hit0.pt,
                    kind=NodeKind.CORNER,
                    in_handle=None,
                    out_handle=out_h_cut0_a,
                )
                node_cut1_a = Node(
                    id=new_id(),
                    position=hit1.pt,
                    kind=NodeKind.CORNER,
                    in_handle=in_h_cut1_a,
                    out_handle=None,
                )

                out_h_cut1_b = (hit1.right_ctrls[1][0] - hit1.pt[0], hit1.right_ctrls[1][1] - hit1.pt[1]) if hit1.is_curved else None
                in_h_cut0_b = (hit0.left_ctrls[2][0] - hit0.pt[0], hit0.left_ctrls[2][1] - hit0.pt[1]) if hit0.is_curved else None

                node_cut1_b = Node(
                    id=new_id(),
                    position=hit1.pt,
                    kind=NodeKind.CORNER,
                    in_handle=None,
                    out_handle=out_h_cut1_b,
                )
                node_cut0_b = Node(
                    id=new_id(),
                    position=hit0.pt,
                    kind=NodeKind.CORNER,
                    in_handle=in_h_cut0_b,
                    out_handle=None,
                )

                # パスAの構築: node_cut0_a -> nodes[idx0+1..idx1] -> node_cut1_a
                path_a_nodes: list[Node] = [node_cut0_a]
                orig_to_new_a: dict[str, str] = {}
                curr = (idx0 + 1) % n_nodes
                while curr != (idx1 + 1) % n_nodes:
                    orig_n = nodes[curr]
                    in_h = orig_n.in_handle
                    out_h = orig_n.out_handle

                    # 分割境界に隣接するハンドルの調整
                    if curr == (idx0 + 1) % n_nodes and hit0.is_curved:
                        in_h = (hit0.right_ctrls[2][0] - orig_n.position[0], hit0.right_ctrls[2][1] - orig_n.position[1])
                    if curr == idx1 and hit1.is_curved:
                        out_h = (hit1.left_ctrls[1][0] - orig_n.position[0], hit1.left_ctrls[1][1] - orig_n.position[1])

                    nid = new_id()
                    orig_to_new_a[orig_n.id] = nid
                    path_a_nodes.append(Node(
                        id=nid,
                        position=orig_n.position,
                        kind=orig_n.kind,
                        in_handle=in_h,
                        out_handle=out_h,
                    ))
                    curr = (curr + 1) % n_nodes
                path_a_nodes.append(node_cut1_a)

                # パスBの構築: node_cut1_b -> nodes[idx1+1..idx0] -> node_cut0_b
                path_b_nodes: list[Node] = [node_cut1_b]
                orig_to_new_b: dict[str, str] = {}
                curr = (idx1 + 1) % n_nodes
                while curr != (idx0 + 1) % n_nodes:
                    orig_n = nodes[curr]
                    in_h = orig_n.in_handle
                    out_h = orig_n.out_handle

                    if curr == (idx1 + 1) % n_nodes and hit1.is_curved:
                        in_h = (hit1.right_ctrls[2][0] - orig_n.position[0], hit1.right_ctrls[2][1] - orig_n.position[1])
                    if curr == idx0 and hit0.is_curved:
                        out_h = (hit0.left_ctrls[1][0] - orig_n.position[0], hit0.left_ctrls[1][1] - orig_n.position[1])

                    nid = new_id()
                    orig_to_new_b[orig_n.id] = nid
                    path_b_nodes.append(Node(
                        id=nid,
                        position=orig_n.position,
                        kind=orig_n.kind,
                        in_handle=in_h,
                        out_handle=out_h,
                    ))
                    curr = (curr + 1) % n_nodes
                path_b_nodes.append(node_cut0_b)

                if len(path_a_nodes) < 3 or len(path_b_nodes) < 3:
                    continue

                # 2つの新規ピースを構築
                id_a, id_b = new_id(), new_id()
                path_id_a, path_id_b = new_id(), new_id()

                piece_a = Piece(
                    id=id_a,
                    name=f"{piece.name}_A",
                    paths={path_id_a: Path(id=path_id_a, role=PathRole.OUTER, nodes=path_a_nodes, closed=True)},
                    outer_path_id=path_id_a,
                    transform=piece.transform,
                    target_edge_length=piece.target_edge_length,
                    layer_id=piece.layer_id,
                )
                piece_b = Piece(
                    id=id_b,
                    name=f"{piece.name}_B",
                    paths={path_id_b: Path(id=path_id_b, role=PathRole.OUTER, nodes=path_b_nodes, closed=True)},
                    outer_path_id=path_id_b,
                    transform=piece.transform,
                    target_edge_length=piece.target_edge_length,
                    layer_id=piece.layer_id,
                )

                # 配置プロキシの複製
                orig_place = doc.placements.get(piece.id)
                if orig_place:
                    doc.placements[id_a] = orig_place.copy_for_piece(id_a)
                    doc.placements[id_b] = orig_place.copy_for_piece(id_b)

                # カットライン上の自動縫合を生成
                seam_id = new_id()
                cut_seam = Seam(
                    id=seam_id,
                    a=EdgeRange(piece_id=id_a, path_id=path_id_a, start_node_id=node_cut1_a.id, end_node_id=node_cut0_a.id),
                    b=EdgeRange(piece_id=id_b, path_id=path_id_b, start_node_id=node_cut0_b.id, end_node_id=node_cut1_b.id),
                )

                # 既存参照（Seam, Notch, Anchor）の付け替え
                self._remap_references(
                    doc,
                    orig_piece_id=piece.id,
                    id_a=id_a,
                    path_id_a=path_id_a,
                    orig_to_new_a=orig_to_new_a,
                    id_b=id_b,
                    path_id_b=path_id_b,
                    orig_to_new_b=orig_to_new_b,
                    hit0=hit0,
                    hit1=hit1,
                    node_cut0_a=node_cut0_a,
                    node_cut1_a=node_cut1_a,
                    node_cut0_b=node_cut0_b,
                    node_cut1_b=node_cut1_b,
                    nodes=nodes,
                    piece_a=piece_a,
                )

                # 元ピースの削除と新ピース・縫合の登録
                del doc.pieces[piece.id]
                if piece.id in doc.placements:
                    del doc.placements[piece.id]

                doc.pieces[id_a] = piece_a
                doc.pieces[id_b] = piece_b
                doc.seams[seam_id] = cut_seam

                controller.selection.clear()
                controller.selection.select_piece(id_a)
                controller.history.record_change(
                    description="Cut Piece",
                    before_doc=before_doc,
                    after_doc=doc,
                )
                return True

        return False

    def _remap_references(
        self,
        doc: Document,
        *,
        orig_piece_id: str,
        id_a: str,
        path_id_a: str,
        orig_to_new_a: dict[str, str],
        id_b: str,
        path_id_b: str,
        orig_to_new_b: dict[str, str],
        hit0: _CutHit,
        hit1: _CutHit,
        node_cut0_a: Node,
        node_cut1_a: Node,
        node_cut0_b: Node,
        node_cut1_b: Node,
        nodes: list[Node],
        piece_a: Piece,
    ) -> None:
        """裁断された元ピースに属していた縫合・ノッチ・アンカーを新ピースA・Bへ引き継ぐ."""
        # 1. 縫合 (Seams) の引き継ぎ
        dead_seam_ids: list[str] = []
        for sid, seam in list(doc.seams.items()):
            all_ranges = seam.iter_unique_edge_ranges()
            seam_broken = False
            for er in all_ranges:
                if er.piece_id != orig_piece_id:
                    continue
                # 1. カットされていない辺の判定
                s_a = orig_to_new_a.get(er.start_node_id)
                e_a = orig_to_new_a.get(er.end_node_id)
                if s_a and e_a:
                    er.piece_id = id_a
                    er.path_id = path_id_a
                    er.start_node_id = s_a
                    er.end_node_id = e_a
                    continue

                s_b = orig_to_new_b.get(er.start_node_id)
                e_b = orig_to_new_b.get(er.end_node_id)
                if s_b and e_b:
                    er.piece_id = id_b
                    er.path_id = path_id_b
                    er.start_node_id = s_b
                    er.end_node_id = e_b
                    continue

                # 2. カットされた辺上の部分 EdgeRange
                remapped = False
                for hit, target_piece_id, target_path_id, seg_start_nid, seg_end_nid, is_first_half in (
                    (hit0, id_b, path_id_b, orig_to_new_b.get(nodes[hit0.seg_idx].id), node_cut0_b.id, True),
                    (hit0, id_a, path_id_a, node_cut0_a.id, orig_to_new_a.get(nodes[(hit0.seg_idx + 1) % len(nodes)].id), False),
                    (hit1, id_a, path_id_a, orig_to_new_a.get(nodes[hit1.seg_idx].id), node_cut1_a.id, True),
                    (hit1, id_b, path_id_b, node_cut1_b.id, orig_to_new_b.get(nodes[(hit1.seg_idx + 1) % len(nodes)].id), False),
                ):
                    if not seg_start_nid or not seg_end_nid:
                        continue
                    n0_id = nodes[hit.seg_idx].id
                    n1_id = nodes[(hit.seg_idx + 1) % len(nodes)].id
                    if er.start_node_id == n0_id and er.end_node_id == n1_id:
                        if is_first_half and er.t0 <= hit.edge_t + 1e-5 and er.t1 <= hit.edge_t + 1e-5:
                            er.piece_id = target_piece_id
                            er.path_id = target_path_id
                            er.start_node_id = seg_start_nid
                            er.end_node_id = seg_end_nid
                            er.t0 = max(0.0, min(1.0, er.t0 / max(1e-6, hit.edge_t)))
                            er.t1 = max(0.0, min(1.0, er.t1 / max(1e-6, hit.edge_t)))
                            remapped = True
                            break
                        elif not is_first_half and er.t0 >= hit.edge_t - 1e-5 and er.t1 >= hit.edge_t - 1e-5:
                            er.piece_id = target_piece_id
                            er.path_id = target_path_id
                            er.start_node_id = seg_start_nid
                            er.end_node_id = seg_end_nid
                            er.t0 = max(0.0, min(1.0, (er.t0 - hit.edge_t) / max(1e-6, 1.0 - hit.edge_t)))
                            er.t1 = max(0.0, min(1.0, (er.t1 - hit.edge_t) / max(1e-6, 1.0 - hit.edge_t)))
                            remapped = True
                            break
                    elif er.start_node_id == n1_id and er.end_node_id == n0_id:
                        # 逆方向指定の場合
                        p_t0 = 1.0 - er.t0
                        p_t1 = 1.0 - er.t1
                        if is_first_half and p_t0 <= hit.edge_t + 1e-5 and p_t1 <= hit.edge_t + 1e-5:
                            er.piece_id = target_piece_id
                            er.path_id = target_path_id
                            er.start_node_id = seg_end_nid
                            er.end_node_id = seg_start_nid
                            sub_t0 = p_t0 / max(1e-6, hit.edge_t)
                            sub_t1 = p_t1 / max(1e-6, hit.edge_t)
                            er.t0 = max(0.0, min(1.0, 1.0 - sub_t0))
                            er.t1 = max(0.0, min(1.0, 1.0 - sub_t1))
                            remapped = True
                            break
                        elif not is_first_half and p_t0 >= hit.edge_t - 1e-5 and p_t1 >= hit.edge_t - 1e-5:
                            er.piece_id = target_piece_id
                            er.path_id = target_path_id
                            er.start_node_id = seg_end_nid
                            er.end_node_id = seg_start_nid
                            sub_t0 = (p_t0 - hit.edge_t) / max(1e-6, 1.0 - hit.edge_t)
                            sub_t1 = (p_t1 - hit.edge_t) / max(1e-6, 1.0 - hit.edge_t)
                            er.t0 = max(0.0, min(1.0, 1.0 - sub_t0))
                            er.t1 = max(0.0, min(1.0, 1.0 - sub_t1))
                            remapped = True
                            break

                if not remapped:
                    seam_broken = True

            if seam_broken:
                dead_seam_ids.append(sid)

        for sid in dead_seam_ids:
            if sid in doc.seams:
                del doc.seams[sid]

        # 2. ノッチ (Notches) の引き継ぎ
        dead_notch_ids: list[str] = []
        for notch in list(doc.notches.values()):
            if notch.piece_id != orig_piece_id:
                continue

            # カットされていない辺上のノッチ
            if notch.segment_start_id in orig_to_new_a:
                n_idx = next((i for i, n in enumerate(nodes) if n.id == notch.segment_start_id), None)
                if n_idx is not None and n_idx != hit0.seg_idx and n_idx != hit1.seg_idx:
                    notch.piece_id = id_a
                    notch.path_id = path_id_a
                    notch.segment_start_id = orig_to_new_a[notch.segment_start_id]
                    continue
            if notch.segment_start_id in orig_to_new_b:
                n_idx = next((i for i, n in enumerate(nodes) if n.id == notch.segment_start_id), None)
                if n_idx is not None and n_idx != hit0.seg_idx and n_idx != hit1.seg_idx:
                    notch.piece_id = id_b
                    notch.path_id = path_id_b
                    notch.segment_start_id = orig_to_new_b[notch.segment_start_id]
                    continue

            # カットされた辺 (hit0 または hit1) 上のノッチ
            handled = False
            for hit, seg_start_node_b, cut_node_b, cut_node_a, seg_end_node_a in (
                (hit0, nodes[hit0.seg_idx], node_cut0_b, node_cut0_a, nodes[(hit0.seg_idx + 1) % len(nodes)]),
                (hit1, nodes[hit1.seg_idx], node_cut1_a, node_cut1_b, nodes[(hit1.seg_idx + 1) % len(nodes)]),
            ):
                if notch.segment_start_id == seg_start_node_b.id:
                    # カット点直上または端点直上のノッチは削除
                    if abs(notch.t - hit.edge_t) < 1e-3 or notch.t < 1e-3 or notch.t > 1.0 - 1e-3:
                        dead_notch_ids.append(notch.id)
                        handled = True
                        break

                    if notch.t < hit.edge_t:
                        # 前半区間
                        if hit == hit0:
                            notch.piece_id = id_b
                            notch.path_id = path_id_b
                            if seg_start_node_b.id in orig_to_new_b:
                                notch.segment_start_id = orig_to_new_b[seg_start_node_b.id]
                        else:
                            notch.piece_id = id_a
                            notch.path_id = path_id_a
                            if seg_start_node_b.id in orig_to_new_a:
                                notch.segment_start_id = orig_to_new_a[seg_start_node_b.id]
                        notch.t = max(0.001, min(0.999, notch.t / max(1e-6, hit.edge_t)))
                    else:
                        # 後半区間
                        if hit == hit0:
                            notch.piece_id = id_a
                            notch.path_id = path_id_a
                            notch.segment_start_id = cut_node_a.id
                        else:
                            notch.piece_id = id_b
                            notch.path_id = path_id_b
                            notch.segment_start_id = cut_node_b.id
                        notch.t = max(0.001, min(0.999, (notch.t - hit.edge_t) / max(1e-6, 1.0 - hit.edge_t)))
                    handled = True
                    break

            if not handled:
                dead_notch_ids.append(notch.id)

        for nid in dead_notch_ids:
            if nid in doc.notches:
                del doc.notches[nid]

        # 3. アンカー (Anchors) の引き継ぎ
        poly_a = [n.position for n in piece_a.paths[path_id_a].nodes]
        for anchor in doc.anchors.values():
            if anchor.piece_id != orig_piece_id:
                continue
            if point_in_polygon(anchor.position, poly_a):
                anchor.piece_id = id_a
            else:
                anchor.piece_id = id_b
