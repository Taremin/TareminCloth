# -*- coding: utf-8 -*-
"""型紙エディタの要素ヒットテストおよび選択ロジック (Blender非依存)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .document import Document, NodeKind, PathRole
from .geometry import (
    point_in_polygon,
    signed_area,
    build_arc_table,
    closest_point_on_table,
    evaluate_cubic,
)


class HitTargetKind(str, Enum):
    """ヒットテストで検出される対象要素の優先度順種別."""
    HANDLE = "HANDLE"
    NODE = "NODE"
    NOTCH = "NOTCH"
    SEAM = "SEAM"
    EDGE = "EDGE"
    INTERNAL_PATH = "INTERNAL_PATH"
    PIECE_BODY = "PIECE_BODY"


@dataclass
class HitResult:
    """ヒットテストの検出結果."""
    kind: HitTargetKind
    piece_id: str
    path_id: Optional[str] = None
    node_id: Optional[str] = None
    notch_id: Optional[str] = None
    seam_id: Optional[str] = None
    handle_type: Optional[str] = None  # "in" または "out"
    segment_start_node_id: Optional[str] = None
    segment_end_node_id: Optional[str] = None
    edge_t: float = 0.0
    world_position: tuple[float, float] = (0.0, 0.0)
    distance: float = 0.0


def _dist_sq(p1: tuple[float, float], p2: tuple[float, float]) -> float:  # audit: allow-positional
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy


def _dist_to_segment_sq(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:  # audit: allow-positional
    ab_x = b[0] - a[0]
    ab_y = b[1] - a[1]
    len_sq = ab_x * ab_x + ab_y * ab_y
    if len_sq < 1e-12:
        return _dist_sq(p, a)
    ap_x = p[0] - a[0]
    ap_y = p[1] - a[1]
    t = max(0.0, min(1.0, (ap_x * ab_x + ap_y * ab_y) / len_sq))
    proj_x = a[0] + t * ab_x
    proj_y = a[1] + t * ab_y
    return _dist_sq(p, (proj_x, proj_y))


def hit_test(
    doc: Document,
    world_pos: tuple[float, float],
    *,
    handle_radius: float = 0.006,
    node_radius: float = 0.008,
    edge_threshold: float = 0.006,
    selected_node_ids: Optional[set[str]] = None,
    allowed_kinds: Optional[set[HitTargetKind]] = None,
) -> Optional[HitResult]:
    """ワールド座標系におけるヒットテストを行い、優先順位に従って最前面の要素を返す.
    
    優先順位:
    1. ハンドル (HANDLE): 選択中ノードのベジェハンドル
    2. ノード (NODE): パス上の制御点ノード
    3. ノッチ (NOTCH): 合印
    4. 辺 (EDGE / SEGMENT): 外周輪郭の辺
    5. 内部線 (INTERNAL_PATH): ピース内の内部線
    6. ピース内部 (PIECE_BODY): ピース多角形の内側
    """
    sel_nodes = selected_node_ids or set()
    best_handle: Optional[HitResult] = None
    best_node: Optional[HitResult] = None
    best_notch: Optional[HitResult] = None
    best_seam: Optional[HitResult] = None
    best_edge: Optional[HitResult] = None
    best_internal: Optional[HitResult] = None
    best_body: Optional[HitResult] = None
    best_body_area: float = float("inf")

    # 各ピースを走査（後から描画されるピースを優先するため逆順も考慮可能）
    for piece_id, piece in doc.pieces.items():
        local_pos = piece.transform.inverse_apply(world_pos)

        # 1. ハンドル & 2. ノード の判定
        check_nodes = allowed_kinds is None or HitTargetKind.NODE in allowed_kinds
        check_handles = allowed_kinds is None or HitTargetKind.HANDLE in allowed_kinds
        if check_nodes or check_handles:
            for path_id, path in piece.paths.items():
                n_nodes = len(path.nodes)
                if n_nodes == 0:
                    continue

                for i, node in enumerate(path.nodes):
                    w_node_pos = piece.transform.transform_point(node.position)
                    d_sq = _dist_sq(world_pos, w_node_pos)

                    if check_nodes and d_sq <= node_radius * node_radius:
                        dist = math.sqrt(d_sq)
                        if best_node is None or dist < best_node.distance:
                            best_node = HitResult(
                                kind=HitTargetKind.NODE,
                                piece_id=piece_id,
                                path_id=path_id,
                                node_id=node.id,
                                world_position=w_node_pos,
                                distance=dist,
                            )

                    # 選択中ノードのハンドルのヒット判定
                    if check_handles and node.id in sel_nodes and (
                        node.kind != NodeKind.CORNER or node.in_handle is not None or node.out_handle is not None
                    ):
                        if node.in_handle is not None:
                            in_pos = (node.position[0] + node.in_handle[0], node.position[1] + node.in_handle[1])
                            w_in_pos = piece.transform.transform_point(in_pos)
                            h_dist_sq = _dist_sq(world_pos, w_in_pos)
                            if h_dist_sq <= handle_radius * handle_radius:
                                h_dist = math.sqrt(h_dist_sq)
                                if best_handle is None or h_dist < best_handle.distance:
                                    best_handle = HitResult(
                                        kind=HitTargetKind.HANDLE,
                                        piece_id=piece_id,
                                        path_id=path_id,
                                        node_id=node.id,
                                        handle_type="in",
                                        world_position=w_in_pos,
                                        distance=h_dist,
                                    )

                        if node.out_handle is not None:
                            out_pos = (node.position[0] + node.out_handle[0], node.position[1] + node.out_handle[1])
                            w_out_pos = piece.transform.transform_point(out_pos)
                            h_dist_sq = _dist_sq(world_pos, w_out_pos)
                            if h_dist_sq <= handle_radius * handle_radius:
                                h_dist = math.sqrt(h_dist_sq)
                                if best_handle is None or h_dist < best_handle.distance:
                                    best_handle = HitResult(
                                        kind=HitTargetKind.HANDLE,
                                        piece_id=piece_id,
                                        path_id=path_id,
                                        node_id=node.id,
                                        handle_type="out",
                                        world_position=w_out_pos,
                                        distance=h_dist,
                                    )

        # 4. 辺 (EDGE) & 5. 内部線 (INTERNAL_PATH)
        check_edges = allowed_kinds is None or HitTargetKind.EDGE in allowed_kinds
        check_internal = allowed_kinds is None or HitTargetKind.INTERNAL_PATH in allowed_kinds
        if check_edges or check_internal:
            for path_id, path in piece.paths.items():
                is_outer = (path.role == PathRole.OUTER)
                if is_outer and not check_edges:
                    continue
                if not is_outer and not check_internal:
                    continue

                n_nodes = len(path.nodes)
                if n_nodes >= 2:
                    num_segs = n_nodes if path.closed else n_nodes - 1
                    for i in range(num_segs):
                        n0 = path.nodes[i]
                        n1 = path.nodes[(i + 1) % n_nodes]

                        ctrls = path.get_segment_controls(i)
                        table = build_arc_table(*ctrls)
                        # ローカル座標系でベジェ曲線への最短距離とパラメータ t を探索
                        _, t_best, local_dist = closest_point_on_table(table, local_pos)

                        # ワールド座標系での距離を近似評価（アフィン変換のスケール考慮）
                        scale = (piece.transform.sx + piece.transform.sy) * 0.5
                        world_dist = local_dist * scale

                        if world_dist <= edge_threshold:
                            target_kind = HitTargetKind.EDGE if is_outer else HitTargetKind.INTERNAL_PATH
                            cand = HitResult(
                                kind=target_kind,
                                piece_id=piece_id,
                                path_id=path_id,
                                segment_start_node_id=n0.id,
                                segment_end_node_id=n1.id,
                                edge_t=t_best,
                                distance=world_dist,
                            )
                            if is_outer:
                                if best_edge is None or world_dist < best_edge.distance:
                                    best_edge = cand
                            else:
                                if best_internal is None or world_dist < best_internal.distance:
                                    best_internal = cand

        # 6. ピース内部 (PIECE_BODY) 判定 (ベジェ曲線サンプリング対応・重なり時は面積の小さい内側ピースを優先)
        check_body = allowed_kinds is None or HitTargetKind.PIECE_BODY in allowed_kinds
        if check_body:
            outer_path = piece.paths.get(piece.outer_path_id)
            if outer_path and len(outer_path.nodes) >= 3:
                poly_pts = outer_path.sample_outline_points(steps_per_segment=16)
                if point_in_polygon(local_pos, poly_pts):
                    scale_area = abs(piece.transform.sx * piece.transform.sy)
                    body_area = abs(signed_area(poly_pts)) * max(1e-6, scale_area)
                    if best_body is None or body_area <= best_body_area * 1.001:
                        best_body_area = body_area
                        best_body = HitResult(
                            kind=HitTargetKind.PIECE_BODY,
                            piece_id=piece_id,
                            path_id=piece.outer_path_id,
                            world_position=world_pos,
                            distance=0.0,
                        )

    # 3. ノッチ (NOTCH) 判定
    check_notches = allowed_kinds is None or HitTargetKind.NOTCH in allowed_kinds
    if check_notches:
        notch_radius = node_radius
        for notch_id, notch in doc.notches.items():
            piece = doc.pieces.get(notch.piece_id)
            if not piece:
                continue
            path = piece.paths.get(notch.path_id)
            if not path or len(path.nodes) < 2:
                continue

            n_nodes = len(path.nodes)
            num_segs = n_nodes if path.closed else n_nodes - 1
            for i in range(num_segs):
                n0 = path.nodes[i]
                if n0.id == notch.segment_start_id:
                    n1 = path.nodes[(i + 1) % n_nodes]
                    ctrls = path.get_segment_controls(i)
                    local_notch_pos = evaluate_cubic(ctrls[0], ctrls[1], ctrls[2], ctrls[3], notch.t)
                    w_pos = piece.transform.transform_point(local_notch_pos)
                    d_sq = _dist_sq(world_pos, w_pos)
                    if d_sq <= notch_radius * notch_radius:
                        dist = math.sqrt(d_sq)
                        if best_notch is None or dist < best_notch.distance:
                            best_notch = HitResult(
                                kind=HitTargetKind.NOTCH,
                                piece_id=notch.piece_id,
                                path_id=notch.path_id,
                                notch_id=notch.id,
                                segment_start_node_id=n0.id,
                                segment_end_node_id=n1.id,
                                edge_t=notch.t,
                                world_position=w_pos,
                                distance=dist,
                            )
                    break

    # 3.5. 縫合 (SEAM) 判定 (両端線および中点線への最近接)
    check_seams = allowed_kinds is None or HitTargetKind.SEAM in allowed_kinds
    if check_seams:
        for seam_id, seam in doc.seams.items():
            w_a0, mid_a, w_a1, match_b0, mid_b, match_b1 = doc.get_seam_geometry(seam)

            d0_sq = _dist_to_segment_sq(world_pos, w_a0, match_b0)
            d1_sq = _dist_to_segment_sq(world_pos, w_a1, match_b1)
            dm_sq = _dist_to_segment_sq(world_pos, mid_a, mid_b)
            min_d_sq = min(d0_sq, d1_sq, dm_sq)

            if min_d_sq <= edge_threshold * edge_threshold:
                dist = math.sqrt(min_d_sq)
                if best_seam is None or dist < best_seam.distance:
                    best_seam = HitResult(
                        kind=HitTargetKind.SEAM,
                        piece_id=seam.a.piece_id,
                        seam_id=seam_id,
                        world_position=world_pos,
                        distance=dist,
                    )

    # 優先順位に従って最優先のものを返す
    if best_handle is not None:
        return best_handle
    if best_node is not None:
        return best_node
    if best_notch is not None:
        return best_notch
    if best_seam is not None:
        return best_seam
    if best_edge is not None:
        return best_edge
    if best_internal is not None:
        return best_internal
    if best_body is not None:
        return best_body

    return None
