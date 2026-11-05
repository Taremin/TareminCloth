# -*- coding: utf-8 -*-
"""型紙エディタの要素ヒットテストおよび選択ロジック (Blender非依存)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .document import Document, NodeKind, PathRole
from .geometry import (
    Vec2,
    point_in_polygon,
    build_arc_table,
    closest_point_on_table,
    make_segment_controls,
    compute_auto_handles,
)


class HitTargetKind(str, Enum):
    """ヒットテストで検出される対象要素の優先度順種別."""
    HANDLE = "HANDLE"
    NODE = "NODE"
    NOTCH = "NOTCH"
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
    handle_type: Optional[str] = None  # "in" または "out"
    segment_start_node_id: Optional[str] = None
    segment_end_node_id: Optional[str] = None
    edge_t: float = 0.0
    world_position: tuple[float, float] = (0.0, 0.0)
    distance: float = 0.0


def _dist_sq(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    # audit: allow-positional
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy


def hit_test(
    doc: Document,
    world_pos: tuple[float, float],
    *,
    handle_radius: float = 0.006,
    node_radius: float = 0.008,
    edge_threshold: float = 0.006,
    selected_node_ids: Optional[set[str]] = None,
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
    best_edge: Optional[HitResult] = None
    best_internal: Optional[HitResult] = None
    best_body: Optional[HitResult] = None

    # 各ピースを走査（後から描画されるピースを優先するため逆順も考慮可能）
    for piece_id, piece in doc.pieces.items():
        local_pos = piece.transform.inverse_apply(world_pos)

        # 1. ハンドル & 2. ノード の判定
        for path_id, path in piece.paths.items():
            is_outer = (path.role == PathRole.OUTER)
            n_nodes = len(path.nodes)
            if n_nodes == 0:
                continue

            for i, node in enumerate(path.nodes):
                w_node_pos = piece.transform.transform_point(node.position)
                d_sq = _dist_sq(world_pos, w_node_pos)

                if d_sq <= node_radius * node_radius:
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
                if node.id in sel_nodes and node.kind != NodeKind.CORNER:
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
            if n_nodes >= 2:
                num_segs = n_nodes if path.closed else n_nodes - 1
                for i in range(num_segs):
                    n0 = path.nodes[i]
                    n1 = path.nodes[(i + 1) % n_nodes]

                    out_h = n0.out_handle
                    in_h = n1.in_handle
                    if n0.kind == NodeKind.AUTO:
                        p_prev = path.nodes[(i - 1) % n_nodes].position if path.closed else path.nodes[max(0, i - 1)].position
                        _, out_h = compute_auto_handles(p_prev, n0.position, n1.position)
                    if n1.kind == NodeKind.AUTO:
                        p_next = path.nodes[(i + 2) % n_nodes].position if path.closed else path.nodes[min(n_nodes - 1, i + 2)].position
                        in_h, _ = compute_auto_handles(n0.position, n1.position, p_next)

                    ctrls = make_segment_controls(n0.position, out_h, in_h, n1.position)
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

        # 6. ピース内部 (PIECE_BODY) 判定
        outer_path = piece.paths.get(piece.outer_path_id)
        if outer_path and len(outer_path.nodes) >= 3:
            poly_pts = [nd.position for nd in outer_path.nodes]
            if point_in_polygon(local_pos, poly_pts):
                # 内部面ヒット。距離は0として扱う
                best_body = HitResult(
                    kind=HitTargetKind.PIECE_BODY,
                    piece_id=piece_id,
                    path_id=piece.outer_path_id,
                    world_position=world_pos,
                    distance=0.0,
                )

    # 優先順位に従って最優先のものを返す
    if best_handle is not None:
        return best_handle
    if best_node is not None:
        return best_node
    if best_notch is not None:
        return best_notch
    if best_edge is not None:
        return best_edge
    if best_internal is not None:
        return best_internal
    if best_body is not None:
        return best_body

    return None
