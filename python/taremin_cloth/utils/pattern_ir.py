# -*- coding: utf-8 -*-
"""Milestone 2: 2D型紙中間表現 (IR) & ノッチ同期分割 & 中継ガイド縫合Loose Edge生成 (Blender非依存)."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class NotchPointIR:
    """境界ループ上の固定点（コーナー頂点または合印ノッチ）."""

    uid: str
    arc_length_ratio: float  # 外周ループ全体の弧長比率 [0.0, 1.0)
    position_2d: np.ndarray  # shape (2,), float32 [u, v] (単位: メートル)
    is_sharp_corner: bool = False


@dataclass
class BoundarySegmentIR:
    """隣接する2つのNotch/Cornerで区切られた1つの境界区間."""

    segment_id: str
    start_uid: str
    end_uid: str
    polyline_2d: np.ndarray  # 区間を構成する点列 shape [M, 2], float32
    arc_length: float  # 区間の実寸弧長 (m)
    target_subdivisions: Optional[int] = None  # 縫合ペア同期で決定された分割数 N (頂点数は N+1)


@dataclass
class PatternContour2D:
    """単一2D型紙パーツの輪郭中間表現."""

    piece_name: str
    outer_segments: list[BoundarySegmentIR]
    internal_lines: list[np.ndarray] = field(default_factory=list)
    target_edge_length: float = 0.015
    layer_id: int = 0
    arrangement_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )


@dataclass
class WaypointGuideIR:
    """3D空間上で縫合エッジをアバター周囲へ迂回させる中継ガイド曲線."""

    guide_id: str
    polyline_3d: np.ndarray  # shape [P, 3], float32
    notch_ratios: list[float] = field(default_factory=list)


@dataclass
class SeamPairIR:
    """2つの境界区間（および任意の中継ガイド群）を結ぶ縫合定義."""

    pair_id: str
    piece_a: str
    segment_a_id: str
    piece_b: str
    segment_b_id: str
    reverse_b: bool = False
    waypoint_guides: list[WaypointGuideIR] = field(default_factory=list)


def synchronize_seam_subdivisions(
    contours: list[PatternContour2D],
    seams: list[SeamPairIR],
) -> None:
    """縫合ペアで結ばれた境界区間の分割数 N を相互同期し、未設定区間には目標長から分割数を設定する."""
    contour_map = {c.piece_name: c for c in contours}
    segment_map: dict[tuple[str, str], BoundarySegmentIR] = {}
    for c in contours:
        for seg in c.outer_segments:
            segment_map[(c.piece_name, seg.segment_id)] = seg

    # 1. 縫合ペアごとの同期
    for seam in seams:
        seg_a = segment_map.get((seam.piece_a, seam.segment_a_id))
        seg_b = segment_map.get((seam.piece_b, seam.segment_b_id))
        if not seg_a or not seg_b:
            continue

        c_a = contour_map.get(seam.piece_a)
        c_b = contour_map.get(seam.piece_b)
        l_a = c_a.target_edge_length if c_a else 0.015
        l_b = c_b.target_edge_length if c_b else 0.015
        l_eff = 0.5 * (l_a + l_b)

        l_total = seg_a.arc_length + seg_b.arc_length
        n_seg = max(1, int(round(l_total / (2.0 * l_eff))))

        seg_a.target_subdivisions = n_seg
        seg_b.target_subdivisions = n_seg

    # 2. 縫合ペアに属さない自由区間の分割数を設定
    for c in contours:
        for seg in c.outer_segments:
            if seg.target_subdivisions is None:
                n_free = max(1, int(round(seg.arc_length / max(1e-6, c.target_edge_length))))
                seg.target_subdivisions = n_free


def _resample_polyline_2d(pts: np.ndarray, num_subdivisions: int) -> np.ndarray:
    """2D折れ線を累積弧長等間隔で num_subdivisions 等分 (頂点数 num_subdivisions + 1) にリサンプリングする."""
    n_pts = len(pts)
    if n_pts == 0:
        return np.empty((0, 2), dtype=np.float32)
    if n_pts == 1:
        return np.tile(pts[0], (num_subdivisions + 1, 1)).astype(np.float32)

    diffs = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_dist[-1]

    if total_len < 1e-9:
        return np.tile(pts[0], (num_subdivisions + 1, 1)).astype(np.float32)

    target_dists = np.linspace(0.0, total_len, num_subdivisions + 1)
    rx = np.interp(target_dists, cum_dist, pts[:, 0])
    ry = np.interp(target_dists, cum_dist, pts[:, 1])
    return np.column_stack([rx, ry]).astype(np.float32)


def _resample_polyline_3d(pts: np.ndarray, num_points: int) -> np.ndarray:
    """3D折れ線を累積弧長等間隔で num_points 個の点にリサンプリングする."""
    n_pts = len(pts)
    if n_pts == 0:
        return np.empty((0, 3), dtype=np.float32)
    if n_pts == 1 or num_points == 1:
        return np.tile(pts[0], (num_points, 1)).astype(np.float32)

    diffs = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_dist[-1]

    if total_len < 1e-9:
        return np.tile(pts[0], (num_points, 1)).astype(np.float32)

    target_dists = np.linspace(0.0, total_len, num_points)
    rx = np.interp(target_dists, cum_dist, pts[:, 0])
    ry = np.interp(target_dists, cum_dist, pts[:, 1])
    rz = np.interp(target_dists, cum_dist, pts[:, 2])
    return np.column_stack([rx, ry, rz]).astype(np.float32)


def sample_pattern_boundary_loop(
    contour: PatternContour2D,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """各 outer_segments を target_subdivisions 等分し、単一反時計回り境界ループと頂点インデックス辞書を返す."""
    segments = contour.outer_segments
    if not segments:
        return np.empty((0, 2), dtype=np.float32), {}

    loop_points: list[np.ndarray] = []
    seg_vertex_map: dict[str, list[int]] = {}

    num_segs = len(segments)

    for k, seg in enumerate(segments):
        n_sub = seg.target_subdivisions
        if n_sub is None:
            n_sub = max(1, int(round(seg.arc_length / max(1e-6, contour.target_edge_length))))

        sampled_pts = _resample_polyline_2d(seg.polyline_2d, n_sub)
        seg_indices: list[int] = []

        if k == 0:
            # 最初の区間の始点
            loop_points.append(sampled_pts[0])
            start_idx = 0
        else:
            # 前の区間の終点インデックス
            prev_seg_id = segments[k - 1].segment_id
            start_idx = seg_vertex_map[prev_seg_id][-1]

        seg_indices.append(start_idx)

        # 中間点 1..n_sub-1 を追加
        for i in range(1, n_sub):
            idx = len(loop_points)
            loop_points.append(sampled_pts[i])
            seg_indices.append(idx)

        # 終点
        if k == num_segs - 1:
            # 最終区間の終点は第0区間の始点 (0) に一致
            end_idx = 0
        else:
            # 次の区間と共有する新規頂点
            end_idx = len(loop_points)
            loop_points.append(sampled_pts[n_sub])

        seg_indices.append(end_idx)
        seg_vertex_map[seg.segment_id] = seg_indices

    boundary_loop_2d = np.array(loop_points, dtype=np.float32)
    return boundary_loop_2d, seg_vertex_map


def build_sewing_loose_edges(
    base_positions: np.ndarray,
    segment_vertex_map: dict[str, dict[str, list[int]]],
    *,
    seams: list[SeamPairIR],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """統合メッシュの頂点配列と縫合定義から直線縫合および中継ガイド付き Loose Edge 列を生成する."""
    all_positions = [base_positions[i].copy() for i in range(len(base_positions))]
    loose_edges: list[tuple[int, int]] = []

    for seam in seams:
        map_a = segment_vertex_map.get(seam.piece_a, {})
        map_b = segment_vertex_map.get(seam.piece_b, {})
        idx_a = map_a.get(seam.segment_a_id)
        idx_b = map_b.get(seam.segment_b_id)

        if not idx_a or not idx_b:
            continue

        if seam.reverse_b:
            idx_b = list(reversed(idx_b))

        m = min(len(idx_a), len(idx_b))
        if m == 0:
            continue

        if not seam.waypoint_guides:
            # 直線縫合エッジ
            for i in range(m):
                loose_edges.append((idx_a[i], idx_b[i]))
        else:
            # 中継ガイド付きチェーン: 各頂点ペアごとにガイド曲線をサンプリングしてチェーン構築
            num_guides = len(seam.waypoint_guides)
            sampled_guides = [
                _resample_polyline_3d(g.polyline_3d, m)
                for g in seam.waypoint_guides
            ]

            for i in range(m):
                chain_nodes = [idx_a[i]]
                for g_idx in range(num_guides):
                    wp_pos = sampled_guides[g_idx][i]
                    node_idx = len(all_positions)
                    all_positions.append(wp_pos)
                    chain_nodes.append(node_idx)
                chain_nodes.append(idx_b[i])

                for k in range(len(chain_nodes) - 1):
                    loose_edges.append((chain_nodes[k], chain_nodes[k + 1]))

    all_pos_arr = np.array(all_positions, dtype=np.float32)
    return all_pos_arr, loose_edges
