# -*- coding: utf-8 -*-
"""型紙境界の弧長等間隔サンプリングおよび縫合分割数同期ユーティリティ."""

import math
from typing import Sequence
import numpy as np

from .bezier import ArcLengthTable, point_at_arc_length


def compute_seam_subdivision_count(
    len_a: float,
    len_b: float,
    *,
    spacing: float = 0.015,
) -> int:
    """2つの縫合辺長 (m) とサンプリング間隔から共通分割数 N を決定する.
    
    spacing が [0.003, 0.20] (3mm 〜 20cm) の範囲外の場合は ValueError を送出する.
    """
    if not (0.003 <= spacing <= 0.20):
        raise ValueError(f"Sampling spacing must be in [0.003, 0.20] meters, got {spacing}")

    max_len = max(float(len_a), float(len_b), 1e-6)
    return max(1, int(math.ceil(max_len / spacing)))


def sample_edge_range_by_count(
    tables: Sequence[ArcLengthTable],
    count: int,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    reverse: bool = False,
) -> np.ndarray:
    """1つまたは複数の連続セグメントに対し、指定範囲 [t0, t1] の合計弧長を count 等分した (count + 1, 2) の座標列を返す.
    
    reverse=True の場合は逆順にする.
    """
    if count < 1:
        raise ValueError(f"Subdivision count must be >= 1, got {count}")
    if not tables:
        return np.empty((0, 2), dtype=np.float32)

    total_span_length = sum(t.total_length for t in tables)
    if total_span_length < 1e-9:
        first_pt = tables[0].points[0]
        pts = np.tile([first_pt[0], first_pt[1]], (count + 1, 1)).astype(np.float32)
        return pts

    # [t0, t1] の範囲制限
    s_start = max(0.0, min(1.0, float(t0))) * total_span_length
    s_end = max(0.0, min(1.0, float(t1))) * total_span_length
    if s_end < s_start:
        s_start, s_end = s_end, s_start

    span_len = s_end - s_start
    target_s_list = [s_start + (float(i) / count) * span_len for i in range(count + 1)]

    # 複数テーブルにまたがる弧長探索
    out_pts: list[tuple[float, float]] = []

    for s_global in target_s_list:
        accum = 0.0
        found = False
        for tbl in tables:
            t_len = tbl.total_length
            if accum + t_len >= s_global or tbl is tables[-1]:
                s_local = max(0.0, min(t_len, s_global - accum))
                pt, _ = point_at_arc_length(tbl, s_local)
                out_pts.append(pt)
                found = True
                break
            accum += t_len
        if not found:
            out_pts.append(tables[-1].points[-1])

    if reverse:
        out_pts.reverse()

    return np.array(out_pts, dtype=np.float32)
