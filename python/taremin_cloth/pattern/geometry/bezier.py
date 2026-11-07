# -*- coding: utf-8 -*-
"""3次ベジェ曲線評価・弧長分割テーブル・最近点探索モジュール (Blender非依存)."""

import functools
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from taremin_cloth.utils.args_guard import allow_positional

Vec2 = Tuple[float, float]


@allow_positional
def make_segment_controls(
    p0: Vec2,
    out_handle: Vec2 | None,
    in_handle: Vec2 | None,
    p1: Vec2,
) -> tuple[Vec2, Vec2, Vec2, Vec2]:
    """端点とハンドルから3次ベジェの4制御点を構築する.
    
    out_handle は p0 からの相対ベクトル、in_handle は p1 からの相対ベクトル。
    None の場合は直線区間として 1/3, 2/3 分割点を制御点とする。
    """
    c0 = (float(p0[0]), float(p0[1]))
    c3 = (float(p1[0]), float(p1[1]))

    if out_handle is not None:
        c1 = (c0[0] + float(out_handle[0]), c0[1] + float(out_handle[1]))
    else:
        c1 = (c0[0] + (c3[0] - c0[0]) / 3.0, c0[1] + (c3[1] - c0[1]) / 3.0)

    if in_handle is not None:
        c2 = (c3[0] + float(in_handle[0]), c3[1] + float(in_handle[1]))
    else:
        c2 = (c0[0] + 2.0 * (c3[0] - c0[0]) / 3.0, c0[1] + 2.0 * (c3[1] - c0[1]) / 3.0)

    return (c0, c1, c2, c3)


@allow_positional
def compute_auto_handles(
    prev_pos: Vec2,
    curr_pos: Vec2,
    next_pos: Vec2,
    *,
    tension: float = 1.0 / 3.0,
) -> tuple[Vec2, Vec2]:
    """AUTOノード向けに前後頂点から接線方向を求め、滑らかな相対ハンドルペア (in_handle, out_handle) を返す."""
    dx = float(next_pos[0] - prev_pos[0])
    dy = float(next_pos[1] - prev_pos[1])
    dist = math.hypot(dx, dy)

    if dist < 1e-9:
        return ((0.0, 0.0), (0.0, 0.0))

    tx = dx / dist
    ty = dy / dist

    # 入射側・出射側の弦長
    d_in = math.hypot(curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
    d_out = math.hypot(next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])

    in_h = (-tx * d_in * tension, -ty * d_in * tension)
    out_h = (tx * d_out * tension, ty * d_out * tension)
    return (in_h, out_h)


@allow_positional
def evaluate_cubic(
    c0: Vec2,
    c1: Vec2,
    c2: Vec2,
    c3: Vec2,
    t: float,
) -> Vec2:
    """3次ベジェ曲線を t in [0.0, 1.0] で評価する."""
    u = 1.0 - t
    uu = u * u
    uuu = uu * u
    tt = t * t
    ttt = tt * t

    x = uuu * c0[0] + 3.0 * uu * t * c1[0] + 3.0 * u * tt * c2[0] + ttt * c3[0]
    y = uuu * c0[1] + 3.0 * uu * t * c1[1] + 3.0 * u * tt * c2[1] + ttt * c3[1]
    return (x, y)


@allow_positional
def subdivide_cubic(
    c0: Vec2,
    c1: Vec2,
    c2: Vec2,
    c3: Vec2,
    t: float = 0.5,
) -> tuple[tuple[Vec2, Vec2, Vec2, Vec2], tuple[Vec2, Vec2, Vec2, Vec2]]:
    """De Casteljau のアルゴリズムにより3次ベジェ曲線をパラメータ t で2分割する."""
    p01 = ((1 - t) * c0[0] + t * c1[0], (1 - t) * c0[1] + t * c1[1])
    p12 = ((1 - t) * c1[0] + t * c2[0], (1 - t) * c1[1] + t * c2[1])
    p23 = ((1 - t) * c2[0] + t * c3[0], (1 - t) * c2[1] + t * c3[1])

    p012 = ((1 - t) * p01[0] + t * p12[0], (1 - t) * p01[1] + t * p12[1])
    p123 = ((1 - t) * p12[0] + t * p23[0], (1 - t) * p12[1] + t * p23[1])

    p0123 = ((1 - t) * p012[0] + t * p123[0], (1 - t) * p012[1] + t * p123[1])

    left = (c0, p01, p012, p0123)
    right = (p0123, p123, p23, c3)
    return (left, right)


@allow_positional
def split_cubic_bezier(
    p0: Vec2,
    c1: Vec2,
    c2: Vec2,
    p3: Vec2,
    t: float,
) -> tuple[tuple[Vec2, Vec2, Vec2, Vec2], tuple[Vec2, Vec2, Vec2, Vec2]]:
    """ド・カステリョのアルゴリズムにより3次ベジェ曲線をパラメータ t (0..1) で2つに厳密分割する."""
    return subdivide_cubic(p0, c1, c2, p3, t)


@dataclass(frozen=True)
class ArcLengthTable:
    """ベジェ曲線の高精度累積弧長サンプリングテーブル."""

    controls: tuple[Vec2, Vec2, Vec2, Vec2]
    t_values: tuple[float, ...]
    arc_lengths: tuple[float, ...]  # 0.0 から total_length への単調増加
    points: tuple[Vec2, ...]
    total_length: float


def _dist(p0: Vec2, p1: Vec2) -> float:
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])


@allow_positional
def _round_vec2(v: Vec2, decimals: int = 7) -> Vec2:
    return (round(v[0], decimals), round(v[1], decimals))


@allow_positional
@functools.lru_cache(maxsize=2048)
def _build_arc_table_cached(
    c0: Vec2,
    c1: Vec2,
    c2: Vec2,
    c3: Vec2,
    tolerance: float = 0.00005,
) -> ArcLengthTable:
    """再帰2分割によって許容誤差内の弧長テーブルを構築する (LRUキャッシュ内部用)."""
    samples: list[tuple[float, Vec2]] = [(0.0, c0)]

    def recurse(b_ctrl: tuple[Vec2, Vec2, Vec2, Vec2], t_start: float, t_end: float, depth: int):  # audit: allow-positional
        p0, p1, p2, p3 = b_ctrl
        poly_len = _dist(p0, p1) + _dist(p1, p2) + _dist(p2, p3)
        chord_len = _dist(p0, p3)

        if (poly_len - chord_len <= tolerance) or depth >= 12:
            samples.append((t_end, p3))
            return

        t_mid = 0.5 * (t_start + t_end)
        left, right = subdivide_cubic(p0, p1, p2, p3, 0.5)
        recurse(left, t_start, t_mid, depth + 1)
        recurse(right, t_mid, t_end, depth + 1)

    recurse((c0, c1, c2, c3), 0.0, 1.0, 0)

    t_list: list[float] = []
    pts_list: list[Vec2] = []
    arc_list: list[float] = [0.0]
    total = 0.0

    t_list.append(samples[0][0])
    pts_list.append(samples[0][1])

    for i in range(1, len(samples)):
        t_val, pt = samples[i]
        d = _dist(samples[i - 1][1], pt)
        total += d
        t_list.append(t_val)
        pts_list.append(pt)
        arc_list.append(total)

    return ArcLengthTable(
        controls=(c0, c1, c2, c3),
        t_values=tuple(t_list),
        arc_lengths=tuple(arc_list),
        points=tuple(pts_list),
        total_length=total,
    )


@allow_positional
def build_arc_table(
    c0: Vec2,
    c1: Vec2,
    c2: Vec2,
    c3: Vec2,
    *,
    tolerance: float = 0.00005,
    steps: Optional[int] = None,
) -> ArcLengthTable:
    """制御点を丸めてLRUキャッシュを通し、高精度弧長テーブルを返す.
    steps が指定された場合は指定分割数の等間隔テーブルを返す.
    """
    if steps is not None and steps > 0:
        t_vals = [i / steps for i in range(steps + 1)]
        pts = [evaluate_cubic(c0, c1, c2, c3, t) for t in t_vals]
        arcs = [0.0]
        tot = 0.0
        for i in range(len(pts) - 1):
            tot += _dist(pts[i], pts[i + 1])
            arcs.append(tot)
        return ArcLengthTable(
            controls=(c0, c1, c2, c3),
            t_values=tuple(t_vals),
            arc_lengths=tuple(arcs),
            points=tuple(pts),
            total_length=tot,
        )

    rc0 = _round_vec2(c0)
    rc1 = _round_vec2(c1)
    rc2 = _round_vec2(c2)
    rc3 = _round_vec2(c3)
    return _build_arc_table_cached(rc0, rc1, rc2, rc3, tolerance)


def point_at_arc_length(
    table: ArcLengthTable,
    target_s: float,
) -> tuple[Vec2, float]:
    """累積弧長 target_s (0 <= target_s <= total_length) に対応する (座標, パラメータt) を二分探索で返す."""
    total = table.total_length
    if target_s <= 0.0 or total < 1e-9:
        return (table.points[0], table.t_values[0])
    if target_s >= total:
        return (table.points[-1], table.t_values[-1])

    # 二分探索 (bisect_right 相当)
    lengths = table.arc_lengths
    low = 0
    high = len(lengths) - 1

    while low < high:
        mid = (low + high) // 2
        if lengths[mid] <= target_s:
            low = mid + 1
        else:
            high = mid

    idx = low
    s0 = lengths[idx - 1]
    s1 = lengths[idx]
    ds = max(1e-12, s1 - s0)
    alpha = (target_s - s0) / ds

    p0 = table.points[idx - 1]
    p1 = table.points[idx]
    x = p0[0] + alpha * (p1[0] - p0[0])
    y = p0[1] + alpha * (p1[1] - p0[1])

    t0 = table.t_values[idx - 1]
    t1 = table.t_values[idx]
    t_val = t0 + alpha * (t1 - t0)

    return ((x, y), t_val)


@allow_positional
def arc_length_at_t(
    table: ArcLengthTable,
    t: float,
) -> float:
    """パラメータ t in [0.0, 1.0] に対応する累積弧長 (メートル) を二分探索で返す."""
    if t <= 0.0 or table.total_length < 1e-9:
        return 0.0
    if t >= 1.0:
        return float(table.total_length)

    t_vals = table.t_values
    low = 0
    high = len(t_vals) - 1

    while low < high:
        mid = (low + high) // 2
        if t_vals[mid] <= t:
            low = mid + 1
        else:
            high = mid

    idx = low
    t0 = t_vals[idx - 1]
    t1 = t_vals[idx]
    dt = max(1e-12, t1 - t0)
    alpha = (t - t0) / dt

    s0 = table.arc_lengths[idx - 1]
    s1 = table.arc_lengths[idx]
    return float(s0 + alpha * (s1 - s0))


@allow_positional
def sample_sub_segment_points(
    c0: Vec2,
    c1: Vec2,
    c2: Vec2,
    c3: Vec2,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    steps: int = 16,
) -> list[Vec2]:
    """3次ベジェ曲線の部分区間 [t0, t1] をサンプリングした点列を返す."""
    step_count = max(1, steps)
    pts: list[Vec2] = []
    for i in range(step_count + 1):
        t = t0 + (t1 - t0) * (i / step_count)
        pts.append(evaluate_cubic(c0, c1, c2, c3, t))
    return pts


def closest_point_on_table(
    table: ArcLengthTable,
    query: Vec2,
) -> tuple[Vec2, float, float]:
    """クエリ点から弧長テーブル折れ線への最短点 (最近点座標, パラメータt, 最短距離) を返す."""
    pts = table.points
    t_vals = table.t_values
    qx, qy = float(query[0]), float(query[1])

    best_dist_sq = float("inf")
    best_pt = pts[0]
    best_t = t_vals[0]

    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-12:
            proj_x, proj_y = x0, y0
            alpha = 0.0
        else:
            alpha = max(0.0, min(1.0, ((qx - x0) * dx + (qy - y0) * dy) / seg_len_sq))
            proj_x = x0 + alpha * dx
            proj_y = y0 + alpha * dy

        dist_sq = (qx - proj_x) ** 2 + (qy - proj_y) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_pt = (proj_x, proj_y)
            best_t = t_vals[i] + alpha * (t_vals[i + 1] - t_vals[i])

    return (best_pt, best_t, math.sqrt(best_dist_sq))


def merge_breakpoints_by_arc_length(
    table: ArcLengthTable,
    raw_ts: Sequence[float],
    *,
    min_arc_gap: float = 0.001,
) -> list[float]:
    """候補パラメータ列 raw_ts を昇順走査し、実寸弧長で min_arc_gap (既定1mm) 以上離れた分割点リスト [0.0, ..., 1.0] を返す."""
    tot_len = table.total_length
    if tot_len <= min_arc_gap * 2.0:
        return [0.0, 1.0]

    candidates = sorted({float(t) for t in raw_ts if 0.0 < float(t) < 1.0})
    result: list[float] = [0.0]
    last_s = 0.0
    for t in candidates:
        s = arc_length_at_t(table, t)
        if (s - last_s) >= min_arc_gap and (tot_len - s) >= min_arc_gap:
            result.append(t)
            last_s = s
    result.append(1.0)
    return result

