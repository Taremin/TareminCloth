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


@allow_positional
def extract_sub_cubic(
    c0: Vec2,
    c1: Vec2,
    c2: Vec2,
    c3: Vec2,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
) -> tuple[Vec2, Vec2, Vec2, Vec2]:
    """3次ベジェ曲線の部分区間 [t0, t1] を単一の3次ベジェ4制御点として厳密抽出する."""
    a = max(0.0, min(1.0, float(t0)))
    b = max(0.0, min(1.0, float(t1)))
    is_rev = a > b
    if is_rev:
        a, b = b, a
    if b - a < 1e-9:
        pt = evaluate_cubic(c0, c1, c2, c3, a)
        return (pt, pt, pt, pt)

    cur = (c0, c1, c2, c3)
    if b < 1.0 - 1e-9:
        cur, _ = subdivide_cubic(cur[0], cur[1], cur[2], cur[3], b)
    if a > 1e-9:
        rel_a = max(0.0, min(1.0, a / max(1e-9, b)))
        _, cur = subdivide_cubic(cur[0], cur[1], cur[2], cur[3], rel_a)

    if is_rev:
        return (cur[3], cur[2], cur[1], cur[0])
    return cur


@allow_positional
def match_segment_shape(
    source_ctrls: tuple[Vec2, Vec2, Vec2, Vec2],
    target_p0: Vec2,
    target_p1: Vec2,
    *,
    reverse: bool = False,
    flip_curvature: bool = False,
) -> tuple[Vec2, Vec2]:
    """参照元3次ベジェ制御点列の相対ハンドル無次元比率を、対象弦 target_p0 -> target_p1 上へ相似射影し中間制御点 (c1, c2) を返す."""
    s0, s1, s2, s3 = source_ctrls
    if reverse:
        s0, s1, s2, s3 = s3, s2, s1, s0

    t0 = (float(target_p0[0]), float(target_p0[1]))
    t3 = (float(target_p1[0]), float(target_p1[1]))

    t_vx = t3[0] - t0[0]
    t_vy = t3[1] - t0[1]
    t_len = math.hypot(t_vx, t_vy)

    s_vx = s3[0] - s0[0]
    s_vy = s3[1] - s0[1]
    s_len = math.hypot(s_vx, s_vy)

    if t_len < 1e-9 or s_len < 1e-9:
        c1_lin = (t0[0] + t_vx / 3.0, t0[1] + t_vy / 3.0)
        c2_lin = (t0[0] + 2.0 * t_vx / 3.0, t0[1] + 2.0 * t_vy / 3.0)
        return (c1_lin, c2_lin)

    s_ux, s_uy = s_vx / s_len, s_vy / s_len
    s_nx, s_ny = -s_uy, s_ux

    h_out_x, h_out_y = s1[0] - s0[0], s1[1] - s0[1]
    h_in_x, h_in_y = s2[0] - s3[0], s2[1] - s3[1]

    alpha_out = (h_out_x * s_ux + h_out_y * s_uy) / s_len
    beta_out = (h_out_x * s_nx + h_out_y * s_ny) / s_len
    alpha_in = (h_in_x * s_ux + h_in_y * s_uy) / s_len
    beta_in = (h_in_x * s_nx + h_in_y * s_ny) / s_len

    if flip_curvature:
        beta_out = -beta_out
        beta_in = -beta_in

    t_ux, t_uy = t_vx / t_len, t_vy / t_len
    t_nx, t_ny = -t_uy, t_ux

    c1 = (
        t0[0] + t_len * (alpha_out * t_ux + beta_out * t_nx),
        t0[1] + t_len * (alpha_out * t_uy + beta_out * t_ny),
    )
    c2 = (
        t3[0] + t_len * (alpha_in * t_ux + beta_in * t_nx),
        t3[1] + t_len * (alpha_in * t_uy + beta_in * t_ny),
    )
    return (c1, c2)


@allow_positional
def adjust_segment_arc_length(
    c0: Vec2,
    c1: Vec2,
    c2: Vec2,
    c3: Vec2,
    *,
    target_length: float,
    mode: str = "LENGTH_ADJUST_BULGE",
    flip_curvature: bool = False,
) -> tuple[Vec2, Vec2, Vec2, Vec2]:
    """3次ベジェ曲線の弧長が target_length に一致するよう制御点を調整して返す.

    mode:
      - "SIMILAR": 変更せずそのまま返す。
      - "LENGTH_MOVE_END": 始点 c0 を固定し、相似拡大縮小で終点 c3 と制御点を伸縮させる。
      - "LENGTH_ADJUST_BULGE": 両端点 c0, c3 を固定し、弦からの偏差を二分探索（32回）でスケールする。
        なお target_length が弦長以下の場合は幾何学的下限である直線制御点を返す。
    """
    goal = max(0.001, float(target_length))
    if mode == "SIMILAR":
        return (c0, c1, c2, c3)

    if mode == "LENGTH_MOVE_END":
        cur_len = build_arc_table(c0, c1, c2, c3).total_length
        if cur_len < 1e-9:
            return (c0, c1, c2, c3)
        scale = goal / cur_len
        nc1 = (c0[0] + (c1[0] - c0[0]) * scale, c0[1] + (c1[1] - c0[1]) * scale)
        nc2 = (c0[0] + (c2[0] - c0[0]) * scale, c0[1] + (c2[1] - c0[1]) * scale)
        nc3 = (c0[0] + (c3[0] - c0[0]) * scale, c0[1] + (c3[1] - c0[1]) * scale)
        return (c0, nc1, nc2, nc3)

    if mode == "LENGTH_ADJUST_BULGE":
        vx = c3[0] - c0[0]
        vy = c3[1] - c0[1]
        chord = math.hypot(vx, vy)
        if chord < 1e-9:
            return (c0, c1, c2, c3)

        b1 = (c0[0] + vx / 3.0, c0[1] + vy / 3.0)
        b2 = (c0[0] + 2.0 * vx / 3.0, c0[1] + 2.0 * vy / 3.0)
        if goal <= chord + 1e-6:
            return (c0, b1, b2, c3)

        d1x, d1y = c1[0] - b1[0], c1[1] - b1[1]
        d2x, d2y = c2[0] - b2[0], c2[1] - b2[1]
        if math.hypot(d1x, d1y) + math.hypot(d2x, d2y) < 1e-7:
            nx, ny = -vy / chord, vx / chord
            if flip_curvature:
                nx, ny = -nx, -ny
            bulge = 0.25 * chord
            d1x, d1y = bulge * nx, bulge * ny
            d2x, d2y = bulge * nx, bulge * ny

        def eval_len(s: float) -> float:
            tc1 = (b1[0] + s * d1x, b1[1] + s * d1y)
            tc2 = (b2[0] + s * d2x, b2[1] + s * d2y)
            return build_arc_table(c0, tc1, tc2, c3).total_length

        lo = 0.0
        hi = 1.0
        while eval_len(hi) < goal and hi < 64.0:
            hi *= 2.0

        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if eval_len(mid) < goal:
                lo = mid
            else:
                hi = mid

        s_final = 0.5 * (lo + hi)
        return (
            c0,
            (b1[0] + s_final * d1x, b1[1] + s_final * d1y),
            (b2[0] + s_final * d2x, b2[1] + s_final * d2y),
            c3,
        )

    raise ValueError(f"Unsupported arc length adjustment mode: '{mode}'")


