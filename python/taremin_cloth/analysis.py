"""
メッシュ幾何解析・異常値検出ライブラリ
Blender非依存で、NumPy配列ベースの高速幾何計算を提供します。
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import taremin_cloth_core as _core

    _RUST_GEOMETRY = all(
        hasattr(_core, n)
        for n in (
            "tri_tri_intersect",
            "find_triangle_intersections",
            "find_proximity_violations",
        )
    )
    _RUST_GEOMETRY_SEGMENTS = all(
        hasattr(_core, n)
        for n in (
            "tri_tri_intersection_segment",
            "find_triangle_intersection_segments",
            "find_mesh_collider_intersections",
        )
    )
except Exception:
    _core = None
    _RUST_GEOMETRY = False
    _RUST_GEOMETRY_SEGMENTS = False

try:
    from .utils import allow_positional
except ImportError:
    try:
        from taremin_cloth.utils import allow_positional
    except ImportError:
        def allow_positional(fn):
            return fn


@allow_positional
def coplanar_tri_tri_2d(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
    q0: np.ndarray, q1: np.ndarray, q2: np.ndarray,
    eps: float = 1e-5
) -> bool:
    """
    同一平面上にある2つの2D三角形が真に交差・重なり合っているかを判定します。
    エッジ共有や頂点接触は交差とみなしません。
    """
    @allow_positional
    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    @allow_positional
    def edge_cross_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)
        tol_2d = min(float(eps), 1e-6)
        tol_ab = float(np.hypot(b[0] - a[0], b[1] - a[1])) * tol_2d
        tol_cd = float(np.hypot(d[0] - c[0], d[1] - c[1])) * tol_2d
        return (
            ((o1 > tol_ab and o2 < -tol_ab) or (o1 < -tol_ab and o2 > tol_ab))
            and ((o3 > tol_cd and o4 < -tol_cd) or (o3 < -tol_cd and o4 > tol_cd))
        )

    @allow_positional
    def point_in_tri(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        o1 = orient(a, b, p)
        o2 = orient(b, c, p)
        o3 = orient(c, a, p)
        tol_2d = min(float(eps), 1e-6)
        tol_ab = float(np.hypot(b[0] - a[0], b[1] - a[1])) * tol_2d
        tol_bc = float(np.hypot(c[0] - b[0], c[1] - b[1])) * tol_2d
        tol_ca = float(np.hypot(a[0] - c[0], a[1] - c[1])) * tol_2d
        return (
            (o1 > tol_ab and o2 > tol_bc and o3 > tol_ca)
            or (o1 < -tol_ab and o2 < -tol_bc and o3 < -tol_ca)
        )

    # 1. エッジ交差テスト (3x3 = 9ペア)
    edges_p = [(p0, p1), (p1, p2), (p2, p0)]
    edges_q = [(q0, q1), (q1, q2), (q2, q0)]
    for ep in edges_p:
        for eq in edges_q:
            if edge_cross_2d(ep[0], ep[1], eq[0], eq[1]):
                return True

    # 2. 包含テスト
    for q in (q0, q1, q2):
        if point_in_tri(q, p0, p1, p2):
            return True
    for p in (p0, p1, p2):
        if point_in_tri(p, q0, q1, q2):
            return True

    return False


@allow_positional
def _tri_tri_intersection_segment_numpy(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
    u0: np.ndarray, u1: np.ndarray, u2: np.ndarray,
    *,
    eps: float = 1e-12,
    coplanar_dist_eps: float = 1e-6,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Rust `tri_tri_intersection_segment` と同一式の NumPy リファレンス実装。"""
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    u0 = np.asarray(u0, dtype=np.float64)
    u1 = np.asarray(u1, dtype=np.float64)
    u2 = np.asarray(u2, dtype=np.float64)

    n1_raw = np.cross(v1 - v0, v2 - v0)
    n1_len = float(np.linalg.norm(n1_raw))
    if n1_len <= 1e-15:
        return None
    n1 = n1_raw / n1_len
    d1 = -float(np.dot(n1, v0))
    du0_raw = float(np.dot(n1, u0)) + d1
    du1_raw = float(np.dot(n1, u1)) + d1
    du2_raw = float(np.dot(n1, u2)) + d1

    n2_raw = np.cross(u1 - u0, u2 - u0)
    n2_len = float(np.linalg.norm(n2_raw))
    if n2_len <= 1e-15:
        return None
    n2 = n2_raw / n2_len
    d2 = -float(np.dot(n2, u0))
    dv0_raw = float(np.dot(n2, v0)) + d2
    dv1_raw = float(np.dot(n2, v1)) + d2
    dv2_raw = float(np.dot(n2, v2)) + d2

    d = np.cross(n1, n2)
    if float(np.dot(d, d)) < 1e-10:
        coplanar_u = max(abs(du0_raw), abs(du1_raw), abs(du2_raw)) < coplanar_dist_eps
        coplanar_v = max(abs(dv0_raw), abs(dv1_raw), abs(dv2_raw)) < coplanar_dist_eps
        if not (coplanar_u and coplanar_v):
            return None
        max_idx = int(np.argmax(np.abs(n1)))
        axes = [i for i in range(3) if i != max_idx]
        if coplanar_tri_tri_2d(v0[axes], v1[axes], v2[axes], u0[axes], u1[axes], u2[axes], coplanar_dist_eps):
            return _coplanar_intersection_segment_3d(
                v0, v1, v2, u0, u1, u2, axes[0], axes[1], eps=coplanar_dist_eps
            )
        return None

    du0 = 0.0 if abs(du0_raw) < eps else du0_raw
    du1 = 0.0 if abs(du1_raw) < eps else du1_raw
    du2 = 0.0 if abs(du2_raw) < eps else du2_raw
    du0du1 = du0 * du1
    du0du2 = du0 * du2
    if du0du1 > 0.0 and du0du2 > 0.0:
        return None

    dv0 = 0.0 if abs(dv0_raw) < eps else dv0_raw
    dv1 = 0.0 if abs(dv1_raw) < eps else dv1_raw
    dv2 = 0.0 if abs(dv2_raw) < eps else dv2_raw
    dv0dv1 = dv0 * dv1
    dv0dv2 = dv0 * dv2
    if dv0dv1 > 0.0 and dv0dv2 > 0.0:
        return None

    all_du_zero = (du0 == 0.0 and du1 == 0.0 and du2 == 0.0)
    all_dv_zero = (dv0 == 0.0 and dv1 == 0.0 and dv2 == 0.0)
    if all_du_zero or all_dv_zero:
        max_idx = int(np.argmax(np.abs(n1)))
        axes = [i for i in range(3) if i != max_idx]
        if coplanar_tri_tri_2d(v0[axes], v1[axes], v2[axes], u0[axes], u1[axes], u2[axes], coplanar_dist_eps):
            return _coplanar_intersection_segment_3d(
                v0, v1, v2, u0, u1, u2, axes[0], axes[1], eps=coplanar_dist_eps
            )
        return None

    max_idx = int(np.argmax(np.abs(d)))

    @allow_positional
    def intervals_3d(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                     d0: float, d1v: float, d2v: float, d0d1: float, d0d2: float) -> Tuple[np.ndarray, np.ndarray]:
        if d0d1 > 0.0:
            return (p2 + (p0 - p2) * (d2v / (d2v - d0)), p2 + (p1 - p2) * (d2v / (d2v - d1v)))
        elif d0d2 > 0.0:
            return (p1 + (p0 - p1) * (d1v / (d1v - d0)), p1 + (p2 - p1) * (d1v / (d1v - d2v)))
        elif d1v * d2v > 0.0 or d0 != 0.0:
            return (p0 + (p1 - p0) * (d0 / (d0 - d1v)), p0 + (p2 - p0) * (d0 / (d0 - d2v)))
        elif d1v != 0.0:
            return (p1 + (p0 - p1) * (d1v / (d1v - d0)), p1 + (p2 - p1) * (d1v / (d1v - d2v)))
        else:
            return (p2 + (p0 - p2) * (d2v / (d2v - d0)), p2 + (p1 - p2) * (d2v / (d2v - d1v)))

    pa0, pa1 = intervals_3d(v0, v1, v2, dv0, dv1, dv2, dv0dv1, dv0dv2)
    pb0, pb1 = intervals_3d(u0, u1, u2, du0, du1, du2, du0du1, du0du2)

    if float(pa0[max_idx]) <= float(pa1[max_idx]):
        t1_min, p1_min, t1_max, p1_max = float(pa0[max_idx]), pa0, float(pa1[max_idx]), pa1
    else:
        t1_min, p1_min, t1_max, p1_max = float(pa1[max_idx]), pa1, float(pa0[max_idx]), pa0

    if float(pb0[max_idx]) <= float(pb1[max_idx]):
        t2_min, p2_min, t2_max, p2_max = float(pb0[max_idx]), pb0, float(pb1[max_idx]), pb1
    else:
        t2_min, p2_min, t2_max, p2_max = float(pb1[max_idx]), pb1, float(pb0[max_idx]), pb0

    if t1_max < t2_min - eps or t2_max < t1_min - eps:
        return None

    seg_start = p1_min if t1_min >= t2_min else p2_min
    seg_end = p1_max if t1_max <= t2_max else p2_max
    return seg_start.astype(np.float32), seg_end.astype(np.float32)


@allow_positional
def tri_tri_intersection_moller(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
    u0: np.ndarray, u1: np.ndarray, u2: np.ndarray,
    eps: float = 1e-12
) -> bool:
    """
    Tomas Möller (1997) "A Fast Triangle-Triangle Intersection Test" による厳密交差判定。
    CG・物理シミュレーション標準。同一平面上の場合は2D射影による包含・交差判定へフォールバックします。
    Rust核 (`taremin_cloth_core.tri_tri_intersect`) を正本とし、利用可能な場合はそちらを優先する。
    """
    if _RUST_GEOMETRY and eps == 1e-12:
        try:
            return bool(
                _core.tri_tri_intersect(
                    np.ascontiguousarray(v0, dtype=np.float32).tolist(),
                    np.ascontiguousarray(v1, dtype=np.float32).tolist(),
                    np.ascontiguousarray(v2, dtype=np.float32).tolist(),
                    np.ascontiguousarray(u0, dtype=np.float32).tolist(),
                    np.ascontiguousarray(u1, dtype=np.float32).tolist(),
                    np.ascontiguousarray(u2, dtype=np.float32).tolist(),
                )
            )
        except Exception:
            pass
    return _tri_tri_intersection_segment_numpy(
        v0, v1, v2, u0, u1, u2, eps=eps, coplanar_dist_eps=max(float(eps), 1e-6)
    ) is not None


# 後方互換性エイリアス
tri_tri_intersection_sat = tri_tri_intersection_moller


@allow_positional
def _coplanar_intersection_segment_3d(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
    u0: np.ndarray, u1: np.ndarray, u2: np.ndarray,
    ax0: int, ax1: int,
    *,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """同一平面上で交差する2三角形から可視化用の代表3D交差線分を抽出する。"""
    tol_2d = min(float(eps), 1e-6)

    @allow_positional
    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    @allow_positional
    def edge_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)
        tol_ab = float(np.hypot(b[0] - a[0], b[1] - a[1])) * tol_2d
        tol_cd = float(np.hypot(d[0] - c[0], d[1] - c[1])) * tol_2d
        return (
            ((o1 > tol_ab and o2 < -tol_ab) or (o1 < -tol_ab and o2 > tol_ab))
            and ((o3 > tol_cd and o4 < -tol_cd) or (o3 < -tol_cd and o4 > tol_cd))
        )

    @allow_positional
    def pt_in_tri(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        o1 = orient(a, b, p)
        o2 = orient(b, c, p)
        o3 = orient(c, a, p)
        tol_ab = float(np.hypot(b[0] - a[0], b[1] - a[1])) * tol_2d
        tol_bc = float(np.hypot(c[0] - b[0], c[1] - b[1])) * tol_2d
        tol_ca = float(np.hypot(a[0] - c[0], a[1] - c[1])) * tol_2d
        return (
            (o1 > tol_ab and o2 > tol_bc and o3 > tol_ca)
            or (o1 < -tol_ab and o2 < -tol_bc and o3 < -tol_ca)
        )

    axes = [ax0, ax1]
    pv0, pv1, pv2 = v0[axes], v1[axes], v2[axes]
    pu0, pu1, pu2 = u0[axes], u1[axes], u2[axes]
    pts: List[np.ndarray] = []

    for ep0, ep1 in ((v0, v1), (v1, v2), (v2, v0)):
        a, b = ep0[axes], ep1[axes]
        for eq0, eq1 in ((u0, u1), (u1, u2), (u2, u0)):
            c, d = eq0[axes], eq1[axes]
            if edge_cross(a, b, c, d):
                r0, r1 = float(b[0] - a[0]), float(b[1] - a[1])
                s0, s1 = float(d[0] - c[0]), float(d[1] - c[1])
                denom = r0 * s1 - r1 * s0
                if abs(denom) > 1e-15:
                    t = float(((c[0] - a[0]) * s1 - (c[1] - a[1]) * s0) / denom)
                    t = min(max(t, 0.0), 1.0)
                    pts.append(ep0 + t * (ep1 - ep0))

    for q in (u0, u1, u2):
        if pt_in_tri(q[axes], pv0, pv1, pv2):
            pts.append(q)
    for p in (v0, v1, v2):
        if pt_in_tri(p[axes], pu0, pu1, pu2):
            pts.append(p)

    if len(pts) >= 2:
        best_d2 = -1.0
        best_a, best_b = pts[0], pts[1]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                diff = pts[j] - pts[i]
                d2 = float(np.dot(diff, diff))
                if d2 > best_d2:
                    best_d2 = d2
                    best_a, best_b = pts[i], pts[j]
        return best_a.astype(np.float32), best_b.astype(np.float32)
    if len(pts) == 1:
        p32 = pts[0].astype(np.float32)
        return p32, p32
    cv = ((v0 + v1 + v2) / 3.0).astype(np.float32)
    cu = ((u0 + u1 + u2) / 3.0).astype(np.float32)
    return cv, cu


@allow_positional
def tri_tri_intersection_segment(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
    u0: np.ndarray, u1: np.ndarray, u2: np.ndarray,
    eps: float = 1e-12,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Möller (1997) による2三角形の交差判定と 3D 交差線分抽出。
    交差する場合は `(p_start, p_end)`（各 `float32[3]`）を返し、非交差時は `None`。
    Rust核 (`taremin_cloth_core.tri_tri_intersection_segment`) を正本とする。
    """
    if _RUST_GEOMETRY_SEGMENTS and eps == 1e-12:
        try:
            got = _core.tri_tri_intersection_segment(
                np.ascontiguousarray(v0, dtype=np.float32).tolist(),
                np.ascontiguousarray(v1, dtype=np.float32).tolist(),
                np.ascontiguousarray(v2, dtype=np.float32).tolist(),
                np.ascontiguousarray(u0, dtype=np.float32).tolist(),
                np.ascontiguousarray(u1, dtype=np.float32).tolist(),
                np.ascontiguousarray(u2, dtype=np.float32).tolist(),
            )
            if got is None:
                return None
            return (
                np.asarray(got[0], dtype=np.float32),
                np.asarray(got[1], dtype=np.float32),
            )
        except Exception:
            pass
    return _tri_tri_intersection_segment_numpy(
        v0, v1, v2, u0, u1, u2, eps=eps, coplanar_dist_eps=max(float(eps), 1e-6)
    )


def _build_cloth_spatial_grid(
    positions: np.ndarray,
    faces: np.ndarray,
    *,
    cell_size: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Dict[Tuple[int, int, int], List[int]], List[int]]]:
    """有効布面のAABBと空間ハッシュグリッドを構築する共通ヘルパー。"""
    if len(faces) == 0 or len(positions) == 0:
        return None
    n = len(positions)
    f_arr = np.asarray(faces, dtype=np.int64)
    valid_mask = (
        (f_arr[:, 0] != f_arr[:, 1])
        & (f_arr[:, 1] != f_arr[:, 2])
        & (f_arr[:, 2] != f_arr[:, 0])
        & (f_arr >= 0).all(axis=1)
        & (f_arr < n).all(axis=1)
    )
    valid_indices = np.where(valid_mask)[0].tolist()
    if not valid_indices:
        return None

    valid_faces = f_arr[valid_indices]
    tris = np.asarray(positions, dtype=np.float64)[valid_faces]  # [M, 3, 3]
    mins = np.min(tris, axis=1)
    maxs = np.max(tris, axis=1)
    global_min = np.min(mins, axis=0)
    global_max = np.max(maxs, axis=0)
    max_span = float(np.max(global_max - global_min))

    if cell_size is None or cell_size <= 0.0:
        e01 = np.linalg.norm(tris[:, 1, :] - tris[:, 0, :], axis=1)
        e12 = np.linalg.norm(tris[:, 2, :] - tris[:, 1, :], axis=1)
        e20 = np.linalg.norm(tris[:, 0, :] - tris[:, 2, :], axis=1)
        mean_len = float(np.mean((e01 + e12 + e20) / 3.0))
        cell = max(mean_len * 2.0, max_span / 128.0, 1e-4)
    else:
        cell = max(float(cell_size), max_span / 128.0, 1e-4)

    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for k, (bmin, bmax) in enumerate(zip(mins, maxs)):
        cmin = np.floor(bmin / cell).astype(int)
        cmax = np.floor(bmax / cell).astype(int)
        for cx in range(cmin[0], cmax[0] + 1):
            for cy in range(cmin[1], cmax[1] + 1):
                for cz in range(cmin[2], cmax[2] + 1):
                    grid.setdefault((cx, cy, cz), []).append(k)

    return tris, mins, maxs, global_min, global_max, cell, grid, valid_indices


def _build_sewing_vertex_clusters(
    num_vertices: int,
    sewing_springs: np.ndarray,
) -> np.ndarray:
    """縫合スプリングで連結された頂点を Union-Find でクラスタ代表IDへ統合する。"""
    parent = list(range(num_vertices))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in np.asarray(sewing_springs, dtype=np.int64):
        a, b = int(row[0]), int(row[1])
        if 0 <= a < num_vertices and 0 <= b < num_vertices:
            ra, rb = find(a), find(b)
            if ra != rb:
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

    for v in range(num_vertices):
        parent[v] = find(v)
    return np.asarray(parent, dtype=np.int64)


def find_triangle_intersection_segments(
    positions: np.ndarray,
    faces: np.ndarray,
    *,
    ignore_adjacent: bool = True,
    cell_size: Optional[float] = None,
    sewing_springs: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    メッシュ内の自己交差している三角形ペアと 3D 交差線分を検出します。
    `ignore_adjacent=True` かつ `sewing_springs` 指定時は、縫合スプリングで結合された
    頂点クラスタを共有する境界隣接面も除外します。
    戻り値: `(pairs, segments)`
    - `pairs`: `[(f0, f1), ...]` (`f0 < f1` 昇順)
    - `segments`: shape `(K, 2, 3)` `dtype=np.float32`
    """
    sew_arr = None
    if sewing_springs is not None and len(sewing_springs) > 0:
        sew_arr = np.ascontiguousarray(sewing_springs, dtype=np.uint32)

    if _RUST_GEOMETRY_SEGMENTS:
        try:
            call_kwargs = {}
            if sew_arr is not None:
                call_kwargs["sewing_springs"] = sew_arr
            pairs_raw, segs = _core.find_triangle_intersection_segments(
                np.ascontiguousarray(positions, dtype=np.float32),
                np.ascontiguousarray(faces, dtype=np.uint32),
                bool(ignore_adjacent),
                float(cell_size) if cell_size else None,
                **call_kwargs,
            )
            pairs = [(int(a), int(b)) for (a, b) in pairs_raw]
            return pairs, np.ascontiguousarray(segs, dtype=np.float32).reshape((-1, 2, 3))
        except Exception:
            pass

    built = _build_cloth_spatial_grid(positions, faces, cell_size=cell_size)
    if built is None:
        return [], np.empty((0, 2, 3), dtype=np.float32)
    tris, mins, maxs, _gmin, _gmax, _cell, grid, valid_indices = built

    clusters = None
    if ignore_adjacent and sew_arr is not None:
        clusters = _build_sewing_vertex_clusters(len(positions), sew_arr)

    tested = set()
    results = []
    for cell_faces in grid.values():
        m_cell = len(cell_faces)
        if m_cell < 2:
            continue
        for i in range(m_cell):
            ka = cell_faces[i]
            for j in range(i + 1, m_cell):
                kb = cell_faces[j]
                fa = valid_indices[ka]
                fb = valid_indices[kb]
                if fa < fb:
                    pair, k0, k1 = (fa, fb), ka, kb
                else:
                    pair, k0, k1 = (fb, fa), kb, ka
                if pair in tested:
                    continue
                tested.add(pair)

                if ignore_adjacent:
                    v0 = faces[pair[0]]
                    v1 = faces[pair[1]]
                    if clusters is not None:
                        c0 = clusters[v0]
                        c1 = clusters[v1]
                        if c0[0] in c1 or c0[1] in c1 or c0[2] in c1:
                            continue
                    elif v0[0] in v1 or v0[1] in v1 or v0[2] in v1:
                        continue

                if np.any(mins[k0] > maxs[k1]) or np.any(mins[k1] > maxs[k0]):
                    continue

                p0, p1, p2 = tris[k0]
                q0, q1, q2 = tris[k1]
                seg = tri_tri_intersection_segment(p0, p1, p2, q0, q1, q2)
                if seg is not None:
                    results.append((pair, seg))

    results.sort(key=lambda item: item[0])
    if not results:
        return [], np.empty((0, 2, 3), dtype=np.float32)
    pairs = [item[0] for item in results]
    segs = np.array([[item[1][0], item[1][1]] for item in results], dtype=np.float32)
    return pairs, segs


@allow_positional
def find_mesh_collider_intersections(
    positions: np.ndarray,
    faces: np.ndarray,
    collider_tris: np.ndarray,
    *,
    cell_size: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    布メッシュとコライダー三角形メッシュ間の交差ペアと 3D 交差線分を検出します。
    戻り値: `(pairs, segments)`
    - `pairs`: `[(cloth_face_idx, collider_tri_idx), ...]`
    - `segments`: shape `(K, 2, 3)` `dtype=np.float32`
    """
    if collider_tris is None or len(collider_tris) == 0:
        return [], np.empty((0, 2, 3), dtype=np.float32)

    cols = np.asarray(collider_tris, dtype=np.float32)
    if cols.ndim == 3 and cols.shape[1] == 3 and cols.shape[2] == 3:
        cols = cols.reshape((-1, 9))
    cols = np.ascontiguousarray(cols, dtype=np.float32)
    if len(cols) == 0:
        return [], np.empty((0, 2, 3), dtype=np.float32)

    if _RUST_GEOMETRY_SEGMENTS:
        try:
            pairs_raw, segs = _core.find_mesh_collider_intersections(
                np.ascontiguousarray(positions, dtype=np.float32),
                np.ascontiguousarray(faces, dtype=np.uint32),
                cols,
                float(cell_size) if cell_size else None,
            )
            pairs = [(int(a), int(b)) for (a, b) in pairs_raw]
            return pairs, np.ascontiguousarray(segs, dtype=np.float32).reshape((-1, 2, 3))
        except Exception:
            pass

    built = _build_cloth_spatial_grid(positions, faces, cell_size=cell_size)
    if built is None:
        return [], np.empty((0, 2, 3), dtype=np.float32)
    tris, mins, maxs, global_min, global_max, cell, grid, valid_indices = built

    m = len(valid_indices)
    visited_stamp = [0] * m
    results = []

    for ci, tri_raw in enumerate(cols):
        if not np.all(np.isfinite(tri_raw)):
            continue
        q = tri_raw.reshape((3, 3)).astype(np.float64)
        c_min = np.min(q, axis=0)
        c_max = np.max(q, axis=0)
        if np.any(c_min > global_max) or np.any(global_min > c_max):
            continue

        clamped_min = np.maximum(c_min, global_min)
        clamped_max = np.minimum(c_max, global_max)
        cmin = np.floor(clamped_min / cell).astype(int)
        cmax = np.floor(clamped_max / cell).astype(int)
        cell_count = int(max(0, cmax[0] - cmin[0] + 1)) * int(max(0, cmax[1] - cmin[1] + 1)) * int(max(0, cmax[2] - cmin[2] + 1))
        stamp = ci + 1

        if cell_count > 4096:
            for k in range(m):
                if np.any(mins[k] > c_max) or np.any(c_min > maxs[k]):
                    continue
                p0, p1, p2 = tris[k]
                seg = tri_tri_intersection_segment(p0, p1, p2, q[0], q[1], q[2])
                if seg is not None:
                    results.append(((valid_indices[k], ci), seg))
        else:
            for cx in range(cmin[0], cmax[0] + 1):
                for cy in range(cmin[1], cmax[1] + 1):
                    for cz in range(cmin[2], cmax[2] + 1):
                        for k in grid.get((cx, cy, cz), ()):
                            if visited_stamp[k] == stamp:
                                continue
                            visited_stamp[k] = stamp
                            if np.any(mins[k] > c_max) or np.any(c_min > maxs[k]):
                                continue
                            p0, p1, p2 = tris[k]
                            seg = tri_tri_intersection_segment(p0, p1, p2, q[0], q[1], q[2])
                            if seg is not None:
                                results.append(((valid_indices[k], ci), seg))

    results.sort(key=lambda item: item[0])
    if not results:
        return [], np.empty((0, 2, 3), dtype=np.float32)
    pairs = [item[0] for item in results]
    segs = np.array([[item[1][0], item[1][1]] for item in results], dtype=np.float32)
    return pairs, segs


@allow_positional
def find_triangle_intersections(
    positions: np.ndarray,
    faces: np.ndarray,
    ignore_adjacent: bool = True,
    cell_size: Optional[float] = None,
    sewing_springs: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """
    メッシュ内の自己交差している三角形ペアを検出します。
    AABB空間ハッシュによるブロードフェーズとSAT判定によるナローフェーズを組み合わせた高速実装です。

    Args:
        positions: shape [N, 3] の頂点座標配列
        faces: shape [M, 3] の三角形インデックス配列
        ignore_adjacent: 頂点またはエッジを共有する隣接三角形同士を判定から除外するか
        cell_size: 空間ハッシュのグリッドセルサイズ（未指定時はAABBと面サイズから自動推定）
        sewing_springs: 縫合スプリング配列 [S, 2]。指定かつ ignore_adjacent=True 時は
            縫合結合された頂点クラスタを共有する面ペアも隣接面として除外します。

    Returns:
        交差している面インデックスのペアリスト [(f0, f1), ...] (f0 < f1)
    Rust核 (`taremin_cloth_core.find_triangle_intersections`) を正本とし、
    利用可能な場合はそちらを優先する。NumPy実装はフォールバック。
    """
    sew_arr = None
    if sewing_springs is not None and len(sewing_springs) > 0:
        sew_arr = np.ascontiguousarray(sewing_springs, dtype=np.uint32)

    if _RUST_GEOMETRY:
        try:
            call_kwargs = {}
            if sew_arr is not None:
                call_kwargs["sewing_springs"] = sew_arr
            got = _core.find_triangle_intersections(
                np.ascontiguousarray(positions, dtype=np.float32),
                np.ascontiguousarray(faces, dtype=np.uint32),
                bool(ignore_adjacent),
                float(cell_size) if cell_size else None,
                **call_kwargs,
            )
            return sorted([(int(a), int(b)) for (a, b) in got])
        except Exception:
            pass

    if sew_arr is not None:
        pairs, _ = find_triangle_intersection_segments(
            positions,
            faces,
            ignore_adjacent=ignore_adjacent,
            cell_size=cell_size,
            sewing_springs=sew_arr,
        )
        return pairs

    if len(faces) == 0:
        return []

    tris = positions[faces]  # [M, 3, 3]
    mins = np.min(tris, axis=1)  # [M, 3]
    maxs = np.max(tris, axis=1)  # [M, 3]

    # 隣接面の検索用マップ
    vert_to_faces: Dict[int, List[int]] = {}
    if ignore_adjacent:
        for f_idx, f in enumerate(faces):
            for v in f:
                vert_to_faces.setdefault(int(v), []).append(f_idx)

    # セルサイズの自動推定（3辺平均と全体スパン下限により巨大グリッドOOMを防止）
    max_span = float(np.max(np.max(maxs, axis=0) - np.min(mins, axis=0))) if len(tris) > 0 else 0.0
    if cell_size is None or cell_size <= 0.0:
        e01 = np.linalg.norm(tris[:, 1, :] - tris[:, 0, :], axis=1)
        e12 = np.linalg.norm(tris[:, 2, :] - tris[:, 1, :], axis=1)
        e20 = np.linalg.norm(tris[:, 0, :] - tris[:, 2, :], axis=1)
        mean_len = float(np.mean((e01 + e12 + e20) / 3.0)) if len(tris) > 0 else 0.05
        cell_size = max(mean_len * 2.0, max_span / 128.0, 1e-4)
    else:
        cell_size = max(float(cell_size), max_span / 128.0, 1e-4)

    # 空間ハッシュグリッドへの登録
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, (bmin, bmax) in enumerate(zip(mins, maxs)):
        cmin = np.floor(bmin / cell_size).astype(int)
        cmax = np.floor(bmax / cell_size).astype(int)
        for cx in range(cmin[0], cmax[0] + 1):
            for cy in range(cmin[1], cmax[1] + 1):
                for cz in range(cmin[2], cmax[2] + 1):
                    grid.setdefault((cx, cy, cz), []).append(idx)

    tested = set()
    intersections = []

    for cell_faces in grid.values():
        n = len(cell_faces)
        if n < 2:
            continue
        for i in range(n):
            f0 = cell_faces[i]
            for j in range(i + 1, n):
                f1 = cell_faces[j]
                pair = (f0, f1) if f0 < f1 else (f1, f0)
                if pair in tested:
                    continue
                tested.add(pair)

                # 隣接判定のチェック
                if ignore_adjacent:
                    v0 = faces[pair[0]]
                    v1 = faces[pair[1]]
                    if v0[0] in v1 or v0[1] in v1 or v0[2] in v1:
                        continue

                # AABB 重複判定
                if np.any(mins[pair[0]] > maxs[pair[1]]) or np.any(mins[pair[1]] > maxs[pair[0]]):
                    continue

                # ナローフェーズ: 三角形同士の交差判定
                p0, p1, p2 = tris[pair[0]]
                q0, q1, q2 = tris[pair[1]]
                if tri_tri_intersection_sat(p0, p1, p2, q0, q1, q2):
                    intersections.append(pair)

    return sorted(intersections)


@allow_positional
def find_displacement_spikes(
    prev_pos: np.ndarray,
    curr_pos: np.ndarray,
    threshold_mm: float = 5.0
) -> List[Dict[str, Any]]:
    """
    2フレーム間の全頂点変位を計算し、指定した閾値（mm単位）を超えた頂点を検出します。

    Returns:
        降順にソートされた変位スパイク情報リスト
        [{"vertex_idx": int, "displacement_mm": float, "position": [x, y, z], "prev_position": [x, y, z]}, ...]
    """
    disps = np.linalg.norm(curr_pos - prev_pos, axis=1) * 1000.0  # mm
    spike_indices = np.where(disps > threshold_mm)[0]
    spikes = []
    for idx in spike_indices:
        spikes.append({
            "vertex_idx": int(idx),
            "displacement_mm": float(disps[idx]),
            "position": curr_pos[idx].tolist(),
            "prev_position": prev_pos[idx].tolist(),
        })

    spikes.sort(key=lambda s: s["displacement_mm"], reverse=True)
    return spikes


@allow_positional
def detect_inverted_faces(
    positions: np.ndarray,
    faces: np.ndarray,
    reference_normals: Optional[np.ndarray] = None
) -> List[int]:
    """
    各面の法線方向を計算し、参照法線（または初期状態）に対して反転（裏返り）している面を検出します。
    reference_normals が指定されない場合は、各面の初期法線との内積が負（dot < 0）である面を検出します。
    """
    v0 = positions[faces[:, 0]]
    v1 = positions[faces[:, 1]]
    v2 = positions[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    normals = normals / norms

    if reference_normals is None:
        return []

    dots = np.sum(normals * reference_normals, axis=1)
    inverted_indices = np.where(dots < -0.1)[0]
    return [int(idx) for idx in inverted_indices]


@allow_positional
def find_proximity_violations(
    positions: np.ndarray,
    faces: np.ndarray,
    thickness: float = 0.005,
    ignore_adjacent: bool = True
) -> List[Tuple[int, int, float]]:
    """
    非隣接面間で、厚み（thickness）未満に異常接近・食い込んでいる頂点-面ペアを検出します。
    戻り値: [(vertex_idx, face_idx, distance), ...]
    Rust核 (`taremin_cloth_core.find_proximity_violations`) を正本とし、
    利用可能な場合はそちらを優先する。NumPy実装はフォールバック。
    """
    if _RUST_GEOMETRY:
        try:
            got = _core.find_proximity_violations(
                np.ascontiguousarray(positions, dtype=np.float32),
                np.ascontiguousarray(faces, dtype=np.uint32),
                float(thickness),
                bool(ignore_adjacent),
            )
            return [(int(v), int(f), float(d)) for (v, f, d) in got]
        except Exception:
            pass
    violations = []
    tris = positions[faces]  # [M, 3, 3]
    face_centers = np.mean(tris, axis=1)

    # 簡易重心近接判定
    for v_idx, p in enumerate(positions):
        dists = np.linalg.norm(face_centers - p, axis=1)
        close_faces = np.where(dists < thickness)[0]
        for f_idx in close_faces:
            if ignore_adjacent and v_idx in faces[f_idx]:
                continue
            violations.append((int(v_idx), int(f_idx), float(dists[f_idx])))

    return violations


@allow_positional
def export_obj(filepath: str, positions: np.ndarray, faces: np.ndarray) -> None:
    """
    頂点配列と面インデックス配列を標準Wavefront OBJ形式として保存します。
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Exported by taremin_cloth analysis tools\n")
        for p in positions:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")
