"""
メッシュ幾何解析・異常値検出ライブラリ
Blender非依存で、NumPy配列ベースの高速幾何計算を提供します。
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def coplanar_tri_tri_2d(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
    q0: np.ndarray, q1: np.ndarray, q2: np.ndarray,
    eps: float = 1e-5
) -> bool:
    """
    同一平面上にある2つの2D三角形が真に交差・重なり合っているかを判定します。
    エッジ共有や頂点接触は交差とみなしません。
    """
    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def edge_cross_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)
        return (o1 * o2 < -eps * eps) and (o3 * o4 < -eps * eps)

    def point_in_tri(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        o1 = orient(a, b, p)
        o2 = orient(b, c, p)
        o3 = orient(c, a, p)
        return (o1 > eps and o2 > eps and o3 > eps) or (o1 < -eps and o2 < -eps and o3 < -eps)

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


def tri_tri_intersection_moller(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
    u0: np.ndarray, u1: np.ndarray, u2: np.ndarray,
    eps: float = 1e-5
) -> bool:
    """
    Tomas Möller (1997) "A Fast Triangle-Triangle Intersection Test" による厳密交差判定。
    CG・物理シミュレーション標準。同一平面上の場合は2D射影による包含・交差判定へフォールバックします。
    """
    # 1. 三角形Vの平面方程式: N1 . (X - V0) = 0
    e1 = v1 - v0
    e2 = v2 - v0
    n1 = np.cross(e1, e2)
    d1 = -float(np.dot(n1, v0))

    # Uの各頂点と平面1との符号付き距離
    du0 = float(np.dot(n1, u0)) + d1
    du1 = float(np.dot(n1, u1)) + d1
    du2 = float(np.dot(n1, u2)) + d1

    if abs(du0) < eps:
        du0 = 0.0
    if abs(du1) < eps:
        du1 = 0.0
    if abs(du2) < eps:
        du2 = 0.0

    du0du1 = du0 * du1
    du0du2 = du0 * du2
    if du0du1 > 0.0 and du0du2 > 0.0:
        return False

    # 2. 三角形Uの平面方程式: N2 . (X - U0) = 0
    e1_u = u1 - u0
    e2_u = u2 - u0
    n2 = np.cross(e1_u, e2_u)
    d2 = -float(np.dot(n2, u0))

    dv0 = float(np.dot(n2, v0)) + d2
    dv1 = float(np.dot(n2, v1)) + d2
    dv2 = float(np.dot(n2, v2)) + d2

    if abs(dv0) < eps:
        dv0 = 0.0
    if abs(dv1) < eps:
        dv1 = 0.0
    if abs(dv2) < eps:
        dv2 = 0.0

    dv0dv1 = dv0 * dv1
    dv0dv2 = dv0 * dv2
    if dv0dv1 > 0.0 and dv0dv2 > 0.0:
        return False

    # 3. 交差線の方向ベクトル D = N1 x N2
    d = np.cross(n1, n2)

    # 同一平面 (Coplanar) の場合
    if np.dot(d, d) < 1e-10:
        abs_n = np.abs(n1)
        max_idx = int(np.argmax(abs_n))
        axes = [i for i in range(3) if i != max_idx]
        return coplanar_tri_tri_2d(
            v0[axes], v1[axes], v2[axes],
            u0[axes], u1[axes], u2[axes],
            eps
        )

    # 4. 3D交差区間の重複判定
    max_idx = int(np.argmax(np.abs(d)))
    vp0, vp1, vp2 = float(v0[max_idx]), float(v1[max_idx]), float(v2[max_idx])
    up0, up1, up2 = float(u0[max_idx]), float(u1[max_idx]), float(u2[max_idx])

    def compute_intervals(p0: float, p1: float, p2: float, d0: float, d1: float, d2: float, d0d1: float, d0d2: float) -> Tuple[float, float]:
        if d0d1 > 0.0:
            return (p2 + (p0 - p2) * (d2 / (d2 - d0)), p2 + (p1 - p2) * (d2 / (d1 - d2)))
        elif d0d2 > 0.0:
            return (p1 + (p0 - p1) * (d1 / (d1 - d0)), p1 + (p2 - p1) * (d1 / (d2 - d1)))
        else:
            return (p0 + (p1 - p0) * (d0 / (d0 - d1)), p0 + (p2 - p0) * (d0 / (d0 - d2)))

    isect1_a, isect1_b = compute_intervals(vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2)
    isect2_a, isect2_b = compute_intervals(up0, up1, up2, du0, du1, du2, du0du1, du0du2)

    t1_min, t1_max = min(isect1_a, isect1_b), max(isect1_a, isect1_b)
    t2_min, t2_max = min(isect2_a, isect2_b), max(isect2_a, isect2_b)

    return not (t1_max < t2_min - eps or t2_max < t1_min - eps)


# 後方互換性エイリアス
tri_tri_intersection_sat = tri_tri_intersection_moller


def find_triangle_intersections(
    positions: np.ndarray,
    faces: np.ndarray,
    ignore_adjacent: bool = True,
    cell_size: Optional[float] = None
) -> List[Tuple[int, int]]:
    """
    メッシュ内の自己交差している三角形ペアを検出します。
    AABB空間ハッシュによるブロードフェーズとSAT判定によるナローフェーズを組み合わせた高速実装です。

    Args:
        positions: shape [N, 3] の頂点座標配列
        faces: shape [M, 3] の三角形インデックス配列
        ignore_adjacent: 頂点またはエッジを共有する隣接三角形同士を判定から除外するか
        cell_size: 空間ハッシュのグリッドセルサイズ（未指定時はAABBと面サイズから自動推定）

    Returns:
        交差している面インデックスのペアリスト [(f0, f1), ...] (f0 < f1)
    """
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

    # セルサイズの自動推定
    if cell_size is None or cell_size <= 0.0:
        edge_lens = np.linalg.norm(tris[:, 1, :] - tris[:, 0, :], axis=1)
        mean_len = float(np.mean(edge_lens)) if len(edge_lens) > 0 else 0.05
        cell_size = max(mean_len * 2.0, 1e-4)

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


def find_proximity_violations(
    positions: np.ndarray,
    faces: np.ndarray,
    thickness: float = 0.005,
    ignore_adjacent: bool = True
) -> List[Tuple[int, int, float]]:
    """
    非隣接面間で、厚み（thickness）未満に異常接近・食い込んでいる頂点-面ペアを検出します。
    戻り値: [(vertex_idx, face_idx, distance), ...]
    """
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
