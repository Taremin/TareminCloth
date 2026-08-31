"""
メッシュ幾何解析・異常値検出ライブラリ
Blender非依存で、NumPy配列ベースの高速幾何計算を提供します。
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def tri_tri_intersection_sat(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
    q0: np.ndarray, q1: np.ndarray, q2: np.ndarray,
    eps: float = 1e-5
) -> bool:
    """
    Separating Axis Theorem (SAT) に基づく2つの3D三角形の交差判定。
    同一平面または接しているだけの場合は False を返します。
    """
    # 1. 三角形Pの平面に対する三角形Qの頂点の符号付き距離
    n1 = np.cross(p1 - p0, p2 - p0)
    norm1 = np.linalg.norm(n1)
    if norm1 < 1e-7:
        return False
    n1 /= norm1
    d_q = [np.dot(n1, q0 - p0), np.dot(n1, q1 - p0), np.dot(n1, q2 - p0)]
    if (d_q[0] > eps and d_q[1] > eps and d_q[2] > eps) or (d_q[0] < -eps and d_q[1] < -eps and d_q[2] < -eps):
        return False

    # 2. 三角形Qの平面に対する三角形Pの頂点の符号付き距離
    n2 = np.cross(q1 - q0, q2 - q0)
    norm2 = np.linalg.norm(n2)
    if norm2 < 1e-7:
        return False
    n2 /= norm2
    d_p = [np.dot(n2, p0 - q0), np.dot(n2, p1 - q0), np.dot(n2, p2 - q0)]
    if (d_p[0] > eps and d_p[1] > eps and d_p[2] > eps) or (d_p[0] < -eps and d_p[1] < -eps and d_p[2] < -eps):
        return False

    # 同一平面判定: 両方の三角形の全頂点が他方の平面上にある場合 (|d| <= eps)
    is_coplanar = all(abs(d) <= eps for d in d_q) and all(abs(d) <= eps for d in d_p)
    if is_coplanar:
        # 同一平面上の場合は、各辺に垂直な平面内法線（計6軸）で分離できるか判定
        edges1 = [p1 - p0, p2 - p1, p0 - p2]
        edges2 = [q1 - q0, q2 - q1, q0 - q2]
        for e in edges1 + edges2:
            axis = np.cross(e, n1)
            norm = np.linalg.norm(axis)
            if norm < 1e-6:
                continue
            axis /= norm
            p_proj = [np.dot(axis, p0), np.dot(axis, p1), np.dot(axis, p2)]
            q_proj = [np.dot(axis, q0), np.dot(axis, q1), np.dot(axis, q2)]
            if min(p_proj) > max(q_proj) + eps or min(q_proj) > max(p_proj) + eps:
                return False
        return True

    # 3. 各エッジの外積軸（9軸）に対する射影重複判定
    edges1 = [p1 - p0, p2 - p1, p0 - p2]
    edges2 = [q1 - q0, q2 - q1, q0 - q2]
    for e1 in edges1:
        for e2 in edges2:
            axis = np.cross(e1, e2)
            norm = np.linalg.norm(axis)
            if norm < 1e-6:
                continue
            axis /= norm
            p_proj = [np.dot(axis, p0), np.dot(axis, p1), np.dot(axis, p2)]
            q_proj = [np.dot(axis, q0), np.dot(axis, q1), np.dot(axis, q2)]
            min_p, max_p = min(p_proj), max(p_proj)
            min_q, max_q = min(q_proj), max(q_proj)
            if min_p > max_q + eps or min_q > max_p + eps:
                return False

    return True


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
