"""
接触候補ペアのカバレッジ監査 (GPU収集 vs CPUグラウンドトゥルース)

`ClothSimulator.get_active_pairs()` で読戻した V-T / E-E ペア集合を、
CPU参照実装によるグラウンドトゥルースと比較し、以下に分類する:

- hit: 収集され、現フレームでも有効なペア
- stale: 収集されたが現フレームでは bound 外 (陳腐化・無駄計算)
- missed/horizon: 未収集で、収集時の予測 bound 外だったもの (horizon不足)
- missed/other: 未収集で、収集時の予測 bound 内だったもの (飽和競合等の疑い)
- saturation drops: カウンタ超過による無言破棄数 (カウンタ差分から算出)

トポロジー近傍 (2ホップ) 除外はシェーダー (`is_topologically_near`) と
同一定義で再現する。グラウンドトゥルースは保守的 (除外は広め) に寄せ、
誤検出 (false miss) が出ない設計としている。
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

try:
    import taremin_cloth_core as _core

    _RUST_AUDIT = hasattr(_core, "audit_pairs")
except Exception:
    _core = None
    _RUST_AUDIT = False


def build_two_hop_sets(edges: np.ndarray, num_vertices: int) -> List[Set[int]]:
    """各頂点の2ホップ以内近傍集合を構築する (自身を含む)。"""
    adj: List[Set[int]] = [set() for _ in range(num_vertices)]
    for e in edges:
        a, b = int(e[0]), int(e[1])
        if 0 <= a < num_vertices and 0 <= b < num_vertices and a != b:
            adj[a].add(b)
            adj[b].add(a)
    near: List[Set[int]] = [set() for _ in range(num_vertices)]
    for v in range(num_vertices):
        seen = {v} | adj[v]
        for u in adj[v]:
            seen |= adj[u]
        near[v] = seen
    return near


def per_vertex_mean_rest_length(
    edges: np.ndarray,
    rest_lengths: Optional[np.ndarray],
    num_vertices: int,
    positions: np.ndarray,
) -> np.ndarray:
    """頂点ローカル辺長 (シェーダーの local_edge_lengths 相当)。"""
    lens = np.zeros(num_vertices, dtype=np.float64)
    cnts = np.zeros(num_vertices, dtype=np.int64)
    if rest_lengths is not None and len(rest_lengths) == len(edges):
        vals = np.asarray(rest_lengths, dtype=np.float64)
    else:
        vals = np.linalg.norm(
            positions[np.asarray(edges)[:, 0]] - positions[np.asarray(edges)[:, 1]], axis=1
        )
    for (a, b), L in zip(np.asarray(edges), vals):
        a, b = int(a), int(b)
        if 0 <= a < num_vertices and 0 <= b < num_vertices and L > 0:
            lens[a] += L
            lens[b] += L
            cnts[a] += 1
            cnts[b] += 1
    out = np.divide(lens, np.maximum(cnts, 1),
                    out=np.full(num_vertices, 0.05), where=(cnts > 0))
    return out


def point_triangle_distances(p: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """点 p (3,) と三角形群 (M,3,3) の距離 (M,)。Ericson 5.1.5。"""
    a = tris[:, 0, :]
    b = tris[:, 1, :]
    c = tris[:, 2, :]
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    out = np.empty(len(tris), dtype=np.float64)

    mask_a = (d1 <= 0.0) & (d2 <= 0.0)
    out[mask_a] = np.linalg.norm((p - a[mask_a]), axis=1)

    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    mask_b = (~mask_a) & (d3 >= 0.0) & (d4 <= d3)
    out[mask_b] = np.linalg.norm((p - b[mask_b]), axis=1)

    rest = ~(mask_a | mask_b)
    idx = np.where(rest)[0]
    if len(idx):
        ab_r, ac_r, ap_r = ab[idx], ac[idx], ap[idx]
        d1r, d2r, d3r, d4r = d1[idx], d2[idx], d3[idx], d4[idx]
        vc = d1r * d4r - d3r * d2r
        m_edge_ab = (vc <= 0.0) & (d1r >= 0.0) & (d3r <= 0.0)
        v = np.zeros_like(d1r)
        nz = m_edge_ab & ((d1r - d3r) != 0.0)
        v[nz] = d1r[nz] / (d1r[nz] - d3r[nz])
        closest = np.empty_like(ab_r)
        closest[m_edge_ab] = a[idx][m_edge_ab] + v[m_edge_ab][:, None] * ab_r[m_edge_ab]

        cp = p - c[idx]
        d5 = np.einsum("ij,ij->i", ab_r, cp)
        d6 = np.einsum("ij,ij->i", ac_r, cp)
        m_c = (~m_edge_ab) & (d6 >= 0.0) & (d5 <= d6)
        closest[m_c] = c[idx][m_c]
        vb = d5 * d2r - d1r * d6
        m_edge_ac = (~m_edge_ab) & (~m_c) & (vb <= 0.0) & (d2r >= 0.0) & (d6 <= 0.0)
        w = np.zeros_like(d1r)
        nz2 = m_edge_ac & ((d2r - d6) != 0.0)
        w[nz2] = d2r[nz2] / (d2r[nz2] - d6[nz2])
        closest[m_edge_ac] = a[idx][m_edge_ac] + w[m_edge_ac][:, None] * ac_r[m_edge_ac]

        m_edge_bc = (~m_edge_ab) & (~m_c) & (~m_edge_ac)
        va = d3r * d6 - d5 * d4r
        m_edge_bc = (m_edge_bc & (va <= 0.0)
                     & ((d4r - d3r) >= 0.0) & ((d5 - d6) >= 0.0))
        if np.any(m_edge_bc):
            d4m = d4r[m_edge_bc]
            d3m = d3r[m_edge_bc]
            d5m = d5[m_edge_bc]
            d6m = d6[m_edge_bc]
            denom = (d4m - d3m) + (d5m - d6m)
            wbc = np.zeros_like(d4m)
            nzb = denom != 0.0
            wbc[nzb] = (d4m[nzb] - d3m[nzb]) / denom[nzb]
            bb = b[idx][m_edge_bc]
            cc = c[idx][m_edge_bc]
            closest[m_edge_bc] = bb + wbc[:, None] * (cc - bb)

        done = m_edge_ab | m_c | m_edge_ac | m_edge_bc
        rem = ~done
        if np.any(rem):
            va = d3r * d6 - d5 * d4r
            denom = va + vb + vc
            safe = denom != 0.0
            vv = np.zeros_like(d1r)
            ww = np.zeros_like(d1r)
            vv[rem & safe] = vb[rem & safe] / denom[rem & safe]
            ww[rem & safe] = vc[rem & safe] / denom[rem & safe]
            closest[rem] = (
                a[idx][rem]
                + ab_r[rem] * vv[rem][:, None]
                + ac_r[rem] * ww[rem][:, None]
            )
        out[idx] = np.linalg.norm(p - closest, axis=1)
    return out


def segment_segment_distances(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> np.ndarray:
    """線分 (p1-q1 固定) と線分群 (p2-q2: (M,3)) の距離 (M,)。Ericson 5.1.9。"""
    d1 = q1 - p1  # (3,)
    d2 = q2 - p2  # (M,3)
    r = p1 - p2  # (M,3)
    a = float(np.dot(d1, d1))
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    eps = 1e-12
    s = np.zeros(len(p2))
    t = np.zeros(len(p2))
    if a <= eps:
        t = np.clip(f / np.maximum(e, eps), 0.0, 1.0)
        c1 = np.broadcast_to(p1, q2.shape)
        c2 = p2 + t[:, None] * d2
        return np.linalg.norm(c1 - c2, axis=1)
    c = d1 @ r.T  # (M,)
    valid_e = e > eps
    b = d2 @ d1  # (M,)
    denom = a * e - b * b
    use = valid_e & (np.abs(denom) > eps)
    s[use] = np.clip((b[use] * f[use] - c[use] * e[use]) / denom[use], 0.0, 1.0)
    t[~valid_e] = np.clip(-c[~valid_e] / a, 0.0, 1.0)
    s[~valid_e] = 0.0
    nurse = use | valid_e
    t[nurse] = (b[nurse] * s[nurse] + f[nurse]) / np.maximum(e[nurse], eps)
    lo = nurse & (t < 0.0)
    t[lo] = 0.0
    s[lo] = np.clip(-c[lo] / a, 0.0, 1.0)
    hi = nurse & (t > 1.0)
    t[hi] = 1.0
    s[hi] = np.clip((b[hi] - c[hi]) / a, 0.0, 1.0)
    c1 = p1 + s[:, None] * d1
    c2 = p2 + t[:, None] * d2
    return np.linalg.norm(c1 - c2, axis=1)


def _grid_candidates_points(
    points: np.ndarray, boxes_min: np.ndarray, boxes_max: np.ndarray, cell: float
) -> List[List[int]]:
    """各点についてAABBが範囲内のボックス索引を返す (単純グリッド)。"""
    inv = 1.0 / max(cell, 1e-9)
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for bi, (bmin, bmax) in enumerate(zip(boxes_min, boxes_max)):
        cmin = np.floor(bmin * inv).astype(int)
        cmax = np.floor(bmax * inv).astype(int)
        for cx in range(cmin[0], cmax[0] + 1):
            for cy in range(cmin[1], cmax[1] + 1):
                for cz in range(cmin[2], cmax[2] + 1):
                    grid.setdefault((cx, cy, cz), []).append(bi)
    out: List[List[int]] = []
    for p in points:
        key = tuple(np.floor(p * inv).astype(int).tolist())
        out.append(grid.get(key, []))
    return out


def audit_frame(
    prev_pos: np.ndarray,
    prev_vel: Optional[np.ndarray],
    curr_pos: np.ndarray,
    vt_pairs: np.ndarray,
    ee_pairs: np.ndarray,
    vt_count: int,
    ee_count: int,
    faces: np.ndarray,
    edges: np.ndarray,
    rest_lengths: Optional[np.ndarray],
    thickness: float,
    safety_margin: float,
    horizon_scale: float,
    max_horizon: float,
    margin_mode: int,
    dt_frame: float,
    max_vt_pairs: int,
    max_ee_pairs: int,
) -> Dict[str, Any]:
    """1フレーム分のペアカバレッジ監査を実行する。
    Rust核 (`taremin_cloth_core.audit_pairs`) を正本とし、利用可能な場合は
    そちらを優先する。NumPy実装はフォールバック。
    """
    if _RUST_AUDIT:
        try:
            vt_arr = np.asarray(vt_pairs, dtype=np.uint32).reshape((-1, 4))
            ee_arr = np.asarray(ee_pairs, dtype=np.uint32).reshape((-1, 4))
            vel_arr = (
                None
                if prev_vel is None or len(np.asarray(prev_vel)) == 0
                else np.ascontiguousarray(prev_vel, dtype=np.float64)
            )
            got = _core.audit_pairs(
                np.ascontiguousarray(prev_pos, dtype=np.float64),
                vel_arr,
                np.ascontiguousarray(curr_pos, dtype=np.float64),
                np.ascontiguousarray(vt_arr),
                np.ascontiguousarray(ee_arr),
                int(vt_count),
                int(ee_count),
                np.ascontiguousarray(faces, dtype=np.uint32),
                np.ascontiguousarray(edges, dtype=np.uint32),
                float(thickness),
                float(safety_margin),
                float(horizon_scale),
                float(max_horizon),
                int(margin_mode),
                float(dt_frame),
                int(max_vt_pairs),
                int(max_ee_pairs),
            )
            return {k: (float(v) if isinstance(v, float) else int(v)) for k, v in dict(got).items()}
        except Exception:
            pass
    n = len(curr_pos)
    prev_pos = np.asarray(prev_pos, dtype=np.float64)
    curr_pos = np.asarray(curr_pos, dtype=np.float64)
    if prev_vel is None:
        prev_vel = np.zeros_like(prev_pos)
    else:
        prev_vel = np.asarray(prev_vel, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64)

    eff_thick = float(thickness) * 2.0
    check_static = eff_thick + float(safety_margin)
    mode_f = 0.0 if int(margin_mode) == 0 else 1.0
    speed = np.linalg.norm(prev_vel, axis=1)
    horizon = np.minimum(speed * float(dt_frame) * float(horizon_scale),
                         float(max_horizon)) * mode_f

    near = build_two_hop_sets(edges, n)

    face_of = {tuple(sorted(f.tolist())): fi for fi, f in enumerate(faces)}
    edge_of = {}
    for ei, e in enumerate(edges):
        edge_of[(min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1])))] = ei
    face_tris = curr_pos[faces]  # (F,3,3)

    # ---- V-T グラウンドトゥルース ----
    fmin = face_tris.min(axis=1) - check_static
    fmax = face_tris.max(axis=1) + check_static
    cand_faces = _grid_candidates_points(curr_pos, fmin, fmax, max(check_static, 1e-4))
    gt_vt: Set[Tuple[int, int]] = set()
    for i, cands in enumerate(cand_faces):
        if not cands:
            continue
        cands = list(dict.fromkeys(cands))
        tris = face_tris[cands]
        d = point_triangle_distances(curr_pos[i], tris)
        for fi, dist in zip(cands, d):
            f = faces[fi]
            if i in f:
                continue
            if all(int(v) in near[i] for v in f):
                continue  # 保守的除外
            if dist < check_static:
                gt_vt.add((i, fi))

    cached_vt: Set[Tuple[int, int]] = set()
    for pr in np.asarray(vt_pairs).reshape((-1, 4)):
        i, j, v0, v1 = (int(x) for x in pr)
        fi = face_of.get(tuple(sorted((j, v0, v1))))
        if fi is not None:
            cached_vt.add((i, fi))

    vt_hit = gt_vt & cached_vt
    vt_missed = gt_vt - cached_vt
    vt_miss_horizon = 0
    vt_miss_other = 0
    prev_tris = prev_pos[faces]
    for (i, fi) in vt_missed:
        f = faces[fi]
        d0 = float(point_triangle_distances(prev_pos[i], prev_tris[fi:fi + 1])[0])
        hj = max(float(horizon[int(v)]) for v in f)
        # 収集時の受理 bound (シェーダーの check_thick 相当)
        check_start = eff_thick + float(safety_margin) + float(horizon[i]) + hj
        if d0 > check_start:
            vt_miss_horizon += 1
        else:
            vt_miss_other += 1

    stale_vt = 0
    for (i, fi) in cached_vt:
        if i >= n or fi >= len(faces):
            continue
        f = faces[fi]
        if i in f:
            continue
        d = float(point_triangle_distances(curr_pos[i], face_tris[fi:fi + 1])[0])
        if d > check_static:
            stale_vt += 1

    # ---- E-E グラウンドトゥルース ----
    emid = (curr_pos[edges[:, 0]] + curr_pos[edges[:, 1]]) * 0.5
    emin = np.minimum(curr_pos[edges[:, 0]], curr_pos[edges[:, 1]]) - check_static
    emax = np.maximum(curr_pos[edges[:, 0]], curr_pos[edges[:, 1]]) + check_static
    cand_edges = _grid_candidates_points(emid, emin, emax, max(check_static, 1e-4))
    gt_ee: Set[Tuple[int, int]] = set()
    ep0 = curr_pos[edges[:, 0]]
    ep1 = curr_pos[edges[:, 1]]
    for ei, cands in enumerate(cand_edges):
        cands = [c for c in dict.fromkeys(cands) if c != ei]
        if not cands:
            continue
        a, b = int(edges[ei][0]), int(edges[ei][1])
        d = segment_segment_distances(ep0[ei], ep1[ei], ep0[cands], ep1[cands])
        for ej, dist in zip(cands, d):
            if ej <= ei:
                continue  # 無順序正規化
            c, dd = int(edges[ej][0]), int(edges[ej][1])
            if len({a, b, c, dd}) < 4:
                continue
            # GPU条件 (始点同士が非近傍) の近似
            if c in near[a]:
                continue
            if dist < check_static and dist > 1e-9:
                gt_ee.add((ei, ej))

    cached_ee: Set[Tuple[int, int]] = set()
    for pr in np.asarray(ee_pairs).reshape((-1, 4)):
        vi, vui, vj, vvj = (int(x) for x in pr)
        ei = edge_of.get((min(vi, vui), max(vi, vui)))
        ej = edge_of.get((min(vj, vvj), max(vj, vvj)))
        if ei is not None and ej is not None and ei != ej:
            cached_ee.add((min(ei, ej), max(ei, ej)))

    ee_hit = gt_ee & cached_ee
    ee_missed = gt_ee - cached_ee
    ee_miss_horizon = 0
    ee_miss_other = 0
    prev_ep0 = prev_pos[edges[:, 0]]
    prev_ep1 = prev_pos[edges[:, 1]]
    for (ei, ej) in ee_missed:
        a, b = int(edges[ei][0]), int(edges[ei][1])
        d0 = float(segment_segment_distances(
            prev_ep0[ei], prev_ep1[ei],
            prev_ep0[ej:ej + 1], prev_ep1[ej:ej + 1])[0])
        # 収集時の受理 bound (シェーダーの check_thick 相当)
        check_start = (eff_thick + float(safety_margin)
                       + float(horizon[a]) + float(horizon[int(edges[ej][0])]))
        if d0 > check_start:
            ee_miss_horizon += 1
        else:
            ee_miss_other += 1

    stale_ee = 0
    for (ei, ej) in cached_ee:
        if ei >= len(edges) or ej >= len(edges):
            continue
        d = float(segment_segment_distances(ep0[ei], ep1[ei], ep0[ej:ej + 1], ep1[ej:ej + 1])[0])
        if d > check_static:
            stale_ee += 1

    vt_dropped = max(0, int(vt_count) - len(np.asarray(vt_pairs).reshape((-1, 4))))
    ee_dropped = max(0, int(ee_count) - len(np.asarray(ee_pairs).reshape((-1, 4))))
    # カウンタ自体が論理上限超過なら追加ドロップ
    vt_dropped = max(vt_dropped, max(0, int(vt_count) - int(max_vt_pairs)))
    ee_dropped = max(ee_dropped, max(0, int(ee_count) - int(max_ee_pairs)))

    def ratio(a: int, b: int) -> float:
        return (a / b) if b else 1.0

    return {
        "vt_ground": len(gt_vt),
        "vt_hit": len(vt_hit),
        "vt_coverage": ratio(len(vt_hit), len(gt_vt)),
        "vt_missed": len(vt_missed),
        "vt_miss_horizon": vt_miss_horizon,
        "vt_miss_other": vt_miss_other,
        "vt_stale": stale_vt,
        "vt_cached": len(cached_vt),
        "vt_dropped": vt_dropped,
        "ee_ground": len(gt_ee),
        "ee_hit": len(ee_hit),
        "ee_coverage": ratio(len(ee_hit), len(gt_ee)),
        "ee_missed": len(ee_missed),
        "ee_miss_horizon": ee_miss_horizon,
        "ee_miss_other": ee_miss_other,
        "ee_stale": stale_ee,
        "ee_cached": len(cached_ee),
        "ee_dropped": ee_dropped,
        "check_static": check_static,
    }


def attribute_penetration(n_log: int, n_on: int, n_off: int) -> Tuple[str, str]:
    """ログ・キャッシュON再現・キャッシュOFF再現の交差数から起因を判定する。

    Returns:
        (verdict_key, message)。verdict_key は以下:
        - "cache-attributable": ONでのみ残存 (ペアキャッシュ起因の疑い)
        - "not-cache-attributable": OFFでも残存 (キャッシュ非起因)
        - "no-penetration": いずれにも貫通なし
        - "replay-mismatch": ログに貫通があるが両再現とも解消
          (非決定性・条件差異の疑い。要再実行)
        - "unexpected": OFFの方が悪い (想定外。要調査)
    """
    n_log, n_on, n_off = int(n_log), int(n_on), int(n_off)
    if n_on > 0 and n_off == 0:
        return ("cache-attributable",
                "penetration persists with pair cache but resolves without it")
    if n_on > 0 and n_off > 0:
        return ("not-cache-attributable",
                "penetration persists with and without pair cache")
    if n_on == 0 and n_off == 0:
        if n_log > 0:
            return ("replay-mismatch",
                    "log shows penetration but neither replay reproduces it")
        return ("no-penetration", "no penetration in log or replays")
    return ("unexpected",
            "penetration only in cache-OFF replay (requires investigation)")


def collect_ground_truth_details(
    prev_pos: np.ndarray,
    prev_vel: Optional[np.ndarray],
    curr_pos: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    thickness: float,
    safety_margin: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """グラウンドトゥルースペアの特徴量リストを収集する (方策実験用)。

    各VT要素: {"key": (i, fi), "owner": i, "d_curr": float, "d_start": float,
               "approach": float}。approach > 0 は接近中。
    各EE要素: {"key": (ei, ej), "owner": min頂点, "d_curr", "d_start", "approach"}。
    除外定義は audit_frame と同一 (保守的)。
    """
    n = len(curr_pos)
    prev_pos = np.asarray(prev_pos, dtype=np.float64)
    curr_pos = np.asarray(curr_pos, dtype=np.float64)
    if prev_vel is None:
        prev_vel = np.zeros_like(prev_pos)
    else:
        prev_vel = np.asarray(prev_vel, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64)

    eff_thick = float(thickness) * 2.0
    check_static = eff_thick + float(safety_margin)
    near = build_two_hop_sets(edges, n)
    face_tris = curr_pos[faces]
    prev_tris = prev_pos[faces]

    vt_out: List[Dict[str, Any]] = []
    fmin = face_tris.min(axis=1) - check_static
    fmax = face_tris.max(axis=1) + check_static
    cand_faces = _grid_candidates_points(curr_pos, fmin, fmax, max(check_static, 1e-4))
    for i, cands in enumerate(cand_faces):
        if not cands:
            continue
        cands = list(dict.fromkeys(cands))
        tris = face_tris[cands]
        d_curr = point_triangle_distances(curr_pos[i], tris)
        for fi, dc in zip(cands, d_curr):
            if dc >= check_static:
                continue
            f = faces[fi]
            if i in f or all(int(v) in near[i] for v in f):
                continue
            ds = float(point_triangle_distances(prev_pos[i], prev_tris[fi:fi + 1])[0])
            # 接触法線 (開始時) と相対速度
            closest = _closest_pt_single(prev_pos[i], prev_tris[fi])
            nrm = prev_pos[i] - closest
            nl = float(np.linalg.norm(nrm))
            if nl > 1e-12:
                nrm = nrm / nl
                rel_v = prev_vel[i] - np.mean(prev_vel[f], axis=0)
                approach = float(-np.dot(rel_v, nrm))
            else:
                approach = 0.0
            vt_out.append({"key": (i, int(fi)), "owner": i, "d_curr": float(dc),
                           "d_start": ds, "approach": approach})

    ee_out: List[Dict[str, Any]] = []
    emid = (curr_pos[edges[:, 0]] + curr_pos[edges[:, 1]]) * 0.5
    emin = np.minimum(curr_pos[edges[:, 0]], curr_pos[edges[:, 1]]) - check_static
    emax = np.maximum(curr_pos[edges[:, 0]], curr_pos[edges[:, 1]]) + check_static
    cand_edges = _grid_candidates_points(emid, emin, emax, max(check_static, 1e-4))
    ep0 = curr_pos[edges[:, 0]]
    ep1 = curr_pos[edges[:, 1]]
    pp0 = prev_pos[edges[:, 0]]
    pp1 = prev_pos[edges[:, 1]]
    for ei, cands in enumerate(cand_edges):
        cands = [c for c in dict.fromkeys(cands) if c != ei]
        if not cands:
            continue
        a, b = int(edges[ei][0]), int(edges[ei][1])
        d = segment_segment_distances(ep0[ei], ep1[ei], ep0[cands], ep1[cands])
        for ej, dc in zip(cands, d):
            if ej <= ei or dc >= check_static or dc <= 1e-9:
                continue
            c, dd = int(edges[ej][0]), int(edges[ej][1])
            if len({a, b, c, dd}) < 4 or c in near[a]:
                continue
            ds = float(segment_segment_distances(
                pp0[ei], pp1[ei], pp0[ej:ej + 1], pp1[ej:ej + 1])[0])
            st = _seg_params_single(pp0[ei], pp1[ei], pp0[ej], pp1[ej])
            pt1 = pp0[ei] + st[0] * (pp1[ei] - pp0[ei])
            pt2 = pp0[ej] + st[1] * (pp1[ej] - pp0[ej])
            nrm = pt1 - pt2
            nl = float(np.linalg.norm(nrm))
            if nl > 1e-12:
                nrm = nrm / nl
                rel_v = 0.5 * (prev_vel[a] + prev_vel[b] - prev_vel[c] - prev_vel[dd])
                approach = float(-np.dot(rel_v, nrm))
            else:
                approach = 0.0
            owner = min(a, b, c, dd)
            ee_out.append({"key": (int(ei), int(ej)), "owner": owner,
                           "d_curr": float(dc), "d_start": ds, "approach": approach})
    return vt_out, ee_out


def _closest_pt_single(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """単一三角形への最近傍点 (3,3)。"""
    a, b, c = tri[0], tri[1], tri[2]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy()
    bp = p - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b.copy()
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3) if d1 != d3 else 0.0
        return a + v * ab
    cp = p - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c.copy()
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6) if d2 != d6 else 0.0
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / denom if denom != 0.0 else 0.0
        return b + w * (c - b)
    denom = va + vb + vc
    v = vb / denom if denom != 0.0 else 0.0
    w = vc / denom if denom != 0.0 else 0.0
    return a + ab * v + ac * w


def _seg_params_single(p1: np.ndarray, q1: np.ndarray,
                       p2: np.ndarray, q2: np.ndarray) -> Tuple[float, float]:
    """2線分の最近傍パラメータ (s, t)。"""
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    a, e = float(np.dot(d1, d1)), float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    eps = 1e-12
    if a <= eps:
        return 0.0, float(np.clip(f / max(e, eps), 0.0, 1.0))
    c = float(np.dot(d1, r))
    if e <= eps:
        return float(np.clip(-c / a, 0.0, 1.0)), 0.0
    b = float(np.dot(d1, d2))
    denom = a * e - b * b
    s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) if abs(denom) > eps else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        return float(np.clip(-c / a, 0.0, 1.0)), 0.0
    if t > 1.0:
        return float(np.clip((b - c) / a, 0.0, 1.0)), 1.0
    return s, t


def _score(pair: Dict[str, Any], mode: str, thickness: float) -> float:
    # GPU収集時に見えるのは開始時距離のみ (d_start基準が忠実な上限推定)
    if mode == "distance":
        return -pair["d_start"]
    if mode == "approach":
        return pair["approach"]
    if mode == "combined":
        return pair["approach"] - pair["d_start"] / max(float(thickness), 1e-9)
    raise ValueError(f"unknown score mode: {mode}")


def simulate_quota_policy(
    vt_details: List[Dict[str, Any]],
    ee_details: List[Dict[str, Any]],
    num_vertices: int,
    thickness: float,
    budgets: Tuple[int, ...] = (8192, 32768, 65536),
    per_vertex_k: Tuple[int, ...] = (1, 2, 4, 8),
    deep_factor: float = 1.0,
) -> Dict[str, Any]:
    """保持方策のカバレッジをシミュレーションする (CPU実験用)。

    方策: global/{distance,approach,combined} (全体上位B件)、
          per-vertex/{distance,approach,combined} (頂点毎上位K件)。
    deep coverage は d_curr < eff_thick*deep_factor の接触中ペアに限定した値。
    """
    eff = float(thickness) * 2.0 * float(deep_factor)
    out: Dict[str, Any] = {"vt_total": len(vt_details), "ee_total": len(ee_details)}
    for name, details in (("vt", vt_details), ("ee", ee_details)):
        total = len(details)
        deep_total = sum(1 for p in details if p["d_curr"] < eff)
        out[f"{name}_deep_total"] = deep_total
        for mode in ("distance", "approach", "combined"):
            ranked = sorted(details, key=lambda p: _score(p, mode, thickness), reverse=True)
            for b in budgets:
                kept = set(p["key"] for p in ranked[:b])
                deep_hit = sum(1 for p in ranked[:b] if p["d_curr"] < eff)
                out[f"{name}_global_{mode}_B{b}"] = {
                    "kept": len(kept), "coverage": len(kept) / total if total else 1.0,
                    "deep_coverage": deep_hit / deep_total if deep_total else 1.0}
            by_owner: Dict[int, List[Dict[str, Any]]] = {}
            for p in details:
                by_owner.setdefault(int(p["owner"]), []).append(p)
            for k in per_vertex_k:
                kept = set()
                for plist in by_owner.values():
                    top = sorted(plist, key=lambda p, m=mode: _score(p, m, thickness),
                                 reverse=True)[:k]
                    kept.update(p["key"] for p in top)
                deep_hit = sum(1 for p in details if p["key"] in kept and p["d_curr"] < eff)
                out[f"{name}_pervertex_{mode}_K{k}"] = {
                    "kept": len(kept), "coverage": len(kept) / total if total else 1.0,
                    "deep_coverage": deep_hit / deep_total if deep_total else 1.0}
    return out


def format_policy_report(pol: Dict[str, Any], baseline: Dict[str, float]) -> str:
    """方策シミュレーション結果の比較表。baseline は {"vt": cov, "ee": cov}。"""
    lines = [f"=== Quota Policy Simulation (vt_total={pol['vt_total']}, "
             f"ee_total={pol['ee_total']}) ===",
             f"baseline(current first-come): vt={baseline.get('vt', 0) * 100:.1f}% "
             f"ee={baseline.get('ee', 0) * 100:.1f}%"]
    for name in ("vt", "ee"):
        lines.append(f" -- {name} (deep_total={pol[f'{name}_deep_total']}) --")
        for mode in ("distance", "approach", "combined"):
            for b in (8192, 32768, 65536):
                r = pol[f"{name}_global_{mode}_B{b}"]
                lines.append(f"  global/{mode:8s} B{b:<6d}: kept={r['kept']:<7d} "
                             f"cov={r['coverage'] * 100:5.1f}% deep={r['deep_coverage'] * 100:5.1f}%")
            for k in (1, 2, 4, 8):
                r = pol[f"{name}_pervertex_{mode}_K{k}"]
                lines.append(f"  pervert/{mode:8s} K{k}: kept={r['kept']:<7d} "
                             f"cov={r['coverage'] * 100:5.1f}% deep={r['deep_coverage'] * 100:5.1f}%")
    return "\n".join(lines)


def format_report(rep: Dict[str, Any]) -> str:
    """監査結果の人間可読サマリー。"""
    lines = [
        "=== PairCache Coverage Audit ===",
        f" VT: ground={rep['vt_ground']} hit={rep['vt_hit']} "
        f"coverage={rep['vt_coverage'] * 100:.1f}% "
        f"missed={rep['vt_missed']} (horizon={rep['vt_miss_horizon']}, "
        f"other={rep['vt_miss_other']}) stale={rep['vt_stale']} "
        f"dropped={rep['vt_dropped']}",
        f" EE: ground={rep['ee_ground']} hit={rep['ee_hit']} "
        f"coverage={rep['ee_coverage'] * 100:.1f}% "
        f"missed={rep['ee_missed']} (horizon={rep['ee_miss_horizon']}, "
        f"other={rep['ee_miss_other']}) stale={rep['ee_stale']} "
        f"dropped={rep['ee_dropped']}",
        f" check bound (static): {rep['check_static'] * 1000:.2f} mm",
    ]
    verdicts = []
    if rep["vt_dropped"] + rep["ee_dropped"] > 0:
        verdicts.append("saturation drops detected -> increase pair_max_pairs")
    if rep["vt_miss_horizon"] + rep["ee_miss_horizon"] > 0:
        verdicts.append("horizon misses -> raise horizon_scale/max_horizon or safety_margin")
    if rep["vt_miss_other"] + rep["ee_miss_other"] > 0:
        verdicts.append("unexplained misses -> inspect hash collisions / narrowphase")
    if rep["vt_stale"] + rep["ee_stale"] > max(rep["vt_cached"] + rep["ee_cached"], 1) * 0.5:
        verdicts.append("high stale ratio -> motion too fast for frame-start caching")
    if not verdicts:
        verdicts.append("healthy")
    lines.append(" verdict: " + "; ".join(verdicts))
    return "\n".join(lines)
