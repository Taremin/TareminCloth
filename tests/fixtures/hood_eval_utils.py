"""
Pattern_Hood 評価用共通ユーティリティ (E2Eテスト・ベンチマーク共用)。

フード層マスク抽出、縫合境界ホップ距離計算、品質メトリクス算出
（交差数・歪み率・速度・左右対称性・背面食い込み）、背面めくり摂動
（動的ピンによる牽引・保持・解放）、および GPU プロファイル集計を一元化する。
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Callable, Sequence

import numpy as np

from taremin_cloth.analysis import find_triangle_intersections
from taremin_cloth.engine.profiling import aggregate_profile, is_parent_scope
from taremin_cloth.utils.mesh_extract import (
    extract_cloth_mesh_data,
    extract_mesh_vertices_and_triangles,
)


def extract_hood_layer_masks(mesh) -> dict[str, np.ndarray]:
    """Blender メッシュの `cloth_layer` POINT 属性からフード層と身頃層のマスク・インデックスを抽出する。"""
    n_verts = len(mesh.vertices)
    if n_verts == 0:
        raise ValueError("メッシュに頂点が存在しません。")

    attr = mesh.attributes.get("cloth_layer") if hasattr(mesh, "attributes") else None
    if attr is None:
        raise ValueError("メッシュ属性 'cloth_layer' が見つかりません。")
    if getattr(attr, "domain", None) != "POINT":
        raise ValueError(f"属性 'cloth_layer' のドメインが POINT ではありません: {getattr(attr, 'domain', None)!r}")
    if len(attr.data) != n_verts:
        raise ValueError(
            f"属性 'cloth_layer' の長さ ({len(attr.data)}) がメッシュ頂点数 ({n_verts}) と一致しません。"
        )

    layer_ids = np.empty(n_verts, dtype=np.int32)
    attr.data.foreach_get("value", layer_ids)

    hood_mask = layer_ids == 1
    bodice_mask = layer_ids == 0
    hood_indices = np.flatnonzero(hood_mask)
    bodice_indices = np.flatnonzero(bodice_mask)

    if len(hood_indices) == 0:
        raise ValueError("cloth_layer == 1（フード層）の頂点が存在しません。")
    if len(bodice_indices) == 0:
        raise ValueError("cloth_layer == 0（身頃層）の頂点が存在しません。")

    positions, _ = extract_mesh_vertices_and_triangles(mesh)
    left_hood_indices = hood_indices[positions[hood_indices, 0] > 1e-4]
    right_hood_indices = hood_indices[positions[hood_indices, 0] < -1e-4]
    if len(left_hood_indices) == 0 or len(right_hood_indices) == 0:
        raise ValueError(
            "フード層の左右パネル頂点が分離抽出できませんでした "
            f"(left={len(left_hood_indices)}, right={len(right_hood_indices)})。"
        )

    return {
        "layer_ids": layer_ids,
        "hood_mask": hood_mask,
        "bodice_mask": bodice_mask,
        "hood_indices": hood_indices,
        "bodice_indices": bodice_indices,
        "left_hood_indices": left_hood_indices,
        "right_hood_indices": right_hood_indices,
    }


def _compute_seam_hop_distances(
    *,
    n_verts: int,
    faces: np.ndarray,
    sewing_edges: np.ndarray | None,
) -> np.ndarray:
    """面辺および縫合辺上のグラフ距離から、各頂点の縫合境界からの最短ホップ数を算出する。"""
    dist = np.full(n_verts, 999, dtype=np.int32)
    if sewing_edges is None or len(sewing_edges) == 0:
        return dist

    sew_arr = np.asarray(sewing_edges, dtype=np.int64)
    valid_sew = (sew_arr[:, 0] < n_verts) & (sew_arr[:, 1] < n_verts)
    sew_arr = sew_arr[valid_sew]
    if len(sew_arr) == 0:
        return dist

    adj: list[set[int]] = [set() for _ in range(n_verts)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)

    queue: deque[int] = deque()
    for u, v in sew_arr:
        iu, iv = int(u), int(v)
        adj[iu].add(iv)
        adj[iv].add(iu)
        if dist[iu] != 0:
            dist[iu] = 0
            queue.append(iu)
        if dist[iv] != 0:
            dist[iv] = 0
            queue.append(iv)

    while queue:
        cur = queue.popleft()
        next_d = int(dist[cur]) + 1
        for nxt in adj[cur]:
            if dist[nxt] > next_d:
                dist[nxt] = next_d
                queue.append(nxt)

    return dist


def extract_hood_eval_geometry(hood_obj) -> dict[str, Any]:
    """Pattern_Hood オブジェクトから引き紐を除いたメッシュ本体幾何情報・層マスク・縫合距離を抽出する。"""
    mesh = hood_obj.data
    n_mesh_verts = len(mesh.vertices)
    cloth_data = extract_cloth_mesh_data(hood_obj, apply_world_matrix=True)

    initial_positions = np.asarray(cloth_data.positions[:n_mesh_verts], dtype=np.float32).copy()
    raw_edges = np.asarray(cloth_data.normal_edges, dtype=np.int64)
    if len(raw_edges) > 0:
        valid_mask = (raw_edges[:, 0] < n_mesh_verts) & (raw_edges[:, 1] < n_mesh_verts)
        edges = raw_edges[valid_mask].astype(np.uint32)
    else:
        edges = np.empty((0, 2), dtype=np.uint32)

    if cloth_data.faces is None or len(cloth_data.faces) == 0:
        raise ValueError(f"オブジェクト '{hood_obj.name}' に三角形面が存在しません。")
    faces = np.asarray(cloth_data.faces, dtype=np.uint32).copy()

    seam_hop_dist = _compute_seam_hop_distances(
        n_verts=n_mesh_verts,
        faces=faces,
        sewing_edges=cloth_data.sewing_edges,
    )

    masks = extract_hood_layer_masks(mesh)
    return {
        "n_mesh_verts": n_mesh_verts,
        "initial_positions": initial_positions,
        "edges": edges,
        "faces": faces,
        "seam_hop_dist": seam_hop_dist,
        **masks,
    }


def slice_mesh_positions(coords: np.ndarray, *, n_mesh_verts: int) -> np.ndarray:
    """step_cloth_scene() が返す座標配列（引き紐連結含む）からメッシュ本体座標 [N, 3] を切り出す。"""
    arr = np.asarray(coords, dtype=np.float32)
    if arr.ndim == 1:
        expected = n_mesh_verts * 3
        if len(arr) < expected:
            raise ValueError(f"座標配列長が不足しています: len={len(arr)}, expected>={expected}")
        return arr[:expected].reshape((n_mesh_verts, 3)).copy()
    if len(arr) < n_mesh_verts:
        raise ValueError(f"座標行数が不足しています: rows={len(arr)}, expected>={n_mesh_verts}")
    return arr[:n_mesh_verts, :3].copy()


def _compute_bidirectional_symmetry_error(
    positions: np.ndarray,
    *,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> tuple[float, float]:
    """左右フード頂点群の X 軸鏡像反転に対する双方向最近傍距離（平均・最大）を算出する。"""
    left_pts = positions[left_indices].astype(np.float64)
    right_mirrored = positions[right_indices].astype(np.float64).copy()
    right_mirrored[:, 0] *= -1.0

    d2 = ((left_pts[:, None, :] - right_mirrored[None, :, :]) ** 2).sum(axis=2)
    d_l_to_r = np.sqrt(np.min(d2, axis=1))
    d_r_to_l = np.sqrt(np.min(d2, axis=0))
    all_dists = np.concatenate([d_l_to_r, d_r_to_l])
    return float(np.mean(all_dists)), float(np.max(all_dists))


def evaluate_hood_frame_metrics(
    *,
    positions: np.ndarray,
    initial_positions: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
    hood_mask: np.ndarray,
    bodice_mask: np.ndarray,
    prev_positions: np.ndarray | None = None,
    dt: float = 1.0 / 60.0,
    left_hood_indices: np.ndarray | None = None,
    right_hood_indices: np.ndarray | None = None,
    seam_hop_dist: np.ndarray | None = None,
    seam_exclude_hops: int = 2,
    check_intersections: bool = True,
) -> dict[str, Any]:
    """単一フレームにおける Pattern_Hood の物理・幾何・自己衝突メトリクスを算出する。"""
    pos = np.asarray(positions, dtype=np.float32)
    init_pos = np.asarray(initial_positions, dtype=np.float32)
    has_non_finite = bool(not np.all(np.isfinite(pos)))

    if has_non_finite:
        return {
            "has_non_finite": True,
            "max_strain": float("nan"),
            "min_strain": float("nan"),
            "p99_strain": float("nan"),
            "max_speed": float("nan") if prev_positions is not None else None,
            "symmetry_x_mean_error": float("nan"),
            "symmetry_x_max_error": float("nan"),
            "total_intersections": None,
            "cross_layer_intersections": None,
            "hood_self_intersections": None,
            "bodice_self_intersections": None,
            "raw_total_intersections": None,
            "raw_cross_layer_intersections": None,
            "raw_hood_self_intersections": None,
            "raw_bodice_self_intersections": None,
            "hood_centroid_y": float("nan"),
            "hood_centroid_z": float("nan"),
            "hood_max_y": float("nan"),
            "hood_min_z": float("nan"),
            "hood_back_contact_count": 0,
            "back_bodice_intrusion_count": 0,
            "back_bodice_max_intrusion_m": float("nan"),
        }

    # 1. エッジ歪み率（現在長 / 初期長）
    if len(edges) > 0:
        e0 = edges[:, 0]
        e1 = edges[:, 1]
        rest_len = np.linalg.norm(init_pos[e1] - init_pos[e0], axis=1)
        cur_len = np.linalg.norm(pos[e1] - pos[e0], axis=1)
        strains = cur_len / np.maximum(rest_len, 1e-6)
        max_strain = float(np.max(strains))
        min_strain = float(np.min(strains))
        p99_strain = float(np.percentile(strains, 99.0))
    else:
        max_strain = 1.0
        min_strain = 1.0
        p99_strain = 1.0

    # 2. 最大速度
    max_speed: float | None = None
    if prev_positions is not None:
        prev = np.asarray(prev_positions, dtype=np.float32)
        if np.all(np.isfinite(prev)):
            speeds = np.linalg.norm(pos - prev, axis=1) / max(float(dt), 1e-8)
            max_speed = float(np.max(speeds))
        else:
            max_speed = float("nan")

    # 3. 左右フード対称誤差
    if (
        left_hood_indices is not None
        and right_hood_indices is not None
        and len(left_hood_indices) > 0
        and len(right_hood_indices) > 0
    ):
        sym_mean, sym_max = _compute_bidirectional_symmetry_error(
            pos,
            left_indices=left_hood_indices,
            right_indices=right_hood_indices,
        )
    else:
        sym_mean, sym_max = 0.0, 0.0

    # 4. 三角形自己交差（生カウントおよび縫合境界帯除外後の本体面カウント）
    total_isect: int | None = None
    cross_isect: int | None = None
    hood_self_isect: int | None = None
    bodice_self_isect: int | None = None
    raw_total_isect: int | None = None
    raw_cross_isect: int | None = None
    raw_hood_self_isect: int | None = None
    raw_bodice_self_isect: int | None = None

    if check_intersections:
        pairs = find_triangle_intersections(pos, faces, ignore_adjacent=True)
        raw_total_isect = len(pairs)
        face_is_hood = np.all(hood_mask[faces], axis=1)
        face_seam_hops = (
            np.min(seam_hop_dist[faces], axis=1) if seam_hop_dist is not None else None
        )

        raw_cross = 0
        raw_hood = 0
        raw_bodice = 0
        body_cross = 0
        body_hood = 0
        body_bodice = 0

        for f0, f1 in pairs:
            h0 = bool(face_is_hood[f0])
            h1 = bool(face_is_hood[f1])
            if h0 != h1:
                raw_cross += 1
            elif h0:
                raw_hood += 1
            else:
                raw_bodice += 1

            # 縫合境界近傍 (< seam_exclude_hops) の縫い合わせ幾何重なりを除外して本体面同士のみ評価
            if face_seam_hops is not None:
                if int(face_seam_hops[f0]) < seam_exclude_hops or int(face_seam_hops[f1]) < seam_exclude_hops:
                    continue
            if h0 != h1:
                body_cross += 1
            elif h0:
                body_hood += 1
            else:
                body_bodice += 1

        raw_cross_isect = raw_cross
        raw_hood_self_isect = raw_hood
        raw_bodice_self_isect = raw_bodice

        cross_isect = body_cross
        hood_self_isect = body_hood
        bodice_self_isect = body_bodice
        total_isect = body_cross + body_hood + body_bodice

    # 5. フード位置＆後身頃との接触・裏側食い込み指標
    hood_pts = pos[hood_mask]
    hood_centroid_y = float(np.mean(hood_pts[:, 1]))
    hood_centroid_z = float(np.mean(hood_pts[:, 2]))
    hood_max_y = float(np.max(hood_pts[:, 1]))
    hood_min_z = float(np.min(hood_pts[:, 2]))

    back_bodice_mask = bodice_mask & (np.abs(init_pos[:, 1] - 0.075) < 0.015)
    if not np.any(back_bodice_mask):
        back_bodice_mask = bodice_mask & (init_pos[:, 1] > 0.02)

    hood_back_contact_count = 0
    back_intrusion_count = 0
    back_max_intrusion = 0.0

    upper_hood_mask = hood_mask & (init_pos[:, 2] > 1.055)
    if np.any(back_bodice_mask) and np.any(upper_hood_mask):
        back_pts = pos[back_bodice_mask]
        upper_pts = pos[upper_hood_mask]
        back_z_max = float(np.max(back_pts[:, 2]))

        # XZ 平面上の最近傍後身頃頂点を特定して 3D 接触数・Y 食い込み量を評価
        dx = upper_pts[:, None, 0] - back_pts[None, :, 0]
        dz = upper_pts[:, None, 2] - back_pts[None, :, 2]
        d2_xz = dx * dx + dz * dz
        nn_idx = np.argmin(d2_xz, axis=1)
        nn_dist_xz = np.sqrt(d2_xz[np.arange(len(upper_pts)), nn_idx])
        nn_back = back_pts[nn_idx]

        dist_3d = np.linalg.norm(upper_pts - nn_back, axis=1)
        hood_back_contact_count = int(np.sum((dist_3d < 0.025) & (upper_pts[:, 1] > 0.02)))

        # 首付け根の背面傾斜での誤検知を避けつつ、後身頃背面から 10mm 超裏側 (-Y) へ食い込んだ頂点を検出
        overlap_mask = (
            (nn_dist_xz < 0.015)
            & (upper_pts[:, 2] < back_z_max - 0.045)
            & (upper_pts[:, 1] > -0.02)
        )
        if np.any(overlap_mask):
            intrusion_depths = nn_back[overlap_mask, 1] - upper_pts[overlap_mask, 1]
            penetrating = intrusion_depths > 0.010
            back_intrusion_count = int(np.sum(penetrating))
            if back_intrusion_count > 0:
                back_max_intrusion = float(np.max(intrusion_depths[penetrating]))

    return {
        "has_non_finite": False,
        "max_strain": max_strain,
        "min_strain": min_strain,
        "p99_strain": p99_strain,
        "max_speed": max_speed,
        "symmetry_x_mean_error": sym_mean,
        "symmetry_x_max_error": sym_max,
        "total_intersections": total_isect,
        "cross_layer_intersections": cross_isect,
        "hood_self_intersections": hood_self_isect,
        "bodice_self_intersections": bodice_self_isect,
        "raw_total_intersections": raw_total_isect,
        "raw_cross_layer_intersections": raw_cross_isect,
        "raw_hood_self_intersections": raw_hood_self_isect,
        "raw_bodice_self_intersections": raw_bodice_self_isect,
        "hood_centroid_y": hood_centroid_y,
        "hood_centroid_z": hood_centroid_z,
        "hood_max_y": hood_max_y,
        "hood_min_z": hood_min_z,
        "hood_back_contact_count": hood_back_contact_count,
        "back_bodice_intrusion_count": back_intrusion_count,
        "back_bodice_max_intrusion_m": back_max_intrusion,
    }


def _get_world_mesh_positions(obj) -> np.ndarray:
    """オブジェクトの現在メッシュ頂点座標をワールド座標系 [N, 3] で取得する。"""
    local_pos, _ = extract_mesh_vertices_and_triangles(obj.data)
    mat_world = getattr(obj, "matrix_world", None)
    if mat_world is not None:
        try:
            mat = np.array(mat_world, dtype=np.float32)
            if mat.ndim == 2 and mat.shape == (4, 4):
                return ((local_pos @ mat[:3, :3].T) + mat[:3, 3]).astype(np.float32)
        except (ValueError, TypeError):
            pass
    return local_pos.astype(np.float32).copy()


def _rotate_around_x_pivot(
    points: np.ndarray,
    *,
    pivot: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """ピボット座標を中心に X 軸周りの剛体回転を適用する。"""
    rel = points - pivot[None, :]
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    out = rel.copy()
    out[:, 1] = rel[:, 1] * c - rel[:, 2] * s
    out[:, 2] = rel[:, 1] * s + rel[:, 2] * c
    return (out + pivot[None, :]).astype(np.float32)


def apply_hood_back_flip_perturbation(
    *,
    sim,
    hood_obj,
    scene,
    step_fn: Callable[[], np.ndarray],
    hood_mask: np.ndarray,
    initial_positions: np.ndarray,
    bodice_mask: np.ndarray | None = None,
    edges: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    left_hood_indices: np.ndarray | None = None,
    right_hood_indices: np.ndarray | None = None,
    seam_hop_dist: np.ndarray | None = None,
    flip_frames: int = 20,
    hold_frames: int = 10,
    settle_frames: int = 30,
    dt: float = 1.0 / 60.0,
) -> list[dict[str, Any]]:
    """フード上端クラウンを背中側 (+Y / -Z) へめくり押し付けた後に解放する摂動シーケンスを実行する。"""
    _ = scene  # step_fn がシーン参照を保持する規約のため引数契約のみ維持
    n_mesh_verts = len(hood_obj.data.vertices)
    hood_indices = np.flatnonzero(hood_mask)
    if len(hood_indices) == 0:
        raise ValueError("hood_mask に有効な頂点がありません。")

    init_z = initial_positions[hood_indices, 2]
    crown_threshold = float(np.max(init_z)) - 0.025
    grab_indices = hood_indices[init_z >= crown_threshold]
    if len(grab_indices) == 0:
        raise ValueError("背面めくり対象のクラウン頂点を選定できませんでした。")
    grab_indices_list = [int(i) for i in grab_indices]

    start_pos = _get_world_mesh_positions(hood_obj)[:n_mesh_verts]

    # ドレープ後の現在座標上でフード下端後側の中点（X=0 対称）をピボットに決定し、実生地長に沿った円弧軌道にする
    hood_init_y = initial_positions[hood_indices, 1]
    back_base_mask = (init_z <= float(np.min(init_z)) + 0.015) & (
        hood_init_y >= float(np.median(hood_init_y))
    )
    back_base_indices = hood_indices[back_base_mask]
    if len(back_base_indices) > 0:
        pivot = np.mean(start_pos[back_base_indices], axis=0).astype(np.float32)
        pivot[0] = 0.0
    else:
        pivot = np.array([0.0, 0.06, 1.02], dtype=np.float32)

    grab_start = start_pos[grab_indices].copy()
    # 右手系 X 軸周りで -135 度回転し、過度な引っ張りなしに後身頃背中側へ折り返す
    total_angle_rad = -math.radians(135.0)
    final_targets = _rotate_around_x_pivot(grab_start, pivot=pivot, angle_rad=total_angle_rad)
    weights = [0.75] * len(grab_indices_list)

    eval_enabled = edges is not None and faces is not None and bodice_mask is not None
    records: list[dict[str, Any]] = []
    prev_pos = start_pos.copy()

    def _run_step(
        *,
        phase: str,
        step_idx: int,
        check_isect: bool,
    ) -> None:
        nonlocal prev_pos
        t0 = time.perf_counter()
        cur_pos = step_fn()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        cur_pos_mesh = slice_mesh_positions(cur_pos, n_mesh_verts=n_mesh_verts)

        metrics = None
        if eval_enabled:
            metrics = evaluate_hood_frame_metrics(
                positions=cur_pos_mesh,
                initial_positions=initial_positions,
                edges=edges,
                faces=faces,
                hood_mask=hood_mask,
                bodice_mask=bodice_mask,
                prev_positions=prev_pos,
                dt=dt,
                left_hood_indices=left_hood_indices,
                right_hood_indices=right_hood_indices,
                seam_hop_dist=seam_hop_dist,
                check_intersections=check_isect,
            )
        prev_pos = cur_pos_mesh
        records.append(
            {
                "frame_index": len(records),
                "phase": phase,
                "phase_step": step_idx,
                "wall_ms": wall_ms,
                "metrics": metrics,
            }
        )

    # Phase 1: フードを背中側へ滑らかにめくり倒す
    for i in range(flip_frames):
        raw_t = (i + 1) / max(1, flip_frames)
        smooth_t = raw_t * raw_t * (3.0 - 2.0 * raw_t)
        angle_i = total_angle_rad * smooth_t
        targets_i = _rotate_around_x_pivot(grab_start, pivot=pivot, angle_rad=angle_i)
        sim.set_pins_batch(grab_indices_list, np.ascontiguousarray(targets_i, dtype=np.float32), weights)
        _run_step(phase="flip", step_idx=i, check_isect=(i == flip_frames - 1))

    # Phase 2: 後身頃側へ倒した状態で保持し、接触を定着させる
    final_targets_c = np.ascontiguousarray(final_targets, dtype=np.float32)
    for i in range(hold_frames):
        sim.set_pins_batch(grab_indices_list, final_targets_c, weights)
        _run_step(phase="hold", step_idx=i, check_isect=(i == hold_frames - 1))

    # Phase 3: 掴みピンを解除し、重力・弾性・自己衝突下で自然安定化させる
    sim.release_pins_batch(grab_indices_list)
    for i in range(settle_frames):
        _run_step(phase="settle", step_idx=i, check_isect=(i == settle_frames - 1))

    return records


def collect_gpu_profile_summary(
    entries_per_frame: Sequence[Sequence[tuple[str, float]]],
    *,
    skip_first_frame: bool = True,
) -> dict[str, Any]:
    """複数フレームの GPU プロファイル列を集計し、親スコープ二重計上を除外した平均値を返す。"""
    sliced = list(entries_per_frame[1:] if skip_first_frame else entries_per_frame)
    valid_frames = [list(entries) for entries in sliced if entries]

    if not valid_frames:
        return {
            "available": False,
            "sample_frames": 0,
            "net_ms_avg": 0.0,
            "flat_ms_avg": 0.0,
            "self_collision_outer_ms_avg": 0.0,
            "passes": {},
        }

    net_totals: list[float] = []
    flat_totals: list[float] = []
    outer_totals: list[float] = []
    pass_frame_sums: dict[str, list[float]] = {}

    for entries in valid_frames:
        totals, _counts, net_total, flat_total = aggregate_profile(entries)
        net_totals.append(net_total)
        flat_totals.append(flat_total)
        outer_ms = sum(v for k, v in totals.items() if is_parent_scope(k))
        outer_totals.append(outer_ms)
        for label, ms in totals.items():
            pass_frame_sums.setdefault(label, []).append(float(ms))

    n_samples = len(valid_frames)
    net_ms_avg = float(np.mean(net_totals))
    flat_ms_avg = float(np.mean(flat_totals))
    outer_ms_avg = float(np.mean(outer_totals))

    passes: dict[str, dict[str, Any]] = {}
    sorted_labels = sorted(
        pass_frame_sums.keys(),
        key=lambda lbl: sum(pass_frame_sums[lbl]) / n_samples,
        reverse=True,
    )
    for label in sorted_labels:
        avg_ms = float(sum(pass_frame_sums[label]) / n_samples)
        share_pct = (avg_ms / net_ms_avg * 100.0) if net_ms_avg > 0.0 else 0.0
        passes[label] = {
            "avg_ms": avg_ms,
            "share_pct": float(share_pct),
            "is_parent": is_parent_scope(label),
        }

    return {
        "available": True,
        "sample_frames": n_samples,
        "net_ms_avg": net_ms_avg,
        "flat_ms_avg": flat_ms_avg,
        "self_collision_outer_ms_avg": outer_ms_avg,
        "passes": passes,
    }
