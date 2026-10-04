# -*- coding: utf-8 -*-
"""
Coupled Collider (反復内同調) vs Decoupled Collider (反復外1回) の比較ベンチマーク
球体コライダー上に落下・接触する布のシミュレーションにおいて、
FPS・GPUプロファイル時間・エッジ歪み率（伸び）・貫通耐性を定量比較する。
"""

import time
import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
python_pkg = os.path.join(root_dir, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

import numpy as np
try:
    import taremin_cloth_core as core
except ImportError:
    from taremin_cloth import taremin_cloth_core as core


def create_cloth_grid(nx=25, ny=25, size=0.8, z=0.35, areal_density=0.15):
    """正方格子布メッシュと物理単位の逆質量を生成"""
    dx = size / (nx - 1)
    dy = size / (ny - 1)
    pos = []
    for y in range(ny):
        for x in range(nx):
            pos.append([x * dx - size * 0.5, y * dy - size * 0.5, z])
    pos = np.array(pos, dtype=np.float32)

    edges = []
    faces = []
    for y in range(ny):
        for x in range(nx):
            idx = y * nx + x
            if x + 1 < nx:
                edges.append([idx, idx + 1])
            if y + 1 < ny:
                edges.append([idx, idx + nx])
            if x + 1 < nx and y + 1 < ny:
                faces.append([idx, idx + 1, idx + nx + 1])
                faces.append([idx, idx + nx + 1, idx + nx])
    edges = np.array(edges, dtype=np.uint32)
    faces = np.array(faces, dtype=np.uint32)

    # 物理質量 (面積 × areal_density)
    masses = np.zeros(len(pos), dtype=np.float64)
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        ab = pos[b] - pos[a]
        ac = pos[c] - pos[a]
        area = 0.5 * np.linalg.norm(np.cross(ab, ac))
        share = area * areal_density / 3.0
        masses[a] += share
        masses[b] += share
        masses[c] += share
    masses = np.maximum(masses, 1e-9)
    inv_masses = (1.0 / masses).astype(np.float32)

    return pos, edges, faces, inv_masses


def create_sphere_collider_triangles(radius=0.25, subdivisions=2):
    """Icosphere状の三角形コライダー配列 [N, 3, 3] を生成"""
    # 正20面体の頂点
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = [
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ]
    verts = [v / np.linalg.norm(v) for v in verts]
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    for _ in range(subdivisions):
        midpoint_cache = {}
        new_faces = []
        for tri in faces:
            def get_mid(i1, i2):
                key = tuple(sorted((i1, i2)))
                if key in midpoint_cache:
                    return midpoint_cache[key]
                m = (verts[i1] + verts[i2]) * 0.5
                m = m / np.linalg.norm(m)
                idx = len(verts)
                verts.append(m)
                midpoint_cache[key] = idx
                return idx
            a = get_mid(tri[0], tri[1])
            b = get_mid(tri[1], tri[2])
            c = get_mid(tri[2], tri[0])
            new_faces.extend([
                [tri[0], a, c],
                [tri[1], b, a],
                [tri[2], c, b],
                [a, b, c],
            ])
        faces = new_faces

    verts = np.array(verts, dtype=np.float32) * radius
    tris = []
    for tri in faces:
        tris.append([verts[tri[0]], verts[tri[1]], verts[tri[2]]])
    return np.array(tris, dtype=np.float32)


def run_benchmark_case(coupled_collider: bool, n_frames=120, substeps=10, solver_iters=2):
    pos, edges, faces, inv_m = create_cloth_grid(nx=25, ny=25, size=0.8, z=0.35)
    col_tris = create_sphere_collider_triangles(radius=0.25, subdivisions=2)

    v0 = pos[edges[:, 0]]
    v1 = pos[edges[:, 1]]
    rest_lens = np.linalg.norm(v1 - v0, axis=1)

    sim = core.ClothSimulator(
        positions=pos.copy(),
        edges=edges,
        faces=faces,
        inv_masses=inv_m,
        thickness=0.01,
        stiffness=1000.0,
        workgroup_size=32,
    )
    sim.set_gravity(0.0, 0.0, -9.81)
    sim.set_enable_self_collision(False)  # 純粋なコライダー性能・剛性比較のため自己衝突OFF
    sim.set_mesh_collider_triangles(col_tris, friction=0.3, thickness=0.01, restitution=0.0, single_sided=False, attributes=None)
    sim.set_coupled_collider(coupled_collider)
    sim.set_solver_iterations(solver_iters)

    # プロファイリング有効化 (対応GPU環境)
    has_profiling = sim.set_profiling_enabled(True)

    # ウォームアップ (初回シェーダービルド等)
    sim.step(dt=1.0 / 60.0, substeps=substeps)

    times = []
    out_coords = np.empty(len(pos) * 3, dtype=np.float32)

    for _ in range(n_frames):
        t0 = time.perf_counter()
        sim.step(dt=1.0 / 60.0, substeps=substeps)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    sim.get_positions(out_coords)
    final_pos = out_coords.reshape((-1, 3))
    cur_v0 = final_pos[edges[:, 0]]
    cur_v1 = final_pos[edges[:, 1]]
    cur_lens = np.linalg.norm(cur_v1 - cur_v0, axis=1)
    strains = (cur_lens - rest_lens) / rest_lens * 100.0

    # 球体中心 (0, 0, 0) からの距離でコライダー侵入（めり込み）を評価
    # 球体半径 0.25m, thickness 0.01m -> 表面距離 0.26m
    dists = np.linalg.norm(final_pos, axis=1)
    # 布が球体上部に接触している頂点（Z > -0.1 かつ 半径近傍）
    contact_mask = final_pos[:, 2] > -0.15
    min_dist_contact = float(np.min(dists[contact_mask])) if np.any(contact_mask) else 0.26

    # プロファイル内訳の回収
    profile_data = sim.take_profile() if has_profiling else []

    return {
        "coupled": coupled_collider,
        "mean_step_ms": float(np.mean(times[5:])),  # 初動数フレーム除外
        "fps": 1000.0 / float(np.mean(times[5:])),
        "max_strain": float(np.max(strains)),
        "mean_strain": float(np.mean(strains)),
        "min_dist": min_dist_contact,
        "penetration_depth": max(0.0, (0.25 + 0.01) - min_dist_contact) * 1000.0,  # mm
        "profile": dict(profile_data),
    }


def main():
    print("=" * 70)
    print("【Coupled vs Decoupled Collider 性能・剛性ベンチマーク】")
    print(f"GPU: {core.get_gpu_device_name()}")
    print("=" * 70)

    # 1. Coupled モード (反復内同調: 既存動作)
    print("\n[1/2] Coupled Collider (反復内同調 / Coupled XPBD) 計測中...")
    res_coupled = run_benchmark_case(coupled_collider=True, n_frames=120)

    # 2. Decoupled モード (反復外1回解決: 高速化)
    print("[2/2] Decoupled Collider (反復外1回解決) 計測中...")
    res_decoupled = run_benchmark_case(coupled_collider=False, n_frames=120)

    print("\n" + "=" * 70)
    print("【ベンチマーク結果サマリー】")
    print("=" * 70)
    print(f"{'項目':<24} | {'Coupled (反復内)':<18} | {'Decoupled (反復外)':<18} | {'差異 / 改善比':<15}")
    print("-" * 75)

    step_c = res_coupled["mean_step_ms"]
    step_d = res_decoupled["mean_step_ms"]
    fps_c = res_coupled["fps"]
    fps_d = res_decoupled["fps"]
    speedup = ((step_c - step_d) / step_c) * 100.0
    print(f"{'平均ステップ所要時間':<20} | {step_c:.2f} ms{'':<11} | {step_d:.2f} ms{'':<11} | {speedup:+.1f}% ({step_d/step_c:.2f}x)")
    print(f"{'換算 FPS':<24} | {fps_c:.1f} FPS{'':<10} | {fps_d:.1f} FPS{'':<10} | {fps_d - fps_c:+.1f} FPS")

    max_s_c = res_coupled["max_strain"]
    max_s_d = res_decoupled["max_strain"]
    mean_s_c = res_coupled["mean_strain"]
    mean_s_d = res_decoupled["mean_strain"]
    print(f"{'最大エッジ伸長率 (Max)':<20} | {max_s_c:.3f}%{'':<12} | {max_s_d:.3f}%{'':<12} | {max_s_d - max_s_c:+.3f}%")
    print(f"{'平均エッジ伸長率 (Mean)':<19} | {mean_s_c:.3f}%{'':<12} | {mean_s_d:.3f}%{'':<12} | {mean_s_d - mean_s_c:+.3f}%")

    pen_c = res_coupled["penetration_depth"]
    pen_d = res_decoupled["penetration_depth"]
    print(f"{'最大めり込み深さ':<22} | {pen_c:.2f} mm{'':<13} | {pen_d:.2f} mm{'':<13} | {pen_d - pen_c:+.2f} mm")

    print("\n【GPUプロファイル内訳 (平均 ms/frame)】")
    all_keys = sorted(set(list(res_coupled["profile"].keys()) + list(res_decoupled["profile"].keys())))
    for k in all_keys:
        vc = res_coupled["profile"].get(k, 0.0)
        vd = res_decoupled["profile"].get(k, 0.0)
        print(f"  - {k:<25}: Coupled={vc:.3f} ms, Decoupled={vd:.3f} ms")


if __name__ == "__main__":
    main()
