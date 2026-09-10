"""
Coupled自己衝突 アプローチ別 性能・品質比較ベンチマークスクリプト
1. 現行方式 (反復外自己衝突)
2. Post-Self-Collision Relaxation (自己衝突後に距離拘束を協調再解決: 1, 2, 4 iters)
3. 反復内実行 (Coupled 参考上限: 各反復で自己衝突を実行)
におけるエッジ伸長率（品質）とフレーム時間・FPS（パフォーマンス）を定量比較する。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, ".")

import numpy as np
import taremin_cloth_core


def create_pressing_cloth_pair(nx=30, ny=30, dx=0.04):
    """上下に対向して互いに強く押し付け合う2枚の布メッシュを生成"""
    n_per = nx * ny
    pos = []
    inv_m = []

    # 1. 下側の布（Z=0.0、外周を固定）
    for y in range(ny):
        for x in range(nx):
            pos.append([x * dx, y * dx, 0.0])
            if y == 0 or y == ny - 1 or x == 0 or x == nx - 1:
                inv_m.append(0.0)
            else:
                inv_m.append(1.0)

    # 2. 上側の布（Z=0.02、厚み thickness=0.02 に対して押し付け状態、外周固定）
    for y in range(ny):
        for x in range(nx):
            pos.append([x * dx, y * dx, 0.02])
            if y == 0 or y == ny - 1 or x == 0 or x == nx - 1:
                inv_m.append(0.0)
            else:
                inv_m.append(1.0)

    edges = []
    faces = []

    for offset in [0, n_per]:
        for y in range(ny):
            for x in range(nx):
                idx = offset + y * nx + x
                if x + 1 < nx:
                    edges.append([idx, idx + 1])
                if y + 1 < ny:
                    edges.append([idx, idx + nx])
                if x + 1 < nx and y + 1 < ny:
                    faces.append([idx, idx + 1, idx + nx + 1])
                    faces.append([idx, idx + nx + 1, idx + nx])

    return (
        np.array(pos, dtype=np.float32),
        np.array(edges, dtype=np.uint32),
        np.array(faces, dtype=np.uint32),
        np.array(inv_m, dtype=np.float32),
    )


def create_falling_wave_mesh(nx=50, ny=50, dx=0.04):
    """初期波打ちを持った標準布メッシュ（自重落下・シワ形成用）"""
    pos = []
    inv_m = []

    for y in range(ny):
        for x in range(nx):
            z = 0.05 * np.sin(x * 0.3) * np.cos(y * 0.3)
            pos.append([x * dx, y * dx, z])
            if y == ny - 1 and (x == 0 or x == nx - 1):
                inv_m.append(0.0)  # 上端2点固定
            else:
                inv_m.append(1.0)

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

    return (
        np.array(pos, dtype=np.float32),
        np.array(edges, dtype=np.uint32),
        np.array(faces, dtype=np.uint32),
        np.array(inv_m, dtype=np.float32),
    )


def run_benchmarks():
    dev_name = taremin_cloth_core.get_gpu_device_name()
    print("=" * 88)
    print(f"【Coupled自己衝突 アプローチ別 性能・品質比較ベンチマーク】")
    print(f"GPUデバイス: {dev_name}")
    print("=" * 88)

    modes = [
        ("① 現行方式 (反復外自己衝突)", 0, 0),
        ("② Post-Relaxation (1 iter)", 1, 1),
        ("③ Post-Relaxation (2 iters)", 1, 2),
        ("④ Post-Relaxation (4 iters)", 1, 4),
        ("⑤ 反復内実行 (Coupled 単体)", 2, 0),
        ("⑥ ハイブリッド (反復内 + Relax 1)", 3, 1),
        ("⑦ ハイブリッド (反復内 + Relax 2)", 3, 2),
    ]

    # -------------------------------------------------------------
    # ベンチマーク 1: 押し付け自己衝突 (エッジ伸長率 & パフォーマンス)
    # -------------------------------------------------------------
    print("\n" + "▼ [ベンチマーク 1] 押し付け自己衝突シーン (2枚対向布: 計1,800頂点, 3,480エッジ)")
    print("  ※自己衝突の強い反発力下で、エッジがどれだけ伸びずに保たれるか（品質）を検証")
    print("-" * 88)
    print(f"{'方式':<36} | {'時間 (ms/f)':<11} | {'FPS':<7} | {'最大伸長率':<12} | {'平均伸長率':<12}")
    print("-" * 88)

    pos_p, edges_p, faces_p, inv_m_p = create_pressing_cloth_pair(30, 30)
    v0_p = pos_p[edges_p[:, 0]]
    v1_p = pos_p[edges_p[:, 1]]
    rest_lens_p = np.linalg.norm(v1_p - v0_p, axis=1)

    substeps = 20
    solver_iters = 2
    n_frames = 60
    n_warmup = 10

    for name, mode, relax_iters in modes:
        sim = taremin_cloth_core.ClothSimulator(
            positions=pos_p.copy(),
            edges=edges_p,
            faces=faces_p,
            inv_masses=inv_m_p,
            thickness=0.02,
            stiffness=1000.0,
            workgroup_size=64,
        )
        sim.set_gravity(0.0, 0.0, 0.0)
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=1.0,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=False,
        )
        sim.set_coupled_self_collision_options(mode, relax_iters)

        # ウォームアップ
        for _ in range(n_warmup):
            sim.step(dt=1.0 / 60.0, substeps=substeps, solver_iterations=solver_iters)

        # 計測
        t0 = time.perf_counter()
        for _ in range(n_frames):
            sim.step(dt=1.0 / 60.0, substeps=substeps, solver_iterations=solver_iters)
        t1 = time.perf_counter()

        total_sec = t1 - t0
        ms_per_f = (total_sec / n_frames) * 1000.0
        fps = n_frames / total_sec

        out_coords = np.empty(len(pos_p) * 3, dtype=np.float32)
        sim.get_positions(out_coords)
        cur_pos = out_coords.reshape((-1, 3))
        cur_v0 = cur_pos[edges_p[:, 0]]
        cur_v1 = cur_pos[edges_p[:, 1]]
        cur_lens = np.linalg.norm(cur_v1 - cur_v0, axis=1)

        strains = (cur_lens - rest_lens_p) / rest_lens_p
        max_strain = np.max(strains) * 100.0
        mean_strain = np.mean(strains[strains > 0]) * 100.0 if np.any(strains > 0) else 0.0

        print(f"{name:<36} | {ms_per_f:6.2f} ms   | {fps:5.1f} | {max_strain:8.3f} %   | {mean_strain:8.3f} %")

    # -------------------------------------------------------------
    # ベンチマーク 2: 標準落下シワ自己衝突 (解像度 2,500頂点)
    # -------------------------------------------------------------
    print("\n" + "▼ [ベンチマーク 2] 標準落下シワシーン (中型メッシュ: 2,500頂点, 4,802面)")
    print("  ※実用的な衣服シミュレーションにおけるフレーム時間とFPSの推移を検証")
    print("-" * 88)
    print(f"{'方式':<36} | {'時間 (ms/f)':<11} | {'FPS':<7} | {'サブステップ':<12} | {'従来比':<8}")
    print("-" * 88)

    pos_w, edges_w, faces_w, inv_m_w = create_falling_wave_mesh(50, 50)
    baseline_time = None

    for name, mode, relax_iters in modes:
        sim = taremin_cloth_core.ClothSimulator(
            positions=pos_w.copy(),
            edges=edges_w,
            faces=faces_w,
            inv_masses=inv_m_w,
            thickness=0.015,
            stiffness=1000.0,
            workgroup_size=64,
        )
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=0.2,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
        )
        sim.set_coupled_self_collision_options(mode, relax_iters)

        for _ in range(n_warmup):
            sim.step(dt=1.0 / 60.0, substeps=substeps, solver_iterations=solver_iters)

        t0 = time.perf_counter()
        for _ in range(n_frames):
            sim.step(dt=1.0 / 60.0, substeps=substeps, solver_iterations=solver_iters)
        t1 = time.perf_counter()

        total_sec = t1 - t0
        ms_per_f = (total_sec / n_frames) * 1000.0
        fps = n_frames / total_sec
        us_per_sub = (total_sec / (n_frames * substeps)) * 1_000_000.0

        if baseline_time is None:
            baseline_time = ms_per_f
            rel_str = "1.00x (基準)"
        else:
            ratio = ms_per_f / baseline_time
            diff_pct = (ratio - 1.0) * 100.0
            rel_str = f"{ratio:.2f}x ({diff_pct:+.1f}%)"

        print(f"{name:<36} | {ms_per_f:6.2f} ms   | {fps:5.1f} | {us_per_sub:6.1f} μs     | {rel_str:<8}")

    print("\n" + "=" * 88)
    print("【ベンチマーク完了】")
    print("=" * 88)


if __name__ == "__main__":
    run_benchmarks()
