"""
自己衝突およびUntangling・貫通解消新機能のパフォーマンスベンチマーク
新機能の各オプション (Soft Relief, Exclude Neighbors, Normal Untangling) の ON/OFF による
フレーム計算時間、サブステップ計算時間、FPS、オーバーヘッドを計測する。
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, ".")

import numpy as np
import taremin_cloth_core


def create_grid_mesh(nx=50, ny=50, dx=0.04):
    positions = []
    inv_masses = []

    for y in range(ny):
        for x in range(nx):
            # 初期状態で少し波打たせて自己接触を生じやすくする
            z = 0.05 * np.sin(x * 0.3) * np.cos(y * 0.3)
            positions.append([x * dx, y * dx, z])
            # 上端の2点を固定
            if y == ny - 1 and (x == 0 or x == nx - 1):
                inv_masses.append(0.0)
            else:
                inv_masses.append(1.0)

    positions_np = np.array(positions, dtype=np.float32)
    inv_masses_np = np.array(inv_masses, dtype=np.float32)

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

    edges_np = np.array(edges, dtype=np.uint32)
    faces_np = np.array(faces, dtype=np.uint32)

    return positions_np, edges_np, faces_np, inv_masses_np


def run_benchmark():
    dev_name = taremin_cloth_core.get_gpu_device_name()
    print("=" * 80)
    print(f"【自己衝突 & Untangling 新機能 パフォーマンスベンチマーク】")
    print(f"GPUデバイス: {dev_name}")
    print("=" * 80)

    resolutions = [
        (30, 30, "小型 (900頂点 / 1,682面)"),
        (50, 50, "中型 (2,500頂点 / 4,802面)"),
        (70, 70, "大型 (4,900頂点 / 9,522面)"),
    ]

    n_warmup = 20
    n_frames = 100
    substeps = 20

    configs = [
        ("① Self Collision OFF (ベースライン)", False, 1.0, False, False),
        ("② 従来方式 (Self Collision ON / 全新機能OFF)", True, 1.0, False, False),
        ("③ + Exclude Neighbors のみON", True, 1.0, True, False),
        ("④ + Normal Untangling のみON", True, 1.0, False, True),
        ("⑤ + Soft Relief のみON", True, 0.2, False, False),
        ("⑥ 新機能 ALL ON (推奨標準設定)", True, 0.2, True, True),
    ]

    for nx, ny, label in resolutions:
        print(f"\n▼ メッシュ解像度: {label} (nx={nx}, ny={ny}, 計測={n_frames}フレーム, サブステップ={substeps})")
        print("-" * 80)
        print(f"{'設定':<42} | {'時間 (ms/f)':<11} | {'FPS':<7} | {'サブステップ':<12} | {'従来比':<8}")
        print("-" * 80)

        pos, edges, faces, inv_m = create_grid_mesh(nx, ny)

        baseline_time = None
        legacy_time = None

        for name, enable_sc, relief, excl_n, norm_untangle in configs:
            sim = taremin_cloth_core.ClothSimulator(
                positions=pos,
                edges=edges,
                faces=faces,
                inv_masses=inv_m,
                thickness=0.015,
                stiffness=1000.0,
                workgroup_size=64,
            )
            sim.set_enable_self_collision(enable_sc)
            sim.set_self_collision_options(
                relief_factor=relief,
                max_displacement_ratio=0.2,
                exclude_neighbors=excl_n,
                enable_normal_untangling=norm_untangle,
            )

            # ウォームアップ
            for _ in range(n_warmup):
                sim.step(dt=1.0 / 60.0, substeps=substeps)

            # 計測
            t0 = time.perf_counter()
            for _ in range(n_frames):
                sim.step(dt=1.0 / 60.0, substeps=substeps)
            t1 = time.perf_counter()

            total_sec = t1 - t0
            ms_per_frame = (total_sec / n_frames) * 1000.0
            fps = n_frames / total_sec
            us_per_substep = (total_sec / (n_frames * substeps)) * 1_000_000.0

            if not enable_sc:
                baseline_time = ms_per_frame
                rel_str = "基準(OFF)"
            elif legacy_time is None:
                legacy_time = ms_per_frame
                rel_str = "1.00x (従来)"
            else:
                ratio = ms_per_frame / legacy_time
                diff_pct = (ratio - 1.0) * 100.0
                rel_str = f"{ratio:.2f}x ({diff_pct:+.1f}%)"

            print(f"{name:<42} | {ms_per_frame:6.2f} ms   | {fps:5.1f} | {us_per_substep:6.1f} μs     | {rel_str:<8}")

    print("\n" + "=" * 80)
    print("【ベンチマーク完了】")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
