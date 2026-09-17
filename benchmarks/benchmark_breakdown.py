"""
自己衝突の各コンポーネント（V-V, V-T, E-E, Post-Relaxation）の負荷内訳プロファイリング
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, ".")

import numpy as np
import taremin_cloth_core
from benchmarks.benchmark_self_collision_scaling import create_grid_mesh


def profile_breakdown():
    dev_name = taremin_cloth_core.get_gpu_device_name()
    print("=" * 80)
    print(f"[Self-Collision Breakdown Profile (80k Verts / 158.4k Tris)]")
    print(f"GPU: {dev_name}")
    print("=" * 80)

    nx, ny = 200, 200
    pos, edges, faces, inv_m = create_grid_mesh(nx, ny)
    out_coords = np.empty(len(pos) * 3, dtype=np.float32)

    substeps = 10
    n_warmup = 2
    n_frames = 10

    test_cases = [
        ("1. SC OFF", False, 0, 0),
        ("2. SC ON (Mode 0: No Post-Relax)", True, 0, 0),
        ("3. SC ON (Mode 1: Post-Relax 1 iter)", True, 1, 1),
        ("4. SC ON (Mode 1: Post-Relax 2 iters)", True, 1, 2),
    ]

    print(f"{'Configuration':<40} | {'ms/frame':<10} | {'FPS':<8} | {'Delta vs OFF':<12}")
    print("-" * 80)

    ms_baseline = 0.0
    for name, enable_sc, mode, relax_iters in test_cases:
        sim = taremin_cloth_core.ClothSimulator(
            positions=pos,
            edges=edges,
            faces=faces,
            inv_masses=inv_m,
            thickness=0.005,
            stiffness=1000.0,
            workgroup_size=64,
        )
        sim.set_enable_self_collision(enable_sc)
        if enable_sc:
            sim.set_self_collision_options(
                relief_factor=0.2,
                max_displacement_ratio=0.2,
                exclude_neighbors=True,
                enable_normal_untangling=True,
            )
            sim.set_coupled_self_collision_options(mode, relax_iters)

        for _ in range(n_warmup):
            sim.step(dt=1.0 / 60.0, substeps=substeps)
            sim.get_positions(out_coords)

        t0 = time.perf_counter()
        for _ in range(n_frames):
            sim.step(dt=1.0 / 60.0, substeps=substeps)
            sim.get_positions(out_coords)
        t1 = time.perf_counter()

        ms_frame = ((t1 - t0) / n_frames) * 1000.0
        fps = n_frames / (t1 - t0)

        if not enable_sc:
            ms_baseline = ms_frame
            delta_str = "Baseline"
        else:
            delta = ms_frame - ms_baseline
            delta_str = f"{delta:+7.2f} ms"

        print(f"{name:<40} | {ms_frame:7.2f} ms | {fps:6.1f} | {delta_str:<12}")


if __name__ == "__main__":
    profile_breakdown()
