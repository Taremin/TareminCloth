"""
自己衝突の各コンポーネント（V-V, V-T, E-E, Post-Relaxation）の負荷内訳プロファイリング
"""

import argparse
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, ".")

import numpy as np
import taremin_cloth_core
from benchmarks.benchmark_self_collision_scaling import create_grid_mesh


def profile_breakdown(n_frames=20, n_trials=3):
    dev_name = taremin_cloth_core.get_gpu_device_name()
    print("=" * 80)
    print(f"[Self-Collision Breakdown Profile (80k Verts / 158.4k Tris)]")
    print(f"GPU: {dev_name} | frames={n_frames} trials={n_trials} (中央値を報告)")
    print("=" * 80)

    nx, ny = 200, 200
    pos, edges, faces, inv_m = create_grid_mesh(nx, ny)
    out_coords = np.empty(len(pos) * 3, dtype=np.float32)

    substeps = 10
    n_warmup = 3

    # 装置予熱: パイプライン遅延生成とGPUクロックを安定させる (計測外)。
    pre = taremin_cloth_core.ClothSimulator(
        positions=pos, edges=edges, faces=faces, inv_masses=inv_m,
        thickness=0.005, stiffness=1000.0, workgroup_size=64,
    )
    pre.set_enable_self_collision(True)
    pre.set_coupled_self_collision_options(1, 2)
    for _ in range(5):
        pre.step(dt=1.0 / 60.0, substeps=substeps)
        pre.get_positions(out_coords)
    del pre

    test_cases = [
        ("1. SC OFF", False, 0, 0, True, 1),
        ("2. SC ON (Mode 0: No Post-Relax)", True, 0, 0, True, 1),
        ("3. SC ON (Mode 1: Post-Relax 1 iter)", True, 1, 1, True, 1),
        ("4. SC ON (Mode 1: Post-Relax 2 iters)", True, 1, 2, True, 1),
        # Phase 0 拡張: 法線展開処理の寄与と間引きの寄与を分離する。
        # 判定式自体は変えず、既存トグルの組み合わせのみで内訳を確定する。
        ("5. SC ON (Mode 1x2, Untangle OFF)", True, 1, 2, False, 1),
        ("6. SC ON (Mode 1x2, Untangle OFF, int=2)", True, 1, 2, False, 2),
    ]

    print(f"{'Configuration':<40} | {'ms/frame':<10} | {'FPS':<8} | {'Delta vs OFF':<12}")
    print("-" * 80)

    ms_baseline = 0.0
    for name, enable_sc, mode, relax_iters, untangle, interval in test_cases:
        trial_ms = []
        for _ in range(n_trials):
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
                    enable_normal_untangling=untangle,
                )
                sim.set_coupled_self_collision_options(mode, relax_iters)
                sim.set_self_collision_substep_interval(interval)

            for _ in range(n_warmup):
                sim.step(dt=1.0 / 60.0, substeps=substeps)
                sim.get_positions(out_coords)

            t0 = time.perf_counter()
            for _ in range(n_frames):
                sim.step(dt=1.0 / 60.0, substeps=substeps)
                sim.get_positions(out_coords)
            t1 = time.perf_counter()

            trial_ms.append(((t1 - t0) / n_frames) * 1000.0)

        trial_ms.sort()
        # 定常状態の比較には最小値を用いる (中央値は参考として括弧表示)。
        ms_frame = trial_ms[0]
        fps = 1000.0 / ms_frame
        spread = trial_ms[-1] - trial_ms[0]

        if not enable_sc:
            ms_baseline = ms_frame
            delta_str = "Baseline"
        else:
            delta = ms_frame - ms_baseline
            delta_str = f"{delta:+7.2f} ms"

        print(f"{name:<40} | {ms_frame:7.2f} ms | {fps:6.1f} | {delta_str:<12} | spread {spread:5.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    profile_breakdown(n_frames=args.frames, n_trials=args.trials)
