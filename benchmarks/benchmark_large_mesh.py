"""
大規模メッシュ (10,000〜40,000頂点) における
自己衝突 & Untangling 新機能 パフォーマンスベンチマーク
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, ".")

import numpy as np
import taremin_cloth_core


def create_grid_mesh(nx=100, ny=100, dx=0.02):
    positions = []
    inv_masses = []

    for y in range(ny):
        for x in range(nx):
            z = 0.05 * np.sin(x * 0.15) * np.cos(y * 0.15)
            positions.append([x * dx, y * dx, z])
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
    print("=" * 85)
    print(f"[Large Scale Benchmark: Self-Collision & Untangling]")
    print(f"GPU Device: {dev_name}")
    print("=" * 85)

    resolutions = [
        (100, 100, "10k Verts (100x100 / 19,602 Tris)"),
        (150, 150, "22.5k Verts (150x150 / 44,402 Tris)"),
        (200, 200, "40k Verts (200x200 / 79,202 Tris)"),
    ]

    n_warmup = 10
    n_frames = 50
    substeps = 20

    configs = [
        ("1. Self Collision OFF (Baseline)", False, 1.0, False, False),
        ("2. Legacy Self Collision (All New Features OFF)", True, 1.0, False, False),
        ("3. + Exclude Neighbors Only", True, 1.0, True, False),
        ("4. + Normal Untangling Only", True, 1.0, False, True),
        ("5. + Soft Relief Only", True, 0.2, False, False),
        ("6. New Features ALL ON (Recommended Standard)", True, 0.2, True, True),
    ]

    for nx, ny, label in resolutions:
        print(f"\n--- Resolution: {label} (nx={nx}, ny={ny}, frames={n_frames}, substeps={substeps}) ---")
        print(f"{'Config':<48} | {'Time (ms/f)':<11} | {'FPS':<7} | {'Substep(us)':<11} | {'vs Legacy':<8}")
        print("-" * 90)

        pos, edges, faces, inv_m = create_grid_mesh(nx, ny)

        legacy_time = None

        for name, enable_sc, relief, excl_n, norm_untangle in configs:
            sim = taremin_cloth_core.ClothSimulator(
                positions=pos,
                edges=edges,
                faces=faces,
                inv_masses=inv_m,
                thickness=0.01,
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

            # Warmup
            for _ in range(n_warmup):
                sim.step(dt=1.0 / 60.0, substeps=substeps)

            # Benchmark
            t0 = time.perf_counter()
            for _ in range(n_frames):
                sim.step(dt=1.0 / 60.0, substeps=substeps)
            t1 = time.perf_counter()

            total_sec = t1 - t0
            ms_per_frame = (total_sec / n_frames) * 1000.0
            fps = n_frames / total_sec
            us_per_substep = (total_sec / (n_frames * substeps)) * 1_000_000.0

            if not enable_sc:
                rel_str = "N/A(OFF)"
            elif legacy_time is None:
                legacy_time = ms_per_frame
                rel_str = "1.00x"
            else:
                ratio = ms_per_frame / legacy_time
                diff_pct = (ratio - 1.0) * 100.0
                rel_str = f"{ratio:.2f}x ({diff_pct:+.1f}%)"

            print(f"{name:<48} | {ms_per_frame:6.2f} ms   | {fps:5.1f} | {us_per_substep:6.1f} us   | {rel_str:<8}")

    print("\n" + "=" * 85)
    print("[Benchmark Completed]")
    print("=" * 85)


if __name__ == "__main__":
    run_benchmark()
