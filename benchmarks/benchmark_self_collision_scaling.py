"""
自己衝突スケーリング性能ベンチマーク (2.5k〜62.5k頂点 / 最大12.5万ポリゴン)
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, ".")

import numpy as np
import taremin_cloth_core


def create_grid_mesh(nx=100, ny=100, dx=0.01):
    positions = []
    inv_masses = []

    # 二重布 (Layer 0 と Layer 1 が間隔 0.008m で近接対向、かつ波打つ)
    n_single = nx * ny
    for layer in range(2):
        z_base = layer * 0.008
        for y in range(ny):
            for x in range(nx):
                # プリーツ状（アコーディオン状）のシワ
                z = z_base + 0.003 * np.sin(x * 0.5) * np.cos(y * 0.5)
                positions.append([x * dx, y * dx, z])
                if y == ny - 1 and (x == 0 or x == nx - 1):
                    inv_masses.append(0.0)
                else:
                    inv_masses.append(1.0)

    positions_np = np.array(positions, dtype=np.float32)
    inv_masses_np = np.array(inv_masses, dtype=np.float32)

    edges = []
    faces = []
    for layer in range(2):
        base_idx = layer * n_single
        for y in range(ny):
            for x in range(nx):
                idx = base_idx + y * nx + x
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
    print(f"[Self-Collision Scaling Benchmark]")
    print(f"GPU: {dev_name}")
    print("=" * 80)

    resolutions = [
        (50, 50, "  5.0k Verts (  9.6k Tris)"),
        (100, 100, " 20.0k Verts ( 39.2k Tris)"),
        (150, 150, " 45.0k Verts ( 88.2k Tris)"),
        (200, 200, " 80.0k Verts (158.4k Tris)"),
        (224, 224, "100.4k Verts (198.9k Tris)"),
    ]

    n_warmup = 3
    n_frames = 15
    substeps = 10

    print(f"{'Mesh Size':<25} | {'SC Mode':<10} | {'ms/frame':<10} | {'FPS':<8} | {'SC Overhead':<12}")
    print("-" * 80)

    for nx, ny, label in resolutions:
        pos, edges, faces, inv_m = create_grid_mesh(nx, ny)

        # 1. SC OFF
        sim_off = taremin_cloth_core.ClothSimulator(
            positions=pos,
            edges=edges,
            faces=faces,
            inv_masses=inv_m,
            thickness=0.005,
            stiffness=1000.0,
            workgroup_size=64,
        )
        sim_off.set_enable_self_collision(False)
        out_coords = np.empty(len(pos) * 3, dtype=np.float32)
        for _ in range(n_warmup):
            sim_off.step(dt=1.0 / 60.0, substeps=substeps)
            sim_off.get_positions(out_coords)
        t0 = time.perf_counter()
        for _ in range(n_frames):
            sim_off.step(dt=1.0 / 60.0, substeps=substeps)
            sim_off.get_positions(out_coords)
        t1 = time.perf_counter()
        ms_off = ((t1 - t0) / n_frames) * 1000.0
        fps_off = n_frames / (t1 - t0)

        print(f"{label:<25} | {'OFF':<10} | {ms_off:7.2f} ms | {fps_off:6.1f} | {'Baseline':<12}")

        # 2. SC ON (Standard: Mode 1 RELAXATION, iters=2)
        sim_on = taremin_cloth_core.ClothSimulator(
            positions=pos,
            edges=edges,
            faces=faces,
            inv_masses=inv_m,
            thickness=0.005,
            stiffness=1000.0,
            workgroup_size=64,
        )
        sim_on.set_enable_self_collision(True)
        sim_on.set_self_collision_options(
            relief_factor=0.2,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
        )
        sim_on.set_coupled_self_collision_options(1, 2)

        for _ in range(n_warmup):
            sim_on.step(dt=1.0 / 60.0, substeps=substeps)
            sim_on.get_positions(out_coords)
        t0 = time.perf_counter()
        for _ in range(n_frames):
            sim_on.step(dt=1.0 / 60.0, substeps=substeps)
            sim_on.get_positions(out_coords)
        t1 = time.perf_counter()
        ms_on = ((t1 - t0) / n_frames) * 1000.0
        fps_on = n_frames / (t1 - t0)

        sc_overhead = ms_on - ms_off
        print(f"{label:<25} | {'ON':<10} | {ms_on:7.2f} ms | {fps_on:6.1f} | {sc_overhead:+7.2f} ms")


if __name__ == "__main__":
    run_benchmark()
