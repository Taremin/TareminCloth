"""
Taremin Cloth チューニングオプション比較ベンチマークスクリプト
各パフォーマンス最適化オプション単体および組み合わせでの効果を定量測定・比較する。
"""

import sys
import time
from pathlib import Path
import numpy as np

# アドオンのロード
sys.path.insert(0, '.')
sys.path.insert(0, 'python')
import taremin_cloth
try:
    taremin_cloth.register()
except Exception:
    pass

import bpy
from python.taremin_cloth.operators import (
    cloth_frame_handler,
    clear_simulators,
    get_or_create_simulator,
    sync_colliders,
    sync_cloth_parameters,
    sync_attachment_pins,
)
import taremin_cloth_core


def measure_real_model(options, n_frames=30):
    """model.blendでのベンチマーク測定"""
    blend_path = r"./tmp/model.blend"
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    apron = bpy.data.objects.get("Cloth_Benchmark")
    if not apron:
        print("[Error] Cloth_Benchmarkオブジェクトが見つかりません")
        return None

    settings = apron.taremin_cloth
    scene = bpy.context.scene

    # オプション適用
    settings.solver_mode = options.get("solver_mode", "COLORING")
    settings.workgroup_size = str(options.get("workgroup_size", 64))
    settings.enable_async_readback = options.get("enable_async_readback", False)
    settings.enable_compact_readback = options.get("enable_compact_readback", True)
    settings.enable_frame_buffering = options.get("enable_frame_buffering", False)
    settings.frame_buffer_size = options.get("frame_buffer_size", 2)
    scene.taremin_cloth_fast_playback = options.get("fast_playback", False)

    clear_simulators()
    scene.frame_set(1)

    # 1. タイムライン再生方式の計測
    # ウォームアップ (5フレーム)
    for f in range(2, 7):
        scene.frame_set(f)

    # 計測
    frame_times = []
    for f in range(7, 7 + n_frames):
        t0 = time.perf_counter()
        scene.frame_set(f)
        t1 = time.perf_counter()
        frame_times.append((t1 - t0) * 1000.0)

    avg_handler_time = float(np.mean(frame_times))
    timeline_fps = 1000.0 / avg_handler_time if avg_handler_time > 0 else 0

    # 2. 内訳詳細プロファイル (各工程の分解計測)
    clear_simulators()
    scene.frame_set(1)
    sim, coords = get_or_create_simulator(apron)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    step_times = []
    read_times = []
    total_loop_times = []

    substeps = settings.substeps
    iters = settings.solver_iterations
    dt = 1.0 / scene.render.fps
    is_async = settings.enable_async_readback

    for _ in range(n_frames):
        t0 = time.perf_counter()
        sync_colliders(sim, scene, depsgraph, cloth_obj=apron)
        sync_cloth_parameters(sim, apron, scene)
        sync_attachment_pins(sim, apron, scene)

        t_before_step = time.perf_counter()
        if is_async:
            sim.step_async(dt=dt, substeps=substeps, solver_iterations=iters)
            t_after_step = time.perf_counter()
            sim.fetch_positions(coords)
            t_after_read = time.perf_counter()
        else:
            sim.step(dt=dt, substeps=substeps, solver_iterations=iters)
            t_after_step = time.perf_counter()
            sim.get_positions(coords)
            t_after_read = time.perf_counter()

        apron.data.vertices.foreach_set("co", coords)
        apron.data.update()
        t_end = time.perf_counter()

        step_times.append((t_after_step - t_before_step) * 1000.0)
        read_times.append((t_after_read - t_after_step) * 1000.0)
        total_loop_times.append((t_end - t0) * 1000.0)

    avg_step = float(np.mean(step_times))
    avg_read = float(np.mean(read_times))
    avg_total = float(np.mean(total_loop_times))
    sim_loop_fps = 1000.0 / avg_total if avg_total > 0 else 0

    return {
        "timeline_ms": avg_handler_time,
        "timeline_fps": timeline_fps,
        "step_ms": avg_step,
        "read_ms": avg_read,
        "sim_loop_ms": avg_total,
        "sim_loop_fps": sim_loop_fps,
    }


def main():
    print("=" * 80)
    print("【Taremin Cloth パフォーマンスチューニング オプション比較ベンチマーク】")
    print(f"GPUデバイス: {taremin_cloth_core.get_gpu_device_name()}")
    print(f"Blenderバージョン: {bpy.app.version_string}")
    print("対象モデル: model.blend (Cloth_Benchmark 1,004頂点, 996面, コライダー4個 4,580面)")
    print("=" * 80)

    configs = [
        ("Baseline (現行標準: Coloring, Sync, WG64)", {
            "solver_mode": "COLORING",
            "workgroup_size": 64,
            "enable_async_readback": False,
            "fast_playback": False,
        }),
        ("Opt 1: Async Readback (非同期ダブルバッファ)", {
            "solver_mode": "COLORING",
            "workgroup_size": 64,
            "enable_async_readback": True,
            "fast_playback": False,
        }),
        ("Opt 2: Workgroup Size 32 (Wave32)", {
            "solver_mode": "COLORING",
            "workgroup_size": 32,
            "enable_async_readback": False,
            "fast_playback": False,
        }),
        ("Opt 3: Atomic Jacobi (単一パス拘束ソルバー)", {
            "solver_mode": "ATOMIC",
            "workgroup_size": 64,
            "enable_async_readback": False,
            "fast_playback": False,
        }),
        ("Opt 4: Fast Playback (Depsgraphバイパス)", {
            "solver_mode": "COLORING",
            "workgroup_size": 64,
            "enable_async_readback": False,
            "fast_playback": True,
        }),
        ("Opt 1+4: Async Readback + Fast Playback", {
            "solver_mode": "COLORING",
            "workgroup_size": 64,
            "enable_async_readback": True,
            "fast_playback": True,
        }),
        ("All Optimizations Combined (全最適化有効)", {
            "solver_mode": "ATOMIC",
            "workgroup_size": 32,
            "enable_async_readback": True,
            "fast_playback": True,
        }),
        ("Opt 6: Frame Buffering (Buffer Size: 2)", {
            "solver_mode": "COLORING",
            "workgroup_size": 64,
            "enable_async_readback": False,
            "fast_playback": False,
            "enable_frame_buffering": True,
            "frame_buffer_size": 2,
        }),
        ("Opt 6: Frame Buffering (Buffer Size: 3)", {
            "solver_mode": "COLORING",
            "workgroup_size": 64,
            "enable_async_readback": False,
            "fast_playback": False,
            "enable_frame_buffering": True,
            "frame_buffer_size": 3,
        }),
        ("All Optimizations + Frame Buffering (Size 2)", {
            "solver_mode": "ATOMIC",
            "workgroup_size": 32,
            "enable_async_readback": False,
            "fast_playback": True,
            "enable_frame_buffering": True,
            "frame_buffer_size": 2,
        }),
    ]

    results = []
    for name, opts in configs:
        print(f"\n>> 測定中: {name} ...")
        res = measure_real_model(opts, n_frames=30)
        results.append((name, res))

    # 結果サマリー出力
    print("\n" + "=" * 85)
    print("【ベンチマーク結果一覧サマリー】")
    print("=" * 85)
    header = f"{'設定構成':<42} | {'TL FPS':>7} | {'TL時間':>7} | {'SimLoop':>8} | {'GPU発行':>7} | {'同期待機':>7}"
    print(header)
    print("-" * 85)

    base_tl_fps = results[0][1]["timeline_fps"] if results[0][1] else 1.0

    for name, r in results:
        if not r:
            continue
        tl_fps = r['timeline_fps']
        tl_ms = r['timeline_ms']
        sim_ms = r['sim_loop_ms']
        step_ms = r['step_ms']
        read_ms = r['read_ms']
        speedup = tl_fps / base_tl_fps if base_tl_fps > 0 else 1.0

        print(f"{name:<42} | {tl_fps:6.1f} | {tl_ms:6.1f}ms | {sim_ms:6.2f}ms | {step_ms:5.2f}ms | {read_ms:5.2f}ms (x{speedup:.2f})")

    print("=" * 85)


if __name__ == "__main__":
    main()
