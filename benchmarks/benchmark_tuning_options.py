"""
Taremin Cloth チューニングオプション比較ベンチマークスクリプト
各パフォーマンス最適化オプション単体および組み合わせでの効果を定量測定・比較する。

デフォルトではプロシージャルに生成されたマネキン＆布シーンで実行されます。
外部の .blend ファイルを指定して実行することも可能です：
  blender -b --python benchmarks/benchmark_tuning_options.py -- --blend path/to/model.blend --cloth ClothName
"""

import argparse
import math
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


def create_procedural_benchmark_scene():
    """外部blend不要のプロシージャルなマネキン＆布メッシュシーンを生成"""
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 1. コライダー4個の作成（人体マネキン状: 計約4,000面）
    # 胴体
    bpy.ops.mesh.primitive_cylinder_add(radius=0.25, depth=0.8, vertices=64, location=(0, 0, 0.9))
    torso = bpy.context.active_object
    torso.name = "Collider_Torso"

    # 頭部 (UV球: 64x48 = 約3,000面)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=48, radius=0.2, location=(0, 0, 1.5))
    head = bpy.context.active_object
    head.name = "Collider_Head"

    # 左腕
    bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=0.6, vertices=48, location=(-0.35, 0, 1.0))
    arm_l = bpy.context.active_object
    arm_l.name = "Collider_Arm_L"
    arm_l.rotation_euler = (0, math.radians(25), 0)

    # 右腕
    bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=0.6, vertices=48, location=(0.35, 0, 1.0))
    arm_r = bpy.context.active_object
    arm_r.name = "Collider_Arm_R"
    arm_r.rotation_euler = (0, math.radians(-25), 0)

    colliders = [torso, head, arm_l, arm_r]
    for col in colliders:
        col.taremin_cloth_collider.is_collider = True
        col.taremin_cloth_collider.collider_type = 'MESH'

    # 2. 布オブジェクト作成（32x32 = 1,089頂点、1,024面のケープ状メッシュ）
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=32, y_subdivisions=32, size=0.9, location=(0, 0, 1.55))
    cloth = bpy.context.active_object
    cloth.name = "Cloth_Benchmark"
    cloth.taremin_cloth.is_cloth = True
    cloth.taremin_cloth.substeps = 20
    cloth.taremin_cloth.solver_iterations = 2
    cloth.taremin_cloth.tension_stiffness = 1.0
    cloth.taremin_cloth.bending_stiffness = 0.5
    cloth.taremin_cloth.air_damping = 0.01

    return cloth


def setup_benchmark_scene(blend_path=None, cloth_name=None):
    """ベンチマークシーンを準備する"""
    if blend_path and Path(blend_path).exists():
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))
        cloth_obj = bpy.data.objects.get(cloth_name) if cloth_name else None
        if not cloth_obj:
            cloth_objs = [
                o for o in bpy.data.objects
                if getattr(o, "taremin_cloth", None) and o.taremin_cloth.is_cloth
            ]
            cloth_obj = cloth_objs[0] if cloth_objs else None
        if not cloth_obj:
            raise ValueError(f"指定されたファイル内に布オブジェクトが見つかりませんでした: {blend_path}")
        return cloth_obj, str(blend_path)
    else:
        cloth = create_procedural_benchmark_scene()
        return cloth, "Procedural Mannequin Scene"


def measure_scene_options(options, blend_path=None, cloth_name=None, n_frames=30):
    """指定オプションでのベンチマーク測定"""
    cloth, scene_desc = setup_benchmark_scene(blend_path, cloth_name)

    settings = cloth.taremin_cloth
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
    for f in range(2, 7):
        scene.frame_set(f)

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
    sim, coords = get_or_create_simulator(cloth)
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
        sync_colliders(sim, scene, depsgraph, cloth_obj=cloth)
        sync_cloth_parameters(sim, cloth, scene)
        sync_attachment_pins(sim, cloth, scene)

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

        cloth.data.vertices.foreach_set("co", coords)
        cloth.data.update()
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
    args_list = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Taremin Cloth チューニングオプション比較ベンチマーク")
    parser.add_argument("--blend", type=str, default=None, help=".blendファイルパス（省略時はプロシージャルモデル）")
    parser.add_argument("--cloth", type=str, default=None, help="布オブジェクト名（省略時は自動探索）")
    parser.add_argument("--frames", type=int, default=30, help="計測フレーム数 (デフォルト: 30)")
    args = parser.parse_args(args_list)

    print("=" * 80)
    print("【Taremin Cloth パフォーマンスチューニング オプション比較ベンチマーク】")
    print(f"GPUデバイス: {taremin_cloth_core.get_gpu_device_name()}")
    print(f"Blenderバージョン: {bpy.app.version_string}")
    print(f"対象シーン: {args.blend if args.blend else 'Procedural Mannequin Scene'}")
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
        ("Opt 2: Workgroup Size 128", {
            "solver_mode": "COLORING",
            "workgroup_size": 128,
            "enable_async_readback": False,
            "fast_playback": False,
        }),
        ("Opt 3: Workgroup Size 256", {
            "solver_mode": "COLORING",
            "workgroup_size": 256,
            "enable_async_readback": False,
            "fast_playback": False,
        }),
        ("Opt 4: Fast Playback Mode (UI描画抑制)", {
            "solver_mode": "COLORING",
            "workgroup_size": 64,
            "enable_async_readback": False,
            "fast_playback": True,
        }),
        ("Opt 5: Async + WG128 + Fast Playback (フル最適化)", {
            "solver_mode": "COLORING",
            "workgroup_size": 128,
            "enable_async_readback": True,
            "fast_playback": True,
        }),
    ]

    results = []
    for name, options in configs:
        print(f"\n>>> 測定中: {name} ...", flush=True)
        res = measure_scene_options(options, blend_path=args.blend, cloth_name=args.cloth, n_frames=args.frames)
        if res:
            results.append((name, res))
            print(f"    タイムライン: {res['timeline_fps']:5.1f} FPS ({res['timeline_ms']:5.1f} ms) | "
                  f"シミュレーション実効: {res['sim_loop_fps']:5.1f} FPS ({res['sim_loop_ms']:5.1f} ms) "
                  f"[Step: {res['step_ms']:4.1f}ms, Read: {res['read_ms']:4.1f}ms]")

    print("\n" + "=" * 80)
    print("【総合比較サマリー】")
    print("=" * 80)
    baseline_fps = results[0][1]["timeline_fps"] if results else 1.0

    print(f"{'構成名':<45} | {'タイムラインFPS':<12} | {'高速化率':<8} | {'シミュレーションFPS':<14} | {'Step時間':<8}")
    print("-" * 95)
    for name, r in results:
        speedup = r["timeline_fps"] / baseline_fps if baseline_fps > 0 else 1.0
        print(f"{name:<45} | {r['timeline_fps']:6.1f} FPS     | {speedup:6.2f}x  | {r['sim_loop_fps']:6.1f} FPS        | {r['step_ms']:5.1f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
