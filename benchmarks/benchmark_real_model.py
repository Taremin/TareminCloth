"""
Taremin Cloth 実モデル / 複合シーン ベンチマークスクリプト

実用規模（布: 約1,000頂点、複数コライダー: 約4,000面）のシーンにおいて
シミュレーションFPSおよび各工程の内訳時間を詳細計測するスクリプト。

デフォルトではプロシージャルに生成されたマネキン＆ケープシーンで実行されます。
外部の .blend ファイルを指定して実行することも可能です：
  blender -b --python benchmarks/benchmark_real_model.py -- --blend path/to/model.blend --cloth ClothName
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

    # 2. 布オブジェクト作成（32x32 = 1,089頂点、1,024面のケープ/Cloth_Benchmark状メッシュ）
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


def run_real_model_benchmark(blend_path=None, cloth_name=None, n_frames=30):
    cloth, scene_desc = setup_benchmark_scene(blend_path, cloth_name)

    # コライダーオブジェクトの確認
    colliders = [
        obj for obj in bpy.data.objects
        if getattr(obj, "taremin_cloth_collider", None) and obj.taremin_cloth_collider.is_collider
    ]
    col_tri_count = sum(len(col.data.polygons) for col in colliders)

    settings = cloth.taremin_cloth
    n_verts = len(cloth.data.vertices)
    n_faces = len(cloth.data.polygons)

    print("=" * 70)
    print(f"【実モデル / 複合シーン ベンチマーク】: {scene_desc}")
    print(f"GPUデバイス: {taremin_cloth_core.get_gpu_device_name()}")
    print(f"Blenderバージョン: {bpy.app.version_string}")
    print(f"布メッシュ: '{cloth.name}' ({n_verts:,} 頂点, {n_faces:,} 面)")
    print(
        f"布パラメータ: substeps={settings.substeps}, iters={settings.solver_iterations}, "
        f"tension={settings.tension_stiffness}, bend={settings.bending_stiffness}, "
        f"air_damp={settings.air_damping}"
    )
    print(f"コライダー数: {len(colliders)} 個 (合計 {col_tri_count:,} 面)")
    for col in colliders:
        print(f"  - {col.name:15}: {len(col.data.vertices):,d} 頂点, {len(col.data.polygons):,d} 面")
    print("=" * 70)

    # 1. タイムライン再生方式（cloth_frame_handler）での計測
    print("\n--- 測定 1: タイムライン再生 (frame_change_post ハンドラー) ---")
    clear_simulators()
    scene = bpy.context.scene
    scene.frame_set(1)

    # ウォームアップ (5フレーム)
    for f in range(2, 7):
        scene.frame_set(f)

    # 計測 (n_frames フレーム)
    frame_times = []
    for f in range(7, 7 + n_frames):
        t0 = time.perf_counter()
        scene.frame_set(f)
        t1 = time.perf_counter()
        frame_times.append((t1 - t0) * 1000.0)

    avg_handler_time = np.mean(frame_times)
    handler_fps = 1000.0 / avg_handler_time if avg_handler_time > 0 else 0
    print(f"  - 1フレーム平均処理時間: {avg_handler_time:6.2f} ms")
    print(f"  => タイムライン実効FPS: {handler_fps:6.1f} FPS")

    # 2. 内訳詳細プロファイル（各工程の時間を分解計測）
    print("\n--- 測定 2: 内訳詳細プロファイル (各工程の分解計測) ---")
    clear_simulators()
    scene.frame_set(1)
    sim, coords = get_or_create_simulator(cloth)

    depsgraph = bpy.context.evaluated_depsgraph_get()

    sync_col_times = []
    sync_param_times = []
    step_times = []
    read_times = []
    mesh_update_times = []
    total_times = []

    for _ in range(n_frames):
        t0 = time.perf_counter()
        sync_colliders(sim, scene, depsgraph, cloth_obj=cloth)
        t1 = time.perf_counter()

        sync_cloth_parameters(sim, cloth, scene)
        sync_attachment_pins(sim, cloth, scene)
        t2 = time.perf_counter()

        sim.step(
            dt=1.0 / scene.render.fps,
            substeps=settings.substeps,
            solver_iterations=settings.solver_iterations,
        )
        t3 = time.perf_counter()

        sim.get_positions(coords)
        t4 = time.perf_counter()

        cloth.data.vertices.foreach_set("co", coords)
        cloth.data.update()
        t5 = time.perf_counter()

        sync_col_times.append((t1 - t0) * 1000.0)
        sync_param_times.append((t2 - t1) * 1000.0)
        step_times.append((t3 - t2) * 1000.0)
        read_times.append((t4 - t3) * 1000.0)
        mesh_update_times.append((t5 - t4) * 1000.0)
        total_times.append((t5 - t0) * 1000.0)

    avg_sync_col = np.mean(sync_col_times)
    avg_sync_param = np.mean(sync_param_times)
    avg_step = np.mean(step_times)
    avg_read = np.mean(read_times)
    avg_mesh = np.mean(mesh_update_times)
    avg_total = np.mean(total_times)
    profile_fps = 1000.0 / avg_total if avg_total > 0 else 0

    print(f"内訳詳細:")
    print(f"  - コライダー同期 (sync_colliders)     : {avg_sync_col:6.2f} ms")
    print(f"  - パラメータ同期 (sync_parameters)    : {avg_sync_param:6.2f} ms")
    print(f"  - GPUシミュレーション (sim.step)      : {avg_step:6.2f} ms")
    print(f"  - GPU待機 & 頂点取得 (get_positions)  : {avg_read:6.2f} ms")
    print(f"  - Blenderメッシュ反映 (mesh.update)   : {avg_mesh:6.2f} ms")
    print(f"  -------------------------------------------------------------")
    print(f"  - 1フレーム合計時間                   : {avg_total:6.2f} ms")
    print(f"  => 実効シミュレーションFPS            : {profile_fps:6.1f} FPS")
    print("=" * 70)


def parse_args():
    # Blenderスクリプト実行時、"--" 以降がスクリプト用引数
    args_list = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Taremin Cloth 実モデルベンチマーク")
    parser.add_argument("--blend", type=str, default=None, help=".blendファイルパス（省略時はプロシージャルモデル）")
    parser.add_argument("--cloth", type=str, default=None, help="布オブジェクト名（省略時は自動探索）")
    parser.add_argument("--frames", type=int, default=30, help="計測フレーム数 (デフォルト: 30)")
    return parser.parse_args(args_list)


if __name__ == "__main__":
    args = parse_args()
    run_real_model_benchmark(blend_path=args.blend, cloth_name=args.cloth, n_frames=args.frames)
