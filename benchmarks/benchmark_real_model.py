"""
前回の会話で使用された実モデル (model.blend) および条件と完全に揃えて
シミュレーションFPSを計測するスクリプト
"""

import sys
import time
from pathlib import Path
import numpy as np

# アドオンのロード
sys.path.insert(0, '.')
sys.path.insert(0, 'python')
import taremin_cloth
taremin_cloth.register()

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

def run_real_model_benchmark(n_frames=30):
    blend_path = r"./tmp/model.blend"
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    apron = bpy.data.objects.get("Cloth_Benchmark")
    if not apron:
        print("[Error] Cloth_Benchmarkオブジェクトが見つかりません")
        return

    # コライダーオブジェクトの確認
    colliders = [obj for obj in bpy.data.objects if getattr(obj, "taremin_cloth_collider", None) and obj.taremin_cloth_collider.is_collider]
    col_tri_count = 0
    for col in colliders:
        col_tri_count += len(col.data.polygons)

    settings = apron.taremin_cloth
    n_verts = len(apron.data.vertices)
    n_faces = len(apron.data.polygons)

    print("=" * 70)
    print("【実モデルベンチマーク: model.blend】")
    print(f"GPUデバイス: {taremin_cloth_core.get_gpu_device_name()}")
    print(f"Blenderバージョン: {bpy.app.version_string}")
    print(f"布メッシュ: '{apron.name}' ({n_verts:,} 頂点, {n_faces:,} 面)")
    print(f"布パラメータ: substeps={settings.substeps}, iters={settings.solver_iterations}, tension={settings.tension_stiffness}, bend={settings.bending_stiffness}, air_damp={settings.air_damping}")
    print(f"コライダー数: {len(colliders)} 個 (合計 {col_tri_count:,} 面)")
    for col in colliders:
        print(f"  - {col.name:10}: {len(col.data.vertices):,d} 頂点, {len(col.data.polygons):,d} 面")
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
    sim, coords = get_or_create_simulator(apron)

    depsgraph = bpy.context.evaluated_depsgraph_get()

    sync_col_times = []
    sync_param_times = []
    step_times = []
    read_times = []
    mesh_update_times = []
    total_times = []

    for _ in range(n_frames):
        t0 = time.perf_counter()
        sync_colliders(sim, scene, depsgraph, cloth_obj=apron)
        t1 = time.perf_counter()

        sync_cloth_parameters(sim, apron, scene)
        sync_attachment_pins(sim, apron, scene)
        t2 = time.perf_counter()

        sim.step(
            dt=1.0 / scene.render.fps,
            substeps=settings.substeps,
            solver_iterations=settings.solver_iterations,
        )
        t3 = time.perf_counter()

        sim.get_positions(coords)
        t4 = time.perf_counter()

        apron.data.vertices.foreach_set("co", coords)
        apron.data.update()
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


if __name__ == "__main__":
    run_real_model_benchmark()
