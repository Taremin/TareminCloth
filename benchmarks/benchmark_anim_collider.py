"""
コライダー変形アニメーション同期性能ベンチマーク
シェイプキー／ボーン変形時のメッシュ評価・GPU同期の所要時間およびFPSを計測します。
実行方法:
  blender -b --python benchmarks/benchmark_anim_collider.py
"""

import time
import sys
from pathlib import Path
import bpy
import numpy as np

# アドオンルートを sys.path に追加
root_dir = Path(__file__).parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import taremin_cloth
from taremin_cloth.utils import anim_driver
from taremin_cloth.operators import sync_colliders, get_or_create_simulator


def setup_benchmark_scene(subdivisions=2):
    """ベンチマーク用のシーンを構築する（Icosphereコライダー + 平面クロス）"""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    taremin_cloth.register()

    # 1. コライダー用メッシュ (Icosphere)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdivisions, radius=1.0, location=(0, 0, 0))
    col_obj = bpy.context.active_object
    col_obj.name = "ColliderMesh"
    col_obj.taremin_collider.is_collider = True
    col_obj.taremin_collider.collider_type = 'MESH'

    # シェイプキーを追加して変形アニメーションを構成
    basis = col_obj.shape_key_add(name="Basis")
    key1 = col_obj.shape_key_add(name="Morph")
    # key1 で頂点を外側に膨らませる
    for v in key1.data:
        v.co *= 1.3

    anim = col_obj.taremin_collider.anim
    anim.enabled = True
    anim.target_type = 'SHAPE_KEY'
    anim.shape_key_name = "Morph"
    anim.start_value = 0.0
    anim.end_value = 1.0
    anim.cycle_frames = 60
    anim.play_mode = 'PINGPONG'

    # 2. クロス用メッシュ (Plane)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=20, y_subdivisions=20, size=2.5, location=(0, 0, 1.2))
    cloth_obj = bpy.context.active_object
    cloth_obj.name = "ClothMesh"
    cloth_obj.taremin_cloth.is_cloth = True

    return col_obj, cloth_obj


def run_benchmark(n_frames=120):
    print("=" * 70)
    print("【Taremin Cloth コライダー変形同期ベンチマーク】")
    print(f"Blenderバージョン: {bpy.app.version_string}")
    print("=" * 70)

    # 各解像度（subdivisions 1〜5: 約20〜5120ポリゴン）で計測
    test_configs = [
        ("Low-Poly (Proxy)", 1),
        ("Mid-Poly (Standard)", 2),
        ("High-Poly (Dense)", 3),
        ("Ultra-Dense (5,120 Tris)", 5),
    ]

    for label, sub_div in test_configs:
        col_obj, cloth_obj = setup_benchmark_scene(subdivisions=sub_div)
        sim, coords = get_or_create_simulator(cloth_obj)

        n_verts = len(col_obj.data.vertices)
        n_polys = len(col_obj.data.polygons)

        # ウォームアップ
        depsgraph = bpy.context.evaluated_depsgraph_get()
        sync_colliders(sim, bpy.context.scene, depsgraph=depsgraph, force=True, cloth_obj=cloth_obj)

        sync_times = []
        step_times = []

        total_start = time.perf_counter()

        for f in range(n_frames):
            # 1. アニメーションステップ
            t_anim_start = time.perf_counter()
            anim_driver.step_collider_animation(col_obj, f)
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()

            # 2. コライダー同期時間計測
            t_sync_start = time.perf_counter()
            sync_colliders(sim, bpy.context.scene, depsgraph=depsgraph, force=True, cloth_obj=cloth_obj)
            t_sync_end = time.perf_counter()
            sync_times.append((t_sync_end - t_sync_start) * 1000.0)

            # 3. シミュレーションステップ
            sim.step(dt=1.0 / 60.0, substeps=10, solver_iterations=2)
            t_step_end = time.perf_counter()
            step_times.append((t_step_end - t_sync_end) * 1000.0)

        total_elapsed = time.perf_counter() - total_start
        fps = n_frames / total_elapsed

        avg_sync_ms = np.mean(sync_times)
        max_sync_ms = np.max(sync_times)
        avg_step_ms = np.mean(step_times)

        print(f"\n--- 設定: {label} ---")
        print(f"  コライダー頂点数: {n_verts:,} | ポリゴン数: {n_polys:,}")
        print(f"  コライダー同期時間: 平均 {avg_sync_ms:.2f} ms / 最大 {max_sync_ms:.2f} ms")
        print(f"  シミュレーション時間: 平均 {avg_step_ms:.2f} ms")
        print(f"  総合計測フレーム: {n_frames} F | 総合実行FPS: {fps:.1f} FPS")
        if fps >= 60.0:
            print("  判定: 🟢 60 FPSを維持（リアルタイム・インタラクティブ完全対応）")
        elif fps >= 30.0:
            print("  判定: 🟡 30〜60 FPS（実用可能・スムーズ）")
        else:
            print("  判定: 🔴 30 FPS未満（プロキシメッシュの推奨）")

    print("\n" + "=" * 70)
    print("ベンチマーク完了")
    print("=" * 70)


if __name__ == "__main__":
    taremin_cloth.register()
    try:
        run_benchmark(n_frames=60)
    finally:
        taremin_cloth.unregister()
