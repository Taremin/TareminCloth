"""
メッシュコライダー (MESH) vs ボーンSDFコライダー (BONE_SDF) 定量ベンチマークスクリプト
同一素体メッシュ・同一アーマチュア・同一アニメーションシーケンスにおける
CPU同期時間、GPUシミュレーション時間、総所要時間、実効FPSを比較計測する。
"""

import time
import bpy
import numpy as np
import sys
from pathlib import Path

root_dir = Path(__file__).parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import taremin_cloth
import taremin_cloth_core
from taremin_cloth.engine.collider import sync_colliders, clear_collider_cache
from taremin_cloth.engine.sdf_baker import clear_all_cached_sdf


def setup_benchmark_scene(collider_type='MESH', sdf_resolution='64'):
    """ベンチマーク用シーン（アーマチュア、細分化素体、布グリッド）を構築する"""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh, do_unlink=True)
    for arm in list(bpy.data.armatures):
        bpy.data.armatures.remove(arm, do_unlink=True)

    # 1. アーマチュア作成 (3ボーンの多関節)
    bpy.ops.object.armature_add()
    arm_obj = bpy.context.active_object
    arm_obj.name = "Bench_Armature"

    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = arm_obj.data.edit_bones
    b1 = edit_bones[0]
    b1.name = "Bone1"
    b1.head = (0.0, 0.0, 0.0)
    b1.tail = (0.0, 0.0, 0.8)

    b2 = edit_bones.new("Bone2")
    b2.parent = b1
    b2.head = (0.0, 0.0, 0.8)
    b2.tail = (0.0, 0.0, 1.6)

    b3 = edit_bones.new("Bone3")
    b3.parent = b2
    b3.head = (0.0, 0.0, 1.6)
    b3.tail = (0.0, 0.0, 2.4)
    bpy.ops.object.mode_set(mode='OBJECT')

    # 2. 素体メッシュ作成 (細分化円柱: 約2000ポリゴン)
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.35, depth=2.4, location=(0.0, 0.0, 1.2))
    body_obj = bpy.context.active_object
    body_obj.name = "Bench_Body"

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=15)
    bpy.ops.object.mode_set(mode='OBJECT')

    # 頂点グループとスキニングウェイト
    vg1 = body_obj.vertex_groups.new(name="Bone1")
    vg2 = body_obj.vertex_groups.new(name="Bone2")
    vg3 = body_obj.vertex_groups.new(name="Bone3")

    for v in body_obj.data.vertices:
        z = v.co.z + 1.2
        if z <= 0.8:
            vg1.add([v.index], 1.0, 'REPLACE')
        elif z <= 1.6:
            vg2.add([v.index], 1.0, 'REPLACE')
        else:
            vg3.add([v.index], 1.0, 'REPLACE')

    mod = body_obj.modifiers.new(name="Armature", type='ARMATURE')
    mod.object = arm_obj

    # コライダー設定
    body_obj.taremin_cloth_collider.is_collider = True
    body_obj.taremin_cloth_collider.enabled = True
    body_obj.taremin_cloth_collider.collider_type = collider_type
    body_obj.taremin_cloth_collider.thickness = 0.015
    if collider_type == 'BONE_SDF':
        body_obj.taremin_cloth_collider.sdf_resolution = sdf_resolution
        body_obj.taremin_cloth_collider.sdf_cache_enabled = True

    # 3. 布メッシュ作成 (25x25 = 625頂点, 1152三角形)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=24, y_subdivisions=24, size=0.9, location=(0.0, 0.0, 2.5))
    cloth_obj = bpy.context.active_object
    cloth_obj.name = "Bench_Cloth"

    mesh = cloth_obj.data
    n_verts = len(mesh.vertices)
    coords = np.empty(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    pos = coords.reshape((n_verts, 3)) + np.array([0.0, 0.0, 2.5], dtype=np.float32)

    n_edges = len(mesh.edges)
    edge_indices = np.empty(n_edges * 2, dtype=np.uint32)
    mesh.edges.foreach_get("vertices", edge_indices)
    edges = edge_indices.reshape((n_edges, 2))

    tri_list = []
    mesh.calc_loop_triangles()
    for tri in mesh.loop_triangles:
        tri_list.append(tri.vertices)
    faces = np.array(tri_list, dtype=np.uint32)

    inv_masses = np.ones(n_verts, dtype=np.float32)

    sim = taremin_cloth_core.ClothSimulator(
        positions=pos,
        edges=edges,
        faces=faces,
        inv_masses=inv_masses,
        layer_id=0,
        thickness=0.01,
        stiffness=10000.0,
        bending_stiffness=50.0,
    )
    sim.set_gravity(0.0, 0.0, -9.8)

    num_tris = len(body_obj.data.polygons) * 2
    return arm_obj, body_obj, sim, n_verts, num_tris


def run_benchmark(collider_type='MESH', sdf_resolution='64', num_frames=60, substeps=10):
    """指定されたコライダー設定でベンチマークを実行しメトリクスを返す"""
    clear_all_cached_sdf()
    clear_collider_cache()

    arm_obj, body_obj, sim, n_verts, num_tris = setup_benchmark_scene(collider_type, sdf_resolution)

    cpu_sync_times = []
    gpu_sim_times = []

    # 初回同期 (ベイクやバッファ生成含む)
    t0_init = time.perf_counter()
    sync_colliders(sim, bpy.context.scene, force=True)
    init_time = (time.perf_counter() - t0_init) * 1000.0

    out_coords = np.empty(n_verts * 3, dtype=np.float32)

    # 60フレームのアニメーションループ
    for frame in range(num_frames):
        # ボーンの曲げアニメーション (Bone2, Bone3 を回転)
        angle = np.sin(frame * 0.1) * 0.3
        p_b2 = arm_obj.pose.bones["Bone2"]
        p_b2.rotation_mode = 'XYZ'
        p_b2.rotation_euler = (angle, 0.0, 0.0)

        p_b3 = arm_obj.pose.bones["Bone3"]
        p_b3.rotation_mode = 'XYZ'
        p_b3.rotation_euler = (angle * 0.5, 0.0, 0.0)

        bpy.context.view_layer.update()

        # 1. CPU コライダー同期
        t0_sync = time.perf_counter()
        sync_colliders(sim, bpy.context.scene)
        t_sync = (time.perf_counter() - t0_sync) * 1000.0
        cpu_sync_times.append(t_sync)

        # 2. GPU 物理シミュレーション
        t0_step = time.perf_counter()
        sim.step(dt=1.0 / 60.0, substeps=substeps)
        sim.get_positions(out_coords)
        t_step = (time.perf_counter() - t0_step) * 1000.0
        gpu_sim_times.append(t_step)

    avg_sync = np.mean(cpu_sync_times)
    avg_step = np.mean(gpu_sim_times)
    total_frame_time = avg_sync + avg_step
    effective_fps = 1000.0 / max(total_frame_time, 1e-4)

    return {
        "collider_type": collider_type,
        "sdf_resolution": sdf_resolution if collider_type == 'BONE_SDF' else "-",
        "num_body_tris": num_tris,
        "num_cloth_verts": n_verts,
        "init_time_ms": init_time,
        "avg_cpu_sync_ms": avg_sync,
        "avg_gpu_sim_ms": avg_step,
        "total_frame_ms": total_frame_time,
        "fps": effective_fps,
    }


def main():
    print("=" * 80)
    print(" taremin_cloth コライダー性能ベンチマーク (MESH vs BONE_SDF)")
    print("=" * 80)

    taremin_cloth.register()

    # 1. メッシュコライダー (MESH)
    print("\n[1/3] メッシュコライダー (MESH) 計測中 (60 frames)...")
    res_mesh = run_benchmark('MESH')

    # 2. ボーンSDF (32^3 軽量)
    print("[2/3] ボーンSDF (32^3 Low) 計測中 (60 frames)...")
    res_sdf32 = run_benchmark('BONE_SDF', sdf_resolution='32')

    # 3. ボーンSDF (64^3 標準)
    print("[3/3] ボーンSDF (64^3 Standard) 計測中 (60 frames)...")
    res_sdf64 = run_benchmark('BONE_SDF', sdf_resolution='64')

    taremin_cloth.unregister()

    print("\n" + "=" * 80)
    print(" ベンチマーク結果サマリー")
    print("=" * 80)
    header = f"{'コライダー方式':<20} | {'解像度':<8} | {'CPU同期(ms)':<12} | {'GPUシミュ(ms)':<13} | {'総時間(ms)':<11} | {'実効FPS':<9}"
    print(header)
    print("-" * len(header))

    for r in [res_mesh, res_sdf32, res_sdf64]:
        name = "MESH (従来メッシュ)" if r["collider_type"] == 'MESH' else f"BONE_SDF ({r['sdf_resolution']}^3)"
        line = f"{name:<20} | {r['sdf_resolution']:<8} | {r['avg_cpu_sync_ms']:>10.3f} ms | {r['avg_gpu_sim_ms']:>11.3f} ms | {r['total_frame_ms']:>9.3f} ms | {r['fps']:>7.1f}"
        print(line)

    print("=" * 80)
    print(f"初期化/ベイク時間:")
    print(f"  - MESH:               {res_mesh['init_time_ms']:.2f} ms")
    print(f"  - BONE_SDF (32^3):     {res_sdf32['init_time_ms']:.2f} ms")
    print(f"  - BONE_SDF (64^3):     {res_sdf64['init_time_ms']:.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
