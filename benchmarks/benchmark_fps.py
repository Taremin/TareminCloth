"""
Taremin Cloth シミュレーションFPS計測ベンチマークスクリプト
Blender環境および純粋なPython環境で実行可能
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, ".")
import numpy as np

def create_grid_mesh_data(nx=50, ny=50, dx=0.04):
    """指定解像度のグリッドメッシュデータを生成"""
    positions = []
    inv_masses = []

    for y in range(ny):
        for x in range(nx):
            positions.append([x * dx, y * dx, 1.0])
            # 上端の2点を固定
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
                # 2つの三角形
                faces.append([idx, idx + 1, idx + nx + 1])
                faces.append([idx, idx + nx + 1, idx + nx])

    edges_np = np.array(edges, dtype=np.uint32)
    faces_np = np.array(faces, dtype=np.uint32)

    return positions_np, edges_np, faces_np, inv_masses_np


def benchmark_pure_simulator(resolutions=[(30, 30), (50, 50), (70, 70)], n_frames=60, substeps=20, solver_iterations=2):
    """純粋なRust/wgpuコア + PythonバインディングのFPS計測"""
    import taremin_cloth_core

    print("=" * 70)
    print(f"【ベンチマーク 1: 純粋なシミュレータ (GPU + CPU転送)】")
    print(f"GPUデバイス: {taremin_cloth_core.get_gpu_device_name()}")
    print(f"設定: substeps={substeps}, solver_iterations={solver_iterations}, frames={n_frames}")
    print("=" * 70)

    for nx, ny in resolutions:
        positions_np, edges_np, faces_np, inv_masses_np = create_grid_mesh_data(nx, ny)
        n_verts = len(positions_np)
        n_edges = len(edges_np)
        n_faces = len(faces_np)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions_np,
            edges=edges_np,
            faces=faces_np,
            inv_masses=inv_masses_np,
            layer_id=0,
            thickness=0.01,
            stiffness=1000.0,
            compression_stiffness=1000.0,
            shear_stiffness=500.0,
            bending_stiffness=50.0,
            sewing_shrink_speed=0.0,
        )

        out_coords = np.empty(n_verts * 3, dtype=np.float32)

        # ウォームアップ (5フレーム)
        for _ in range(5):
            sim.step(dt=1.0 / 60.0, substeps=substeps, solver_iterations=solver_iterations)
            sim.get_positions(out_coords)

        # 計測
        step_times = []
        read_times = []
        total_times = []

        for _ in range(n_frames):
            t0 = time.perf_counter()
            sim.step(dt=1.0 / 60.0, substeps=substeps, solver_iterations=solver_iterations)
            t1 = time.perf_counter()
            sim.get_positions(out_coords)
            t2 = time.perf_counter()

            step_times.append((t1 - t0) * 1000.0)
            read_times.append((t2 - t1) * 1000.0)
            total_times.append((t2 - t0) * 1000.0)

        avg_step = np.mean(step_times)
        avg_read = np.mean(read_times)
        avg_total = np.mean(total_times)
        fps = 1000.0 / avg_total if avg_total > 0 else 0

        print(f"メッシュ: {nx}x{ny} ({n_verts:,} 頂点, {n_edges:,} エッジ, {n_faces:,} 面)")
        print(f"  - sim.step()        : {avg_step:6.2f} ms")
        print(f"  - sim.get_positions : {avg_read:6.2f} ms")
        print(f"  - 合計フレーム時間    : {avg_total:6.2f} ms")
        print(f"  => シミュレーションFPS : {fps:6.1f} FPS")
        print("-" * 70)


def benchmark_blender_pipeline(resolutions=[(30, 30), (50, 50), (70, 70)], n_frames=60, substeps=20, solver_iterations=2):
    """Blender内での実パイプライン（パラメータ同期 + step + get_positions + foreach_set + update）のFPS計測"""
    try:
        import bpy
    except ImportError:
        print("[Notice] Blender環境外のためBlenderパイプライン計測をスキップします")
        return

    import taremin_cloth_core

    print("=" * 70)
    print(f"【ベンチマーク 2: Blender統合パイプライン (Blender メッシュ更新含む)】")
    print(f"Blenderバージョン: {bpy.app.version_string}")
    print(f"GPUデバイス: {taremin_cloth_core.get_gpu_device_name()}")
    print(f"設定: substeps={substeps}, solver_iterations={solver_iterations}, frames={n_frames}")
    print("=" * 70)

    for nx, ny in resolutions:
        # Blenderメッシュ作成
        mesh = bpy.data.meshes.new(f"BenchMesh_{nx}x{ny}")
        obj = bpy.data.objects.new(f"BenchObj_{nx}x{ny}", mesh)
        bpy.context.scene.collection.objects.link(obj)

        positions_np, edges_np, faces_np, inv_masses_np = create_grid_mesh_data(nx, ny)
        n_verts = len(positions_np)
        n_edges = len(edges_np)

        verts_flat = positions_np.tolist()
        faces_list = faces_np.tolist()
        mesh.from_pydata(verts_flat, [], faces_list)
        mesh.update()

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions_np,
            edges=edges_np,
            faces=faces_np,
            inv_masses=inv_masses_np,
            layer_id=0,
            thickness=0.01,
            stiffness=1000.0,
            compression_stiffness=1000.0,
            shear_stiffness=500.0,
            bending_stiffness=50.0,
            sewing_shrink_speed=0.0,
        )

        out_coords = np.empty(n_verts * 3, dtype=np.float32)

        # ウォームアップ
        for _ in range(5):
            sim.step(dt=1.0 / 60.0, substeps=substeps, solver_iterations=solver_iterations)
            sim.get_positions(out_coords)
            mesh.vertices.foreach_set("co", out_coords)
            mesh.update()

        # 計測
        step_times = []
        read_times = []
        mesh_update_times = []
        total_times = []

        for _ in range(n_frames):
            t0 = time.perf_counter()
            sim.step(dt=1.0 / 60.0, substeps=substeps, solver_iterations=solver_iterations)
            t1 = time.perf_counter()
            sim.get_positions(out_coords)
            t2 = time.perf_counter()
            mesh.vertices.foreach_set("co", out_coords)
            mesh.update()
            t3 = time.perf_counter()

            step_times.append((t1 - t0) * 1000.0)
            read_times.append((t2 - t1) * 1000.0)
            mesh_update_times.append((t3 - t2) * 1000.0)
            total_times.append((t3 - t0) * 1000.0)

        avg_step = np.mean(step_times)
        avg_read = np.mean(read_times)
        avg_mesh = np.mean(mesh_update_times)
        avg_total = np.mean(total_times)
        fps = 1000.0 / avg_total if avg_total > 0 else 0

        print(f"メッシュ: {nx}x{ny} ({n_verts:,} 頂点)")
        print(f"  - sim.step()             : {avg_step:6.2f} ms")
        print(f"  - sim.get_positions      : {avg_read:6.2f} ms")
        print(f"  - mesh.update() (Blender): {avg_mesh:6.2f} ms")
        print(f"  - 合計フレーム時間         : {avg_total:6.2f} ms")
        print(f"  => 総合実効FPS            : {fps:6.1f} FPS")
        print("-" * 70)

        # クリーンアップ
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh, do_unlink=True)


def benchmark_with_mesh_collider(n_frames=60, substeps=20, solver_iterations=2):
    """Blender内でSuzanneコライダーに布が乗る実利用シナリオのFPS計測"""
    try:
        import bpy
    except ImportError:
        return

    import python.taremin_cloth as taremin_cloth
    try:
        taremin_cloth.register()
    except Exception:
        pass

    from python.taremin_cloth.operators import sync_colliders, sync_cloth_parameters
    import taremin_cloth_core

    print("=" * 70)
    print(f"【ベンチマーク 3: 実シナリオ (50x50布 + Suzanne メッシュコライダー衝突)】")
    print("=" * 70)

    # Suzanne作成
    bpy.ops.mesh.primitive_monkey_add(size=1.5, location=(1.0, 1.0, 0.0))
    monkey = bpy.context.active_object
    monkey.taremin_collider.is_collider = True
    monkey.taremin_collider.collider_type = 'MESH'

    # 布メッシュ作成
    nx, ny = 50, 50
    mesh = bpy.data.meshes.new("ClothWithCollider")
    cloth_obj = bpy.data.objects.new("ClothWithColliderObj", mesh)
    bpy.context.scene.collection.objects.link(cloth_obj)

    positions_np, edges_np, faces_np, inv_masses_np = create_grid_mesh_data(nx, ny)
    n_verts = len(positions_np)
    mesh.from_pydata(positions_np.tolist(), [], faces_np.tolist())
    mesh.update()

    cloth_obj.taremin_cloth.is_cloth = True
    cloth_obj.taremin_cloth.substeps = substeps
    cloth_obj.taremin_cloth.solver_iterations = solver_iterations

    sim = taremin_cloth_core.ClothSimulator(
        positions=positions_np,
        edges=edges_np,
        faces=faces_np,
        inv_masses=inv_masses_np,
        layer_id=0,
        thickness=0.01,
        stiffness=1000.0,
        compression_stiffness=1000.0,
        shear_stiffness=500.0,
        bending_stiffness=50.0,
        sewing_shrink_speed=0.0,
    )
    out_coords = np.empty(n_verts * 3, dtype=np.float32)

    # ウォームアップ
    for _ in range(5):
        sync_colliders(sim, bpy.context.scene)
        sync_cloth_parameters(sim, cloth_obj, bpy.context.scene)
        sim.step(dt=1.0 / 60.0, substeps=substeps, solver_iterations=solver_iterations)
        sim.get_positions(out_coords)
        cloth_obj.data.vertices.foreach_set("co", out_coords)
        cloth_obj.data.update()

    # 計測
    sync_times = []
    step_times = []
    read_times = []
    mesh_update_times = []
    total_times = []

    for _ in range(n_frames):
        t0 = time.perf_counter()
        sync_colliders(sim, bpy.context.scene)
        sync_cloth_parameters(sim, cloth_obj, bpy.context.scene)
        t1 = time.perf_counter()

        sim.step(dt=1.0 / 60.0, substeps=substeps, solver_iterations=solver_iterations)
        t2 = time.perf_counter()

        sim.get_positions(out_coords)
        t3 = time.perf_counter()

        cloth_obj.data.vertices.foreach_set("co", out_coords)
        cloth_obj.data.update()
        t4 = time.perf_counter()

        sync_times.append((t1 - t0) * 1000.0)
        step_times.append((t2 - t1) * 1000.0)
        read_times.append((t3 - t2) * 1000.0)
        mesh_update_times.append((t4 - t3) * 1000.0)
        total_times.append((t4 - t0) * 1000.0)

    avg_sync = np.mean(sync_times)
    avg_step = np.mean(step_times)
    avg_read = np.mean(read_times)
    avg_mesh = np.mean(mesh_update_times)
    avg_total = np.mean(total_times)
    fps = 1000.0 / avg_total if avg_total > 0 else 0

    print(f"実シナリオ (50x50 = 2,500 頂点 + Suzanne 968面):")
    print(f"  - コライダー同期 (Python)   : {avg_sync:6.2f} ms")
    print(f"  - sim.step() (GPU)        : {avg_step:6.2f} ms")
    print(f"  - sim.get_positions (GPU) : {avg_read:6.2f} ms")
    print(f"  - mesh.update() (Blender) : {avg_mesh:6.2f} ms")
    print(f"  - 合計フレーム時間          : {avg_total:6.2f} ms")
    print(f"  => 実効FPS                 : {fps:6.1f} FPS")
    print("=" * 70)

    # クリーンアップ
    bpy.data.objects.remove(cloth_obj, do_unlink=True)
    bpy.data.meshes.remove(mesh, do_unlink=True)
    bpy.data.objects.remove(monkey, do_unlink=True)


if __name__ == "__main__":
    benchmark_pure_simulator(resolutions=[(30, 30), (50, 50), (100, 100)])
    benchmark_blender_pipeline(resolutions=[(30, 30), (50, 50), (100, 100)])
    benchmark_with_mesh_collider()
