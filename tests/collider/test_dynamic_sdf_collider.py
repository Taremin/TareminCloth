from pathlib import Path
import sys
import unittest
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import taremin_cloth_core
from taremin_cloth.engine.sdf_baker import bake_bone_sdf_from_data


def _make_closed_box(x_range=(-0.1, 0.1), y_range=(0.0, 0.5), z_range=(-0.1, 0.1)):
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range
    verts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    tris = np.array([
        [0, 2, 1], [0, 3, 2],  # bottom (-Z)
        [4, 5, 6], [4, 6, 7],  # top (+Z)
        [0, 1, 5], [0, 5, 4],  # -Y
        [3, 7, 6], [3, 6, 2],  # +Y
        [0, 4, 7], [0, 7, 3],  # -X
        [1, 2, 6], [1, 6, 5],  # +X
    ], dtype=np.int32)
    # 頂点法線 (外向き中心放射近似)
    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5], dtype=np.float32)
    normals = verts - center
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(norms > 1e-6, normals / norms, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    return verts, normals.astype(np.float32), tris


class TestDynamicSdfCollider(unittest.TestCase):
    def setUp(self):
        if not taremin_cloth_core.is_gpu_available():
            self.skipTest("GPUが利用できない環境のためスキップします")

    def test_dynamic_bone_sdf_pipeline_and_simulation(self):
        """フルGPU動的SDFコライダーの静止支持（非貫通）およびボーン変形による押出応答を検証"""
        # 2ボーン素体メッシュ（直方体 2連: Y=0..0.5 は Bone0, Y=0.5..1.0 は Bone1）
        verts = np.array([
            [-0.1, 0.0, -0.1], [ 0.1, 0.0, -0.1], [ 0.1, 0.5, -0.1], [-0.1, 0.5, -0.1],
            [-0.1, 0.0,  0.1], [ 0.1, 0.0,  0.1], [ 0.1, 0.5,  0.1], [-0.1, 0.5,  0.1],
            [-0.1, 1.0, -0.1], [ 0.1, 1.0, -0.1], [ 0.1, 1.0,  0.1], [-0.1, 1.0,  0.1],
        ], dtype=np.float32)
        normals = np.zeros_like(verts)
        normals[:, 2] = np.where(verts[:, 2] >= 0.0, 1.0, -1.0)

        tris = np.array([
            # Bottom (-Y, y=0)
            [0, 1, 5], [0, 5, 4],
            # Top (+Y, y=1.0)
            [8, 11, 10], [8, 10, 9],
            # Back (-Z, z=-0.1)
            [0, 3, 2], [0, 2, 1],
            [3, 8, 9], [3, 9, 2],
            # Front (+Z, z=+0.1)
            [4, 5, 6], [4, 6, 7],
            [7, 6, 10], [7, 10, 11],
            # Left (-X, x=-0.1)
            [0, 4, 7], [0, 7, 3],
            [3, 7, 11], [3, 11, 8],
            # Right (+X, x=+0.1)
            [1, 2, 6], [1, 6, 5],
            [2, 9, 10], [2, 10, 6],
        ], dtype=np.int32)

        w0 = np.array([1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        w1 = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        bone_weights = {"Bone0": w0, "Bone1": w1}

        b0_mat = np.eye(4, dtype=np.float32)
        b1_mat = np.eye(4, dtype=np.float32)
        b1_mat[1, 3] = -0.5
        bone_bind_matrices = {"Bone0": b0_mat, "Bone1": b1_mat}

        bake_res = bake_bone_sdf_from_data(
            mesh_verts=verts, mesh_tris=tris,
            bone_weights=bone_weights, bone_bind_matrices=bone_bind_matrices,
            resolution=32, margin=0.2, weight_threshold=0.02,
        )
        self.assertIsNotNone(bake_res)

        # 布頂点群: Bone1 上面 (z=+0.15 -> -Z落下), および回転先領域 (-0.30, 0.52, 0.0)
        cloth_verts = np.array([
            [-0.04, 0.75, 0.15],
            [ 0.04, 0.75, 0.15],
            [-0.04, 0.85, 0.15],
            [ 0.04, 0.85, 0.15],
        ], dtype=np.float32)
        edges = np.array([[0, 1], [1, 3], [3, 2], [2, 0], [0, 3]], dtype=np.uint32)
        sim = taremin_cloth_core.ClothSimulator(positions=cloth_verts, edges=edges, thickness=0.01)
        sim.set_gravity(0.0, 0.0, -9.81)

        rest_verts = np.hstack([verts, normals]).astype(np.float32)
        n_verts = len(verts)
        bone_indices = np.zeros((n_verts, 4), dtype=np.uint32)
        bone_weights_arr = np.zeros((n_verts, 4), dtype=np.float32)
        bone_indices[:, 0] = 0
        bone_weights_arr[:, 0] = w0
        bone_indices[:, 1] = 1
        bone_weights_arr[:, 1] = w1

        tri_sources_list, tri_weights_list = [], []
        for b_id, bw in ((0, w0), (1, w1)):
            for tri in tris:
                tw0, tw1, tw2 = bw[tri[0]], bw[tri[1]], bw[tri[2]]
                if tw0 > 0 or tw1 > 0 or tw2 > 0:
                    tri_sources_list.append([tri[0], tri[1], tri[2], b_id])
                    tri_weights_list.append([tw0, tw1, tw2])

        bind_inv_mats = bake_res.bind_matrices.astype(np.float32)
        sim.setup_dynamic_bone_sdf(
            bake_res.width, bake_res.height, bake_res.depth, bake_res.res,
            bake_res.bone_infos, rest_verts, bone_indices, bone_weights_arr,
            np.array(tri_sources_list, dtype=np.uint32),
            np.array(tri_weights_list, dtype=np.float32),
            bind_inv_mats, 1,
        )

        # 1. 静止ポーズでの落下・支持検証 (20ステップ = 自由落下なら z ≈ -0.39)
        w_mats = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)])
        w_mats[1, 1, 3] = 0.5
        inv_mats = np.linalg.inv(w_mats)
        sim.update_bone_transforms(
            np.ascontiguousarray(np.transpose(w_mats, (0, 2, 1)), dtype=np.float32),
            np.ascontiguousarray(np.transpose(inv_mats, (0, 2, 1)), dtype=np.float32),
        )
        for _ in range(20):
            sim.step(1.0 / 60.0, 10)

        out_pos = np.zeros(len(cloth_verts) * 3, dtype=np.float32)
        sim.get_positions(out_pos)
        pos_static = out_pos.reshape((-1, 3))
        self.assertTrue(np.all(np.isfinite(pos_static)))
        self.assertGreater(float(np.min(pos_static[:, 2])), 0.08, "静止ポーズのBone1上面(z=0.1)で布が支持されていること")

        # 2. Bone1 を Z軸回りに段階回転 (90度倒す) し、ボーン変形で布が側面方向へ押出されること
        z_before = pos_static.copy()
        for step_i in range(1, 7):
            theta = (np.pi * 0.5) * (step_i / 6.0)
            r_mat = np.eye(4, dtype=np.float32)
            r_mat[0, 0] = np.cos(theta); r_mat[0, 1] = -np.sin(theta)
            r_mat[1, 0] = np.sin(theta); r_mat[1, 1] = np.cos(theta)
            w_mats[1] = np.eye(4, dtype=np.float32)
            w_mats[1, 1, 3] = 0.5
            w_mats[1] = w_mats[1] @ r_mat
            inv_mats = np.linalg.inv(w_mats)
            sim.update_bone_transforms(
                np.ascontiguousarray(np.transpose(w_mats, (0, 2, 1)), dtype=np.float32),
                np.ascontiguousarray(np.transpose(inv_mats, (0, 2, 1)), dtype=np.float32),
            )
            for _ in range(3):
                sim.step(1.0 / 60.0, 10)

        sim.get_positions(out_pos)
        pos_rot = out_pos.reshape((-1, 3))
        self.assertTrue(np.all(np.isfinite(pos_rot)))
        max_shift = float(np.max(np.linalg.norm(pos_rot - z_before, axis=1)))
        self.assertGreater(max_shift, 0.05, f"ボーン90度回転に伴い布が有意に変位・追従すること: shift={max_shift:.4f}")

    def test_dynamic_sdf_with_update_interval(self):
        """動的SDFの更新間隔（interval=2）でも素体上面での非貫通支持が機能すること"""
        verts, normals, tris = _make_closed_box(x_range=(-0.2, 0.2), y_range=(0.0, 0.5), z_range=(-0.1, 0.1))
        bone_weights = {"Bone0": np.ones(len(verts), dtype=np.float32)}
        bone_bind_matrices = {"Bone0": np.eye(4, dtype=np.float32)}

        bake_res = bake_bone_sdf_from_data(
            mesh_verts=verts, mesh_tris=tris,
            bone_weights=bone_weights, bone_bind_matrices=bone_bind_matrices,
            resolution=32, margin=0.2,
        )

        # 上面 z=0.1 の直上 z=0.15 から落下
        cloth_verts = np.array([[0.0, 0.2, 0.15], [0.08, 0.2, 0.15]], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.uint32)
        sim = taremin_cloth_core.ClothSimulator(positions=cloth_verts, edges=edges, thickness=0.01)
        sim.set_gravity(0.0, 0.0, -9.81)

        rest_verts = np.hstack([verts, normals]).astype(np.float32)
        bone_indices = np.zeros((len(verts), 4), dtype=np.uint32)
        bone_weights_arr = np.zeros((len(verts), 4), dtype=np.float32)
        bone_weights_arr[:, 0] = 1.0
        tri_sources = np.column_stack([tris.astype(np.uint32), np.zeros(len(tris), dtype=np.uint32)])
        tri_weights = np.ones((len(tris), 3), dtype=np.float32)
        bind_inv_mats = np.array([np.eye(4, dtype=np.float32)])

        sim.setup_dynamic_bone_sdf(
            bake_res.width, bake_res.height, bake_res.depth, bake_res.res,
            bake_res.bone_infos, rest_verts, bone_indices, bone_weights_arr,
            tri_sources, tri_weights, bind_inv_mats, 2,  # 2フレームごと更新
        )

        w_mats = np.array([np.eye(4, dtype=np.float32)])
        w_mats_col = np.ascontiguousarray(np.transpose(w_mats, (0, 2, 1)), dtype=np.float32)
        sim.update_bone_transforms(w_mats_col, w_mats_col)

        for _ in range(25):
            sim.step(1.0 / 60.0, 5)

        out_pos = np.zeros(len(cloth_verts) * 3, dtype=np.float32)
        sim.get_positions(out_pos)
        self.assertTrue(np.all(np.isfinite(out_pos)))
        min_z = float(np.min(out_pos.reshape((-1, 3))[:, 2]))
        # 25フレーム自由落下なら z = 0.15 - 0.5*9.81*(25/60)^2 = -0.70m まで落ちるはず
        self.assertGreater(min_z, 0.08, f"interval=2 でもBone0上面を貫通していないこと: min_z={min_z:.4f}")

    def test_dynamic_sdf_coexist_with_primitive_colliders(self):
        """動的SDFコライダーと球・平面コライダーが同時有効で全頂点が貫通せず支持されること"""
        verts, normals, tris = _make_closed_box(x_range=(-0.2, 0.2), y_range=(-0.2, 0.4), z_range=(-0.1, 0.1))
        bone_weights = {"Bone0": np.ones(len(verts), dtype=np.float32)}
        bone_bind_matrices = {"Bone0": np.eye(4, dtype=np.float32)}

        bake_res = bake_bone_sdf_from_data(
            mesh_verts=verts, mesh_tris=tris,
            bone_weights=bone_weights, bone_bind_matrices=bone_bind_matrices,
            resolution=32, margin=0.2,
        )

        # 3頂点: [0] 動的Bone SDF直上, [1] 球コライダー直上, [2] 平面コライダー直上
        cloth_verts = np.array([
            [0.0, 0.1, 0.18],   # Bone0上面 z=0.1 の上
            [2.0, 0.0, 0.58],   # 球 (center=[2,0,0], r=0.5) の上
            [4.0, 0.0, 0.15],   # 平面 (point=[4,0,0], n=[0,0,1]) の上
        ], dtype=np.float32)
        edges = np.empty((0, 2), dtype=np.uint32)
        sim = taremin_cloth_core.ClothSimulator(positions=cloth_verts, edges=edges, thickness=0.01)
        sim.set_gravity(0.0, 0.0, -9.81)

        sim.add_sphere_collider([2.0, 0.0, 0.0], 0.5, 0.3, 0.0)
        sim.add_plane_collider([4.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.3, 0.0)

        rest_verts = np.hstack([verts, normals]).astype(np.float32)
        bone_indices = np.zeros((len(verts), 4), dtype=np.uint32)
        bone_weights_arr = np.zeros((len(verts), 4), dtype=np.float32)
        bone_weights_arr[:, 0] = 1.0
        tri_sources = np.column_stack([tris.astype(np.uint32), np.zeros(len(tris), dtype=np.uint32)])
        tri_weights = np.ones((len(tris), 3), dtype=np.float32)
        bind_inv_mats = np.array([np.eye(4, dtype=np.float32)])

        sim.setup_dynamic_bone_sdf(
            bake_res.width, bake_res.height, bake_res.depth, bake_res.res,
            bake_res.bone_infos, rest_verts, bone_indices, bone_weights_arr,
            tri_sources, tri_weights, bind_inv_mats, 1,
        )

        w_mats = np.array([np.eye(4, dtype=np.float32)])
        w_mats_col = np.ascontiguousarray(np.transpose(w_mats, (0, 2, 1)), dtype=np.float32)
        sim.update_bone_transforms(w_mats_col, w_mats_col)

        # 30フレーム実行 (0.5s: 自由落下なら Δz ≈ -1.225m)
        for _ in range(30):
            sim.step(1.0 / 60.0, 10)

        out_pos = np.zeros(len(cloth_verts) * 3, dtype=np.float32)
        sim.get_positions(out_pos)
        res = out_pos.reshape((-1, 3))
        self.assertTrue(np.all(np.isfinite(res)))

        self.assertGreater(float(res[0, 2]), 0.08, f"頂点0が動的Bone SDF上で支持されていません: z={res[0, 2]:.4f}")
        sphere_dist = float(np.linalg.norm(res[1] - np.array([2.0, 0.0, 0.0], dtype=np.float32)))
        self.assertGreater(sphere_dist, 0.48, f"頂点1が球コライダー上で支持されていません: dist={sphere_dist:.4f}")
        self.assertGreater(float(res[2, 2]), -0.01, f"頂点2が平面コライダー上で支持されていません: z={res[2, 2]:.4f}")

    def test_dynamic_sdf_dirty_bones_differential_update(self):
        """Dirtyボーン指定 (空配列スキップ -> 指定ボーン更新) で衝突位置が正しく追従することを検証"""
        verts, normals, tris = _make_closed_box(x_range=(-0.15, 0.15), y_range=(0.0, 0.5), z_range=(-0.1, 0.1))
        # Bone0 (下) と Bone1 (上: Y=0.5付近) の2ボーン素体
        verts2 = verts.copy()
        verts2[:, 1] += 0.5
        verts_all = np.vstack([verts, verts2])
        normals_all = np.vstack([normals, normals])
        tris_all = np.vstack([tris, tris + len(verts)])
        w0 = np.array([1.0] * len(verts) + [0.0] * len(verts), dtype=np.float32)
        w1 = 1.0 - w0

        b0_mat = np.eye(4, dtype=np.float32)
        b1_mat = np.eye(4, dtype=np.float32)
        b1_mat[1, 3] = -0.5
        bake_res = bake_bone_sdf_from_data(
            mesh_verts=verts_all, mesh_tris=tris_all,
            bone_weights={"Bone0": w0, "Bone1": w1},
            bone_bind_matrices={"Bone0": b0_mat, "Bone1": b1_mat},
            resolution=32, margin=0.2,
        )

        # Bone1 上面直上 (z=0.16) に布頂点を配置
        cloth_verts = np.array([[0.0, 0.75, 0.16], [0.05, 0.75, 0.16]], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.uint32)
        sim = taremin_cloth_core.ClothSimulator(positions=cloth_verts, edges=edges, thickness=0.01)
        sim.set_gravity(0.0, 0.0, -9.81)

        rest_verts = np.hstack([verts_all, normals_all]).astype(np.float32)
        n_v = len(verts_all)
        bone_indices = np.zeros((n_v, 4), dtype=np.uint32)
        bone_weights_arr = np.zeros((n_v, 4), dtype=np.float32)
        bone_indices[:, 0] = 0; bone_weights_arr[:, 0] = w0
        bone_indices[:, 1] = 1; bone_weights_arr[:, 1] = w1

        tri_sources_list, tri_weights_list = [], []
        for b_id, bw in ((0, w0), (1, w1)):
            for tri in tris_all:
                tw0, tw1, tw2 = bw[tri[0]], bw[tri[1]], bw[tri[2]]
                if tw0 > 0 or tw1 > 0 or tw2 > 0:
                    tri_sources_list.append([tri[0], tri[1], tri[2], b_id])
                    tri_weights_list.append([tw0, tw1, tw2])
        bind_inv_mats = bake_res.bind_matrices.astype(np.float32)
        sim.setup_dynamic_bone_sdf(
            bake_res.width, bake_res.height, bake_res.depth, bake_res.res,
            bake_res.bone_infos, rest_verts, bone_indices, bone_weights_arr,
            np.array(tri_sources_list, dtype=np.uint32),
            np.array(tri_weights_list, dtype=np.float32),
            bind_inv_mats, 1,
        )

        w_mats = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)])
        w_mats[1, 1, 3] = 0.5
        inv_mats = np.linalg.inv(w_mats)
        sim.update_bone_transforms(
            np.ascontiguousarray(np.transpose(w_mats, (0, 2, 1)), dtype=np.float32),
            np.ascontiguousarray(np.transpose(inv_mats, (0, 2, 1)), dtype=np.float32),
            None,
        )
        for _ in range(15):
            sim.step(1.0 / 60.0, 5)

        out_pos = np.zeros(len(cloth_verts) * 3, dtype=np.float32)
        sim.get_positions(out_pos)
        pos1 = out_pos.reshape((-1, 3)).copy()
        self.assertGreater(float(np.min(pos1[:, 2])), 0.08)

        # 空 Dirty 配列での静止ステップも破損・貫通なく支持継続すること
        empty_dirty = np.empty(0, dtype=np.uint32)
        sim.update_bone_transforms(
            np.ascontiguousarray(np.transpose(w_mats, (0, 2, 1)), dtype=np.float32),
            np.ascontiguousarray(np.transpose(inv_mats, (0, 2, 1)), dtype=np.float32),
            empty_dirty,
        )
        for _ in range(10):
            sim.step(1.0 / 60.0, 5)
        sim.get_positions(out_pos)
        pos2 = out_pos.reshape((-1, 3)).copy()
        self.assertGreater(float(np.min(pos2[:, 2])), 0.08)

        # Dirty=[1] で Bone1 を Z方向に 0.05 -> 0.10 -> 0.15m へ段階持ち上げすると布が押し上げられること
        for dz in (0.05, 0.10, 0.15):
            w_mats[1, 2, 3] = dz
            inv_mats = np.linalg.inv(w_mats)
            sim.update_bone_transforms(
                np.ascontiguousarray(np.transpose(w_mats, (0, 2, 1)), dtype=np.float32),
                np.ascontiguousarray(np.transpose(inv_mats, (0, 2, 1)), dtype=np.float32),
                np.array([1], dtype=np.uint32),
            )
            for _ in range(5):
                sim.step(1.0 / 60.0, 5)

        sim.get_positions(out_pos)
        pos3 = out_pos.reshape((-1, 3))
        self.assertTrue(np.all(np.isfinite(pos3)))
        self.assertGreater(float(np.min(pos3[:, 2])), float(np.min(pos2[:, 2])) + 0.10, "Dirtyボーン移動で布が押し上げられること")


if __name__ == "__main__":
    unittest.main()

