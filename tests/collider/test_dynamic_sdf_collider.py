import os
import sys
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import taremin_cloth_core
from taremin_cloth.engine.sdf_baker import bake_bone_sdf_from_data, BoneSdfBakeResult


class TestDynamicSdfCollider(unittest.TestCase):
    def setUp(self):
        if not taremin_cloth_core.is_gpu_available():
            self.skipTest("GPUが利用できない環境のためスキップします")

    def test_dynamic_bone_sdf_pipeline_and_simulation(self):
        """フルGPU動的SDFコライダーの初期化、ボーン変形、シミュレーション実行テスト"""
        # 1. 2ボーン素体メッシュ（直方体 2連、高さ 1.0m、幅 0.2m、奥行き 0.2m）
        # 下半身 (y: 0..0.5) は Bone0, 上半身 (y: 0.5..1.0) は Bone1
        verts = np.array([
            [-0.1, 0.0, -0.1], [ 0.1, 0.0, -0.1], [ 0.1, 0.5, -0.1], [-0.1, 0.5, -0.1], # Bone0 底〜中間
            [-0.1, 0.0,  0.1], [ 0.1, 0.0,  0.1], [ 0.1, 0.5,  0.1], [-0.1, 0.5,  0.1],
            [-0.1, 1.0, -0.1], [ 0.1, 1.0, -0.1], [ 0.1, 1.0,  0.1], [-0.1, 1.0,  0.1], # Bone1 上部
        ], dtype=np.float32)

        normals = np.zeros_like(verts)
        normals[:, 1] = 1.0 # 簡易法線

        tris = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 9, 8], [2, 3, 8],
            [6, 10, 9], [6, 9, 2], [7, 11, 10], [7, 10, 6],
            [8, 9, 10], [8, 10, 11]
        ], dtype=np.int32)

        bone_names = ["Bone0", "Bone1"]
        # ウェイト設定 (中間頂点 2, 3, 6, 7 は 5:5 ブレンド)
        w0 = np.array([1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        w1 = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        bone_weights = {"Bone0": w0, "Bone1": w1}

        # Bone0 は原点、Bone1 は y=0.5
        b0_mat = np.eye(4, dtype=np.float32)
        b1_mat = np.eye(4, dtype=np.float32)
        b1_mat[1, 3] = -0.5 # メッシュ -> ボーン1ローカル (y -= 0.5)
        bone_bind_matrices = {"Bone0": b0_mat, "Bone1": b1_mat}

        # 2. 初期静的SDFアトラスのベイク
        bake_res = bake_bone_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            bone_weights=bone_weights,
            bone_bind_matrices=bone_bind_matrices,
            resolution=32,
            margin=0.2,
            weight_threshold=0.02,
        )
        self.assertIsNotNone(bake_res)

        # 3. ClothSimulator の作成と布メッシュ登録
        cloth_verts = np.array([
            [0.0, 0.8, -0.2],
            [0.1, 0.8, -0.2],
            [0.0, 0.9, -0.2],
            [0.1, 0.9, -0.2],
        ], dtype=np.float32)
        edges = np.array([[0, 1], [1, 3], [3, 2], [2, 0], [0, 3]], dtype=np.uint32)
        sim = taremin_cloth_core.ClothSimulator(
            positions=cloth_verts,
            edges=edges,
        )

        # 4. 動的SDFデータの整形
        rest_verts = np.hstack([verts, normals]).astype(np.float32)
        n_verts = len(verts)

        bone_indices = np.zeros((n_verts, 4), dtype=np.uint32)
        bone_weights_arr = np.zeros((n_verts, 4), dtype=np.float32)
        for i in range(n_verts):
            bone_indices[i, 0] = 0
            bone_weights_arr[i, 0] = w0[i]
            bone_indices[i, 1] = 1
            bone_weights_arr[i, 1] = w1[i]

        # 三角形ソース
        tri_sources_list = []
        tri_weights_list = []
        # Bone0 三角形
        for tri in tris:
            tw0 = w0[tri[0]]; tw1 = w0[tri[1]]; tw2 = w0[tri[2]]
            if tw0 > 0 or tw1 > 0 or tw2 > 0:
                tri_sources_list.append([tri[0], tri[1], tri[2], 0])
                tri_weights_list.append([tw0, tw1, tw2])
        # Bone1 三角形
        for tri in tris:
            tw0 = w1[tri[0]]; tw1 = w1[tri[1]]; tw2 = w1[tri[2]]
            if tw0 > 0 or tw1 > 0 or tw2 > 0:
                tri_sources_list.append([tri[0], tri[1], tri[2], 1])
                tri_weights_list.append([tw0, tw1, tw2])

        tri_sources = np.array(tri_sources_list, dtype=np.uint32)
        tri_weights = np.array(tri_weights_list, dtype=np.float32)

        b0_inv = np.linalg.inv(b0_mat)
        b1_inv = np.linalg.inv(b1_mat)
        bind_inv_mats = np.stack([b0_inv, b1_inv]).astype(np.float32)

        # 5. setup_dynamic_bone_sdf の呼び出し
        sim.setup_dynamic_bone_sdf(
            bake_res.width,
            bake_res.height,
            bake_res.depth,
            bake_res.res,
            bake_res.bone_infos,
            rest_verts,
            bone_indices,
            bone_weights_arr,
            tri_sources,
            tri_weights,
            bind_inv_mats,
            1, # 毎フレーム更新
        )

        # 6. 静止ポーズでの初期ステップ進行
        w_mats = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)])
        w_mats[1, 1, 3] = 0.5 # Bone1のワールド位置 y=0.5
        inv_mats = np.linalg.inv(w_mats)

        # 列優先(Column-major)転置
        w_mats_col = np.ascontiguousarray(np.transpose(w_mats, (0, 2, 1)), dtype=np.float32)
        inv_mats_col = np.ascontiguousarray(np.transpose(inv_mats, (0, 2, 1)), dtype=np.float32)
        sim.update_bone_transforms(w_mats_col, inv_mats_col)

        # 5ステップ実行（クラッシュしないこと）
        for _ in range(5):
            sim.step(0.0166667, 10)

        # 7. Bone1 を 90度回転（Z軸回転）させてアニメーション変形
        # 回転行列
        theta = np.pi / 2.0
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        r_mat = np.eye(4, dtype=np.float32)
        r_mat[0, 0] = cos_t; r_mat[0, 1] = -sin_t
        r_mat[1, 0] = sin_t; r_mat[1, 1] = cos_t

        w_mats[1] = np.eye(4, dtype=np.float32)
        w_mats[1, 1, 3] = 0.5
        w_mats[1] = w_mats[1] @ r_mat # 回転適用
        inv_mats = np.linalg.inv(w_mats)

        w_mats_col = np.ascontiguousarray(np.transpose(w_mats, (0, 2, 1)), dtype=np.float32)
        inv_mats_col = np.ascontiguousarray(np.transpose(inv_mats, (0, 2, 1)), dtype=np.float32)
        sim.update_bone_transforms(w_mats_col, inv_mats_col)

        # 変形後も5ステップ実行し、GPUパイプライン（LBSスキニング -> 局所射影 -> SDFベイク -> テクスチャコピー -> 衝突判定）が完全動作することを確認
        for _ in range(5):
            sim.step(0.0166667, 10)

        # 頂点座標が正常に取得できること（NaNがないこと）
        out_pos = np.zeros(len(cloth_verts) * 3, dtype=np.float32)
        sim.get_positions(out_pos)
        self.assertFalse(np.isnan(out_pos).any())
        self.assertFalse(np.isinf(out_pos).any())

    def test_dynamic_sdf_with_update_interval(self):
        """動的SDFの更新間隔（interval=2など）の動作テスト"""
        verts = np.array([
            [-0.1, 0.0, -0.1], [ 0.1, 0.0, -0.1], [ 0.1, 0.5, -0.1], [-0.1, 0.5, -0.1],
        ], dtype=np.float32)
        normals = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float32)
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        bone_weights = {"Bone0": np.ones(4, dtype=np.float32)}
        bone_bind_matrices = {"Bone0": np.eye(4, dtype=np.float32)}

        bake_res = bake_bone_sdf_from_data(
            mesh_verts=verts, mesh_tris=tris,
            bone_weights=bone_weights, bone_bind_matrices=bone_bind_matrices,
            resolution=32,
        )

        cloth_verts = np.array([[0.0, 0.2, 0.0], [0.1, 0.2, 0.0]], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.uint32)
        sim = taremin_cloth_core.ClothSimulator(positions=cloth_verts, edges=edges)

        rest_verts = np.hstack([verts, normals]).astype(np.float32)
        bone_indices = np.zeros((4, 4), dtype=np.uint32)
        bone_weights_arr = np.zeros((4, 4), dtype=np.float32)
        bone_weights_arr[:, 0] = 1.0

        tri_sources = np.array([[0, 1, 2, 0], [0, 2, 3, 0]], dtype=np.uint32)
        tri_weights = np.ones((2, 3), dtype=np.float32)
        bind_inv_mats = np.array([np.eye(4, dtype=np.float32)])

        sim.setup_dynamic_bone_sdf(
            bake_res.width, bake_res.height, bake_res.depth, bake_res.res,
            bake_res.bone_infos, rest_verts, bone_indices, bone_weights_arr,
            tri_sources, tri_weights, bind_inv_mats, 2, # 2フレームごとに更新
        )

        # 複数フレーム実行（奇数・偶数フレームの跨ぎ）
        w_mats = np.array([np.eye(4, dtype=np.float32)])
        w_mats_col = np.ascontiguousarray(np.transpose(w_mats, (0, 2, 1)), dtype=np.float32)
        sim.update_bone_transforms(w_mats_col, w_mats_col)

        for _ in range(10):
            sim.step(0.0166667, 5)

        out_pos = np.zeros(len(cloth_verts) * 3, dtype=np.float32)
        sim.get_positions(out_pos)
        self.assertFalse(np.isnan(out_pos).any())

    def test_dynamic_sdf_coexist_with_primitive_colliders(self):
        """動的SDFコライダーと他のコライダー（球・平面等）が完全共存して動作するテスト"""
        verts = np.array([
            [-0.1, 0.0, -0.1], [ 0.1, 0.0, -0.1], [ 0.1, 0.5, -0.1], [-0.1, 0.5, -0.1],
        ], dtype=np.float32)
        normals = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float32)
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        bone_weights = {"Bone0": np.ones(4, dtype=np.float32)}
        bone_bind_matrices = {"Bone0": np.eye(4, dtype=np.float32)}

        bake_res = bake_bone_sdf_from_data(
            mesh_verts=verts, mesh_tris=tris,
            bone_weights=bone_weights, bone_bind_matrices=bone_bind_matrices,
            resolution=32,
        )

        cloth_verts = np.array([[0.0, 0.5, 0.0], [0.1, 0.5, 0.0]], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.uint32)
        sim = taremin_cloth_core.ClothSimulator(positions=cloth_verts, edges=edges)

        # 球コライダーと平面コライダーを追加
        sim.add_sphere_collider([0.0, -1.0, 0.0], 0.5, 0.3, 0.0)
        sim.add_plane_collider([0.0, -2.0, 0.0], [0.0, 1.0, 0.0], 0.3, 0.0)

        rest_verts = np.hstack([verts, normals]).astype(np.float32)
        bone_indices = np.zeros((4, 4), dtype=np.uint32)
        bone_weights_arr = np.zeros((4, 4), dtype=np.float32)
        bone_weights_arr[:, 0] = 1.0

        tri_sources = np.array([[0, 1, 2, 0], [0, 2, 3, 0]], dtype=np.uint32)
        tri_weights = np.ones((2, 3), dtype=np.float32)
        bind_inv_mats = np.array([np.eye(4, dtype=np.float32)])

        # 動的SDFのセットアップ（球・平面コライダーが存在する状態）
        sim.setup_dynamic_bone_sdf(
            bake_res.width, bake_res.height, bake_res.depth, bake_res.res,
            bake_res.bone_infos, rest_verts, bone_indices, bone_weights_arr,
            tri_sources, tri_weights, bind_inv_mats, 1,
        )

        w_mats = np.array([np.eye(4, dtype=np.float32)])
        w_mats_col = np.ascontiguousarray(np.transpose(w_mats, (0, 2, 1)), dtype=np.float32)
        sim.update_bone_transforms(w_mats_col, w_mats_col)

        # 5ステップ実行
        for _ in range(5):
            sim.step(0.0166667, 10)

    def test_dynamic_sdf_dirty_bones_differential_update(self):
        """Dirtyボーン指定による動的SDFの差分更新およびスキップ（空配列）の動作テスト"""
        verts = np.array([
            [-0.1, 0.0, -0.1], [ 0.1, 0.0, -0.1], [ 0.1, 0.5, -0.1], [-0.1, 0.5, -0.1],
            [-0.1, 0.0,  0.1], [ 0.1, 0.0,  0.1], [ 0.1, 0.5,  0.1], [-0.1, 0.5,  0.1],
            [-0.1, 1.0, -0.1], [ 0.1, 1.0, -0.1], [ 0.1, 1.0,  0.1], [-0.1, 1.0,  0.1],
        ], dtype=np.float32)
        normals = np.zeros_like(verts)
        normals[:, 1] = 1.0
        tris = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 9, 8], [2, 3, 8],
            [6, 10, 9], [6, 9, 2], [7, 11, 10], [7, 10, 6],
            [8, 9, 10], [8, 10, 11]
        ], dtype=np.int32)

        w0 = np.array([1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        w1 = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        bone_weights = {"Bone0": w0, "Bone1": w1}

        b0_mat = np.eye(4, dtype=np.float32)
        b1_mat = np.eye(4, dtype=np.float32)
        b1_mat[1, 3] = -0.5
        bone_bind_matrices = {"Bone0": b0_mat, "Bone1": b1_mat}

        bake_res = bake_bone_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            bone_weights=bone_weights,
            bone_bind_matrices=bone_bind_matrices,
            resolution=32,
            margin=0.2,
        )

        cloth_verts = np.array([[0.0, 0.8, -0.2], [0.1, 0.8, -0.2]], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.uint32)
        sim = taremin_cloth_core.ClothSimulator(positions=cloth_verts, edges=edges)

        rest_verts = np.hstack([verts, normals]).astype(np.float32)
        n_verts = len(verts)
        bone_indices = np.zeros((n_verts, 4), dtype=np.uint32)
        bone_weights_arr = np.zeros((n_verts, 4), dtype=np.float32)
        for i in range(n_verts):
            bone_indices[i, 0] = 0
            bone_weights_arr[i, 0] = w0[i]
            bone_indices[i, 1] = 1
            bone_weights_arr[i, 1] = w1[i]

        tri_sources_list = []
        tri_weights_list = []
        for tri in tris:
            tw0 = w0[tri[0]]; tw1 = w0[tri[1]]; tw2 = w0[tri[2]]
            if tw0 > 0 or tw1 > 0 or tw2 > 0:
                tri_sources_list.append([tri[0], tri[1], tri[2], 0])
                tri_weights_list.append([tw0, tw1, tw2])
        for tri in tris:
            tw0 = w1[tri[0]]; tw1 = w1[tri[1]]; tw2 = w1[tri[2]]
            if tw0 > 0 or tw1 > 0 or tw2 > 0:
                tri_sources_list.append([tri[0], tri[1], tri[2], 1])
                tri_weights_list.append([tw0, tw1, tw2])

        tri_sources = np.array(tri_sources_list, dtype=np.uint32)
        tri_weights = np.array(tri_weights_list, dtype=np.float32)
        bind_inv_mats = np.stack([np.linalg.inv(b0_mat), np.linalg.inv(b1_mat)]).astype(np.float32)

        sim.setup_dynamic_bone_sdf(
            bake_res.width, bake_res.height, bake_res.depth, bake_res.res,
            bake_res.bone_infos, rest_verts, bone_indices, bone_weights_arr,
            tri_sources, tri_weights, bind_inv_mats, 1,
        )

        w_mats = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)])
        w_mats[1, 1, 3] = 0.5
        inv_mats = np.linalg.inv(w_mats)
        w_mats_col = np.ascontiguousarray(np.transpose(w_mats, (0, 2, 1)), dtype=np.float32)
        inv_mats_col = np.ascontiguousarray(np.transpose(inv_mats, (0, 2, 1)), dtype=np.float32)

        # 1. 初期フレーム: 全ボーン Dirty (dirty_bone_indices=None)
        sim.update_bone_transforms(w_mats_col, inv_mats_col, None)
        sim.step(0.0166667, 5)

        # 2. 静止フレーム: ポーズ変化なし (dirty_bone_indices=空配列 -> GPUディスパッチ完全スキップ)
        empty_dirty = np.empty(0, dtype=np.uint32)
        sim.update_bone_transforms(w_mats_col, inv_mats_col, empty_dirty)
        sim.step(0.0166667, 5)

        # 3. 差分変形フレーム: Bone1 のみ回転 (dirty_bone_indices=[1])
        theta = np.pi / 4.0
        r_mat = np.eye(4, dtype=np.float32)
        r_mat[0, 0] = np.cos(theta); r_mat[0, 1] = -np.sin(theta)
        r_mat[1, 0] = np.sin(theta); r_mat[1, 1] = np.cos(theta)
        w_mats[1] = np.eye(4, dtype=np.float32)
        w_mats[1, 1, 3] = 0.5
        w_mats[1] = w_mats[1] @ r_mat
        inv_mats = np.linalg.inv(w_mats)
        w_mats_col = np.ascontiguousarray(np.transpose(w_mats, (0, 2, 1)), dtype=np.float32)
        inv_mats_col = np.ascontiguousarray(np.transpose(inv_mats, (0, 2, 1)), dtype=np.float32)

        dirty_bone1 = np.array([1], dtype=np.uint32)
        sim.update_bone_transforms(w_mats_col, inv_mats_col, dirty_bone1)
        sim.step(0.0166667, 5)

        out_pos = np.zeros(len(cloth_verts) * 3, dtype=np.float32)
        sim.get_positions(out_pos)
        self.assertFalse(np.isnan(out_pos).any())
        self.assertFalse(np.isinf(out_pos).any())


if __name__ == "__main__":
    unittest.main()

