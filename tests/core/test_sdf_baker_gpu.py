import os
import sys
import unittest
import time
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import taremin_cloth_core
from taremin_cloth.engine.sdf_baker import bake_bone_sdf_from_data, BoneSdfBakeResult


class TestSdfBakerGpu(unittest.TestCase):
    def setUp(self):
        if not taremin_cloth_core.is_gpu_available():
            self.skipTest("GPUが利用できない環境のためスキップします")

    def test_gpu_bone_sdf_bake_cube(self):
        """単位キューブに対するGPUボーンSDFベイクのテスト"""
        # 単位立方体メッシュ (8頂点, 12三角形)
        verts = np.array([
            [-0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5],
            [ 0.5,  0.5, -0.5],
            [-0.5,  0.5, -0.5],
            [-0.5, -0.5,  0.5],
            [ 0.5, -0.5,  0.5],
            [ 0.5,  0.5,  0.5],
            [-0.5,  0.5,  0.5],
        ], dtype=np.float32)

        tris = np.array([
            [0, 2, 1], [0, 3, 2],  # -Z
            [4, 5, 6], [4, 6, 7],  # +Z
            [0, 1, 5], [0, 5, 4],  # -Y
            [2, 3, 7], [2, 7, 6],  # +Y
            [0, 4, 7], [0, 7, 3],  # -X
            [1, 2, 6], [1, 6, 5],  # +X
        ], dtype=np.int32)

        bone_names = ["CubeBone"]
        bone_weights = {"CubeBone": np.ones(len(verts), dtype=np.float32)}
        bone_bind_matrices = {"CubeBone": np.eye(4, dtype=np.float32)}

        t0 = time.time()
        result = bake_bone_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            bone_weights=bone_weights,
            bone_bind_matrices=bone_bind_matrices,
            resolution=32,
            margin=0.2,
            weight_threshold=0.02,
        )
        elapsed = time.time() - t0

        self.assertIsInstance(result, BoneSdfBakeResult)
        self.assertEqual(result.width, 32)
        self.assertEqual(result.height, 32)
        self.assertEqual(result.depth, 32)
        self.assertEqual(len(result.bone_names), 1)
        self.assertEqual(result.bone_infos.shape, (1, 20))
        self.assertEqual(len(result.texture_bytes), 32 * 32 * 32 * 4)

        # テクスチャデータを np.float16 にデコード (形状: [D, H, W, 2])
        atlas_3d = np.frombuffer(result.texture_bytes, dtype=np.float16).reshape((32, 32, 32, 2))
        dists = atlas_3d[:, :, :, 0].astype(np.float32)
        alphas = atlas_3d[:, :, :, 1].astype(np.float32)

        # ボーンのローカルAABB情報を取得
        b_info = result.bone_infos[0]
        local_min = b_info[0:3]
        local_max = b_info[4:7]

        # キューブの中心 (0, 0, 0) は立方体の内側なので、SDF値は負であること
        # 中心ボクセルのインデックス
        c_idx = 16
        center_dist = dists[c_idx, c_idx, c_idx]
        self.assertLess(center_dist, 0.0, f"キューブ中心のSDF値は負（内部）であるべきです: {center_dist}")

        # キューブの外側の隅 (margin領域) は正であること
        corner_dist = dists[0, 0, 0]
        self.assertGreater(corner_dist, 0.0, f"マージン領域のSDF値は正（外部）であるべきです: {corner_dist}")

        # アルファ（ボーンウェイト）はすべて1.0近傍であること
        self.assertGreater(np.mean(alphas), 0.9)

        print(f"[テスト成功] GPUボーンSDFベイク (32^3): {elapsed*1000:.2f} ms (中心距離: {center_dist:.3f}, 隅距離: {corner_dist:.3f})")

    def test_gpu_bone_sdf_bake_multi_bone_performance(self):
        """複数ボーン（10ボーン）に対するGPU並列ベイク性能テスト"""
        # シリンダー状メッシュ (約1000ポリゴン)
        n_rings = 20
        n_segs = 16
        verts = []
        for r in range(n_rings):
            z = (r / (n_rings - 1)) * 2.0 - 1.0
            for s in range(n_segs):
                th = (s / n_segs) * 2.0 * np.pi
                verts.append([0.3 * np.cos(th), 0.3 * np.sin(th), z])
        verts = np.array(verts, dtype=np.float32)

        tris = []
        for r in range(n_rings - 1):
            for s in range(n_segs):
                s_next = (s + 1) % n_segs
                v00 = r * n_segs + s
                v01 = r * n_segs + s_next
                v10 = (r + 1) * n_segs + s
                v11 = (r + 1) * n_segs + s_next
                tris.append([v00, v10, v01])
                tris.append([v01, v10, v11])
        tris = np.array(tris, dtype=np.int32)

        # 10本のボーン
        n_bones = 10
        bone_names = [f"Bone_{i}" for i in range(n_bones)]
        bone_weights = {}
        bone_bind_matrices = {}

        for i, b_name in enumerate(bone_names):
            b_z = (i / (n_bones - 1)) * 2.0 - 1.0
            # z座標の近さに基づくウェイト分布
            dist_z = np.abs(verts[:, 2] - b_z)
            w = np.clip(1.0 - dist_z / 0.4, 0.0, 1.0).astype(np.float32)
            bone_weights[b_name] = w
            bone_bind_matrices[b_name] = np.eye(4, dtype=np.float32)

        # 64^3 の高解像度で 10 本のボーンを一括ベイク
        t0 = time.time()
        result = bake_bone_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            bone_weights=bone_weights,
            bone_bind_matrices=bone_bind_matrices,
            resolution=64,
            margin=0.2,
            weight_threshold=0.02,
        )
        elapsed = time.time() - t0

        self.assertIsInstance(result, BoneSdfBakeResult)
        self.assertEqual(len(result.bone_names), 10)
        self.assertGreater(len(result.texture_bytes), 0)

        print(f"[性能テスト成功] 10ボーン x 64^3 GPU並列ベイク所要時間: {elapsed:.3f} 秒 (約 {elapsed*1000:.1f} ms)")
        
        # ソフトウェアアダプタ（WARP/Basic Render Driver/llvmpipe等）やCI環境ではCPUエミュレーション実行のため閾値を緩和
        dev_name = taremin_cloth_core.get_gpu_device_name().lower()
        is_software = any(name in dev_name for name in ["basic render", "warp", "llvmpipe", "lavapipe", "software", "cpu"])
        is_ci = os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"
        max_allowed = 60.0 if (is_software or is_ci) else 2.0

        self.assertLess(elapsed, max_allowed, f"10ボーンの64^3ベイクが{max_allowed}秒未満で完了すること (デバイス: {dev_name})")

    def test_open_boundary_detection_and_capping(self):
        """開放境界チェッカー (detect_open_boundaries) と自動キャップ閉鎖のテスト"""
        from taremin_cloth.utils.validation import detect_open_boundaries, close_open_boundary_loops_numpy

        # 上下端が開いた正8角柱 (首切断端モデル)
        n_seg = 8
        th = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
        bot = np.column_stack([0.045 * np.cos(th), 0.045 * np.sin(th), np.zeros(n_seg)])
        top = np.column_stack([0.055 * np.cos(th), 0.055 * np.sin(th), np.full(n_seg, 0.10)])
        verts = np.vstack([bot, top]).astype(np.float32)

        tris = []
        for i in range(n_seg):
            b0 = i
            b1 = (i + 1) % n_seg
            t0 = n_seg + i
            t1 = n_seg + ((i + 1) % n_seg)
            tris.append([b0, b1, t1])
            tris.append([b0, t1, t0])
        tris = np.array(tris, dtype=np.int32)

        rep = detect_open_boundaries(tris, num_verts=len(verts))
        self.assertTrue(rep.has_open_boundaries)
        self.assertEqual(rep.open_edge_count, n_seg * 2)
        self.assertEqual(rep.open_loop_count, 2)
        self.assertEqual(rep.non_manifold_edge_count, 0)

        c_verts, c_tris, closed_cnt = close_open_boundary_loops_numpy(verts, tris)
        self.assertEqual(closed_cnt, 2)
        self.assertEqual(len(c_verts), len(verts) + 2)
        self.assertEqual(len(c_tris), len(tris) + n_seg * 2)

        rep_after = detect_open_boundaries(c_tris, num_verts=len(c_verts))
        self.assertFalse(rep_after.has_open_boundaries)
        self.assertEqual(rep_after.open_edge_count, 0)


if __name__ == "__main__":
    unittest.main()
