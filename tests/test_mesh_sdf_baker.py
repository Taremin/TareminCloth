"""
単一メッシュ直方体SDFベーカー (Mesh SDF) の単体テスト
"""

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
from taremin_cloth.engine.sdf_baker import (
    bake_mesh_sdf_from_data,
    estimate_mesh_sdf_info,
    MeshSdfBakeResult,
    save_cached_sdf,
    load_cached_sdf,
)


class TestMeshSdfBaker(unittest.TestCase):
    def setUp(self):
        if not taremin_cloth_core.is_gpu_available():
            self.skipTest("GPUが利用できない環境のためスキップします")

    def test_mesh_sdf_bake_cube(self):
        """単位立方体メッシュに対するGPUメッシュSDFベイクテスト"""
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

        t0 = time.time()
        result = bake_mesh_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            voxel_size=0.05,  # 5cm
            margin=0.2,
            friction=0.6,
            thickness=0.01,
            restitution=0.1,
        )
        elapsed = time.time() - t0

        self.assertIsNotNone(result)
        self.assertIsInstance(result, MeshSdfBakeResult)
        self.assertGreaterEqual(result.width, 8)
        self.assertGreaterEqual(result.height, 8)
        self.assertGreaterEqual(result.depth, 8)

        # テクスチャデータを np.float16 にデコード (形状: [D, H, W, 2])
        atlas_3d = np.frombuffer(result.texture_bytes, dtype=np.float16).reshape(
            (result.depth, result.height, result.width, 2)
        )
        dists = atlas_3d[:, :, :, 0].astype(np.float32)
        alphas = atlas_3d[:, :, :, 1].astype(np.float32)

        # 中心ボクセルは内部 (負のSDF)
        cz, cy, cx = result.depth // 2, result.height // 2, result.width // 2
        center_dist = dists[cz, cy, cx]
        self.assertLess(center_dist, 0.0, f"キューブ中心は負のSDF値であるべき: {center_dist}")

        # 端のボクセル (マージン領域) は外部 (正のSDF)
        corner_dist = dists[0, 0, 0]
        self.assertGreater(corner_dist, 0.0, f"マージン領域は正のSDF値であるべき: {corner_dist}")

        # Alphaチャンネルはすべて 1.0 (固定) であること
        self.assertAlmostEqual(float(alphas[cz, cy, cx]), 1.0, places=2)

        # パラメータが正しく設定されていること
        self.assertEqual(result.bone_info[16], 0.6)   # friction
        self.assertEqual(result.bone_info[17], 0.01)  # thickness
        self.assertEqual(result.bone_info[18], 0.1)   # restitution

        print(f"[テスト成功] GPUメッシュSDFベイク (立方体: {result.width}x{result.height}x{result.depth}): {elapsed*1000:.2f} ms")

    def test_mesh_sdf_bake_rectangular_aspect_ratio(self):
        """直方体（アスペクト比 1:2:4 の人体風プロポーション）に対する非等方解像度ベイクテスト"""
        # X: 0.4m (-0.2〜0.2), Y: 0.2m (-0.1〜0.1), Z: 1.6m (-0.8〜0.8)
        verts = np.array([
            [-0.2, -0.1, -0.8],
            [ 0.2, -0.1, -0.8],
            [ 0.2,  0.1, -0.8],
            [-0.2,  0.1, -0.8],
            [-0.2, -0.1,  0.8],
            [ 0.2, -0.1,  0.8],
            [ 0.2,  0.1,  0.8],
            [-0.2,  0.1,  0.8],
        ], dtype=np.float32)

        tris = np.array([
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ], dtype=np.int32)

        result = bake_mesh_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            voxel_size=0.02,  # 2cm
            margin=0.1,
        )

        self.assertIsNotNone(result)
        # Z軸（高さ1.6m）が最も解像度が高く、Y軸（奥行き0.2m）が最も解像度が低いはず
        self.assertGreater(result.depth, result.width)
        self.assertGreater(result.width, result.height)

        print(f"[テスト成功] 直方体SDFアスペクト比検証: X={result.width}, Y={result.height}, Z={result.depth}")

    def test_mesh_sdf_cache_save_and_load(self):
        """メッシュSDFのキャッシュ保存と再ロード整合性テスト"""
        verts = np.array([
            [-0.1, -0.1, -0.1],
            [ 0.1, -0.1, -0.1],
            [ 0.1,  0.1, -0.1],
            [-0.1,  0.1, -0.1],
            [-0.1, -0.1,  0.1],
            [ 0.1, -0.1,  0.1],
            [ 0.1,  0.1,  0.1],
            [-0.1,  0.1,  0.1],
        ], dtype=np.float32)
        tris = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)

        result = bake_mesh_sdf_from_data(verts, tris, voxel_size=0.05)
        self.assertIsNotNone(result)

        test_key = "test_mesh_sdf_cache_unit"
        save_cached_sdf(test_key, result)

        loaded = load_cached_sdf(test_key)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.width, result.width)
        self.assertEqual(loaded.height, result.height)
        self.assertEqual(loaded.depth, result.depth)
        self.assertEqual(len(loaded.texture_bytes), len(result.texture_bytes))
        np.testing.assert_array_almost_equal(loaded.bone_infos, result.bone_infos)

        # 後処理
        from taremin_cloth.engine.sdf_baker import get_cache_dir
        cache_file = os.path.join(get_cache_dir(), f"{test_key}.npz")
        if os.path.exists(cache_file):
            os.remove(cache_file)

    def test_mesh_sdf_collision_simulation(self):
        """ClothSimulatorにメッシュSDFコライダーを登録し、自由落下する布が立方体表面で止まることを検証"""
        # 立方体コライダー (-0.5〜0.5, 上面 z=0.5)
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
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ], dtype=np.int32)

        sdf_res = bake_mesh_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            voxel_size=0.05,
            margin=0.2,
            friction=0.3,
            thickness=0.01,
            restitution=0.0,
        )
        self.assertIsNotNone(sdf_res)

        # z=0.8 から落下する布頂点 (1点)
        cloth_verts = np.array([[0.0, 0.0, 0.8]], dtype=np.float32)
        sim = taremin_cloth_core.ClothSimulator(
            cloth_verts,
            np.empty((0, 2), dtype=np.uint32),
            thickness=0.01,  # cloth thickness: 1cm
        )

        # メッシュSDFコライダーの登録
        sim.set_bone_sdf_colliders(
            sdf_res.width,
            sdf_res.height,
            sdf_res.depth,
            sdf_res.texture_bytes,
            sdf_res.bone_infos,
        )

        # ワールド変換行列（単位行列）の登録
        identity_4x4 = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
        sim.update_bone_transforms(identity_4x4, identity_4x4)

        # 60ステップ（約1秒）シミュレーション
        for _ in range(60):
            sim.step(1.0 / 60.0, 20)

        res_verts = np.zeros(3, dtype=np.float32)
        sim.get_positions(res_verts)
        final_z = float(res_verts[2])

        # 立方体の上面は z=0.5、コライダー厚み 0.01、布厚み 0.01 -> 予想静止位置 z ~= 0.52
        print(f"[テスト成功] メッシュSDF衝突シミュレーション: 初期z=0.8 -> 最終z={final_z:.4f} (期待値 ~= 0.52)")
        self.assertGreater(final_z, 0.50, f"布が立方体表面を貫通してはならない (実測 z={final_z})")
        self.assertLess(final_z, 0.55, f"布が浮き上がりすぎてはならない (実測 z={final_z})")

    def test_large_mesh_sdf_bake_over_256mb(self):
        """256MBを超える大容量SDFバッファ（例: ボクセルサイズ2mm）が正常にベイクできるかのテスト"""
        # 0.8m 立方体メッシュ (ボクセルサイズ 2mm で約 300MB〜400MB)
        s = 0.4
        verts = np.array([
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
        ], dtype=np.float32)
        tris = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ], dtype=np.int32)

        result = bake_mesh_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            voxel_size=0.002,  # 2mm
            margin=0.02,
        )
        self.assertIsNotNone(result, "大容量SDFベイクが成功すること")
        total_mb = len(result.texture_bytes) / (1024 * 1024)
        print(f"[テスト成功] 大容量Mesh SDFベイク: {result.width}x{result.height}x{result.depth} ({total_mb:.1f} MB > 256MB)")
        self.assertGreater(total_mb, 256.0, "バッファサイズが256MBを超えていること")

    def test_auto_fit_vram_clamping(self):
        """VRAM上限予算（例: 256MB）を超えるボクセルサイズが指定された場合、自動で安全なサイズに調整されるテスト"""
        from taremin_cloth.engine.sdf_baker import compute_effective_voxel_size

        # サイズ 0.86m x 0.86m x 0.86m の立方体
        size = np.array([0.86, 0.86, 0.86], dtype=np.float32)

        # 2mm (0.002) を要求すると 約 320MB になる
        eff_v, is_clamped, raw_vram = compute_effective_voxel_size(
            size, requested_voxel_size=0.002, max_vram_mb=256, auto_scale=True
        )
        self.assertTrue(is_clamped, "256MB超過のためクランプされること")
        self.assertGreater(eff_v, 0.002, "ボクセルサイズが安全側にスケールアップされていること")
        self.assertGreater(raw_vram, 256.0, "元の要求VRAMは256MBを超えていること")

        # 調整後のVRAM容量を計算
        w = int(np.clip(np.ceil(size[0] / eff_v), 8, 2048))
        h = int(np.clip(np.ceil(size[1] / eff_v), 8, 2048))
        d = int(np.clip(np.ceil(size[2] / eff_v), 8, 2048))
        clamped_vram = (w * h * d * 4) / (1024 * 1024)
        self.assertLessEqual(clamped_vram, 256.0, f"調整後のVRAMは上限256MB以下であること (実測: {clamped_vram:.1f}MB)")
        print(f"[テスト成功] Auto Fit VRAM自動最適化: 2.0mm (推定{raw_vram:.1f}MB) -> {eff_v*1000:.2f}mm ({clamped_vram:.1f}MB <= 256MB)")

        # 上限を 512MB に引き上げた場合はクランプされないこと
        eff_v_512, is_clamped_512, _ = compute_effective_voxel_size(
            size, requested_voxel_size=0.002, max_vram_mb=512, auto_scale=True
        )
        self.assertFalse(is_clamped_512, "上限512MBなら320MBは収まるためクランプされないこと")
        self.assertAlmostEqual(eff_v_512, 0.002, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
