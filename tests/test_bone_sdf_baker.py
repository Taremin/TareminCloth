"""
taremin_cloth.engine.sdf_baker モジュールのスタンドアロン・ユニットテスト
ボーン局所SDFベイク処理、内外符号、Alpha減衰、キャッシュ保存・復元の検証
"""

import os
import sys
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

from taremin_cloth.engine.sdf_baker import (
    bake_bone_sdf_from_data,
    BoneSdfBakeResult,
    compute_mesh_signature,
    save_cached_sdf,
    load_cached_sdf,
    clear_all_cached_sdf,
)


def create_cube_mesh(center=(0.0, 0.0, 0.0), size=1.0):
    """幅 size の立方体メッシュ（8頂点, 12三角形）を生成する"""
    cx, cy, cz = center
    h = size * 0.5
    verts = np.array([
        [cx - h, cy - h, cz - h],
        [cx + h, cy - h, cz - h],
        [cx + h, cy + h, cz - h],
        [cx - h, cy + h, cz - h],
        [cx - h, cy - h, cz + h],
        [cx + h, cy - h, cz + h],
        [cx + h, cy + h, cz + h],
        [cx - h, cy + h, cz + h],
    ], dtype=np.float32)

    # 6面 x 2三角形 = 12三角形 (外向き法線)
    tris = np.array([
        # Bottom (-Z)
        [0, 2, 1], [0, 3, 2],
        # Top (+Z)
        [4, 5, 6], [4, 6, 7],
        # Front (-Y)
        [0, 1, 5], [0, 5, 4],
        # Back (+Y)
        [2, 3, 7], [2, 7, 6],
        # Left (-X)
        [0, 4, 7], [0, 7, 3],
        # Right (+X)
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)
    return verts, tris


class TestBoneSdfBaker(unittest.TestCase):
    def setUp(self):
        clear_all_cached_sdf()

    def tearDown(self):
        clear_all_cached_sdf()

    def test_single_bone_sdf_bake(self):
        """単一ボーンに対するSDFベイクの幾何的妥当性を検証"""
        verts, tris = create_cube_mesh(center=(0.0, 0.0, 0.0), size=1.0)
        n_verts = len(verts)

        # 全頂点がボーン"Bone_Root"にウェイト1.0でアタッチ
        bone_weights = {"Bone_Root": np.ones(n_verts, dtype=np.float32)}
        bone_bind_matrices = {"Bone_Root": np.eye(4, dtype=np.float32)}

        res = 16  # テスト用に小型解像度
        result = bake_bone_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            bone_weights=bone_weights,
            bone_bind_matrices=bone_bind_matrices,
            resolution=res,
            margin=0.2,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.width, res)
        self.assertEqual(result.height, res)
        self.assertEqual(result.depth, res)
        self.assertEqual(len(result.bone_names), 1)
        self.assertEqual(result.bone_names[0], "Bone_Root")

        # テクスチャバイト列を (depth, height, width, 2) の float16 配列に復元
        arr = np.frombuffer(result.texture_bytes, dtype=np.float16).reshape((res, res, res, 2))
        dist_field = arr[:, :, :, 0].astype(np.float32)
        weight_field = arr[:, :, :, 1].astype(np.float32)

        # 1. 中心部 (res//2, res//2, res//2) は立方体内部なので符号が負であること
        mid = res // 2
        center_dist = dist_field[mid, mid, mid]
        self.assertLess(center_dist, 0.0, f"立方体内部の中心距離は負であるべき: {center_dist}")

        # 2. 立方体の表面境界付近での距離値がほぼ0、外部の端 (0, 0, 0) では正であること
        corner_dist = dist_field[0, 0, 0]
        self.assertGreater(corner_dist, 0.0, f"AABB外側境界の距離は正であるべき: {corner_dist}")

        # 3. ウェイトフィールド（Alpha）が内部で 1.0 に近いこと
        center_weight = weight_field[mid, mid, mid]
        self.assertGreater(center_weight, 0.8, f"中心部ウェイトは1.0に近いべき: {center_weight}")

        # 4. bone_infos 配列の形状と値の妥当性
        self.assertEqual(result.bone_infos.shape, (1, 20))
        b_info = result.bone_infos[0]
        aabb_min = b_info[0:3]
        aabb_max = b_info[4:7]
        # 立方体サイズ1.0にマージン0.2が付くので、[-0.7, 0.7]程度になるはず
        self.assertLess(aabb_min[0], -0.5)
        self.assertGreater(aabb_max[0], 0.5)
        self.assertEqual(b_info[3], 0.0)  # slot_z_offset
        self.assertEqual(b_info[7], float(res))  # slot_depth

    def test_multi_bone_atlas_packing(self):
        """複数ボーン時の3Dアトラスパッキング（深さ方向の連結）の検証"""
        # ボーンA: [-1, 0, 0] 付近の立方体
        # ボーンB: [ 1, 0, 0] 付近の立方体
        verts_a, tris_a = create_cube_mesh(center=(-1.0, 0.0, 0.0), size=0.8)
        verts_b, tris_b = create_cube_mesh(center=(1.0, 0.0, 0.0), size=0.8)

        verts = np.vstack([verts_a, verts_b])
        tris = np.vstack([tris_a, tris_b + len(verts_a)])

        w_a = np.zeros(len(verts), dtype=np.float32)
        w_a[:len(verts_a)] = 1.0
        w_b = np.zeros(len(verts), dtype=np.float32)
        w_b[len(verts_a):] = 1.0

        bone_weights = {"Bone_Left": w_a, "Bone_Right": w_b}
        bone_bind_matrices = {"Bone_Left": np.eye(4, dtype=np.float32), "Bone_Right": np.eye(4, dtype=np.float32)}

        res = 16
        result = bake_bone_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            bone_weights=bone_weights,
            bone_bind_matrices=bone_bind_matrices,
            resolution=res,
        )

        self.assertIsNotNone(result)
        # 3D Brick Atlas 仕様: 2ボーン時は cols=2, rows=1, layers=1
        self.assertEqual(result.width, res * 2)   # width = 32
        self.assertEqual(result.height, res)      # height = 16
        self.assertEqual(result.depth, res)       # depth = 16
        self.assertEqual(len(result.bone_names), 2)

        # 各ボーンの b_idx (3) と res (7) の検証
        self.assertEqual(result.bone_infos[0, 3], 0.0)
        self.assertEqual(result.bone_infos[0, 7], float(res))
        self.assertEqual(result.bone_infos[1, 3], 1.0)
        self.assertEqual(result.bone_infos[1, 7], float(res))

    def test_cache_save_and_load(self):
        """SDFベイク結果のディスクキャッシュ保存・復元・消去の検証"""
        verts, tris = create_cube_mesh(center=(0.0, 0.0, 0.0), size=1.0)
        bone_weights = {"Bone_Root": np.ones(len(verts), dtype=np.float32)}
        bone_bind_matrices = {"Bone_Root": np.eye(4, dtype=np.float32)}

        res = 16
        sig = compute_mesh_signature(verts, tris, ["Bone_Root"], res)

        # 初期状態ではキャッシュなし
        self.assertIsNone(load_cached_sdf(sig))

        result = bake_bone_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            bone_weights=bone_weights,
            bone_bind_matrices=bone_bind_matrices,
            resolution=res,
        )

        # キャッシュに保存
        save_cached_sdf(sig, result)

        # キャッシュから復元
        loaded = load_cached_sdf(sig)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.width, result.width)
        self.assertEqual(loaded.height, result.height)
        self.assertEqual(loaded.depth, result.depth)
        self.assertEqual(loaded.bone_names, result.bone_names)
        self.assertEqual(loaded.texture_bytes, result.texture_bytes)
        np.testing.assert_allclose(loaded.bone_infos, result.bone_infos)
        np.testing.assert_allclose(loaded.bind_matrices, result.bind_matrices)

        # キャッシュ消去
        deleted = clear_all_cached_sdf()
        self.assertGreaterEqual(deleted, 1)
        self.assertIsNone(load_cached_sdf(sig))


if __name__ == "__main__":
    unittest.main()
