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
    extract_joint_mesh_indices,
    extract_limb_joint_mesh_indices,
    extract_hierarchy_joint_mesh_indices,
    extract_joint_mesh_from_sdf,
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

    def test_hybrid_joint_mesh_extraction(self):
        """ハイブリッドモードでの関節面抽出とAlphaフェードアウト、キャッシュ復元を検証"""
        verts, tris = create_cube_mesh(center=(0.0, 0.0, 0.0), size=1.0)
        n_verts = len(verts)

        # 頂点 0〜3 は Bone_A (1.0), 頂点 4〜7 は Bone_A (0.5) と Bone_B (0.5) のブレンド領域
        w_a = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        w_b = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        bone_weights = {"Bone_A": w_a, "Bone_B": w_b}
        bone_bind_matrices = {
            "Bone_A": np.eye(4, dtype=np.float32),
            "Bone_B": np.eye(4, dtype=np.float32),
        }

        # 1. ハイブリッド無効時
        res_standard = bake_bone_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            bone_weights=bone_weights,
            bone_bind_matrices=bone_bind_matrices,
            resolution=16,
            enable_joint_mesh=False,
        )
        self.assertEqual(len(res_standard.joint_face_indices), 0)

        # 2. ハイブリッド有効時 (threshold=0.85)
        res_hybrid = bake_bone_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            bone_weights=bone_weights,
            bone_bind_matrices=bone_bind_matrices,
            resolution=16,
            enable_joint_mesh=True,
            joint_weight_threshold=0.85,
        )
        # 頂点4〜7を含む三角形（Top面、Front面の一部、Back面の一部等）が抽出されること
        self.assertGreater(len(res_hybrid.joint_face_indices), 0)
        self.assertLessEqual(len(res_hybrid.joint_face_indices), len(tris))

        # 3. キャッシュの保存と復元
        sig = compute_mesh_signature(verts, tris, ["Bone_A", "Bone_B"], 16, enable_joint_mesh=True, joint_weight_threshold=0.85)
        save_cached_sdf(sig, res_hybrid)
        loaded = load_cached_sdf(sig)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded.joint_face_indices, res_hybrid.joint_face_indices)

    def test_extract_joint_mesh_indices_complete_coverage(self):
        """第2ウェイトによるブレンド全域抽出および剛体側へののりしろ（overlap_rings）の検証"""
        verts, tris = create_cube_mesh(center=(0.0, 0.0, 0.0), size=1.0)
        n_verts = len(verts)

        # 頂点 0〜3: Bone_A (1.0) 完全剛体
        # 頂点 4: Bone_A (0.95), Bone_B (0.05) - 従来の threshold=0.85 では除外されていた境界頂点
        # 頂点 5: Bone_A (0.50), Bone_B (0.50) - 中心ブレンド頂点
        # 頂点 6, 7: Bone_B (1.0) 完全剛体
        w_a = np.array([1.0, 1.0, 1.0, 1.0, 0.95, 0.50, 0.0, 0.0], dtype=np.float32)
        w_b = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.50, 1.0, 1.0], dtype=np.float32)
        bone_weights = {"Bone_A": w_a, "Bone_B": w_b}

        # 1. overlap_rings=0 (純粋なブレンド頂点 4, 5 のみを含む三角形)
        indices_ring0 = extract_joint_mesh_indices(
            verts, tris, bone_weights, overlap_rings=0, min_blend_weight=0.02
        )
        self.assertGreater(len(indices_ring0), 0)

        # 抽出された各面が頂点 4 または 5 を含んでいることを検証
        for f_idx in indices_ring0:
            face_verts = set(tris[f_idx])
            self.assertTrue(4 in face_verts or 5 in face_verts)

        # 2. overlap_rings=1 (剛体側の隣接頂点へ1リング拡張)
        indices_ring1 = extract_joint_mesh_indices(
            verts, tris, bone_weights, overlap_rings=1, min_blend_weight=0.02
        )
        # 1リング拡張により、より広範囲の面がカバーされること
        self.assertGreater(len(indices_ring1), len(indices_ring0))

        # 3. ボーンが1本のみの場合は空配列が返ること
        single_weights = {"Bone_A": np.ones(n_verts, dtype=np.float32)}
        indices_single = extract_joint_mesh_indices(verts, tris, single_weights)
        self.assertEqual(len(indices_single), 0)

        # 4. 微小ノイズ (w_second < min_blend_weight) では抽出されないこと
        noise_weights = {
            "Bone_A": np.array([1.0, 1.0, 1.0, 1.0, 0.995, 1.0, 1.0, 1.0], dtype=np.float32),
            "Bone_B": np.array([0.0, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        indices_noise = extract_joint_mesh_indices(verts, tris, noise_weights, min_blend_weight=0.02)
        self.assertEqual(len(indices_noise), 0)

    def test_smoothstep_alpha_fade(self):
        """Smoothstep減衰によるAlpha連続遷移の検証"""
        joint_weight_threshold = 0.85
        min_t = joint_weight_threshold * 0.4  # 0.34

        test_alphas = np.array([0.0, 0.2, 0.34, 0.5, 0.85, 1.0], dtype=np.float32)
        t = np.clip((test_alphas - min_t) / (joint_weight_threshold - min_t), 0.0, 1.0)
        fade = t * t * (3.0 - 2.0 * t)
        faded_alphas = test_alphas * fade

        # min_t (0.34) 以下では fade = 0.0
        self.assertEqual(faded_alphas[0], 0.0)
        self.assertEqual(faded_alphas[1], 0.0)
        self.assertEqual(faded_alphas[2], 0.0)

        # threshold (0.85) 以上では fade = 1.0
        self.assertAlmostEqual(faded_alphas[4], 0.85, places=5)
        self.assertAlmostEqual(faded_alphas[5], 1.0, places=5)

        # 中間値 (0.5) では滑らかに減衰 (0 < faded < 0.5)
        self.assertGreater(faded_alphas[3], 0.0)
        self.assertLess(faded_alphas[3], 0.5)

        # 単調増加性の検証
        self.assertTrue(np.all(np.diff(faded_alphas) >= 0.0))

    def test_extract_limb_joint_mesh(self):
        """大関節ペアテスト"""
        verts, tris = create_cube_mesh(center=(0.0, 0.0, 0.0), size=1.0)
        bw_a = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        bw_b = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        bone_weights = {
            "hips": bw_a,
            "upper_leg.L": bw_b,
        }
        bone_bind_matrices = {
            "hips": np.eye(4, dtype=np.float32),
            "upper_leg.L": np.eye(4, dtype=np.float32),
        }

        # ハイブリッド有効でベイク
        res = 16
        bake_res = bake_bone_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            bone_weights=bone_weights,
            bone_bind_matrices=bone_bind_matrices,
            resolution=res,
            enable_joint_mesh=True,
        )

        joint_faces = extract_limb_joint_mesh_indices(
            verts,
            tris,
            bone_weights,
            min_blend_weight=0.05,
        )

        self.assertIsInstance(joint_faces, np.ndarray)
        self.assertEqual(joint_faces.dtype, np.int32)
        self.assertGreater(len(joint_faces), 0)
        np.testing.assert_array_equal(bake_res.joint_face_indices, joint_faces)

    def test_extract_hierarchy_joint_mesh(self):
        """ボーン名不問の親子階層（Parent-Child）に基づく関節抽出およびペア辞書保持の検証"""
        verts, tris = create_cube_mesh(center=(0.0, 0.0, 0.0), size=1.0)
        # 英語名・日本語名に関係なく親子関係から自動抽出できることを検証
        bone_parent_map = {
            "CustomBone_Child": "CustomBone_Parent",
            "CustomBone_Parent": None,
        }
        bw_parent = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        bw_child = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        bone_weights = {
            "CustomBone_Parent": bw_parent,
            "CustomBone_Child": bw_child,
        }
        bone_bind_matrices = {
            "CustomBone_Parent": np.eye(4, dtype=np.float32),
            "CustomBone_Child": np.eye(4, dtype=np.float32),
        }

        # 1. extract_hierarchy_joint_mesh_indices 単体テスト
        all_faces, pair_dict = extract_hierarchy_joint_mesh_indices(
            verts, tris, bone_weights, bone_parent_map, min_blend_weight=0.05, min_verts_per_joint=2
        )
        self.assertGreater(len(all_faces), 0)
        self.assertIn(("CustomBone_Parent", "CustomBone_Child"), pair_dict)
        np.testing.assert_array_equal(all_faces, pair_dict[("CustomBone_Parent", "CustomBone_Child")])

        # 2. bake_bone_sdf_from_data による統合ベイクテスト
        bake_res = bake_bone_sdf_from_data(
            mesh_verts=verts,
            mesh_tris=tris,
            bone_weights=bone_weights,
            bone_bind_matrices=bone_bind_matrices,
            resolution=16,
            enable_joint_mesh=True,
            bone_parent_map=bone_parent_map,
        )
        self.assertEqual(len(bake_res.joint_face_indices), len(all_faces))
        self.assertIn(("CustomBone_Parent", "CustomBone_Child"), bake_res.joint_faces_by_pair)

        # 3. キャッシュ保存と復元によるペア辞書整合性の検証
        sig = compute_mesh_signature(verts, tris, list(bone_weights.keys()), 16, enable_joint_mesh=True)
        save_cached_sdf(sig, bake_res)
        loaded = load_cached_sdf(sig)
        self.assertIsNotNone(loaded)
        self.assertIn(("CustomBone_Parent", "CustomBone_Child"), loaded.joint_faces_by_pair)
        np.testing.assert_array_equal(
            loaded.joint_faces_by_pair[("CustomBone_Parent", "CustomBone_Child")],
            bake_res.joint_faces_by_pair[("CustomBone_Parent", "CustomBone_Child")]
        )


if __name__ == "__main__":
    unittest.main()


