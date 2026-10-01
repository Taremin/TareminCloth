"""
ドレープガイド幾何プロファイル抽出の単体テスト
Blender非依存で動作し、円柱座標射影、補間、GPUバイナリ生成の精度を検証します。
"""

import math
import unittest
import numpy as np

from taremin_cloth.engine.wrinkle_field import (
    WrinkleProfile,
    WrinkleFieldParams,
    build_orthonormal_basis,
    project_points_to_cylindrical,
    interpolate_curve_to_profile,
    build_wrinkle_profile_from_curves,
)


class TestWrinkleProfile(unittest.TestCase):
    def test_orthonormal_basis(self):
        """直交基底の正規性・直交性を検証"""
        axis = [0.0, 1.0, 0.0]
        a, n, bn = build_orthonormal_basis(axis)
        self.assertAlmostEqual(np.linalg.norm(a), 1.0, places=5)
        self.assertAlmostEqual(np.linalg.norm(n), 1.0, places=5)
        self.assertAlmostEqual(np.linalg.norm(bn), 1.0, places=5)
        self.assertAlmostEqual(np.dot(a, n), 0.0, places=5)
        self.assertAlmostEqual(np.dot(a, bn), 0.0, places=5)
        self.assertAlmostEqual(np.dot(n, bn), 0.0, places=5)

    def test_cylindrical_projection(self):
        """円柱座標系への射影が幾何学的に正確であることを検証"""
        origin = [0.0, 0.0, 0.0]
        axis = [0.0, 0.0, 1.0]      # Z軸がボーン軸
        normal = [1.0, 0.0, 0.0]    # X軸がθ=0基準

        # X=0.05, Y=0.0, Z=0.1 -> theta=0, z=0.1, r=0.05
        # X=0.0, Y=0.05, Z=0.2 -> theta=pi/2, z=0.2, r=0.05
        pts = np.array([
            [0.05, 0.0, 0.1],
            [0.0, 0.05, 0.2],
            [-0.05, 0.0, 0.3],
            [0.0, -0.05, 0.4],
        ], dtype=np.float32)

        thetas, zs, rs = project_points_to_cylindrical(pts, origin, axis, normal)

        self.assertAlmostEqual(thetas[0], 0.0, places=4)
        self.assertAlmostEqual(thetas[1], math.pi * 0.5, places=4)
        self.assertAlmostEqual(thetas[2], math.pi, places=4)
        self.assertAlmostEqual(thetas[3], math.pi * 1.5, places=4)

        np.testing.assert_allclose(zs, [0.1, 0.2, 0.3, 0.4], atol=1e-5)
        np.testing.assert_allclose(rs, [0.05, 0.05, 0.05, 0.05], atol=1e-5)

    def test_circular_curve_interpolation(self):
        """円形カーブの補間が全周で滑らかに抽出されることを検証"""
        origin = [0.0, 0.0, 0.0]
        axis = [0.0, 0.0, 1.0]
        normal = [1.0, 0.0, 0.0]

        # 肘の内側 (theta=pi) で z が波打つカーブ: z(theta) = 0.1 + 0.02 * cos(theta)
        num_pts = 32
        thetas = np.linspace(0.0, 2.0 * math.pi, num_pts, endpoint=False)
        r = 0.06
        zs = 0.1 + 0.02 * np.cos(thetas)
        xs = r * np.cos(thetas)
        ys = r * np.sin(thetas)
        curve_pts = np.stack([xs, ys, zs], axis=-1).astype(np.float32)

        num_samples = 64
        z_prof, r_prof, w_prof = interpolate_curve_to_profile(
            curve_pts, origin, axis, normal, num_samples=num_samples, is_closed=True
        )

        self.assertEqual(len(z_prof), num_samples)
        self.assertEqual(len(r_prof), num_samples)
        # 全周でウェイト 1.0
        np.testing.assert_allclose(w_prof, 1.0, atol=1e-5)
        # 半径が一定
        np.testing.assert_allclose(r_prof, 0.06, atol=1e-4)
        # theta=0 で z=0.12, theta=pi で z=0.08
        self.assertAlmostEqual(z_prof[0], 0.12, places=3)
        self.assertAlmostEqual(z_prof[num_samples // 2], 0.08, places=3)

    def test_wrinkle_profile_to_gpu_bytes(self):
        """GPUバイナリレイアウト (32バイト/サンプル) の整合性検証"""
        num_samples = 4
        v_z = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        v_r = np.array([0.04, 0.045, 0.05, 0.055], dtype=np.float32)
        c_z = np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float32)
        c_r = np.array([0.06, 0.07, 0.08, 0.09], dtype=np.float32)
        v_w = np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float32)
        c_w = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        profile = WrinkleProfile(
            valley_z=v_z,
            crest_radius=c_r,
            valley_weight=v_w,
            crest_weight=c_w,
            valley_radius=v_r,
            crest_z=c_z,
        )

        raw_bytes = profile.to_gpu_bytes()
        # 4 samples * 8 floats * 4 bytes = 128 bytes
        self.assertEqual(len(raw_bytes), 128)

        # 復元して検証
        unpacked = np.frombuffer(raw_bytes, dtype=np.float32).reshape(num_samples, 8)
        np.testing.assert_allclose(unpacked[:, 0], v_z)
        np.testing.assert_allclose(unpacked[:, 1], v_r)
        np.testing.assert_allclose(unpacked[:, 2], c_z)
        np.testing.assert_allclose(unpacked[:, 3], c_r)
        np.testing.assert_allclose(unpacked[:, 4], v_w)
        np.testing.assert_allclose(unpacked[:, 5], c_w)
        np.testing.assert_allclose(unpacked[:, 6], 0.0)
        np.testing.assert_allclose(unpacked[:, 7], 0.0)

    def test_crest_and_root_integration(self):
        """山カーブと谷カーブを統合した WrinkleProfile の生成テスト"""
        origin = [0.0, 0.0, 0.0]
        axis = [0.0, 0.0, 1.0]

        # 谷カーブ (Z=0.10, R=0.05)
        th = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        root_pts = np.stack([0.05 * np.cos(th), 0.05 * np.sin(th), np.full_like(th, 0.10)], axis=-1)

        # 山カーブ (Z=0.12, R=0.07)
        crest_pts = np.stack([0.07 * np.cos(th), 0.07 * np.sin(th), np.full_like(th, 0.12)], axis=-1)

        profile = build_wrinkle_profile_from_curves(
            crest_curves=[crest_pts],
            root_curves=[root_pts],
            origin=origin,
            axis=axis,
            num_samples=32,
            base_radius=0.05
        )

        self.assertEqual(profile.num_samples, 32)
        # 谷のターゲットZが0.10, 谷のターゲット半径が0.05
        np.testing.assert_allclose(profile.valley_z, 0.10, atol=1e-4)
        np.testing.assert_allclose(profile.valley_radius, 0.05, atol=1e-4)
        # 山のターゲットZが0.12, 山のターゲット半径が0.07
        np.testing.assert_allclose(profile.crest_z, 0.12, atol=1e-4)
        np.testing.assert_allclose(profile.crest_radius, 0.07, atol=1e-4)
        # ウェイトが有効
        self.assertTrue(np.all(profile.valley_weight > 0.0))
        self.assertTrue(np.all(profile.crest_weight > 0.0))

    def test_metaball_blending_curves(self):
        """複数カーブが重なる箇所でのメタボール的ポテンシャル合成（加重平均＋合算重み）の検証"""
        origin = [0.0, 0.0, 0.0]
        axis = [0.0, 0.0, 1.0]

        # 2本の山カーブ: 一方は Z=0.10, R=0.06、もう一方は Z=0.20, R=0.08
        th = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
        curve1 = np.stack([0.06 * np.cos(th), 0.06 * np.sin(th), np.full_like(th, 0.10)], axis=-1)
        curve2 = np.stack([0.08 * np.cos(th), 0.08 * np.sin(th), np.full_like(th, 0.20)], axis=-1)

        profile = build_wrinkle_profile_from_curves(
            crest_curves=[curve1, curve2],
            root_curves=[],
            origin=origin,
            axis=axis,
            num_samples=32,
            base_radius=0.05
        )

        # 2本のカーブが全周で等しい重み（1.0）を持つため、合成された Z は中間の 0.15、R は中間の 0.07
        np.testing.assert_allclose(profile.crest_z, 0.15, atol=1e-4)
        np.testing.assert_allclose(profile.crest_radius, 0.07, atol=1e-4)
        # 重みは 1.0 に飽和クリップ
        np.testing.assert_allclose(profile.crest_weight, 1.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
