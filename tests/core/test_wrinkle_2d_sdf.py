"""
円柱UV展開2D-SDFテクスチャベイク機能の単体テスト
Blender非依存で動作し、円柱UV射影、周期的2D-SDF距離変換、RGBAテクスチャ生成の整合性を検証します。
"""

import math
import unittest
import numpy as np

from taremin_cloth.engine.wrinkle_field import (
    WrinkleTexture2D,
    bake_wrinkle_2d_sdf_texture,
)


class TestWrinkle2DSdf(unittest.TestCase):
    def test_circular_curves_bake(self):
        """円形山・谷カーブから2D-SDFテクスチャが正確に生成されることを検証"""
        num_pts = 64
        thetas = np.linspace(0.0, 2.0 * math.pi, num_pts, endpoint=False)
        r_col = 0.05
        r_cloth = 0.06

        # 谷カーブ: Z=-0.02, 半径 5.2cm
        root_z = np.full_like(thetas, -0.02)
        root_r = np.full_like(thetas, r_col + 0.002)
        root_pts = np.stack([
            root_r * np.cos(thetas),
            root_r * np.sin(thetas),
            root_z
        ], axis=-1).astype(np.float32)

        # 山カーブ: Z=+0.03, 半径 7.0cm
        crest_z = np.full_like(thetas, 0.03)
        crest_r = np.full_like(thetas, r_cloth + 0.010)
        crest_pts = np.stack([
            crest_r * np.cos(thetas),
            crest_r * np.sin(thetas),
            crest_z
        ], axis=-1).astype(np.float32)

        tex = bake_wrinkle_2d_sdf_texture(
            crest_curves=[crest_pts],
            root_curves=[root_pts],
            origin=[0.0, 0.0, 0.0],
            axis=[0.0, 0.0, 1.0],
            normal=[1.0, 0.0, 0.0],
            width=256,
            height=128,
            influence_radius=0.03,
            bone_radius=r_col,
            cloth_radius=r_cloth,
        )

        self.assertIsInstance(tex, WrinkleTexture2D)
        self.assertEqual(tex.width, 256)
        self.assertEqual(tex.height, 128)
        self.assertEqual(len(tex.texture_bytes), 256 * 128 * 4)

        rgba = tex.rgba_array
        self.assertEqual(rgba.shape, (128, 256, 4))

        # (1) 谷カーブの位置 (Z = -0.02) において、Rチャンネル (谷ウェイト) が最大 (255近傍) であること
        # V座標を計算
        v_root = ((-0.02) - tex.z_min) / (tex.z_max - tex.z_min)
        y_root = int(round(v_root * (tex.height - 1)))
        self.assertGreaterEqual(np.mean(rgba[y_root, :, 0]), 200, "谷カーブ位置のRチャンネルが高いこと")

        # (2) 山カーブの位置 (Z = +0.03) において、Gチャンネル (山ウェイト) が最大 (255近傍) であること
        v_crest = ((0.03) - tex.z_min) / (tex.z_max - tex.z_min)
        y_crest = int(round(v_crest * (tex.height - 1)))
        self.assertGreaterEqual(np.mean(rgba[y_crest, :, 1]), 200, "山カーブ位置のGチャンネルが高いこと")

        # (3) 円周方向 (U=0 と U=width-1) の周期性・シームレス性の検証
        # 最も外側の列同士の差分が極めて小さいこと
        diff_u_edge = np.abs(rgba[:, 0, :].astype(int) - rgba[:, -1, :].astype(int))
        self.assertLess(np.max(diff_u_edge), 15, "U軸両端の周期境界でテクスチャ値がシームレスであること")

    def test_y_branch_curves_bake(self):
        """「Y字型」分岐シワカーブが平均化で潰れずに2D-SDFとして正常にベイクされることを検証"""
        # 幹: Z=-0.03 から Z=0.0 (theta=pi)
        # 枝1: Z=0.0 から Z=+0.03 (theta=pi -> pi/2)
        # 枝2: Z=0.0 から Z=+0.03 (theta=pi -> 3pi/2)
        r = 0.06
        stem_z = np.linspace(-0.03, 0.0, 16)
        stem_th = np.full_like(stem_z, math.pi)
        stem_pts = np.stack([r * np.cos(stem_th), r * np.sin(stem_th), stem_z], axis=-1).astype(np.float32)

        branch1_z = np.linspace(0.0, 0.03, 16)
        branch1_th = np.linspace(math.pi, 0.5 * math.pi, 16)
        branch1_pts = np.stack([r * np.cos(branch1_th), r * np.sin(branch1_th), branch1_z], axis=-1).astype(np.float32)

        branch2_z = np.linspace(0.0, 0.03, 16)
        branch2_th = np.linspace(math.pi, 1.5 * math.pi, 16)
        branch2_pts = np.stack([r * np.cos(branch2_th), r * np.sin(branch2_th), branch2_z], axis=-1).astype(np.float32)

        tex = bake_wrinkle_2d_sdf_texture(
            crest_curves=[stem_pts, branch1_pts, branch2_pts],
            root_curves=[],
            origin=[0.0, 0.0, 0.0],
            axis=[0.0, 0.0, 1.0],
            width=256,
            height=128,
            influence_radius=0.02,
        )

        rgba = tex.rgba_array
        # Gチャンネル (山ウェイト) の非ゼロ画素数が十分にあること (Y字全体がプロットされている)
        active_pixels = np.count_nonzero(rgba[:, :, 1] > 100)
        self.assertGreater(active_pixels, 500, "Y字シワが正常に2D-SDF領域を形成していること")

    def test_per_curve_strength_and_influence_radius(self):
        """カーブごとの個別強度 (strength) と個別影響半径 (influence_radius) の反映を検証"""
        from taremin_cloth.engine.wrinkle_field import WrinkleCurveItem

        num_pts = 64
        thetas = np.linspace(0.0, 2.0 * math.pi, num_pts, endpoint=False)
        r = 0.06

        # カーブ A (広くて強い山): Z = +0.05, strength = 1.0, influence_radius = 0.04m
        cA_z = np.full_like(thetas, 0.05)
        cA_pts = np.stack([r * np.cos(thetas), r * np.sin(thetas), cA_z], axis=-1).astype(np.float32)
        item_A = WrinkleCurveItem(points=cA_pts, strength=1.0, influence_radius=0.04)

        # カーブ B (狭くて弱い山): Z = -0.05, strength = 0.5, influence_radius = 0.015m
        cB_z = np.full_like(thetas, -0.05)
        cB_pts = np.stack([r * np.cos(thetas), r * np.sin(thetas), cB_z], axis=-1).astype(np.float32)
        item_B = WrinkleCurveItem(points=cB_pts, strength=0.5, influence_radius=0.015)

        tex = bake_wrinkle_2d_sdf_texture(
            crest_curves=[item_A, item_B],
            root_curves=[],
            origin=[0.0, 0.0, 0.0],
            axis=[0.0, 0.0, 1.0],
            width=256,
            height=256,
            bone_radius=0.05,
            cloth_radius=r,
        )

        rgba = tex.rgba_array
        # V座標を計算
        v_A = (0.05 - tex.z_min) / (tex.z_max - tex.z_min)
        y_A = int(round(v_A * (tex.height - 1)))

        v_B = (-0.05 - tex.z_min) / (tex.z_max - tex.z_min)
        y_B = int(round(v_B * (tex.height - 1)))

        peak_A = float(np.mean(rgba[y_A, :, 1]))
        peak_B = float(np.mean(rgba[y_B, :, 1]))

        # (1) 強度の検証: カーブAは約255、カーブBは約127 (0.5倍)
        self.assertGreaterEqual(peak_A, 220, "カーブAのピーク強度が約1.0 (255) であること")
        self.assertGreaterEqual(peak_B, 100, "カーブBのピーク強度が約0.5 (127) であること")
        self.assertLessEqual(peak_B, 150, "カーブBのピーク強度が上限を超えないこと")
        self.assertGreater(peak_A, peak_B * 1.5, "カーブAの強度がカーブBより明らかに高いこと")

        # (2) 影響半径（グラデーション幅）の検証:
        # カーブA (0.04m) の影響幅（G > 30 の行数）が、カーブB (0.015m) の影響幅より明らかに広いこと
        rows_A = np.count_nonzero(np.mean(rgba[y_A-30:y_A+30, :, 1], axis=1) > 30)
        rows_B = np.count_nonzero(np.mean(rgba[y_B-30:y_B+30, :, 1], axis=1) > 30)
        self.assertGreater(rows_A, rows_B * 1.5, "カーブAの影響行数がカーブBより明らかに広いこと")


if __name__ == "__main__":
    unittest.main()
