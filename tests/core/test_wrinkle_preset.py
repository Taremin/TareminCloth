"""
シワフィールドプリセット管理モジュール (wrinkle_preset.py) の単体テスト
Blender非依存で動作し、プリセット定義、ボーン正規化・実体化のラウンドトリップ、JSON保存・読込を検証します。
"""

import math
import os
import shutil
import tempfile
import unittest
import numpy as np

from taremin_cloth.engine.wrinkle_preset import (
    WrinklePreset,
    WrinklePresetCurve,
    get_builtin_presets,
    normalize_curves_to_preset,
    instantiate_preset_on_bone,
    save_preset_to_json,
    load_preset_from_json,
)
from taremin_cloth.engine.wrinkle_field import WrinkleCurveItem


class TestWrinklePreset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_builtin_presets_structure(self):
        """組み込みプリセット群が正しくロードされ、整合性のあるカーブ情報を持つことを検証"""
        presets = get_builtin_presets()
        self.assertIn("cinch_single", presets)
        self.assertIn("puff_single", presets)
        self.assertIn("pinch_and_puff", presets)
        self.assertIn("accordion_double", presets)
        self.assertIn("y_branch_joint", presets)

        for key, p in presets.items():
            self.assertIsInstance(p, WrinklePreset)
            self.assertGreater(len(p.curves), 0, f"Preset '{key}' must have at least 1 curve")
            for c in p.curves:
                self.assertIn(c.type, ("crest", "root"))
                self.assertEqual(len(c.thetas), len(c.rel_zs))
                self.assertEqual(len(c.thetas), len(c.rel_rs))
                self.assertGreaterEqual(c.strength, 0.0)
                self.assertGreater(c.influence_radius, 0.0)

    def test_normalize_and_instantiate_roundtrip(self):
        """ワールド座標カーブ群からプリセット化し、同一ボーンへ実体化した際に座標が完全一致することを検証"""
        # 傾いたボーン: Head(0.1, 0.2, 0.3) -> Tail(0.1, 0.5, 0.7)
        b_head = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        b_tail = np.array([0.1, 0.5, 0.7], dtype=np.float32)
        b_radius = 0.05
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # 谷カーブ1本と山カーブ1本をボーン周囲に作成
        axis = b_tail - b_head
        L = float(np.linalg.norm(axis))
        axis_norm = axis / L
        binorm = np.cross(axis_norm, normal)
        binorm /= np.linalg.norm(binorm)

        thetas = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=True)
        # 谷: z = 0.4 * L, r = 1.05 * b_radius
        pts_root = []
        for th in thetas:
            rad = math.cos(th) * normal + math.sin(th) * binorm
            pts_root.append(b_head + axis_norm * (0.4 * L) + rad * (1.05 * b_radius))
        item_root = WrinkleCurveItem(np.array(pts_root, dtype=np.float32), strength=1.2, influence_radius=0.02)

        # 山: z = 0.6 * L, r = 1.35 * b_radius
        pts_crest = []
        for th in thetas:
            rad = math.cos(th) * normal + math.sin(th) * binorm
            pts_crest.append(b_head + axis_norm * (0.6 * L) + rad * (1.35 * b_radius))
        item_crest = WrinkleCurveItem(np.array(pts_crest, dtype=np.float32), strength=0.9, influence_radius=0.035)

        # 1. プリセットへ正規化
        preset = normalize_curves_to_preset(
            curves=[item_root, item_crest],
            curve_types=["root", "crest"],
            bone_head=b_head,
            bone_tail=b_tail,
            bone_radius=b_radius,
            normal=normal,
            name="TestRoundtrip",
        )

        self.assertEqual(len(preset.curves), 2)
        self.assertEqual(preset.curves[0].type, "root")
        self.assertEqual(preset.curves[1].type, "crest")
        self.assertAlmostEqual(preset.curves[0].rel_zs[0], 0.4, places=4)
        self.assertAlmostEqual(preset.curves[1].rel_zs[0], 0.6, places=4)

        # 2. 同一ボーンへ実体化
        c_items, r_items = instantiate_preset_on_bone(
            preset=preset,
            bone_head=b_head,
            bone_tail=b_tail,
            bone_radius=b_radius,
            normal=normal,
        )

        self.assertEqual(len(r_items), 1)
        self.assertEqual(len(c_items), 1)

        # 座標のラウンドトリップ誤差が 0.1mm 未満であること
        diff_root = np.max(np.linalg.norm(r_items[0].points - item_root.points, axis=-1))
        diff_crest = np.max(np.linalg.norm(c_items[0].points - item_crest.points, axis=-1))
        self.assertLess(diff_root, 1e-4, f"谷カーブの復元誤差が小さいこと (got {diff_root})")
        self.assertLess(diff_crest, 1e-4, f"山カーブの復元誤差が小さいこと (got {diff_crest})")

    def test_rescaling_on_different_bone(self):
        """異なる長さ・太さ・向きのボーンに適用した際に正しく自動スケーリングされることを検証"""
        presets = get_builtin_presets()
        preset = presets["pinch_and_puff"]

        # 長さ 0.4m, 半径 0.08m の太いボーン
        b_head = [0.0, 0.0, 0.0]
        b_tail = [0.0, 0.0, 0.4]
        b_radius = 0.08

        c_items, r_items = instantiate_preset_on_bone(
            preset=preset,
            bone_head=b_head,
            bone_tail=b_tail,
            bone_radius=b_radius,
        )

        self.assertEqual(len(r_items), 1)
        self.assertEqual(len(c_items), 1)

        # 谷の位置: rel_z=0.42 -> Z = 0.42 * 0.4 = 0.168m
        # 谷の半径: rel_r=1.05 -> r = 1.05 * 0.08 = 0.084m
        pts_r = r_items[0].points
        self.assertAlmostEqual(float(np.mean(pts_r[:, 2])), 0.168, places=3)
        radii_r = np.linalg.norm(pts_r[:, :2], axis=-1)
        self.assertAlmostEqual(float(np.mean(radii_r)), 0.084, places=3)

    def test_json_serialization(self):
        """JSONファイルへの保存・読み込み整合性を検証"""
        presets = get_builtin_presets()
        orig = presets["y_branch_joint"]

        fpath = os.path.join(self.temp_dir, "y_branch_test.json")
        save_preset_to_json(orig, fpath)
        self.assertTrue(os.path.exists(fpath))

        loaded = load_preset_from_json(fpath)
        self.assertEqual(loaded.name, orig.name)
        self.assertEqual(loaded.category, orig.category)
        self.assertEqual(len(loaded.curves), len(orig.curves))
        for c_orig, c_load in zip(orig.curves, loaded.curves):
            self.assertEqual(c_orig.type, c_load.type)
            self.assertEqual(len(c_orig.thetas), len(c_load.thetas))
            self.assertEqual(c_orig.strength, c_load.strength)


if __name__ == "__main__":
    unittest.main()
