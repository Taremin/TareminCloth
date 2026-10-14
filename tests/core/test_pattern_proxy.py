# -*- coding: utf-8 -*-
"""3D配置プロキシ & クイック配置プリセットの単体テスト."""

import math
import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from taremin_cloth.utils.pattern_proxy import (
    QUICK_PLACEMENT_PRESETS,
    get_quick_placement_transform,
    estimate_body_reference,
)


class TestPatternProxy(unittest.TestCase):
    def test_quick_placement_transforms(self):
        """主要なクイック配置プリセットのオフセットと回転角が正しいこと."""
        # 1. 前身頃 (Front / Chest)
        offset_f, euler_f = get_quick_placement_transform('CHEST_FRONT')
        self.assertAlmostEqual(offset_f.x, 0.0)
        self.assertLess(offset_f.y, 0.0)  # Yマイナス（体の前）
        self.assertGreater(offset_f.z, 0.9)  # 胸の高さ
        self.assertAlmostEqual(euler_f.x, math.pi / 2, places=4)  # 90度起き上がる

        # 2. 後身頃 (Back)
        offset_b, euler_b = get_quick_placement_transform('BACK')
        self.assertAlmostEqual(offset_b.x, 0.0)
        self.assertGreater(offset_b.y, 0.0)  # Yプラス（体の後ろ）
        self.assertAlmostEqual(euler_b.x, math.pi / 2, places=4)
        self.assertAlmostEqual(abs(euler_b.z), math.pi, places=4)  # 180度反転

        # 3. 左袖 (Left Arm)
        offset_l, euler_l = get_quick_placement_transform('ARM_LEFT')
        self.assertLess(offset_l.x, 0.0)  # Xマイナス（左腕側）
        self.assertAlmostEqual(euler_l.z, -math.pi / 2, places=4)

        # 4. 右袖 (Right Arm)
        offset_r, euler_r = get_quick_placement_transform('ARM_RIGHT')
        self.assertGreater(offset_r.x, 0.0)  # Xプラス（右腕側）
        self.assertAlmostEqual(euler_r.z, math.pi / 2, places=4)

    def test_preset_items_defined(self):
        """プリセット定義リストが空でなく主要パーツを網羅していること."""
        keys = [item[0] for item in QUICK_PLACEMENT_PRESETS]
        self.assertIn('CHEST_FRONT', keys)
        self.assertIn('BACK', keys)
        self.assertIn('ARM_LEFT', keys)
        self.assertIn('ARM_RIGHT', keys)
        self.assertIn('COLLAR', keys)
        self.assertIn('WAIST_SKIRT', keys)


if __name__ == '__main__':
    unittest.main()
