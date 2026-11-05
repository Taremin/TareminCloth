# -*- coding: utf-8 -*-
"""ids.py および units.py の単体テスト (Blender非依存・GPU不要)."""

import math
import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from taremin_cloth.pattern.ids import new_id, is_valid_id
from taremin_cloth.pattern.units import m_to_cm, cm_to_m, format_cm, parse_length_input


class TestPatternUnitsAndIds(unittest.TestCase):
    """ID採番および単位変換の正確性を検証するテストケース."""

    def test_new_id_uniqueness_and_format(self):
        """生成されるIDが32文字16進文字列であり、一意であること."""
        generated = {new_id() for _ in range(1000)}
        self.assertEqual(len(generated), 1000)
        for uid in generated:
            self.assertTrue(is_valid_id(uid))
            self.assertEqual(len(uid), 32)
            self.assertTrue(uid.islower() or uid.isdigit())

    def test_is_valid_id_edge_cases(self):
        """不正なIDを正しく拒絶すること."""
        self.assertFalse(is_valid_id(""))
        self.assertFalse(is_valid_id(None))
        self.assertFalse(is_valid_id(12345))
        self.assertFalse(is_valid_id("short_id"))
        self.assertFalse(is_valid_id("g" * 32))  # 'g' is not hex
        self.assertFalse(is_valid_id("A" * 32))  # uppercase rejected
        self.assertFalse(is_valid_id(" " * 32))
        self.assertTrue(is_valid_id("0123456789abcdef0123456789abcdef"))

    def test_m_cm_conversions(self):
        """メートルとセンチメートルの相互変換."""
        self.assertAlmostEqual(m_to_cm(1.0), 100.0)
        self.assertAlmostEqual(m_to_cm(0.254), 25.4)
        self.assertAlmostEqual(cm_to_m(100.0), 1.0)
        self.assertAlmostEqual(cm_to_m(25.4), 0.254)

    def test_format_cm(self):
        """表示用センチメートルフォーマット."""
        self.assertEqual(format_cm(0.254), "25.4 cm")
        self.assertEqual(format_cm(0.254, with_unit=False), "25.4")
        self.assertEqual(format_cm(0.2546, decimals=2), "25.46 cm")
        self.assertEqual(format_cm(1.0), "100.0 cm")

    def test_parse_length_input(self):
        """ユーザー入力文字列のパース."""
        # 既定単位 (cm)
        self.assertAlmostEqual(parse_length_input("25.4"), 0.254)
        self.assertAlmostEqual(parse_length_input(" 25.4 "), 0.254)
        # 明示的単位
        self.assertAlmostEqual(parse_length_input("25.4 cm"), 0.254)
        self.assertAlmostEqual(parse_length_input("25.4cm"), 0.254)
        self.assertAlmostEqual(parse_length_input("0.254 m"), 0.254)
        self.assertAlmostEqual(parse_length_input("254 mm"), 0.254)
        self.assertAlmostEqual(parse_length_input("10 in"), 0.254)

        # 科学表記
        self.assertAlmostEqual(parse_length_input("1e2 cm"), 1.0)

        # 不正入力
        with self.assertRaises(ValueError):
            parse_length_input("")
        with self.assertRaises(ValueError):
            parse_length_input("abc")
        with self.assertRaises(ValueError):
            parse_length_input("25.4 yard")
        with self.assertRaises(ValueError):
            parse_length_input("NaN")
        with self.assertRaises(ValueError):
            parse_length_input("Infinity")


if __name__ == "__main__":
    unittest.main()
