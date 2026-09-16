"""
プロパティ定義のエルゴノミクス改善 (soft_min/soft_max および slider 描画) の単体テスト
"""

import ast
import inspect
import os
import sys
import unittest

# プロジェクトルートおよび python/ ディレクトリをモジュール検索パスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from taremin_cloth import properties, panels


class TestPropertyErgonomics(unittest.TestCase):
    """剛性・減衰プロパティのエルゴノミクス設定（soft_min, soft_max, slider）のテスト"""

    def test_stiffness_and_damping_soft_ranges(self):
        """TareminClothObjectSettings の各プロパティの soft_min/soft_max が常用域に設定されていることを検証"""
        # properties.py のソースコードから AST を解析して FloatProperty の引数を直接検証
        prop_file = inspect.getfile(properties)
        with open(prop_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=prop_file)

        class_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TareminClothObjectSettings":
                class_node = node
                break

        self.assertIsNotNone(class_node, "TareminClothObjectSettings クラスが存在すること")

        # 各プロパティのキーワード引数を辞書化
        prop_kwargs = {}
        for item in class_node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                prop_name = item.target.id
                call_node = item.annotation if isinstance(item.annotation, ast.Call) else item.value
                if isinstance(call_node, ast.Call):
                    kwargs = {}
                    for kw in call_node.keywords:
                        if isinstance(kw.value, ast.Constant):
                            kwargs[kw.arg] = kw.value.value
                        elif isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.operand, ast.Constant):
                            # 負の数値対応
                            kwargs[kw.arg] = -kw.value.operand.value
                    prop_kwargs[prop_name] = kwargs

        # 剛性 (Stiffness) の soft_min / soft_max 検証
        expected_stiffness = {
            "tension_stiffness": {"soft_min": 10.0, "soft_max": 10000.0, "min": 0.1, "max": 100000.0},
            "compression_stiffness": {"soft_min": 10.0, "soft_max": 10000.0, "min": 0.1, "max": 100000.0},
            "shear_stiffness": {"soft_min": 10.0, "soft_max": 5000.0, "min": 0.1, "max": 100000.0},
            "bending_stiffness": {"soft_min": 0.0, "soft_max": 200.0, "min": 0.0, "max": 1000.0},
            "stiffness": {"soft_min": 10.0, "soft_max": 10000.0, "min": 0.1, "max": 100000.0},
        }
        for prop_name, expected in expected_stiffness.items():
            self.assertIn(prop_name, prop_kwargs, f"{prop_name} がプロパティ定義に存在すること")
            actual = prop_kwargs[prop_name]
            self.assertEqual(actual.get("soft_min"), expected["soft_min"], f"{prop_name} の soft_min が {expected['soft_min']} であること")
            self.assertEqual(actual.get("soft_max"), expected["soft_max"], f"{prop_name} の soft_max が {expected['soft_max']} であること")
            self.assertEqual(actual.get("min"), expected["min"], f"{prop_name} のハードリミット min が維持されていること")
            self.assertEqual(actual.get("max"), expected["max"], f"{prop_name} のハードリミット max が維持されていること")

        # 減衰 (Damping) の soft_min / soft_max 検証
        expected_damping = {
            "tension_damping": {"soft_min": 0.0, "soft_max": 25.0, "min": 0.0, "max": 50.0},
            "compression_damping": {"soft_min": 0.0, "soft_max": 25.0, "min": 0.0, "max": 50.0},
            "shear_damping": {"soft_min": 0.0, "soft_max": 25.0, "min": 0.0, "max": 50.0},
            "bending_damping": {"soft_min": 0.0, "soft_max": 10.0, "min": 0.0, "max": 50.0},
            "air_damping": {"soft_min": 0.0, "soft_max": 10.0, "min": 0.0, "max": 50.0},
        }
        for prop_name, expected in expected_damping.items():
            self.assertIn(prop_name, prop_kwargs, f"{prop_name} がプロパティ定義に存在すること")
            actual = prop_kwargs[prop_name]
            self.assertEqual(actual.get("soft_min"), expected["soft_min"], f"{prop_name} の soft_min が {expected['soft_min']} であること")
            self.assertEqual(actual.get("soft_max"), expected["soft_max"], f"{prop_name} の soft_max が {expected['soft_max']} であること")
            self.assertEqual(actual.get("min"), expected["min"], f"{prop_name} のハードリミット min が維持されていること")
            self.assertEqual(actual.get("max"), expected["max"], f"{prop_name} のハードリミット max が維持されていること")

    def test_panels_render_with_slider_true(self):
        """panels.py において剛性と減衰の各プロパティが slider=True で描画されていることを検証"""
        panel_file = inspect.getfile(panels)
        with open(panel_file, "r", encoding="utf-8") as f:
            content = f.read()

        target_props = [
            "tension_stiffness",
            "compression_stiffness",
            "shear_stiffness",
            "bending_stiffness",
            "air_damping",
            "tension_damping",
            "compression_damping",
            "shear_damping",
            "bending_damping",
        ]
        for prop in target_props:
            # layout.prop(..., "prop_name", slider=True) の記述パターンが存在することを確認
            expected_snippet = f'"{prop}", slider=True'
            self.assertIn(
                expected_snippet,
                content,
                f"panels.py 内で {prop} が slider=True を指定して描画されていること"
            )


if __name__ == "__main__":
    unittest.main()
