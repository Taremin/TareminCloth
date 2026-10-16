"""
tools/audit_function_args.py の単体テスト。

AST走査によるキーワード専用引数安全規約、ブールトラップ検知、
除外デコレータ・コメント判定の動作を検証します。
"""

import ast
import tempfile
import unittest
from pathlib import Path

from tools.audit_function_args import FunctionArgAuditor, audit_path


class TestFunctionArgAuditor(unittest.TestCase):
    def _run_audit_on_code(self, code: str):
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            violations, total_funcs, total_skipped = audit_path(temp_path)
            return violations, total_funcs, total_skipped
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_clean_keyword_only_function(self):
        """キーワード専用引数を正しく使用している関数は違反が検出されないこと"""
        code = """
def step_sim(scene, *, dt: float = 1.0 / 60.0, update_mesh: bool = True):
    pass
"""
        violations, total, skipped = self._run_audit_on_code(code)
        self.assertEqual(len(violations), 0)
        self.assertEqual(total, 1)

    def test_detect_boolean_trap_annotation(self):
        """boolアノテーションを持つ位置引数をブールトラップとして検出できること"""
        code = """
def apply_action(obj, flag: bool):
    pass
"""
        violations, total, skipped = self._run_audit_on_code(code)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["rule"], "boolean_trap")
        self.assertEqual(violations[0]["arg"], "flag")

    def test_detect_boolean_trap_default(self):
        """デフォルト値 True/False を持つ位置引数をブールトラップとして検出できること"""
        code = """
def update_view(obj, force=False):
    pass
"""
        violations, total, skipped = self._run_audit_on_code(code)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["rule"], "boolean_trap")
        self.assertEqual(violations[0]["arg"], "force")

    def test_detect_optional_positional_arg(self):
        """複数引数でデフォルト値を持つ引数が位置引数に残っている場合を検出できること"""
        code = """
def calculate(data, factor=1.5):
    pass
"""
        violations, total, skipped = self._run_audit_on_code(code)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["rule"], "optional_positional")
        self.assertEqual(violations[0]["arg"], "factor")

    def test_detect_too_many_positional_args(self):
        """位置引数が3個以上ある場合を検出できること"""
        code = """
def transform_point(p, x, y):
    pass
"""
        violations, total, skipped = self._run_audit_on_code(code)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["rule"], "too_many_positional")

    def test_allow_positional_decorator(self):
        """@allow_positional デコレータが付与された関数はスキップされること"""
        code = """
from taremin_cloth.utils import allow_positional

@allow_positional
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
"""
        violations, total, skipped = self._run_audit_on_code(code)
        self.assertEqual(len(violations), 0)
        self.assertEqual(skipped, 1)

    def test_allow_positional_comment(self):
        """# audit: allow-positional コメントが付与された行はスキップされること"""
        code = """
def clamp(val, min_val, max_val):  # audit: allow-positional
    return max(min_val, min(val, max_val))
"""
        violations, total, skipped = self._run_audit_on_code(code)
        self.assertEqual(len(violations), 0)
        self.assertEqual(skipped, 1)

    def test_blender_callbacks_whitelisted(self):
        """Blenderオペレーターの execute や poll が除外されること"""
        code = """
class SimpleOperator:
    def execute(self, context):
        return {'FINISHED'}

    def poll(self, context):
        return True

def update_property_cb(self, context):
    pass
"""
        violations, total, skipped = self._run_audit_on_code(code)
        self.assertEqual(len(violations), 0)
        self.assertEqual(skipped, 3)

    def test_special_methods_whitelisted(self):
        """Python特殊メソッド __init__ 等が除外されること"""
        code = """
class DataContainer:
    def __init__(self, a, b, c, d):
        pass
"""
        violations, total, skipped = self._run_audit_on_code(code)
        self.assertEqual(len(violations), 0)
        self.assertEqual(skipped, 1)


if __name__ == "__main__":
    unittest.main()

