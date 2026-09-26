# -*- coding: utf-8 -*-
"""profiling集計ヘルパの単体テスト (bpy非依存・GPU不要)。"""

import os
import sys
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

from taremin_cloth.engine.profiling import (
    aggregate_profile,
    format_profile_table,
    is_parent_scope,
)


class TestProfilingHelper(unittest.TestCase):
    def test_parent_detection(self):
        self.assertTrue(is_parent_scope("self_collision_Outer"))
        self.assertFalse(is_parent_scope("sc_solve"))

    def test_aggregate_excludes_parents_from_net(self):
        entries = [
            ("self_collision_Outer", 3.0),
            ("sc_solve", 1.0),
            ("predict", 0.5),
        ]
        totals, counts, net, flat = aggregate_profile(entries)
        self.assertAlmostEqual(flat, 4.5)
        self.assertAlmostEqual(net, 1.5)
        self.assertEqual(counts["sc_solve"], 1)

    def test_format_empty(self):
        self.assertIn("no entries", format_profile_table([]))

    def test_format_marks_parent(self):
        out = format_profile_table([("self_collision_Outer", 2.0), ("sc_solve", 1.0)])
        self.assertIn("(parent)", out)
        self.assertIn("net ms", out)


if __name__ == "__main__":
    unittest.main()
