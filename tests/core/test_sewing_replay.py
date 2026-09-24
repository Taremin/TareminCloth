# -*- coding: utf-8 -*-
"""縫合現在自然長の記録・再現テスト"""

import os
import sys
import tempfile
import unittest

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

# bpyモック登録 (test_sparse_recording.py と同一preamble)
try:
    import tests  # noqa: F401
except Exception:
    from unittest.mock import MagicMock

    class _DummyTypesModule:
        def __init__(self):
            self._types = {
                name: type(name, (object,), {})
                for name in ("Operator", "Panel", "PropertyGroup", "UIList",
                             "AddonPreferences", "Menu", "Header", "Gizmo",
                             "GizmoGroup", "Node", "NodeTree", "NodeSocket", "UILayout")
            }

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            if name not in self._types:
                self._types[name] = type(name, (object,), {})
            return self._types[name]

    if "bpy.types" not in sys.modules:
        sys.modules["bpy.types"] = _DummyTypesModule()
    for _mod_name in ("bpy", "bpy.props", "bpy.ops", "bpy.utils", "bpy.context",
                      "bpy.path", "bpy.app", "bpy_extras", "bpy_extras.view3d_utils",
                      "gpu", "blf", "bmesh", "mathutils"):
        if _mod_name not in sys.modules:
            try:
                __import__(_mod_name)
            except Exception:
                _m = MagicMock()
                _m.__name__ = _mod_name
                if _mod_name == "bpy":
                    _m.types = sys.modules.get("bpy.types")
                sys.modules[_mod_name] = _m


def make_two_patches(dx=0.05, gap=0.02):
    """縫合で結合する2枚の小パッチ。"""
    pos = []
    for ox in (0.0, dx * 2 + gap):
        for y in range(2):
            for x in range(2):
                pos.append([ox + x * dx, y * dx, 1.0])
    pos = np.array(pos, dtype=np.float32)
    edges = np.array([[0, 1], [2, 3], [0, 2], [1, 3],
                      [4, 5], [6, 7], [4, 6], [5, 7]], dtype=np.uint32)
    # facesなし (リプレイヤーはfacesなしでも再現可能)
    sew = np.array([[1, 4], [3, 6]], dtype=np.uint32)
    return pos, edges, sew


class TestSewingReplay(unittest.TestCase):
    def setUp(self):
        try:
            import taremin_cloth_core as core
            self.core = core
        except ImportError:
            try:
                from taremin_cloth import taremin_cloth_core as core
                self.core = core
            except ImportError:
                self.skipTest("taremin_cloth_core が利用できないためスキップします")
        from taremin_cloth import replayer as rp
        self.rp = rp
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sewing_rests_recorded_and_replayed(self):
        pos, edges, sew = make_two_patches()
        inv = np.ones(len(pos), dtype=np.float32)
        sim = self.core.ClothSimulator(
            pos, edges, sewing_springs=sew, sewing_shrink_speed=0.5)
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.start_debug_recording("Sew", max_frames=20)
        for _ in range(4):
            sim.step(1.0 / 60.0, 4)
        lp = os.path.join(self.temp_dir, "sew.jsonl.gz")
        sim.save_debug_recording(lp)
        sim.stop_debug_recording()

        # 記録内容: 現在自然長は単調減少し、shrink_speedが保存される
        frames = {f["frame_index"]: f for f in self.rp.iter_frames(lp)}
        rests0 = frames[0]["sewing_rest_lengths"]
        rests3 = frames[3]["sewing_rest_lengths"]
        self.assertIsInstance(rests0, list)
        self.assertEqual(len(rests0), 2)
        for a, b in zip(rests0, rests3):
            self.assertGreaterEqual(a, b)
        meta = self.rp.read_metadata(lp)
        self.assertAlmostEqual(meta.get("sewing_shrink_speed"), 0.5)

        # 完全再現 (縫合ありでも一致すること)
        rp = self.rp.ClothReplayer(lp)
        results = rp.replay_range(start_frame_idx=0, end_frame_idx=3, compare=True)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertTrue(r["compared"])
            self.assertLess(r["max_diff"], 1e-4, f"F{r['frame_index']}")

        # 途中開始でも折り畳みで一致すること
        rp2 = self.rp.ClothReplayer(lp)
        results2 = rp2.replay_range(start_frame_idx=2, end_frame_idx=3, compare=True)
        for r in results2:
            self.assertLess(r["max_diff"], 1e-4, f"F{r['frame_index']}")

    def test_sewing_priority_replay(self):
        """縫合優先モード (ラッチ進行) の再現。runnerと同順序でupdaterを駆動する。"""
        pos, edges, sew = make_two_patches()
        inv = np.ones(len(pos), dtype=np.float32)
        sim = self.core.ClothSimulator(pos, edges, sewing_springs=sew)
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_sewing_priority_options(True, 0.0, 0.005, 3, 600)
        sim.start_debug_recording("SewPrio", max_frames=20)
        coords = pos.copy()
        for _ in range(4):
            # runner.step_cloth_object と同順序: 直近座標で測定してから進める
            sim.update_sewing_priority(np.ascontiguousarray(coords))
            sim.step(1.0 / 60.0, 4)
            out = np.zeros(len(pos) * 3, dtype=np.float32)
            sim.get_positions(out)
            coords = out.reshape((-1, 3)).copy()
        lp = os.path.join(self.temp_dir, "sewprio.jsonl.gz")
        sim.save_debug_recording(lp)
        sim.stop_debug_recording()

        rp = self.rp.ClothReplayer(lp)
        results = rp.replay_range(start_frame_idx=0, end_frame_idx=3, compare=True)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertTrue(r["compared"])
            self.assertLess(r["max_diff"], 1e-4, f"F{r['frame_index']}")
        # ラッチ進行の裏付け: 重力スケールが復帰していること
        self.assertGreater(sim.get_sewing_priority_scale(), 0.0)
        self.assertGreater(rp.sim.get_sewing_priority_scale(), 0.0)


if __name__ == "__main__":
    unittest.main()
