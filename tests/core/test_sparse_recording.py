# -*- coding: utf-8 -*-
"""2階層デバッグ記録 (スパース+トリガー) の統合テスト"""

import gzip
import json
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

# bpyモック登録 (tests/__init__ と同等。システムPythonの破損bpy対策として
# ImportErrorだけでなく全例外を捕捉し、 недостающийモジュールはMagicMock化する)
try:
    import tests  # noqa: F401 (CI環境ではこちらがbpyモックを登録する)
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


class TestSparseRecording(unittest.TestCase):
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
        # bpyモック (taremin_clothパッケージimport用: 上記preambleで登録済み)
        from taremin_cloth import replayer as rp
        from taremin_cloth import log_tools as lt
        self.rp = rp
        self.lt = lt
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _grid(self, nx=3, ny=3, dx=0.05):
        positions = []
        for y in range(ny):
            for x in range(nx):
                positions.append([x * dx, y * dx, 1.0])
        edges = []
        faces = []
        for y in range(ny):
            for x in range(nx):
                idx = y * nx + x
                if x + 1 < nx:
                    edges.append([idx, idx + 1])
                if y + 1 < ny:
                    edges.append([idx, idx + nx])
                if x + 1 < nx and y + 1 < ny:
                    faces.append([idx, idx + 1, idx + nx + 1])
                    faces.append([idx, idx + nx + 1, idx + nx])
        return (
            np.array(positions, dtype=np.float32),
            np.array(edges, dtype=np.uint32),
            np.array(faces, dtype=np.uint32),
            np.ones(len(positions), dtype=np.float32),
        )

    def _record(self, stride=1, ring=0, lookahead=0, triggers=False, frames=7):
        pos, edges, faces, inv = self._grid()
        sim = self.core.ClothSimulator(pos, edges, faces, inv)
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_debug_recording_options(
            full_stride=stride, ring_size=ring, lookahead=lookahead,
            enable_triggers=triggers,
        )
        sim.start_debug_recording("Sparse", max_frames=50)
        for _ in range(frames):
            sim.step(1.0 / 60.0, 2)
        lp = os.path.join(self.temp_dir, f"sparse_{stride}_{ring}.jsonl.gz")
        sim.save_debug_recording(lp)
        info = sim.get_debug_sparse_info()
        sim.stop_debug_recording()
        return lp, info

    def test_default_is_full(self):
        lp, (total, full, stubs) = self._record()
        self.assertEqual(total, 7)
        self.assertEqual(full, 7)
        self.assertEqual(stubs, 0)

    def test_stride_produces_stubs_with_stats(self):
        lp, (total, full, stubs) = self._record(stride=3, ring=2)
        self.assertEqual(total, 7)
        self.assertEqual(full, 3)  # 0,3,6
        self.assertEqual(stubs, 4)
        with gzip.open(lp, "rt", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(lines[0].get("version"), 3)
        for rec in lines[1:]:
            if rec["frame_index"] % 3 == 0:
                self.assertTrue(len(rec["positions"]) > 0)
            else:
                self.assertEqual(rec["positions"], [])
                self.assertEqual(rec["velocities"], [])
                # スタブでも入力とstatsは保持
                self.assertIn("stats", rec)
                self.assertIn("dt", rec)

    def test_sparse_file_size_smaller(self):
        lp_full, _ = self._record(stride=1)
        lp_sparse, _ = self._record(stride=5, ring=0)
        self.assertLess(
            os.path.getsize(lp_sparse), os.path.getsize(lp_full),
            "スパース記録がフル記録より小さいこと",
        )

    def test_replay_range_marks_stubs_uncompared(self):
        lp, _ = self._record(stride=3, ring=0)
        rp = self.rp.ClothReplayer(lp)
        results = rp.replay_range(start_frame_idx=0, end_frame_idx=6, compare=True)
        by_idx = {r["frame_index"]: r for r in results}
        self.assertTrue(by_idx[3]["compared"])
        self.assertLess(by_idx[3]["max_diff"], 1e-4)
        self.assertFalse(by_idx[1]["compared"])
        self.assertEqual(by_idx[1]["max_diff"], 0.0)

    def test_replay_until_passes_through_stubs(self):
        lp, _ = self._record(stride=3, ring=0)
        rp = self.rp.ClothReplayer(lp)
        hit = rp.replay_until(0, 6, stop_on=lambda ctx: ctx["max_disp_mm"] > 0.0001)
        self.assertIsNotNone(hit)

    def test_trace_substeps_rejects_stub(self):
        lp, _ = self._record(stride=3, ring=0)
        rp = self.rp.ClothReplayer(lp)
        with self.assertRaises(ValueError):
            rp.trace_substeps(target_frame_idx=1)

    def test_trigger_flush_restores_context(self):
        # stride大 + ring + 後からトリガー有効化で文脈復元を検証
        pos, edges, faces, inv = self._grid()
        sim = self.core.ClothSimulator(pos, edges, faces, inv)
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_debug_recording_options(full_stride=100, ring_size=2, lookahead=0, enable_triggers=False)
        sim.start_debug_recording("Trig", max_frames=50)
        for _ in range(3):
            sim.step(1.0 / 60.0, 2)
        # ここでトリガー有効化 (次フレーム以降の大変位を検出)
        sim.set_debug_recording_options(
            full_stride=100, ring_size=2, lookahead=0, enable_triggers=True,
            disp_trigger_mm=0.0001,
        )
        for _ in range(2):
            sim.step(1.0 / 60.0, 2)
        lp = os.path.join(self.temp_dir, "trig.jsonl.gz")
        sim.save_debug_recording(lp)
        sim.stop_debug_recording()
        frames = {f["frame_index"]: f for f in self.rp.iter_frames(lp)}
        # frame 3 のトリガーで ring(1,2) が復元されていること
        self.assertTrue(len(frames[1]["positions"]) > 0)
        self.assertTrue(len(frames[2]["positions"]) > 0)

    def test_inspect_handles_sparse(self):
        import argparse
        lp, _ = self._record(stride=3, ring=0)
        args = argparse.Namespace(log_file=lp, disp_threshold=500.0, vel_threshold=500.0)
        # 例外なく走査できること
        self.lt.cmd_inspect(args)


if __name__ == "__main__":
    unittest.main()
