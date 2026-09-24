# -*- coding: utf-8 -*-
"""フレーム途中パラメータ変更 (param_deltas) の記録・再現テスト"""

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


class TestParamDeltas(unittest.TestCase):
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

    def _grid(self):
        pos, edges, faces = [], [], []
        nx, ny, dx = 4, 4, 0.05
        for y in range(ny):
            for x in range(nx):
                pos.append([x * dx, y * dx, 1.0])
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
        return (np.array(pos, dtype=np.float32), np.array(edges, dtype=np.uint32),
                np.array(faces, dtype=np.uint32), np.ones(len(pos), dtype=np.float32))

    def _record_with_mid_change(self):
        pos, edges, faces, inv = self._grid()
        sim = self.core.ClothSimulator(pos, edges, faces, inv)
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.start_debug_recording("Delta", max_frames=20)
        for _ in range(3):
            sim.step(1.0 / 60.0, 4)
        # 途中で重力反転 + 剛性変更 (Nパネル操作の模擬)
        sim.set_gravity(0.0, 0.0, 5.0)
        sim.set_stiffness_all(3000.0, 3000.0, 1500.0, 20.0)
        for _ in range(3):
            sim.step(1.0 / 60.0, 4)
        lp = os.path.join(self.temp_dir, "delta.jsonl.gz")
        sim.save_debug_recording(lp)
        sim.stop_debug_recording()
        return lp

    def test_deltas_recorded_only_on_change(self):
        lp = self._record_with_mid_change()
        frames = {f["frame_index"]: f for f in self.rp.iter_frames(lp)}
        self.assertEqual(len(frames), 6)
        # frame 0-2: 変更なし
        for i in (0, 1, 2):
            self.assertIsNone(frames[i].get("param_deltas"), f"F{i}")
        # frame 3: 変更あり
        d3 = frames[3].get("param_deltas")
        self.assertIsInstance(d3, dict)
        self.assertEqual(d3["gravity"], [0.0, 0.0, 5.0])
        self.assertEqual(d3["tension_stiffness"], 3000.0)
        # frame 4-5: 変更なし
        self.assertIsNone(frames[4].get("param_deltas"))
        self.assertIsNone(frames[5].get("param_deltas"))

    def test_replay_reproduces_mid_change(self):
        lp = self._record_with_mid_change()
        rp = self.rp.ClothReplayer(lp)
        results = rp.replay_range(start_frame_idx=0, end_frame_idx=5, compare=True)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertTrue(r["compared"])
            self.assertLess(r["max_diff"], 1e-4, f"F{r['frame_index']}")

    def test_replay_from_mid_applies_folded_config(self):
        """途中開始でも変更後設定で初期化されること"""
        lp = self._record_with_mid_change()
        rp = self.rp.ClothReplayer(lp)
        cfg = self.rp.config_at_frame(lp, rp.metadata, 4)
        self.assertEqual(cfg["gravity"], [0.0, 0.0, 5.0])
        cfg0 = self.rp.config_at_frame(lp, rp.metadata, 1)
        self.assertEqual(cfg0["gravity"], [0.0, 0.0, -9.81])
        # 途中開始リプレイも一致すること
        results = rp.replay_range(start_frame_idx=4, end_frame_idx=5, compare=True)
        for r in results:
            self.assertLess(r["max_diff"], 1e-4, f"F{r['frame_index']}")

    def test_watch_config_token(self):
        lp = self._record_with_mid_change()
        rp = self.rp.ClothReplayer(lp)
        pred = self.rp.build_stop_predicate(config_changed=True)
        hit = rp.replay_until(0, 5, stop_on=pred)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["frame_index"], 3)

    def test_elastic_scales_reproduced(self):
        """伸縮スケール変更の記録・適用・再現.

        背景: stiffなスナップ遷移ではGPU実行順序の非決定性がmm級に増幅
        される (双子forward実行同士でも再現しない既知の制限) ため、
        軌道の厳密一致ではなく以下を検証する:
          1. 差分内容の厳密一致 (決定的)
          2. 折り畳み適用後の自然長一致 (決定的)
          3. 再生トポロジーがせん断拘束を正しく再生成 (決定的)
          4. 軌道の sanity bound (緩やかな regime で 1e-3)
        """
        pos, edges, faces, inv = self._grid()
        sim = self.core.ClothSimulator(pos, edges, faces, inv)
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.start_debug_recording("Elastic", max_frames=20)
        for _ in range(2):
            sim.step(1.0 / 60.0, 4)
        n_edge = len(edges)
        sim.set_edge_rest_length_scales(
            np.array([0, 1], dtype=np.uint32),
            np.array([0.99, 0.99], dtype=np.float32),
        )
        for _ in range(2):
            sim.step(1.0 / 60.0, 4)
        lp = os.path.join(self.temp_dir, "elastic.jsonl.gz")
        sim.save_debug_recording(lp)
        sim.stop_debug_recording()

        frames = {f["frame_index"]: f for f in self.rp.iter_frames(lp)}
        self.assertIsNone(frames[0].get("elastic_scales"))
        self.assertIsNone(frames[1].get("elastic_scales"))
        esc = frames[2].get("elastic_scales")
        self.assertIsInstance(esc, list)
        self.assertEqual({e["edge_idx"] for e in esc}, {0, 1})
        for e in esc:
            self.assertAlmostEqual(e["scale"], 0.99, places=5)

        rp = self.rp.ClothReplayer(lp)
        # 折り畳み適用で自然長が一致すること (決定的)
        rp.create_simulator(np.array(frames[0]["positions"], dtype=np.float32))
        self.assertTrue(rp.apply_elastic_at(3))
        for e in (0, 1):
            self.assertAlmostEqual(
                rp.sim.get_edge_rest_length(e),
                sim.get_edge_rest_length(e), places=5)

        # 再生トポロジーがせん断拘束を再生成していること (決定的)。
        # 旧形式 (対角混入45辺) では全拘束がstretch化され shear が250に落ちる。
        import json as _json
        shear = _json.loads(rp.sim.get_config_json())["shear_stiffness"]
        self.assertAlmostEqual(shear, 500.0, delta=1.0)

        # 軌道 sanity (緩やかな regime)
        results = rp.replay_range(start_frame_idx=0, end_frame_idx=3, compare=True)
        for r in results:
            self.assertLess(r["max_diff"], 1e-3, f"F{r['frame_index']}")
        _ = n_edge

    def test_thickness_metadata_recorded(self):
        """頂点毎厚み・レイヤーがメタデータに記録されること"""
        import gzip
        import json
        pos, edges, faces, inv = self._grid()
        sim = self.core.ClothSimulator(pos, edges, faces, inv, thickness=0.007)
        sim.start_debug_recording("Thick", max_frames=5)
        sim.step(1.0 / 60.0, 2)
        lp = os.path.join(self.temp_dir, "thick.jsonl.gz")
        sim.save_debug_recording(lp)
        sim.stop_debug_recording()
        with gzip.open(lp, "rt", encoding="utf-8") as f:
            meta = json.loads(f.readline())["metadata"]
        self.assertEqual(len(meta["thicknesses"]), len(pos))
        self.assertTrue(all(abs(t - 0.007) < 1e-9 for t in meta["thicknesses"]))
        self.assertEqual(len(meta["layer_ids"]), len(pos))
        # 再現も通ること
        rp = self.rp.ClothReplayer(lp)
        results = rp.replay_range(start_frame_idx=0, end_frame_idx=0, compare=True)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
