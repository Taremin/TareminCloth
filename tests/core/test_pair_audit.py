# -*- coding: utf-8 -*-
"""ペアカバレッジ監査 (get_active_pairs + pair_audit) のテスト"""

import argparse
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

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


def make_two_layers(nx=3, ny=3, dx=0.05, gap=0.003):
    """上下2枚のグリッド布 (非連結2成分)。"""
    def grid(z):
        pos, edges, faces, inv = [], [], [], []
        for y in range(ny):
            for x in range(nx):
                pos.append([x * dx, y * dx, z])
                inv.append(1.0)
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
                np.array(faces, dtype=np.uint32), np.array(inv, dtype=np.float32))

    p1, e1, f1, i1 = grid(0.0)
    p2, e2, f2, i2 = grid(gap)
    n1 = len(p1)
    return (np.vstack([p1, p2]),
            np.vstack([e1, e2 + n1]),
            np.vstack([f1, f2 + n1]),
            np.concatenate([i1, i2]))


class TestPairAudit(unittest.TestCase):
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
        from taremin_cloth import pair_audit as pa
        from taremin_cloth import replayer as rp
        from taremin_cloth import log_tools as lt
        self.pa = pa
        self.rp = rp
        self.lt = lt
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_active_pairs_roundtrip(self):
        """読戻しペアの形状・件数整合性"""
        pos, edges, faces, inv = make_two_layers()
        sim = self.core.ClothSimulator(
            pos, edges, faces, inv, thickness=0.005, enable_pair_cache=True)
        sim.set_enable_self_collision(True)
        sim.set_gravity(0.0, 0.0, 0.0)
        sim.set_enable_pair_cache_final_fallback(False)
        sim.step(1.0 / 60.0, 1)
        vt_count, ee_count, vt_pairs, ee_pairs = sim.get_active_pairs()
        vt = np.asarray(vt_pairs, dtype=np.uint32).reshape((-1, 4))
        ee = np.asarray(ee_pairs, dtype=np.uint32).reshape((-1, 4))
        self.assertGreater(len(vt), 0, "V-Tペアが収集されること")
        self.assertGreater(len(ee), 0, "E-Eペアが収集されること")
        self.assertLessEqual(len(vt), vt_count)
        self.assertLessEqual(len(ee), ee_count)
        self.assertTrue(np.all(vt < len(pos)))
        self.assertTrue(np.all(ee < len(pos)))

    def test_static_coverage_near_full(self):
        """静止近接配置ではカバレッジ≒100%・staleなし"""
        pos, edges, faces, inv = make_two_layers(gap=0.003)
        sim = self.core.ClothSimulator(
            pos, edges, faces, inv, thickness=0.005, enable_pair_cache=True)
        sim.set_enable_self_collision(True)
        sim.set_gravity(0.0, 0.0, 0.0)
        sim.set_enable_pair_cache_final_fallback(False)
        sim.step(1.0 / 60.0, 1)
        vt_count, ee_count, vt_pairs, ee_pairs = sim.get_active_pairs()
        rest = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
        rep = self.pa.audit_frame(
            pos.astype(np.float64), np.zeros_like(pos, dtype=np.float64),
            pos.astype(np.float64),
            np.asarray(vt_pairs).reshape((-1, 4)), np.asarray(ee_pairs).reshape((-1, 4)),
            vt_count, ee_count, faces, edges, rest,
            0.005, 0.005, 1.3, 0.02, 1, 1.0 / 60.0, 32768, 32768,
        )
        self.assertGreater(rep["vt_ground"], 0)
        self.assertGreaterEqual(rep["vt_coverage"], 0.9, f"VT coverage: {rep}")
        self.assertGreaterEqual(rep["ee_coverage"], 0.9, f"EE coverage: {rep}")
        self.assertEqual(rep["vt_stale"], 0)
        self.assertEqual(rep["vt_dropped"], 0)

    def test_saturation_drops_reported(self):
        """上限超過時はドロップ数が報告されること"""
        pos, edges, faces, inv = make_two_layers(nx=6, ny=6, gap=0.003)
        n = len(pos)
        sim = self.core.ClothSimulator(
            pos, edges, faces, inv, thickness=0.005, enable_pair_cache=True)
        sim.set_enable_self_collision(True)
        sim.set_gravity(0.0, 0.0, 0.0)
        # 予算64 → 72頂点では枠数K=1 (最低1枠保証のため72スロット)。
        # 旧方式の総数キャップとは異なり、頂点単位の公平性が保たれる。
        sim.set_pair_cache_options(64, 64, 1, 0.005, 1.3, 0.02)
        sim.set_enable_pair_cache_final_fallback(False)
        sim.step(1.0 / 60.0, 1)
        vt_count, ee_count, vt_pairs, ee_pairs = sim.get_active_pairs()
        vt = np.asarray(vt_pairs).reshape((-1, 4))
        self.assertLessEqual(len(vt), n)  # N*K (K=1)
        self.assertGreater(vt_count + ee_count, n * 2, "枠を超過するほど収集されること")
        rest = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
        rep = self.pa.audit_frame(
            pos.astype(np.float64), np.zeros_like(pos, dtype=np.float64),
            pos.astype(np.float64), vt, np.asarray(ee_pairs).reshape((-1, 4)),
            vt_count, ee_count, faces, edges, rest,
            0.005, 0.005, 1.3, 0.02, 1, 1.0 / 60.0, 64, 64,
        )
        self.assertGreater(rep["vt_dropped"] + rep["ee_dropped"], 0)

    def test_motion_miss_classified_as_horizon(self):
        """収集 bound 外からの接近は horizon 分類になること (FIXED)"""
        pos, edges, faces, inv = make_two_layers(gap=0.030)
        n = len(pos) // 2
        vel = np.zeros_like(pos)
        vel[:n, 2] = 0.5
        vel[n:, 2] = -0.5
        sim = self.core.ClothSimulator(
            pos, edges, faces, inv, thickness=0.005, enable_pair_cache=True)
        sim.set_enable_self_collision(True)
        sim.set_gravity(0.0, 0.0, 0.0)
        sim.set_positions_and_velocities(pos, vel)
        sim.set_pair_cache_options(32768, 32768, 0, 0.005, 1.3, 0.02)  # FIXED
        sim.set_enable_pair_cache_final_fallback(False)
        sim.step(1.0 / 60.0, 10)
        vt_count, ee_count, vt_pairs, ee_pairs = sim.get_active_pairs()
        out = np.zeros(len(pos) * 3, dtype=np.float32)
        sim.get_positions(out)
        curr = out.reshape((-1, 3))
        rest = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
        rep = self.pa.audit_frame(
            pos.astype(np.float64), vel.astype(np.float64), curr.astype(np.float64),
            np.asarray(vt_pairs).reshape((-1, 4)), np.asarray(ee_pairs).reshape((-1, 4)),
            vt_count, ee_count, faces, edges, rest,
            0.005, 0.005, 1.3, 0.02, 0, 1.0 / 60.0, 32768, 32768,
        )
        self.assertGreater(rep["vt_ground"], 0, "終端で近接ペアが存在すること")
        self.assertGreater(rep["vt_miss_horizon"], 0, f"horizon分類されること: {rep}")
        self.assertEqual(rep["vt_miss_other"], 0, f"otherはゼロのはず: {rep}")

    def test_audit_pairs_command_end_to_end(self):
        """audit-pairs CLIの端到端動作"""
        pos, edges, faces, inv = make_two_layers(gap=0.003)
        sim = self.core.ClothSimulator(
            pos, edges, faces, inv, thickness=0.005, enable_pair_cache=True)
        sim.set_enable_self_collision(True)
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.start_debug_recording("Audit", max_frames=10)
        for _ in range(3):
            sim.step(1.0 / 60.0, 2)
        lp = os.path.join(self.temp_dir, "audit.jsonl.gz")
        sim.save_debug_recording(lp)
        sim.stop_debug_recording()
        args = argparse.Namespace(log_file=lp, frame=2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.lt.cmd_audit_pairs(args)
        out = buf.getvalue()
        self.assertIn("Coverage Audit", out)
        self.assertIn("VT:", out)


class TestReleasePersistPredicate(unittest.TestCase):
    """リリースゲート＋持続条件述語の純粋テスト (GPU不要)。"""

    def setUp(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        python_pkg = os.path.join(project_root, "python")
        if python_pkg not in sys.path:
            sys.path.insert(0, python_pkg)
        from taremin_cloth import replayer as rp
        self.rp = rp

    def test_fires_after_release_with_persist(self):
        # pins: 1,1,0,0,0 / base hit: 全T -> release at F2, fire at F3
        hits = [True, True, True, True, True]
        pins = [1, 1, 0, 0, 0]
        pred = self.rp.build_release_persist_predicate(
            lambda ctx: "hit" if hits[ctx["frame_index"]] else False,
            persist=2, require_release=True)
        out = []
        for f in range(5):
            out.append(pred({"frame_index": f, "log_frame": {"pins": [{}] * pins[f]}}))
        self.assertFalse(out[0])
        self.assertFalse(out[1])
        self.assertFalse(out[2])  # リリース直後は run=1 < persist
        self.assertTrue(out[3])   # 連続2で発火
        self.assertIn("released at F2", out[3])
        self.assertTrue(out[4])

    def test_run_resets_on_gap(self):
        pred = self.rp.build_release_persist_predicate(
            lambda ctx: "hit" if ctx["frame_index"] in (2, 4) else False,
            persist=2, require_release=True)
        seq = [(0, 1), (1, 1), (2, 0), (3, 0), (4, 0)]
        out = [pred({"frame_index": f, "log_frame": {"pins": [{}] * p}}) for f, p in seq]
        self.assertFalse(out[2])  # run=1
        self.assertFalse(out[3])  # base偽でリセット
        self.assertFalse(out[4])  # run=1 で未達

    def test_no_release_no_fire(self):
        pred = self.rp.build_release_persist_predicate(
            lambda ctx: "always", persist=1, require_release=True)
        out = [pred({"frame_index": f, "log_frame": {"pins": [{}]}}) for f in range(3)]
        self.assertEqual(out, [False, False, False])

    def test_passthrough_without_gate(self):
        pred = self.rp.build_release_persist_predicate(
            lambda ctx: "hit" if ctx["frame_index"] == 1 else False,
            persist=1, require_release=False)
        out = [pred({"frame_index": f, "log_frame": {"pins": []}}) for f in range(3)]
        self.assertEqual(out, [False, "hit", False])

    def test_state_exposed(self):
        pred = self.rp.build_release_persist_predicate(None, persist=3, require_release=True)
        self.assertEqual(pred.state["release_frame"], None)
        pred({"frame_index": 0, "log_frame": {"pins": [{}]}})
        pred({"frame_index": 1, "log_frame": {"pins": []}})
        self.assertEqual(pred.state["release_frame"], 1)


class TestAuditCacheCommand(unittest.TestCase):
    """audit-cache (ON/OFF再現の起因判定) のテスト。"""

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
        from taremin_cloth import pair_audit as pa
        from taremin_cloth import replayer as rp
        from taremin_cloth import log_tools as lt
        self.pa = pa
        self.rp = rp
        self.lt = lt
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_attribute_branches(self):
        pa = self.pa
        self.assertEqual(pa.attribute_penetration(3, 2, 0)[0], "cache-attributable")
        self.assertEqual(pa.attribute_penetration(3, 2, 1)[0], "not-cache-attributable")
        self.assertEqual(pa.attribute_penetration(0, 0, 0)[0], "no-penetration")
        self.assertEqual(pa.attribute_penetration(2, 0, 0)[0], "replay-mismatch")
        self.assertEqual(pa.attribute_penetration(0, 0, 2)[0], "unexpected")

    def test_force_pair_cache_plumbing(self):
        """force指定がsimに反映されること"""
        pos = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.uint32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        inv = np.ones(4, dtype=np.float32)
        sim = self.core.ClothSimulator(pos, edges, faces, inv)
        sim.start_debug_recording("Force", max_frames=10)
        for _ in range(3):
            sim.step(1.0 / 60.0, 2)
        lp = os.path.join(self.temp_dir, "force.jsonl.gz")
        sim.save_debug_recording(lp)
        sim.stop_debug_recording()

        rp_off = self.rp.ClothReplayer(lp)
        rp_off.replay_range(0, 2, compare=False, force_pair_cache=False)
        self.assertFalse(rp_off.sim.get_enable_pair_cache())
        rp_on = self.rp.ClothReplayer(lp)
        rp_on.replay_range(0, 2, compare=False, force_pair_cache=True)
        self.assertTrue(rp_on.sim.get_enable_pair_cache())

    def test_audit_cache_end_to_end(self):
        """audit-cache CLIの端到端動作 (貫通なしログでは no-penetration)"""
        pos = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.uint32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        inv = np.ones(4, dtype=np.float32)
        sim = self.core.ClothSimulator(pos, edges, faces, inv, enable_pair_cache=True)
        sim.set_enable_self_collision(True)
        sim.start_debug_recording("AC", max_frames=10)
        for _ in range(3):
            sim.step(1.0 / 60.0, 2)
        lp = os.path.join(self.temp_dir, "ac.jsonl.gz")
        sim.save_debug_recording(lp)
        sim.stop_debug_recording()
        args = argparse.Namespace(log_file=lp, frame=2, lookback=2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.lt.cmd_audit_cache(args)
        out = buf.getvalue()
        self.assertIn("verdict:", out)

    def test_watch_require_release_end_to_end(self):
        """ピン解放後のみ発火すること (記録ログ＋watch)"""
        pos = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.uint32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        inv = np.ones(4, dtype=np.float32)
        sim = self.core.ClothSimulator(pos, edges, faces, inv)
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_pin(0, [0.0, 0.0, 1.0], 1.0)
        sim.start_debug_recording("Rel", max_frames=10)
        sim.step(1.0 / 60.0, 2)
        sim.step(1.0 / 60.0, 2)
        sim.clear_pins()  # 解放
        sim.step(1.0 / 60.0, 2)
        sim.step(1.0 / 60.0, 2)
        lp = os.path.join(self.temp_dir, "rel.jsonl.gz")
        sim.save_debug_recording(lp)
        sim.stop_debug_recording()

        out_path = os.path.join(self.temp_dir, "hit.jsonl.gz")
        args = argparse.Namespace(
            log_file=lp, stop_on="disp>0.0001", start=0, end=None, stride=1,
            disp=None, intersections=False, vt_sat=False, watch_verts=None,
            watch_disp_mm=2.0, max_diff_mm=None, lookback=1, lookahead=1,
            output=out_path, persist=1, require_release=True,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.lt.cmd_watch(args)
        out = buf.getvalue()
        self.assertIn("HIT Frame 2", out)
        self.assertIn("released at F2", out)
        self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
