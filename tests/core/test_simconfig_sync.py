# -*- coding: utf-8 -*-
"""SimConfig同期一本化 (engine.simconfig) のテスト.

- 収集辞書が Rust SimConfig の全キーを網羅すること (欠落防止)
- collect -> apply -> export の往復一致
- coupled分岐・署名感度・旧バイナリフォールバック
"""

import json
import os
import sys
import unittest
from types import SimpleNamespace

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

import numpy as np


def make_settings(**overrides):
    base = dict(
        tension_stiffness=1000.0, compression_stiffness=100.0, shear_stiffness=100.0,
        bending_stiffness=10.0, air_damping=1.0, tension_damping=5.0,
        compression_damping=5.0, shear_damping=5.0, bending_damping=0.5,
        gravity=1.0, solver_iterations=2, solver_mode="COLORING", workgroup_size="32",
        enable_self_collision=True, self_collision_relief_factor=0.2,
        self_collision_max_displacement_ratio=0.2, self_collision_exclude_neighbors=True,
        enable_normal_untangling=True, self_collision_max_iterations="256",
        coupled_self_collision_mode="RELAXATION", post_collision_relaxation_iters=2,
        self_collision_substep_interval=1, enable_pair_cache=True,
        pair_cache_margin_mode="AUTO", pair_cache_safety_margin=0.005,
        pair_cache_horizon_scale=1.3, pair_cache_max_horizon=0.02,
        pair_cache_max_pairs=32768, enable_pair_cache_final_fallback=True,
        enable_edge_collision=False, edge_margin_scale=1.0, edge_margin_offset=0.0,
        sewing_shrink_speed=1.0, sewing_stiffness=10000.0, enable_sewing_lock=True,
        enable_sewing_priority=False, sewing_priority_threshold=0.9,
        sewing_priority_merge_dist=0.005, sewing_priority_ramp_frames=3,
        sewing_priority_max_frames=600,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_scene(gravity=(0.0, 0.0, -9.81), use_gravity=True):
    return SimpleNamespace(gravity=gravity, use_gravity=use_gravity)


class TestSimconfigSync(unittest.TestCase):
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
        from taremin_cloth.engine import simconfig as sc
        self.sc = sc

    def test_collect_covers_rust_config_keys(self):
        """収集辞書がRust SimConfigの全キーを網羅すること"""
        import taremin_cloth_core as core
        pos = np.zeros((2, 3), dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.uint32)
        sim = core.ClothSimulator(pos, edges)
        rust_keys = set(json.loads(sim.get_config_json()).keys())
        cfg = self.sc.collect_sim_config(make_settings(), make_scene())
        missing = rust_keys - set(cfg.keys())
        self.assertEqual(missing, set(), f"収集漏れ: {missing}")

    def test_collect_apply_export_roundtrip(self):
        """collect -> apply -> export の往復一致 (減衰合成式を考慮)"""
        import taremin_cloth_core as core
        # 面ありグリッド (せん断拘束 type=1 が自動生成される)
        pos, edges, faces = [], [], []
        nx, ny, dx = 3, 3, 0.05
        for y in range(ny):
            for x in range(nx):
                pos.append([x * dx, y * dx, 0.0])
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
        pos = np.array(pos, dtype=np.float32)
        edges = np.array(edges, dtype=np.uint32)
        faces = np.array(faces, dtype=np.uint32)
        sim = core.ClothSimulator(pos, edges, faces)
        cfg = self.sc.collect_sim_config(
            make_settings(tension_stiffness=2000.0, coupled_self_collision_mode="FULL_COUPLED",
                          pair_cache_max_pairs=4096, sewing_stiffness=5000.0),
            make_scene(),
        )
        route = self.sc.apply_sim_config(sim, cfg)
        self.assertEqual(route, "config")
        exported = json.loads(sim.get_config_json())
        for key in self.sc.LIVE_KEYS:
            if key == "damping":
                # 実効値 = air + detail合計*0.05
                expect = (cfg["damping"] + (cfg["tension_damping"] + cfg["compression_damping"]
                                            + cfg["shear_damping"] + cfg["bending_damping"]) * 0.05)
                self.assertAlmostEqual(exported[key], expect, places=5, msg=key)
            elif key in ("tension_damping", "compression_damping", "shear_damping", "bending_damping"):
                continue  # 実効値に折り畳まれるためexport側は0
            elif isinstance(cfg[key], float):
                # f32 compliance往復のため相対誤差で判定
                self.assertAlmostEqual(exported[key], cfg[key], delta=abs(cfg[key]) * 1e-4 + 1e-6, msg=key)
            else:
                self.assertEqual(exported[key], cfg[key], msg=key)

    def test_coupled_mapping(self):
        s = make_settings(coupled_self_collision_mode="OFF", post_collision_relaxation_iters=5)
        self.assertEqual(self.sc.coupled_mode_from_settings(s), (0, 0))
        s = make_settings(coupled_self_collision_mode="FULL_COUPLED", post_collision_relaxation_iters=0)
        self.assertEqual(self.sc.coupled_mode_from_settings(s), (3, 1))
        s = make_settings(coupled_self_collision_mode="RELAXATION", post_collision_relaxation_iters=0)
        self.assertEqual(self.sc.coupled_mode_from_settings(s), (1, 2))
        s = make_settings(coupled_self_collision_mode="RELAXATION", post_collision_relaxation_iters=4)
        self.assertEqual(self.sc.coupled_mode_from_settings(s), (1, 4))

    def test_signature_sensitivity(self):
        """全Liveキーの変更が署名に反映されること"""
        base = self.sc.collect_sim_config(make_settings(), make_scene())
        base_sig = self.sc.sim_config_signature(base)
        for key in self.sc.SIGNATURE_KEYS:
            mod = dict(base)
            v = mod[key]
            if isinstance(v, bool):
                mod[key] = not v
            elif isinstance(v, (int, float)):
                mod[key] = v + 1
            elif isinstance(v, (list, tuple)):
                mod[key] = [x + 1 for x in v]
            else:
                continue
            self.assertNotEqual(
                self.sc.sim_config_signature(mod), base_sig, f"署名不感: {key}")

    def test_legacy_fallback_path(self):
        """apply_config_json欠如時は個別setter経路にフォールバックすること"""
        calls = []

        class FakeSim:
            # apply_config_json を持たせない
            def __getattr__(self, name):
                if name.startswith("set_"):
                    def rec(*a, **k):
                        calls.append((name, a, k))
                    return rec
                raise AttributeError(name)

        cfg = self.sc.collect_sim_config(make_settings(), make_scene())
        route = self.sc.apply_sim_config(FakeSim(), cfg)
        self.assertEqual(route, "legacy")
        called = {c[0] for c in calls}
        for expected in ("set_gravity", "set_damping", "set_damping_all",
                         "set_solver_iterations", "set_enable_self_collision",
                         "set_self_collision_options", "set_coupled_self_collision_options",
                         "set_self_collision_substep_interval", "set_enable_pair_cache",
                         "set_pair_cache_options", "set_enable_pair_cache_final_fallback",
                         "set_enable_edge_collision", "set_stiffness_all",
                         "set_sewing_stiffness", "set_enable_sewing_lock",
                         "set_sewing_priority_options"):
            self.assertIn(expected, called, f"legacy未呼出: {expected}")

    def test_gravity_variants(self):
        vec, scale = self.sc.gravity_from_settings(make_settings(gravity=2.0), make_scene())
        self.assertEqual(vec, [0.0, 0.0, -19.62])
        self.assertEqual(scale, 2.0)
        vec, _ = self.sc.gravity_from_settings(make_settings(), make_scene(use_gravity=False))
        self.assertEqual(vec, [0.0, 0.0, 0.0])
        vec, _ = self.sc.gravity_from_settings(make_settings(gravity=0.5), None)
        self.assertAlmostEqual(vec[2], -4.905)

    def test_gui_wire_mapping_has_max_iterations(self):
        """GUI転送辞書にmax_iterationsが含まれること (旧 omission の回帰防止)"""
        from taremin_cloth.engine.gui_client import extract_self_collision_data
        d = extract_self_collision_data(make_settings(self_collision_max_iterations="512"))
        self.assertEqual(d["max_iterations"], 512)
        self.assertEqual(d["coupled_mode"], 1)
        self.assertEqual(d["substep_interval"], 1)


class TestRecordingOptions(unittest.TestCase):
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
        from taremin_cloth.engine import simconfig as sc
        self.sc = sc

    def test_options_from_prefs(self):
        prefs = SimpleNamespace(debug_sparse_stride=5, debug_sparse_ring=3,
                                debug_sparse_lookahead=2, debug_sparse_triggers=True)
        opts = self.sc.recording_options_from_prefs(prefs)
        self.assertEqual(opts, {"full_stride": 5, "ring_size": 3,
                                "lookahead": 2, "enable_triggers": True})
        # None / 欠損時は既定値
        self.assertEqual(self.sc.recording_options_from_prefs(None),
                         {"full_stride": 1, "ring_size": 0,
                          "lookahead": 0, "enable_triggers": False})
        self.assertEqual(
            self.sc.recording_options_from_prefs(SimpleNamespace())["full_stride"], 1)

    def test_apply_recording_options_roundtrip(self):
        import taremin_cloth_core as core
        pos = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.uint32)
        sim = core.ClothSimulator(pos, edges)
        opts = {"full_stride": 5, "ring_size": 3, "lookahead": 2, "enable_triggers": True}
        self.assertTrue(self.sc.apply_recording_options(sim, opts))
        got = dict(sim.get_debug_recording_options())
        self.assertEqual(got["full_stride"], 5)
        self.assertEqual(got["ring_size"], 3)
        self.assertEqual(got["lookahead"], 2)
        self.assertTrue(got["enable_triggers"])

    def test_apply_recording_options_legacy_sim(self):
        class OldSim:
            pass
        self.assertFalse(self.sc.apply_recording_options(OldSim(), {"full_stride": 5}))


if __name__ == "__main__":
    unittest.main()
