# -*- coding: utf-8 -*-
"""BONE_SDF記録・復元 (テクスチャ+姿勢) のテスト"""

import base64
import gzip
import json
import os
import struct
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


def make_sphere_sdf(w, h, d, center=(0.5, 0.5, 0.5), radius=0.2):
    """Rg16Float密パック (Z->Y->X, [dist, 1.0]) の球SDFを合成する。"""
    buf = bytearray()
    for z in range(d):
        for y in range(h):
            for x in range(w):
                px = (x + 0.5) / w - center[0]
                py = (y + 0.5) / h - center[1]
                pz = (z + 0.5) / d - center[2]
                dist = float(np.sqrt(px * px + py * py + pz * pz) - radius)
                buf += struct.pack("<e", dist)
                buf += struct.pack("<e", 1.0)
    return bytes(buf)


def make_bone_info():
    row = np.zeros(20, dtype=np.float32)
    row[0:3] = [0.0, 0.0, 0.0]      # aabb_min
    row[4:7] = [1.0, 1.0, 1.0]      # aabb_max
    row[8:11] = [1.0, 1.0, 1.0]     # uvw_scale
    row[12:15] = [0.0, 0.0, 0.0]    # uvw_offset
    row[16:20] = [0.3, 0.01, 0.0, 0.0]  # friction, thickness, restitution, blend
    return row.reshape(1, 20)


def translate_x(dx):
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = dx
    return m.reshape(1, 4, 4)


class TestBoneSdfRecording(unittest.TestCase):
    # わざと256非アライン (row=40B) かつ非立方体でパディング剥離を検証
    W, H, D = 10, 12, 14

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
        self.tex = make_sphere_sdf(self.W, self.H, self.D)
        self.infos = make_bone_info()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _cloth_inside(self):
        # 球中心からずらしたパッチ (SDF内部だが勾配特異点を回避)。
        # 中心ちょうどでは押し出し方向が不定となり再現比較できない。
        pos = np.array([[0.48, 0.48, 0.62], [0.52, 0.48, 0.62],
                        [0.52, 0.52, 0.62], [0.48, 0.52, 0.62]], dtype=np.float32)
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.uint32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        return pos, edges, faces, np.ones(4, dtype=np.float32)

    def test_metadata_roundtrip_exact(self):
        """テクスチャバイトの完全一致 (パディング剥離の検証)。"""
        pos, edges, faces, inv = self._cloth_inside()
        sim = self.core.ClothSimulator(pos, edges, faces, inv)
        sim.set_bone_sdf_colliders(self.W, self.H, self.D, self.tex, self.infos)
        eye = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
        sim.update_bone_transforms(eye, eye)
        sim.start_debug_recording("Bone", max_frames=10)
        sim.step(1.0 / 60.0, 2)
        lp = os.path.join(self.temp_dir, "bone.jsonl.gz")
        sim.save_debug_recording(lp)
        sim.stop_debug_recording()

        with gzip.open(lp, "rt", encoding="utf-8") as f:
            meta = json.loads(f.readline())["metadata"]
        rec = meta.get("bone_sdf")
        self.assertIsInstance(rec, dict)
        self.assertEqual((rec["width"], rec["height"], rec["depth"]),
                         (self.W, self.H, self.D))
        self.assertEqual(base64.b64decode(rec["texture_base64"]), self.tex)
        self.assertFalse(rec.get("dynamic_enabled", False))
        got_infos = np.array(rec["bone_infos"], dtype=np.float32)
        np.testing.assert_array_equal(got_infos, self.infos)

    def test_frame_transforms_and_replay(self):
        """毎フレーム姿勢の記録と完全再現。"""
        pos, edges, faces, inv = self._cloth_inside()
        sim = self.core.ClothSimulator(pos, edges, faces, inv)
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_bone_sdf_colliders(self.W, self.H, self.D, self.tex, self.infos)
        sim.start_debug_recording("BoneAnim", max_frames=10)
        mirs = []
        for i in range(3):
            w = translate_x(0.05 * i)
            iw = np.linalg.inv(w.astype(np.float64)).astype(np.float32)
            mirs.append((w, iw))
            sim.update_bone_transforms(w, iw)
            sim.step(1.0 / 60.0, 2)
        lp = os.path.join(self.temp_dir, "bone_anim.jsonl.gz")
        sim.save_debug_recording(lp)
        sim.stop_debug_recording()

        frames = {f["frame_index"]: f for f in self.rp.iter_frames(lp)}
        for i in range(3):
            bt = frames[i].get("bone_transforms")
            self.assertIsInstance(bt, dict, f"F{i}")
            np.testing.assert_allclose(
                np.array(bt["world"], dtype=np.float32), mirs[i][0],
                rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(
                np.array(bt["inv_world"], dtype=np.float32), mirs[i][1],
                rtol=1e-6, atol=1e-7)

        rp = self.rp.ClothReplayer(lp)
        results = rp.replay_range(start_frame_idx=0, end_frame_idx=2, compare=True)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertTrue(r["compared"])
            self.assertLess(r["max_diff"], 1e-4, f"F{r['frame_index']}")

    def test_old_log_without_bone(self):
        """bone_sdfなしログでは復元系が無操作になること。"""
        pos = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.uint32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        sim = self.core.ClothSimulator(pos, edges, faces)
        sim.start_debug_recording("NoBone", max_frames=5)
        sim.step(1.0 / 60.0, 2)
        lp = os.path.join(self.temp_dir, "nobone.jsonl.gz")
        sim.save_debug_recording(lp)
        sim.stop_debug_recording()

        rp = self.rp.ClothReplayer(lp)
        self.assertIsNone(rp.metadata.get("bone_sdf"))
        p0 = np.array([f for f in self.rp.iter_frames(lp)][0]["positions"],
                      dtype=np.float32)
        rp.create_simulator(p0)
        self.assertFalse(rp.setup_bone_sdf(rp.sim))
        self.assertFalse(rp.apply_frame_bone({"bone_transforms": None}))
        self.assertFalse(rp.apply_frame_bone({}))


if __name__ == "__main__":
    unittest.main()
