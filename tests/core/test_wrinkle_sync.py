"""sync_wrinkle_field (engine/params.py) のBlender連携テスト。Blender非依存。

検証: PyO3実API名での呼び出し、正規直交ボーン基底、テクスチャ位置引数、
無効/空カーブ/縮退ボーン時の無効化、署名キャッシュ、捻り変化時の再ベイク。
"""
import math
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from taremin_cloth.engine import params as params_mod
from taremin_cloth.engine.wrinkle_field import WrinkleCurveItem


class _FakeMat:
    def __matmul__(self, vec):
        return list(vec)


def _pb(head, tail, r=0.05):
    pb = MagicMock()
    pb.head = list(head)
    pb.tail = list(tail)
    pb.length = 0.2
    pb.bone = MagicMock()
    pb.bone.head_radius = r
    return pb


def _arm(bones):
    arm = MagicMock()
    arm.pose.bones = dict(bones)
    arm.matrix_world = _FakeMat()
    return arm


def _settings(arm, enabled=True, bone="Bone"):
    s = MagicMock()
    s.use_wrinkle_field = enabled
    col = MagicMock()
    col.name = "TareminCloth_Wrinkles"
    s.wrinkle_collection = col
    s.wrinkle_global_strength = 1.0
    s.wrinkle_armature = arm
    s.wrinkle_bone_name = bone
    return s


def _obj(name, settings):
    o = MagicMock()
    o.name = name
    o.configure_mock(**{"taremin_cloth": settings})
    return o


def _sim():
    return MagicMock(spec=["set_enable_wrinkle_field", "set_wrinkle_field_params",
                           "set_wrinkle_field_texture_2d"])


def _ring(radius=0.06, z=0.0, n=16):
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.stack([radius * np.cos(th), radius * np.sin(th),
                     np.full_like(th, z)], axis=-1).astype(np.float32)
class TestSyncWrinkleField(unittest.TestCase):
    def setUp(self):
        params_mod._prev_wrinkle_signatures.clear()

    def _bpy(self, arms=()):
        fb = MagicMock()
        fb.data.objects = list(arms)
        return patch.object(params_mod, "bpy", fb)

    def _curves(self, items):
        return patch("taremin_cloth.engine.wrinkle_field.extract_curves_from_blender_collection",
                     return_value=items)

    def test_calls_correct_pyo3_names_with_orthonormal_basis(self):
        pb = _pb([0.0, 0.0, 0.0], [0.0, 0.0, 0.2])
        arm = _arm({"Bone": pb})
        obj = _obj("ClothA", _settings(arm))
        sim = _sim()
        with self._bpy([arm]), self._curves(([_crest()], [])):
            params_mod.sync_wrinkle_field(sim, obj)
        sim.set_enable_wrinkle_field.assert_called_once_with(True)
        self.assertTrue(sim.set_wrinkle_field_params.called)
        self.assertTrue(sim.set_wrinkle_field_texture_2d.called)
        self.assertFalse(hasattr(sim, "set_wrinkle_field_enabled"))
        kw = sim.set_wrinkle_field_params.call_args
        p = kw.kwargs if kw.kwargs else dict(zip(
            ("origin", "axis", "normal", "binormal", "influence_radius",
             "bone_radius", "stiffness", "blend_weight"), kw.args))
        ax = np.array(p["axis"], dtype=np.float64)
        no = np.array(p["normal"], dtype=np.float64)
        bi = np.array(p["binormal"], dtype=np.float64)
        self.assertAlmostEqual(float(np.linalg.norm(ax)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(no)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(bi)), 1.0, places=5)
        self.assertAlmostEqual(float(np.dot(ax, no)), 0.0, places=5)
        self.assertAlmostEqual(float(np.dot(ax, bi)), 0.0, places=5)
        self.assertAlmostEqual(float(np.dot(no, bi)), 0.0, places=5)
        self.assertAlmostEqual(float(ax[2]), 1.0, places=5)
        self.assertGreater(float(np.linalg.norm(ax)), 0.5)

    def test_texture_positional_signature(self):
        pb = _pb([0.0, 0.0, 0.0], [0.0, 0.0, 0.2])
        arm = _arm({"Bone": pb})
        obj = _obj("ClothB", _settings(arm))
        sim = _sim()
        with self._bpy([arm]), self._curves(([_crest()], [])):
            params_mod.sync_wrinkle_field(sim, obj)
        args, kwargs = sim.set_wrinkle_field_texture_2d.call_args
        self.assertEqual(kwargs, {})
        self.assertEqual(len(args), 7)
        w, h, buf, z0, z1, r0, r1 = args
        self.assertEqual(len(buf), int(w) * int(h) * 4)
        self.assertLess(float(z0), float(z1))
        self.assertLess(float(r0), float(r1))

    def test_second_call_skipped_by_cache(self):
        pb = _pb([0.0, 0.0, 0.0], [0.0, 0.0, 0.2])
        arm = _arm({"Bone": pb})
        obj = _obj("ClothC", _settings(arm))
        sim = _sim()
        with self._bpy([arm]), self._curves(([_crest()], [])):
            params_mod.sync_wrinkle_field(sim, obj)
            self.assertEqual(sim.set_wrinkle_field_texture_2d.call_count, 1)
            params_mod.sync_wrinkle_field(sim, obj)
            self.assertEqual(sim.set_wrinkle_field_texture_2d.call_count, 1)

    def test_disable_clears_signature(self):
        pb = _pb([0.0, 0.0, 0.0], [0.0, 0.0, 0.2])
        arm = _arm({"Bone": pb})
        st = _settings(arm)
        obj = _obj("ClothD", st)
        sim = _sim()
        with self._bpy([arm]), self._curves(([_crest()], [])):
            params_mod.sync_wrinkle_field(sim, obj)
        self.assertIn("ClothD", params_mod._prev_wrinkle_signatures)
        st.use_wrinkle_field = False
        params_mod.sync_wrinkle_field(sim, obj)
        sim.set_enable_wrinkle_field.assert_called_with(False)
        self.assertNotIn("ClothD", params_mod._prev_wrinkle_signatures)

    def test_empty_curves_disables(self):
        pb = _pb([0.0, 0.0, 0.0], [0.0, 0.0, 0.2])
        arm = _arm({"Bone": pb})
        obj = _obj("ClothE", _settings(arm))
        sim = _sim()
        with self._bpy([arm]), self._curves(([], [])):
            params_mod.sync_wrinkle_field(sim, obj)
        sim.set_wrinkle_field_params.assert_not_called()
        sim.set_wrinkle_field_texture_2d.assert_not_called()
        sim.set_enable_wrinkle_field.assert_not_called()
        self.assertNotIn("ClothE", params_mod._prev_wrinkle_signatures)

    def test_degenerate_bone_disables(self):
        pb = _pb([0.0, 0.0, 0.1], [0.0, 0.0, 0.1])
        arm = _arm({"Bone": pb})
        obj = _obj("ClothF", _settings(arm))
        sim = _sim()
        with self._bpy([arm]), self._curves(([_crest()], [])):
            params_mod.sync_wrinkle_field(sim, obj)
        sim.set_wrinkle_field_params.assert_not_called()
        sim.set_wrinkle_field_texture_2d.assert_not_called()

    def test_twist_change_invalidates_cache(self):
        pb = _pb([0.0, 0.0, 0.0], [0.0, 0.0, 0.2])
        arm = _arm({"Bone": pb})
        obj = _obj("ClothG", _settings(arm))
        sim = _sim()
        ax = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        with self._bpy([arm]), self._curves(([_crest()], [])), \
                patch("taremin_cloth.engine.wrinkle_field.build_orthonormal_basis") as mb:
            mb.return_value = (ax, np.array([1.0, 0.0, 0.0], dtype=np.float32),
                               np.array([0.0, 1.0, 0.0], dtype=np.float32))
            params_mod.sync_wrinkle_field(sim, obj)
            self.assertEqual(sim.set_wrinkle_field_texture_2d.call_count, 1)
            mb.return_value = (ax, np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                               np.array([0.0, -1.0, 0.0], dtype=np.float32))
            params_mod.sync_wrinkle_field(sim, obj)
            self.assertEqual(sim.set_wrinkle_field_texture_2d.call_count, 2)


if __name__ == "__main__":
    unittest.main()



def _crest():
    return WrinkleCurveItem(points=_ring(), strength=1.0,
                            influence_radius=0.03, target_radius=None)
