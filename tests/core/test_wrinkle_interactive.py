"""インタラクティブ・シワ調整の回帰テスト (Blender非依存)。

検証内容:
1. splatベイクと総当たり参照の等価性 (量子化1LSB以内)
2. 骨ローカル形状署名の剛体不変性
3. スライド等価変換の照合 (dz/r_scale/strength_scale) と範囲更新の厳密性
4. sync_wrinkle_field のベイク回避 (pose-only / slide-match で texture未再送)
5. 性能スモーク (フルベイクとpose-onlyの上限)
"""
import math
import os
import sys
import time
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
from taremin_cloth.engine.wrinkle_field import (
    WrinkleCurveDesc,
    WrinkleCurveItem,
    bake_wrinkle_2d_sdf_texture,
    build_orthonormal_basis,
    describe_wrinkle_curves,
    match_slide_transform,
    project_points_to_cylindrical,
    wrinkles_descs_match,
    wrinkle_shape_signature,
)


def _wave_ring(radius=0.06, z=0.0, n=24, wave=0.004):
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.stack([
        radius * np.cos(th),
        radius * np.sin(th),
        np.full_like(th, z) + wave * np.cos(th),
    ], axis=-1).astype(np.float32)


def _slant_crest(n=12):
    th = np.linspace(-1.0, 1.0, n)
    return np.stack([
        0.068 * np.cos(th),
        0.068 * np.sin(th),
        np.linspace(-0.03, 0.03, n),
    ], axis=-1).astype(np.float32)


def _brute_force_R(root_pts, origin, axis, normal, bone_radius, z_min, z_max,
                   width, height, influence):
    """旧総当たり方式の文字通り再現 (テスト参照用・低速のため小規模限定)。"""
    axis_norm, norm, _ = build_orthonormal_basis(axis)
    n = np.array(normal, dtype=np.float32)
    n = n - np.dot(n, axis_norm) * axis_norm
    n = n / (np.linalg.norm(n) + 1e-8)
    ths, zs, rs = project_points_to_cylindrical(root_pts, origin, axis_norm, n)
    xs, ys = [], []
    for i in range(len(ths) - 1):
        dth = ((float(ths[i + 1]) - float(ths[i]) + math.pi) % (2.0 * math.pi)) - math.pi
        dz = float(zs[i + 1]) - float(zs[i])
        seg = math.hypot(dth * bone_radius, dz)
        nsub = max(1, int(math.ceil(seg / 0.002)))
        for s in range(nsub + 1):
            f = s / nsub
            xs.append(((float(ths[i]) + f * dth) % (2.0 * math.pi)) * bone_radius)
            ys.append(float(zs[i]) + f * dz)
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    circ = 2.0 * math.pi * bone_radius
    ax3 = np.concatenate([xs - circ, xs, xs + circ])
    ay3 = np.concatenate([ys, ys, ys])
    u = np.linspace(0.0, 1.0, width, endpoint=False)
    v = np.linspace(0.0, 1.0, height, endpoint=True)
    gx = u * 2.0 * math.pi * bone_radius
    gz = z_min + v * (z_max - z_min)
    out = np.empty((height, width), dtype=np.float32)
    for j in range(height):
        for i in range(width):
            d = np.hypot(gx[i] - ax3, gz[j] - ay3)
            best = float(np.min(d))
            t = min(1.0, max(0.0, 1.0 - best / influence))
            out[j, i] = t * t * (3.0 - 2.0 * t) * 255.0
    return out.astype(np.uint8)


class TestSplatParity(unittest.TestCase):
    def test_matches_brute_force(self):
        root = _wave_ring()
        crest = _slant_crest()
        o, ax, no = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
        tex = bake_wrinkle_2d_sdf_texture([crest], [root], o, ax, no,
                                          width=48, height=48)
        ref_r = _brute_force_R(root, o, ax, no, 0.05, tex.z_min, tex.z_max,
                               48, 48, 0.03)
        got_r = tex.rgba_array[:, :, 0]
        diff = np.abs(got_r.astype(np.int16) - ref_r.astype(np.int16))
        self.assertLessEqual(int(diff.max()), 1,
                             f"splat/brute-force乖離: max={diff.max()}")
        # 山チャンネルも発火していること (，许多ゼロでない)
        self.assertGreater(int(np.count_nonzero(tex.rgba_array[:, :, 1])), 100)


class TestShapeSignature(unittest.TestCase):
    def test_rigid_follow_is_stable(self):
        crest = [WrinkleCurveItem(_wave_ring(), 1.0, 0.03)]
        root = [WrinkleCurveItem(_wave_ring(z=-0.02), 1.2, 0.025)]
        d0 = describe_wrinkle_curves(crest, root, (0, 0, 0), (0, 0, 1), (1, 0, 0), 0.03)
        # 骨とカーブを共に剛体移動 → 記述子は許容値内で一致。
        # 移動量は非表現値 (0.05) をあえて使い、1ULPノイズ耐性も検証する。
        shift = np.array([0, 0, 0.05], np.float32)
        moved_c = [WrinkleCurveItem(c.points + shift, 1.0, 0.03) for c in crest]
        moved_r = [WrinkleCurveItem(c.points + shift, 1.2, 0.025) for c in root]
        d1 = describe_wrinkle_curves(moved_c, moved_r, (0, 0, 0.05), (0, 0, 1), (1, 0, 0), 0.03)
        self.assertTrue(wrinkles_descs_match(d0, d1))
        # 1ULPノイズを直接付加しても一致すること
        noisy = [
            WrinkleCurveDesc(
                kind=d.kind,
                theta=d.theta + np.float32(1e-9),
                z=d.z - np.float32(1e-9),
                r=d.r,
                strength=d.strength,
                influence_radius=d.influence_radius,
            )
            for d in d0
        ]
        self.assertTrue(wrinkles_descs_match(d0, noisy))

    def test_shape_change_detected(self):
        c0 = [WrinkleCurveItem(_wave_ring(), 1.0, 0.03)]
        c1 = [WrinkleCurveItem(_wave_ring(radius=0.07), 1.0, 0.03)]
        d0 = describe_wrinkle_curves(c0, [], (0, 0, 0), (0, 0, 1), (1, 0, 0), 0.03)
        d1 = describe_wrinkle_curves(c1, [], (0, 0, 0), (0, 0, 1), (1, 0, 0), 0.03)
        self.assertNotEqual(
            wrinkle_shape_signature(d0, 128, 128, 0.05, 0.07, 0.002),
            wrinkle_shape_signature(d1, 128, 128, 0.05, 0.07, 0.002))


class TestSlideMatch(unittest.TestCase):
    def _descs(self, dz=0.0, r_scale=1.0, strength=1.0):
        th = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        base = np.stack([0.06 * np.cos(th), 0.06 * np.sin(th),
                         np.full_like(th, 0.01)], axis=-1).astype(np.float32)
        pts = base.copy()
        pts[:, 2] += dz
        pts[:, 0:2] *= r_scale
        item = WrinkleCurveItem(pts, strength, 0.03)
        return describe_wrinkle_curves([item], [], (0, 0, 0), (0, 0, 1), (1, 0, 0), 0.03)

    def test_pure_z_slide(self):
        m = match_slide_transform(self._descs(), self._descs(dz=0.02))
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m["dz"], 0.02, places=4)
        self.assertAlmostEqual(m["r_scale"], 1.0, places=3)
        self.assertAlmostEqual(m["r_offset"], 0.0, places=4)

    def test_radial_scale(self):
        m = match_slide_transform(self._descs(), self._descs(r_scale=1.2))
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m["dz"], 0.0, places=4)
        self.assertAlmostEqual(m["r_scale"], 1.2, places=3)

    def test_strength_scale(self):
        m = match_slide_transform(self._descs(), self._descs(strength=1.5))
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m["strength_scale"], 1.5, places=6)

    def test_shape_change_rejected(self):
        d0 = self._descs()
        th = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        other = np.stack([0.06 * np.cos(th), 0.06 * np.sin(th),
                          0.01 + 0.01 * np.cos(3 * th)], axis=-1).astype(np.float32)
        d1 = describe_wrinkle_curves([WrinkleCurveItem(other, 1.0, 0.03)], [],
                                     (0, 0, 0), (0, 0, 1), (1, 0, 0), 0.03)
        self.assertIsNone(match_slide_transform(d0, d1))

    def test_slide_texture_equivalence(self):
        """スライド後のフルベイク ≒ 正準テクスチャ＋範囲シフト (厳密等価の検証)。"""
        o, ax, no = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
        crest = [_wave_ring()]
        root = [_wave_ring(z=-0.02)]
        tex0 = bake_wrinkle_2d_sdf_texture(crest, root, o, ax, no, width=64, height=64)
        dz = 0.015
        shifted_c = [c + np.array([0, 0, dz], dtype=np.float32) for c in crest]
        shifted_r = [c + np.array([0, 0, dz], dtype=np.float32) for c in root]
        tex1 = bake_wrinkle_2d_sdf_texture(shifted_c, shifted_r, o, ax, no,
                                           width=64, height=64)
        self.assertAlmostEqual(tex1.z_min - tex0.z_min, dz, places=4)
        self.assertAlmostEqual(tex1.z_max - tex0.z_max, dz, places=4)
        d = np.abs(tex1.rgba_array.astype(np.int16) - tex0.rgba_array.astype(np.int16))
        self.assertLessEqual(int(d.max()), 1,
                             f"スライド等価性乖離: max={d.max()}")


class _FakeMat:
    def __matmul__(self, vec):
        return list(vec)


def _make_world(head, tail, name="ClothI", global_str=1.0):
    pb = MagicMock()
    pb.head = list(head)
    pb.tail = list(tail)
    pb.length = 0.2
    pb.bone = MagicMock()
    pb.bone.head_radius = 0.05
    pb.bone.tail_radius = 0.05
    arm = MagicMock()
    arm.pose.bones = {"Bone": pb}
    arm.matrix_world = _FakeMat()
    st = MagicMock()
    st.use_wrinkle_field = True
    col = MagicMock()
    col.name = "TareminCloth_Wrinkles"
    st.wrinkle_collection = col
    st.wrinkle_global_strength = global_str
    st.wrinkle_armature = arm
    st.wrinkle_bone_name = "Bone"
    obj = MagicMock()
    obj.name = name
    obj.configure_mock(**{"taremin_cloth": st})
    sim = MagicMock(spec=["set_enable_wrinkle_field", "set_wrinkle_field_params",
                           "set_wrinkle_field_texture_2d"])
    return pb, arm, obj, sim


def _ring_item(r=0.06, z=0.0, n=16):
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    pts = np.stack([r * np.cos(th), r * np.sin(th),
                    np.full_like(th, z)], axis=-1).astype(np.float32)
    return WrinkleCurveItem(points=pts, strength=1.0,
                            influence_radius=0.03, target_radius=None)


class TestSyncAvoidance(unittest.TestCase):
    def setUp(self):
        params_mod._prev_wrinkle_signatures.clear()

    def _run(self, pb, arm, obj, sim, curves):
        fb = MagicMock()
        fb.data.objects = [arm]
        with patch.object(params_mod, "bpy", fb), patch(
                "taremin_cloth.engine.wrinkle_field.extract_curves_from_blender_collection",
                return_value=curves):
            params_mod.sync_wrinkle_field(sim, obj)

    def test_pose_only_skips_upload(self):
        pb, arm, obj, sim = _make_world([0, 0, 0], [0, 0, 0.2])
        curves = ([_ring_item()], [_ring_item()])
        self._run(pb, arm, obj, sim, curves)
        self.assertEqual(sim.set_wrinkle_field_texture_2d.call_count, 1)
        self._run(pb, arm, obj, sim, curves)
        self.assertEqual(sim.set_wrinkle_field_texture_2d.call_count, 1)
        # uniform は毎回更新される (姿勢追従のため)
        self.assertGreaterEqual(sim.set_wrinkle_field_params.call_count, 2)

    def test_bone_translate_uses_slide_match(self):
        pb, arm, obj, sim = _make_world([0, 0, 0], [0, 0, 0.2])
        curves = ([_ring_item()], [_ring_item()])
        self._run(pb, arm, obj, sim, curves)
        z0 = (params_mod._prev_wrinkle_signatures[obj.name]["z_min"],
              params_mod._prev_wrinkle_signatures[obj.name]["z_max"])
        pb.head = [0, 0, 0.01]
        pb.tail = [0, 0, 0.21]
        self._run(pb, arm, obj, sim, curves)
        self.assertEqual(sim.set_wrinkle_field_texture_2d.call_count, 1)
        z1 = (params_mod._prev_wrinkle_signatures[obj.name]["z_min"],
              params_mod._prev_wrinkle_signatures[obj.name]["z_max"])
        self.assertAlmostEqual(z1[0] - z0[0], -0.01, places=4)
        self.assertAlmostEqual(z1[1] - z0[1], -0.01, places=4)

    def test_global_strength_no_rebake(self):
        pb, arm, obj, sim = _make_world([0, 0, 0], [0, 0, 0.2])
        curves = ([_ring_item()], [_ring_item()])
        self._run(pb, arm, obj, sim, curves)
        obj.taremin_cloth.wrinkle_global_strength = 2.0
        self._run(pb, arm, obj, sim, curves)
        self.assertEqual(sim.set_wrinkle_field_texture_2d.call_count, 1)
        kw = sim.set_wrinkle_field_params.call_args
        p = kw.kwargs or {}
        self.assertAlmostEqual(float(p["stiffness"]), 60.0, places=5)

    def test_depth_windows_plumbed(self):
        """谷/山グラデーション窓が uniform に載ること (再ベイクなし)"""
        pb, arm, obj, sim = _make_world([0, 0, 0], [0, 0, 0.2])
        obj.taremin_cloth.wrinkle_valley_depth = 0.025
        obj.taremin_cloth.wrinkle_crest_depth = 0.030
        curves = ([_ring_item()], [_ring_item()])
        self._run(pb, arm, obj, sim, curves)
        kw = sim.set_wrinkle_field_params.call_args
        p = kw.kwargs or {}
        self.assertAlmostEqual(float(p["valley_window"]), 0.025, places=6)
        self.assertAlmostEqual(float(p["crest_window"]), 0.030, places=6)
        # 窓変更だけではテクスチャ再送しない (uniform 更新のみ)
        n_tex = sim.set_wrinkle_field_texture_2d.call_count
        self._run(pb, arm, obj, sim, curves)
        self.assertEqual(sim.set_wrinkle_field_texture_2d.call_count, n_tex)

    def test_shape_change_rebakes(self):
        pb, arm, obj, sim = _make_world([0, 0, 0], [0, 0, 0.2])
        self._run(pb, arm, obj, sim, ([_ring_item()], [_ring_item()]))
        self._run(pb, arm, obj, sim, ([_ring_item(r=0.075)], [_ring_item()]))
        self.assertEqual(sim.set_wrinkle_field_texture_2d.call_count, 2)

    def _two_bone_world(self, settings_bone="Finger"):
        """設定=指 / 実カーブ=胸 の不一致 with 2-bone armature."""

        def _pb(head, tail):
            pb = MagicMock()
            pb.head = list(head)
            pb.tail = list(tail)
            pb.length = 0.2
            pb.bone = MagicMock()
            pb.bone.head_radius = 0.05
            pb.bone.tail_radius = 0.05
            return pb

        chest = _pb([0, 0, 1.0], [0, 0, 1.2])
        finger = _pb([0.5, 0, 1.0], [0.52, 0, 1.0])
        arm = MagicMock()
        arm.pose.bones = {"chest": chest, "Finger": finger}
        arm.matrix_world = _FakeMat()
        st = MagicMock()
        st.use_wrinkle_field = True
        col = MagicMock()
        col.name = "C"
        st.wrinkle_collection = col
        st.wrinkle_global_strength = 1.0
        st.wrinkle_armature = arm
        st.wrinkle_bone_name = settings_bone
        st.wrinkle_bone_radius = 0.0
        obj = MagicMock()
        obj.name = "ClothBone"
        obj.configure_mock(**{"taremin_cloth": st})
        sim = MagicMock(spec=["set_enable_wrinkle_field", "set_wrinkle_field_params",
                               "set_wrinkle_field_texture_2d"])
        return arm, obj, sim

    def _chest_ring(self):
        th = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        pts = np.stack([0.06 * np.cos(th), 0.06 * np.sin(th),
                        np.full_like(th, 1.1)], axis=-1).astype(np.float32)
        return WrinkleCurveItem(points=pts, strength=1.0,
                                influence_radius=0.03, target_radius=None,
                                target_bone="chest")

    def test_unanimous_curve_bone_overrides_stale_setting(self):
        """全カーブ一致の帰属ボーンが陳腐な設定値を上書きすること"""
        arm, obj, sim = self._two_bone_world(settings_bone="Finger")
        curves = ([], [self._chest_ring()])
        self._run(None, arm, obj, sim, curves)
        kw = sim.set_wrinkle_field_params.call_args
        p = kw.kwargs or {}
        # 胸ボーン頭 (0,0,1.0) が使われること（指(0.5,0,1.0)ではない）
        self.assertAlmostEqual(float(p["origin"][0]), 0.0, places=5)
        self.assertAlmostEqual(float(p["origin"][2]), 1.0, places=5)

    def test_mixed_curve_bones_fall_back_to_setting(self):
        """帰属不一致時は設定ボーンに従うこと"""
        arm, obj, sim = self._two_bone_world(settings_bone="Finger")
        c1 = self._chest_ring()
        c2 = self._chest_ring()
        c2.target_bone = "spine"
        curves = ([], [c1, c2])
        self._run(None, arm, obj, sim, curves)
        kw = sim.set_wrinkle_field_params.call_args
        p = kw.kwargs or {}
        self.assertAlmostEqual(float(p["origin"][0]), 0.5, places=5)


class TestPerfSmoke(unittest.TestCase):
    def test_bake_and_pose_only_budgets(self):
        crest = [_wave_ring(n=48)]
        root = [_wave_ring(n=48, z=-0.02)]
        o, ax, no = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
        t0 = time.perf_counter()
        bake_wrinkle_2d_sdf_texture(crest, root, o, ax, no, width=128, height=128)
        bake_ms = (time.perf_counter() - t0) * 1000.0
        print(f"\n[perf] full bake 128px 2 curves: {bake_ms:.1f}ms")
        self.assertLess(bake_ms, 1000.0, f"フルベイクが遅すぎます: {bake_ms:.1f}ms")

        pb, arm, obj, sim = _make_world([0, 0, 0], [0, 0, 0.2])
        curves = ([_ring_item()], [_ring_item()])
        fb = MagicMock()
        fb.data.objects = [arm]
        with patch.object(params_mod, "bpy", fb), patch(
                "taremin_cloth.engine.wrinkle_field.extract_curves_from_blender_collection",
                return_value=curves):
            params_mod._prev_wrinkle_signatures.clear()
            params_mod.sync_wrinkle_field(sim, obj)
            t0 = time.perf_counter()
            params_mod.sync_wrinkle_field(sim, obj)
            pose_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[perf] pose-only sync: {pose_ms:.2f}ms")
        self.assertLess(pose_ms, 100.0, f"pose-only同期が遅すぎます: {pose_ms:.2f}ms")


if __name__ == "__main__":
    unittest.main()
