"""Phase 3.3 改修項目の回帰・仕様検証テスト (Blender不要)。

対象:
- [ALG-16] / [UX-09] / [UX-14]: ドレープガイドのボーンRoll基底・端点延長スライド・
  スライド強度保持・プリセット保存 curve_types・Crest影響プレビューアルファ向き・複数ボーン絞り込み
- [UX-07] / [UX-08]: プリセット全プロパティ抽出・thickness保護リセット・シミュレータ破棄・
  4種Primary Configメニュー・Dirty Signature補完・アタッチメントピン切替/解除・ベイクESC破棄
- [UX-13]: ClothGuiClient 非ブロッキング部分送信とタイムアウト跨ぎバッファ保持受信
"""

from pathlib import Path
import math
import struct
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tests  # Blender 未導入時の安全なモックセットアップ


def _bone_mat4_from_axis_xz(ax: float, az: float, roll_rad: float) -> list:
    """ローカルY軸がXZ平面の単位ベクトル(ax, 0, az)向きで、ボーン軸周りにroll回転した4x4行列を返す。"""
    y = np.array([ax, 0.0, az], dtype=np.float64)
    y /= np.linalg.norm(y)
    # 基準ローカルX: XZ平面に直交な+Y方向ベース、X=axとの直交方向 (-az, 0, ax) と +Y をRoll混合
    base_z = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    base_x = np.cross(y, base_z)
    base_x /= np.linalg.norm(base_x)
    x = math.cos(roll_rad) * base_x + math.sin(roll_rad) * base_z
    z = np.cross(x, y)
    m4 = np.eye(4, dtype=np.float64)
    m4[:3, 0] = x
    m4[:3, 1] = y
    m4[:3, 2] = z
    return m4.tolist()


class _FakeWorldMat4:
    """matrix_world[i][j] と (matrix_world @ vec3) の両方を満たす単位行列モック。"""
    def __init__(self):
        self._m = np.eye(4, dtype=np.float32)

    def __getitem__(self, idx):
        return self._m[idx]

    def __matmul__(self, vec):
        return list(vec)


class TestWrinkleGeomAndSync(unittest.TestCase):
    def setUp(self):
        from taremin_cloth.engine import params as params_mod
        params_mod.clear_wrinkle_signature_cache(None)

    def tearDown(self):
        from taremin_cloth.engine import params as params_mod
        params_mod.clear_wrinkle_signature_cache(None)

    def test_pose_bone_normal_hint_prevents_90deg_flip_and_follows_roll(self):
        """X=±0.9境界跨ぎでの90度不連続フリップ防止とボーンRoll追従の検証"""
        from taremin_cloth.engine.wrinkle_field import (
            build_orthonormal_basis,
            resolve_pose_bone_normal_hint,
        )

        # ボーン軸をXZ平面で傾け、a[0] が 0.89 -> 0.91 に変化するケース
        ax1, az1 = 0.89, math.sqrt(1.0 - 0.89 ** 2)
        ax2, az2 = 0.91, math.sqrt(1.0 - 0.91 ** 2)
        m1 = _bone_mat4_from_axis_xz(ax1, az1, 0.0)
        m2 = _bone_mat4_from_axis_xz(ax2, az2, 0.0)
        axis1 = [ax1, 0.0, az1]
        axis2 = [ax2, 0.0, az2]

        # 従来 (normal_hintなし): a[0]=0.89 と 0.91 で参照軸がX→Yに切り替わり法線が約90度ジャンプ
        _, n_old1, _ = build_orthonormal_basis(axis1)
        _, n_old2, _ = build_orthonormal_basis(axis2)
        self.assertLess(abs(float(np.dot(n_old1, n_old2))), 0.6)

        # 改善後: PoseBone.matrix のローカルX軸に基づく normal_hint により連続性を維持
        class _BoneProxy:
            def __init__(self, mat):
                self.matrix = mat

        arm = MagicMock()
        arm.matrix_world = _FakeWorldMat4()
        h1 = resolve_pose_bone_normal_hint(arm, _BoneProxy(m1))
        h2 = resolve_pose_bone_normal_hint(arm, _BoneProxy(m2))
        self.assertIsNotNone(h1)
        self.assertIsNotNone(h2)
        _, n_new1, _ = build_orthonormal_basis(axis1, normal_hint=h1)
        _, n_new2, _ = build_orthonormal_basis(axis2, normal_hint=h2)
        self.assertGreater(float(np.dot(n_new1, n_new2)), 0.98)

        # ボーン軸周りの90度Roll回転で法線も90度追従すること
        m_rolled = _bone_mat4_from_axis_xz(ax2, az2, math.pi * 0.5)
        h_roll = resolve_pose_bone_normal_hint(arm, _BoneProxy(m_rolled))
        _, n_roll, _ = build_orthonormal_basis(axis2, normal_hint=h_roll)
        self.assertAlmostEqual(float(np.dot(n_new2, n_roll)), 0.0, delta=1e-4)

    def test_slide_curves_beyond_bone_end_preserves_spacing(self):
        """ボーン端を超えるスライドでも制御点が単一平面へ潰れず軸方向に延長維持されること"""
        from taremin_cloth.engine.wrinkle_slide import BoneChain, slide_curves_along_chain

        chain = BoneChain.from_bones([
            ("Arm", [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.05),
        ])
        th = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
        ring_a = np.stack([0.05 * np.cos(th), 0.05 * np.sin(th), np.full_like(th, 0.7)], axis=-1).astype(np.float32)
        ring_b = np.stack([0.05 * np.cos(th), 0.05 * np.sin(th), np.full_like(th, 0.9)], axis=-1).astype(np.float32)

        # source_t=0.8 (z=0.8) から target_t=1.2 (z=1.2) へスライド -> ring_a=1.1, ring_b=1.3 に延長されるはず
        slid = slide_curves_along_chain(
            [ring_a, ring_b],
            chain,
            source_t=0.8,
            target_t=1.2,
            scale_radius=1.0,
        )
        np.testing.assert_allclose(slid[0][:, 2], 1.1, atol=1e-4)
        np.testing.assert_allclose(slid[1][:, 2], 1.3, atol=1e-4)
        self.assertAlmostEqual(float(np.mean(slid[1][:, 2] - slid[0][:, 2])), 0.2, delta=1e-4)

    def test_sync_wrinkle_field_keeps_slide_strength_across_rigid_follow_frames(self):
        """スライド等価変換で強度変化後、次フレームの剛体追従でもstrength_scaleが消失しないこと"""
        from taremin_cloth.engine import params as params_mod
        from taremin_cloth.engine.wrinkle_field import WrinkleCurveItem

        th = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        pts0 = np.stack([0.06 * np.cos(th), 0.06 * np.sin(th), np.full_like(th, 0.1)], axis=-1).astype(np.float32)
        pts1 = np.stack([0.06 * np.cos(th), 0.06 * np.sin(th), np.full_like(th, 0.12)], axis=-1).astype(np.float32)

        pb = MagicMock()
        pb.head = [0.0, 0.0, 0.0]
        pb.tail = [0.0, 0.0, 0.2]
        pb.length = 0.2
        pb.bone = MagicMock(head_radius=0.05, tail_radius=0.05)

        arm = MagicMock()
        arm.pose.bones = {"Bone": pb}
        arm.matrix_world = _FakeWorldMat4()

        settings = MagicMock()
        settings.use_wrinkle_field = True
        settings.wrinkle_collection = MagicMock(name="Col")
        settings.wrinkle_collection.name = "Col"
        settings.wrinkle_global_strength = 1.0
        settings.wrinkle_armature = arm
        settings.wrinkle_bone_name = "Bone"
        settings.wrinkle_bone_radius = 0.0
        settings.wrinkle_valley_depth = 0.015
        settings.wrinkle_crest_depth = 0.020

        obj = MagicMock()
        obj.name = "ClothStrengthTest"
        obj.configure_mock(**{"taremin_cloth": settings})

        sim = MagicMock(spec=["set_enable_wrinkle_field", "set_wrinkle_field_params", "set_wrinkle_field_texture_2d"])

        # Frame 1: フルベイク (strength = 1.0 -> stiffness = 30.0)
        c1 = [WrinkleCurveItem(points=pts0, strength=1.0, influence_radius=0.03)]
        # Frame 2: スライド + 強度2.0倍 (strength_scale = 2.0 -> stiffness = 60.0)
        c2 = [WrinkleCurveItem(points=pts1, strength=2.0, influence_radius=0.03)]

        mock_bpy = MagicMock()
        mock_bpy.data.objects = [arm]
        with patch.object(params_mod, "bpy", mock_bpy):
            with patch("taremin_cloth.engine.wrinkle_field.extract_curves_from_blender_collection", return_value=(c1, [])):
                params_mod.sync_wrinkle_field(sim, obj)
            self.assertAlmostEqual(sim.set_wrinkle_field_params.call_args.kwargs["stiffness"], 30.0, delta=1e-4)

            with patch("taremin_cloth.engine.wrinkle_field.extract_curves_from_blender_collection", return_value=(c2, [])):
                params_mod.sync_wrinkle_field(sim, obj)
            # テクスチャ再ベイクされずスライドパスを通って stiffness=60.0 になる
            self.assertEqual(sim.set_wrinkle_field_texture_2d.call_count, 1)
            self.assertAlmostEqual(sim.set_wrinkle_field_params.call_args.kwargs["stiffness"], 60.0, delta=1e-4)

            # Frame 3: 同じカーブのまま剛体追従パスにヒットしても stiffness=60.0 が維持されること
            with patch("taremin_cloth.engine.wrinkle_field.extract_curves_from_blender_collection", return_value=(c2, [])):
                params_mod.sync_wrinkle_field(sim, obj)
            self.assertEqual(sim.set_wrinkle_field_texture_2d.call_count, 1)
            self.assertAlmostEqual(sim.set_wrinkle_field_params.call_args.kwargs["stiffness"], 60.0, delta=1e-4)

    def test_sync_wrinkle_field_filters_multi_bone_collection(self):
        """コレクションに複数ボーンのカーブが混在しても対象ボーンのみに絞り込むこと"""
        from taremin_cloth.engine import params as params_mod
        from taremin_cloth.engine.wrinkle_field import WrinkleCurveItem

        th = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        pts_l = np.stack([0.06 * np.cos(th), 0.06 * np.sin(th), np.full_like(th, 0.1)], axis=-1).astype(np.float32)
        pts_r = np.stack([2.0 + 0.06 * np.cos(th), 0.06 * np.sin(th), np.full_like(th, 0.1)], axis=-1).astype(np.float32)

        pb_l = MagicMock(head=[0.0, 0.0, 0.0], tail=[0.0, 0.0, 0.2], length=0.2)
        pb_l.bone = MagicMock(head_radius=0.05, tail_radius=0.05)
        pb_r = MagicMock(head=[2.0, 0.0, 0.0], tail=[2.0, 0.0, 0.2], length=0.2)
        pb_r.bone = MagicMock(head_radius=0.05, tail_radius=0.05)

        arm = MagicMock()
        arm.pose.bones = {"ArmL": pb_l, "ArmR": pb_r}
        arm.matrix_world = _FakeWorldMat4()

        settings = MagicMock(
            use_wrinkle_field=True,
            wrinkle_global_strength=1.0,
            wrinkle_armature=arm,
            wrinkle_bone_name="ArmL",
            wrinkle_bone_radius=0.0,
            wrinkle_valley_depth=0.015,
            wrinkle_crest_depth=0.020,
        )
        settings.wrinkle_collection = MagicMock()
        settings.wrinkle_collection.name = "Wrinkles"

        obj = MagicMock()
        obj.name = "MultiBoneCloth"
        obj.configure_mock(**{"taremin_cloth": settings})

        sim = MagicMock(spec=["set_enable_wrinkle_field", "set_wrinkle_field_params", "set_wrinkle_field_texture_2d"])

        c_l = WrinkleCurveItem(points=pts_l, strength=1.0, influence_radius=0.03, target_bone="ArmL")
        c_r = WrinkleCurveItem(points=pts_r, strength=1.0, influence_radius=0.03, target_bone="ArmR")

        mock_bpy = MagicMock()
        mock_bpy.data.objects = [arm]
        with patch.object(params_mod, "bpy", mock_bpy), \
             patch("taremin_cloth.engine.wrinkle_field.extract_curves_from_blender_collection", return_value=([c_l, c_r], [])):
            params_mod.sync_wrinkle_field(sim, obj)

        # 記録された記述子が ArmL の1本分のみ（ArmR が混入していない）であること
        sig_entry = params_mod._prev_wrinkle_signatures["MultiBoneCloth"]
        self.assertEqual(len(sig_entry["desc"]), 1)

    def test_save_wrinkle_preset_preserves_root_and_crest_types(self):
        """save_wrinkle_preset が谷カーブ (root) と山カーブ (crest) の種別を正確に保存すること"""
        from taremin_cloth.ops.wrinkle import TAREMIN_CLOTH_OT_save_wrinkle_preset
        from taremin_cloth.engine.wrinkle_preset import load_preset_from_json

        def _make_curve_obj(w_type: str, r: float):
            c_obj = MagicMock()
            c_obj.type = 'CURVE'
            c_obj.matrix_world = np.eye(4, dtype=np.float32)
            tw = MagicMock(curve_type=w_type, strength=1.2, influence_radius=0.025, target_radius=0.0)
            c_obj.taremin_wrinkle = tw
            props = {
                "wrinkle_type": w_type,
                "wrinkle_strength": 1.2,
                "wrinkle_influence_radius": 0.025,
                "wrinkle_armature": "Arm",
                "wrinkle_target_bone": "BoneA",
            }
            c_obj.get = lambda k, default=None, _p=props: _p.get(k, default)
            sp = MagicMock()
            p0 = MagicMock(co=[r, 0.0, 0.5, 1.0])
            p1 = MagicMock(co=[0.0, r, 0.5, 1.0])
            sp.points = [p0, p1]
            c_obj.data.splines = [sp]
            return c_obj

        curve_root = _make_curve_obj("root", 0.05)
        curve_crest = _make_curve_obj("crest", 0.07)

        pb = MagicMock(head=[0.0, 0.0, 0.0], tail=[0.0, 0.0, 1.0], length=1.0)
        pb.matrix = np.eye(4, dtype=np.float32).tolist()
        pb.bone = MagicMock(head_radius=0.05, tail_radius=0.05)
        arm = MagicMock(type='ARMATURE')
        arm.matrix_world = np.eye(4, dtype=np.float32).tolist()
        arm.pose.bones = {"BoneA": pb}

        # matrix_world @ vec をリスト内積で処理できるよう差し替え
        class _Mat4:
            def __init__(self):
                self._m = np.eye(4, dtype=np.float32)

            def __getitem__(self, idx):
                return self._m[idx]

            def __matmul__(self, v):
                return list(v)

        arm.matrix_world = _Mat4()
        curve_root.matrix_world = _Mat4()
        curve_crest.matrix_world = _Mat4()

        op = TAREMIN_CLOTH_OT_save_wrinkle_preset()
        op.preset_name = "test_root_and_crest"
        op.category = "user"
        op.description = "desc"
        op.bone_name = "BoneA"
        op.report = MagicMock()

        ctx = MagicMock()
        ctx.selected_objects = [curve_root, curve_crest]
        ctx.scene.objects = [arm]

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("bpy.data.objects.get", return_value=arm), \
             patch("taremin_cloth.engine.wrinkle_preset.get_user_presets_directory", return_value=tmpdir):
            res = op.execute(ctx)
            self.assertEqual(res, {'FINISHED'})
            loaded = load_preset_from_json(str(Path(tmpdir) / "test_root_and_crest.json"))
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.curves), 2)
            self.assertEqual(loaded.curves[0].type, "root")
            self.assertEqual(loaded.curves[1].type, "crest")

    def test_crest_influence_mesh_alpha_increases_from_target_to_edge(self):
        """Crest影響範囲シェルのアルファが目標側 (i=0) から最内縁 (i=-1) へ向かい単調増加すること"""
        from taremin_cloth.utils.drawing import build_wrinkle_influence_meshes

        th = np.linspace(0.0, 2.0 * math.pi, 32)
        pts = [[float(0.07 * np.cos(t)), float(0.07 * np.sin(t)), 0.05] for t in th]
        meshes = build_wrinkle_influence_meshes(
            [("crest", pts, 0.03, 0.0)],
            [0, 0, 0], [0, 0, 1], [1, 0, 0], 0.05,
            crest_depth=0.02, fade_shells=3,
        )
        self.assertEqual(len(meshes), 3)
        alphas = [m[3] for m in meshes]
        self.assertLess(alphas[0], alphas[1])
        self.assertLess(alphas[1], alphas[2])


class TestPresetsCachePinBake(unittest.TestCase):
    def setUp(self):
        from taremin_cloth.engine import cache as cache_mod
        cache_mod.clear_simulators()

    def tearDown(self):
        from taremin_cloth.engine import cache as cache_mod
        cache_mod.clear_simulators()

    def test_extract_properties_for_category_full_keys_no_last_preset(self):
        """extract_properties_for_category がシミュレーション・コライダーの全対象キーを抽出しlast_*_presetを除外すること"""
        from taremin_cloth.presets import (
            extract_properties_for_category,
            CLOTH_SIMULATION_PROPERTIES,
            COLLIDER_RESET_PROPERTIES,
        )

        class _ClothSettings:
            pass

        cloth_s = _ClothSettings()
        for k in CLOTH_SIMULATION_PROPERTIES:
            setattr(cloth_s, k, True if "enable_" in k else 42)
        cloth_s.thickness = 0.005
        cloth_s.tension_stiffness = 1500.0

        col_s = _ClothSettings()
        for k in COLLIDER_RESET_PROPERTIES:
            setattr(col_s, k, True if "enable_" in k else 0.5)
        col_s.collider_type = 'CAPSULE'

        obj = MagicMock()
        obj.taremin_cloth = cloth_s
        obj.taremin_cloth_collider = col_s

        sim_data = extract_properties_for_category(obj, 'simulation')
        self.assertIn("max_substeps", sim_data)
        self.assertIn("solver_mode", sim_data)
        self.assertIn("auto_compensate_iterations", sim_data)
        self.assertIn("gravity", sim_data)
        self.assertNotIn("last_simulation_preset", sim_data)

        col_data = extract_properties_for_category(obj, 'collider')
        self.assertIn("collider_type", col_data)
        self.assertIn("sdf_resolution", col_data)
        self.assertIn("single_sided", col_data)
        self.assertNotIn("last_collider_preset", col_data)

    def test_reset_properties_preserves_thickness_when_purpose_callback_fires(self):
        """COLLISIONリセットでself_collision_purposeの副作用が起きてもthicknessが保護復元されること"""
        from taremin_cloth.presets import reset_properties_to_defaults, CLOTH_COLLISION_PROPERTIES

        class _PropDef:
            def __init__(self, default):
                self.type = 'FLOAT' if isinstance(default, float) else ('BOOL' if isinstance(default, bool) else 'ENUM')
                self.default = default
                self.is_array = False

        class _BlRna:
            properties = {
                "self_collision_purpose": _PropDef("STANDARD"),
                "enable_self_collision": _PropDef(False),
                "self_collision_relief_factor": _PropDef(1.0),
            }

        class _MockClothPG:
            bl_rna = _BlRna()

            def __init__(self):
                self._thickness = 0.012
                self.enable_self_collision = True
                self.self_collision_relief_factor = 0.3
                self._purpose = "GARMENT_SEWING"

            @property
            def thickness(self):
                return self._thickness

            @thickness.setter
            def thickness(self, v):
                self._thickness = v

            @property
            def self_collision_purpose(self):
                return self._purpose

            @self_collision_purpose.setter
            def self_collision_purpose(self, val):
                self._purpose = val
                # 本番 update_self_collision_purpose が thickness を勝手に書き換える挙動を再現
                self._thickness = 0.002

        pg = _MockClothPG()
        reset_properties_to_defaults(pg, CLOTH_COLLISION_PROPERTIES)
        self.assertEqual(pg.self_collision_purpose, "STANDARD")
        self.assertFalse(pg.enable_self_collision)
        self.assertAlmostEqual(pg.thickness, 0.012, delta=1e-6)

    def test_reset_cloth_settings_invokes_clear_simulator_for_object(self):
        """reset_cloth_settings.execute が clear_simulator_for_object を正しく呼び出しキャッシュ破棄すること"""
        from taremin_cloth.presets import TAREMIN_CLOTH_OT_reset_cloth_settings
        from taremin_cloth.engine import cache as cache_mod

        class _PG:
            class bl_rna:
                properties = {}
            is_cloth = True
            thickness = 0.005

        obj = MagicMock()
        obj.name = "ResetTargetCloth"
        obj.taremin_cloth = _PG()

        cache_mod._simulators["ResetTargetCloth"] = (MagicMock(), np.zeros(3, dtype=np.float32))
        ctx = MagicMock()
        ctx.active_object = obj

        op = TAREMIN_CLOTH_OT_reset_cloth_settings()
        op.scope = 'FABRIC'
        op.report = MagicMock()
        res = op.execute(ctx)
        self.assertEqual(res, {'FINISHED'})
        self.assertNotIn("ResetTargetCloth", cache_mod._simulators)

    def test_config_preset_menu_renders_primary_four_only(self):
        """TAREMIN_CLOTH_MT_config_presets.draw が重複エイリアスを含めずPrimary 4件のみ描画すること"""
        from taremin_cloth.presets import TAREMIN_CLOTH_MT_config_presets, PRIMARY_BUILTIN_CONFIG_PRESETS

        recorded_ops = []

        class _OpProxy:
            def __init__(self, op_id):
                self.op_id = op_id
                self.preset_name = None

        class _LayoutProxy:
            def label(self, **kwargs):
                pass

            def separator(self):
                pass

            def operator(self, op_id, **kwargs):
                p = _OpProxy(op_id)
                recorded_ops.append(p)
                return p

        menu = TAREMIN_CLOTH_MT_config_presets()
        menu.layout = _LayoutProxy()
        ctx = MagicMock()
        ctx.scene = None
        menu.draw(ctx)

        preset_names = [op.preset_name for op in recorded_ops if op.op_id == "taremin_cloth.apply_config_preset"]
        self.assertEqual(preset_names, list(PRIMARY_BUILTIN_CONFIG_PRESETS))

    def test_cloth_params_signature_detects_added_fields_and_scene_gravity(self):
        """get_cloth_params_signature が layer_id/wrinkle_*/string.pull_length/triangulation_mode/scene.gravity を検知すること"""
        from taremin_cloth.engine.cache import get_cloth_params_signature
        from taremin_cloth.engine.simconfig import LIVE_KEYS, EXTRA_KEYS

        class _StringPart:
            enabled = True
            curve_object = None
            tail_vert_count_left = 4
            tail_vert_count_right = 4
            eyelet_stride = 2
            linear_density = 0.01
            radius = 0.002
            pull_length = 0.0
            pull_offset_left = 0.0
            pull_offset_right = 0.0
            guide_vert_indices_str = "0,1"
            seam_bridge_pairs_str = ""

        class _Settings:
            is_cloth = True
            pin_vertex_group = ""
            pin_target_object = None
            pin_target_bone = ""
            enable_sewing = False
            thickness = 0.005
            enable_coarse_constraints = False
            self_collision_purpose = "STANDARD"
            substeps = 20
            layer_id = 0
            triangulation_mode = "QUAD"
            use_wrinkle_field = False
            wrinkle_global_strength = 1.0
            wrinkle_bone_name = ""
            wrinkle_bone_radius = 0.0
            wrinkle_valley_depth = 0.015
            wrinkle_crest_depth = 0.020
            wrinkle_collection = None
            wrinkle_armature = None
            elastic_groups = []
            string_parts = [_StringPart()]

        st = _Settings()
        for k in tuple(LIVE_KEYS) + tuple(EXTRA_KEYS):
            if not hasattr(st, k):
                setattr(st, k, 1.0 if "stiffness" in k or "gravity" in k else False)
        st.gravity = 1.0

        obj = MagicMock()
        obj.taremin_cloth = st

        sc = MagicMock()
        sc.use_gravity = True
        sc.gravity = [0.0, 0.0, -9.81]

        base = get_cloth_params_signature(obj, scene=sc)

        for field, val in [
            ("layer_id", 2),
            ("triangulation_mode", "DIAGONAL"),
            ("wrinkle_global_strength", 2.5),
        ]:
            orig = getattr(st, field)
            setattr(st, field, val)
            self.assertNotEqual(get_cloth_params_signature(obj, scene=sc), base, f"{field} not detected")
            setattr(st, field, orig)

        st.string_parts[0].pull_length = 0.15
        self.assertNotEqual(get_cloth_params_signature(obj, scene=sc), base)
        st.string_parts[0].pull_length = 0.0

        sc.gravity = [0.0, 0.0, -3.0]
        self.assertNotEqual(get_cloth_params_signature(obj, scene=sc), base)

    def test_sync_attachment_pins_target_switch_and_release(self):
        """ターゲットオブジェクト変更時のキャッシュ更新とNone解除時のrelease_pin呼び出し検証"""
        from taremin_cloth.engine import params as params_mod

        class _Mat:
            def __init__(self, tx):
                self._tx = float(tx)
                self._m = np.eye(4, dtype=np.float32)
                self._m[0, 3] = self._tx

            @property
            def translation(self):
                return MagicMock(x=self._tx, y=0.0, z=0.0)

            def inverted(self):
                return _Mat(-self._tx)

            def __matmul__(self, v):
                arr = np.array([v[0], v[1], v[2], 1.0], dtype=np.float32)
                res = self._m @ arr
                return _Vec3(res[0], res[1], res[2])

            def __iter__(self):
                for row in self._m:
                    yield row.tolist()

        class _Vec3:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = float(x), float(y), float(z)

            def __getitem__(self, i):
                return (self.x, self.y, self.z)[i]

        vg = MagicMock(index=0)
        g_entry = MagicMock(group=0, weight=1.0)
        v0 = MagicMock(index=0, co=_Vec3(0.0, 0.0, 0.0), groups=[g_entry])
        v1 = MagicMock(index=1, co=_Vec3(0.5, 0.0, 0.0), groups=[g_entry])

        target_a = MagicMock()
        target_a.name = "TargetA"
        target_a.type = "EMPTY"
        target_a.matrix_world = _Mat(1.0)

        target_b = MagicMock()
        target_b.name = "TargetB"
        target_b.type = "EMPTY"
        target_b.matrix_world = _Mat(5.0)

        settings = MagicMock(
            pin_target_object=target_a,
            pin_vertex_group="Pin",
            pin_target_bone="",
        )
        obj = MagicMock()
        obj.name = "AttachedCloth"
        obj.matrix_world = _Mat(0.0)
        obj.taremin_cloth = settings
        obj.vertex_groups.get.return_value = vg
        obj.data.vertices = [v0, v1]

        sim = MagicMock(spec=["set_pin", "release_pin"])

        # 1. TargetA をセット
        params_mod.sync_attachment_pins(sim, obj)
        self.assertEqual(sim.set_pin.call_count, 2)
        first_call_v0 = sim.set_pin.call_args_list[0].args[1]
        np.testing.assert_allclose(first_call_v0, [0.0, 0.0, 0.0], atol=1e-5)

        # 2. TargetB に差し替え (同一トポロジーでも新ターゲット基準で追従)
        sim.set_pin.reset_mock()
        settings.pin_target_object = target_b
        params_mod.sync_attachment_pins(sim, obj)
        self.assertEqual(sim.set_pin.call_count, 2)
        # 初期登録はTargetBに対して行われるため相対座標再計算、続いてTargetB移動すると追従
        sim.set_pin.reset_mock()
        target_b.matrix_world = _Mat(7.0)
        params_mod.sync_attachment_pins(sim, obj)
        moved_v0 = sim.set_pin.call_args_list[0].args[1]
        np.testing.assert_allclose(moved_v0, [2.0, 0.0, 0.0], atol=1e-5)

        # 3. ターゲットを None に解除 -> release_pin(0), release_pin(1) が呼ばれること
        settings.pin_target_object = None
        params_mod.sync_attachment_pins(sim, obj)
        released = {c.args[0] for c in sim.release_pin.call_args_list}
        self.assertEqual(released, {0, 1})

    def test_modal_bake_esc_discards_partial_cache(self):
        """ベイク中のESCキャンセルで途中フレームのキャッシュが破棄されパネルロックが残らないこと"""
        from taremin_cloth.ops.bake import TAREMIN_CLOTH_OT_bake
        from taremin_cloth.engine import cache as cache_mod
        from taremin_cloth.engine.cache import is_scene_baked

        class _VertsProxy:
            def __init__(self, n):
                self._n = n
                self._buf = np.zeros(n * 3, dtype=np.float32)

            def __len__(self):
                return self._n

            def foreach_set(self, attr, vals):
                self._buf = np.asarray(vals, dtype=np.float32)

        obj = MagicMock()
        obj.name = "BakeCloth"
        obj.type = 'MESH'
        obj.taremin_cloth = MagicMock(is_cloth=True, enabled=True)
        obj.data.vertices = _VertsProxy(2)
        obj.get = lambda k, default=None: obj._custom.get(k, default)
        obj._custom = {"_taremin_cloth_rest_positions": np.zeros(6, dtype=np.float32).tobytes()}
        obj.__contains__ = lambda self, k: k in obj._custom
        obj.__getitem__ = lambda self, k: obj._custom[k]
        obj.__setitem__ = lambda self, k, v: obj._custom.__setitem__(k, v)
        obj.__delitem__ = lambda self, k: obj._custom.__delitem__(k)

        scene = MagicMock()
        scene.name = "Scene"
        scene.objects = [obj]

        # ベイク途中で2フレーム分キャッシュに格納された状態をシミュレート
        cache_mod._timeline_frame_cache["BakeCloth"] = {
            1: np.zeros(6, dtype=np.float32),
            2: np.ones(6, dtype=np.float32),
        }
        self.assertTrue(is_scene_baked(scene))

        op = TAREMIN_CLOTH_OT_bake()
        op.bake_ctx = {"scene": scene, "cloth_objs": [obj], "start_frame": 1, "end_frame": 10, "baked_count": 2}
        op._timer = None
        op.current_frame = 3
        op.report = MagicMock()

        ctx = MagicMock()
        ctx.scene = scene
        event = MagicMock(type='ESC', value='PRESS')

        res = op.modal(ctx, event)
        self.assertEqual(res, {'CANCELLED'})
        self.assertFalse(is_scene_baked(scene))


class TestGuiClientNonBlockingIpc(unittest.TestCase):
    def test_send_packet_retries_blocking_and_partial_sends(self):
        """_send_packet が BlockingIOError と部分送信を超えて全バイトを送信すること"""
        from taremin_cloth.engine.gui_client import ClothGuiClient

        sent_bytes = bytearray()

        class _PartialNonBlockingSock:
            def __init__(self):
                self._calls = 0

            def send(self, data):
                self._calls += 1
                if self._calls % 2 == 1:
                    raise BlockingIOError("would block")
                chunk = bytes(data[:17])
                sent_bytes.extend(chunk)
                return len(chunk)

        client = ClothGuiClient()
        client.sock = _PartialNonBlockingSock()
        client._is_connected = True

        payload_obj = {"type": "Play", "blob": "x" * 200}
        self.assertTrue(client._send_packet(payload_obj, timeout=2.0))

        pkt_len = struct.unpack_from(">I", sent_bytes, 0)[0]
        self.assertEqual(len(sent_bytes), 4 + pkt_len)
        import json
        decoded = json.loads(sent_bytes[4:].decode("utf-8"))
        self.assertEqual(decoded, payload_obj)

    def test_recv_packet_preserves_buffer_across_timeout(self):
        """_recv_packet が途中タイムアウト時に読みかけバイトを保持して次回呼出しで復元すること"""
        from taremin_cloth.engine.gui_client import ClothGuiClient
        import json

        msg = json.dumps({"type": "Ack", "data": {"message": "hello_world"}}).encode("utf-8")
        frame = struct.pack(">I", len(msg)) + msg

        # 呼び出しシーケンス: 2バイト -> BlockingIOError (1回目タイムアウト) -> 4バイト -> BlockingIOError (2回目) -> 残り
        chunks = [frame[:2], None, frame[2:6], None, frame[6:]]

        class _ChunkedSock:
            def __init__(self):
                self.idx = 0

            def recv(self, _n):
                if self.idx >= len(chunks):
                    raise BlockingIOError("no more yet")
                item = chunks[self.idx]
                self.idx += 1
                if item is None:
                    raise BlockingIOError("temporary EWOULDBLOCK")
                return item

        client = ClothGuiClient()
        client.sock = _ChunkedSock()
        client._is_connected = True

        # 1回目: 2バイトのみ届き即タイムアウト (timeout=0.0) -> None だが _recv_buf に2バイト残る
        self.assertIsNone(client._recv_packet(timeout=0.0))
        self.assertTrue(client.is_connected)
        self.assertEqual(len(client._recv_buf), 2)

        # 2回目: 残りヘッダ2+ペイロード2を受信し即タイムアウト -> None だが計6バイト保持
        self.assertIsNone(client._recv_packet(timeout=0.0))
        self.assertEqual(len(client._recv_buf), 6)

        # 3回目: 残余を受信して破損なくJSONデコード完了、バッファは空に戻る
        res = client._recv_packet(timeout=0.5)
        self.assertEqual(res, {"type": "Ack", "data": {"message": "hello_world"}})
        self.assertEqual(len(client._recv_buf), 0)


if __name__ == "__main__":
    unittest.main()
