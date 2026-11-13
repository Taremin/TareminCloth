import sys
import unittest
from pathlib import Path

import bpy

# プロジェクトルートと python ディレクトリを sys.path に追加
root_dir = Path(__file__).resolve().parent.parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests
import taremin_cloth
from taremin_cloth.engine import (
    _simulators,
    _timeline_frame_cache,
    clear_simulators,
    get_effective_substeps,
    step_cloth_scene,
)
from taremin_cloth.engine.sdf_baker import clear_all_cached_sdf


def _create_cloth_grid(*, name="AdaptiveCloth", location=(0.0, 0.0, 1.0), pin_all=True):
    """テスト用のグリッド布オブジェクトを作成する。"""
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    settings = obj.taremin_cloth
    settings.is_cloth = True
    settings.thickness = 0.005
    settings.substeps = 20
    settings.enable_adaptive_substep = True
    settings.min_substeps = 4
    settings.max_substeps = 20

    if pin_all:
        vg = obj.vertex_groups.new(name="Pin")
        vg.add(list(range(len(obj.data.vertices))), 1.0, 'REPLACE')
        settings.pin_group = vg.name

    return obj


def _get_sim(obj):
    """_simulators から対象オブジェクトのシミュレータ本体を取り出す。"""
    entry = _simulators.get(obj.name)
    return entry[0] if isinstance(entry, tuple) else entry


@unittest.skipUnless(getattr(tests, "_IS_REAL_BLENDER", False), "Blender実行環境が必要です")
class TestAdaptiveSubstep(unittest.TestCase):
    """Rust 物理コア (cloth_core) と連動する適応型サブステップ制御の検証テスト"""

    def setUp(self):
        clear_all_cached_sdf()
        taremin_cloth.register()
        for obj in list(bpy.context.scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh, do_unlink=True)
        for arm in list(bpy.data.armatures):
            bpy.data.armatures.remove(arm, do_unlink=True)
        clear_simulators()
        _timeline_frame_cache.clear()
        self.cloth_obj = _create_cloth_grid(pin_all=True)

    def tearDown(self):
        for obj in list(bpy.context.scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh, do_unlink=True)
        for arm in list(bpy.data.armatures):
            bpy.data.armatures.remove(arm, do_unlink=True)
        clear_simulators()
        _timeline_frame_cache.clear()
        taremin_cloth.unregister()
        clear_all_cached_sdf()

    def test_disabled_returns_base_substeps(self):
        """適応型サブステップ無効時はシミュレータ生成前後ともに基本 substeps を返すこと"""
        settings = self.cloth_obj.taremin_cloth
        settings.enable_adaptive_substep = False
        settings.substeps = 15

        self.assertEqual(get_effective_substeps(self.cloth_obj), 15)

        step_cloth_scene(bpy.context.scene, dt=1.0 / 60.0)
        self.assertIsNotNone(_get_sim(self.cloth_obj))
        self.assertEqual(get_effective_substeps(self.cloth_obj), 15)

    def test_hysteresis_smooth_decay_from_peak(self):
        """緊急ブースト後、静止状態では1ステップ最大2ずつ滑らかに減衰し設定更新後も履歴が維持されること"""
        scene = bpy.context.scene
        dt = 1.0 / 60.0

        # 初回ステップでシミュレータ生成（静止状態 -> min_substeps=4）
        step_cloth_scene(scene, dt=dt)
        sim = _get_sim(self.cloth_obj)
        self.assertIsNotNone(sim)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 4)

        # 急激な外部変位を通知して上限 (20) へ即時ブースト
        sim.note_external_displacement(0.20)
        step_cloth_scene(scene, dt=dt)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 20)

        # 静止のままステップ進行 -> 最大 -2 ずつ減衰 (20 -> 18)
        step_cloth_scene(scene, dt=dt)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 18)

        # パラメータ変更 (apply_config 経由のオプション再同期) を挟んでもヒステリシス履歴が消えないこと
        self.cloth_obj.taremin_cloth.tension_stiffness = 1600.0
        step_cloth_scene(scene, dt=dt)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 16)

        # さらに減衰を続け、min_substeps=4 で下げ止まること
        history = []
        for _ in range(8):
            step_cloth_scene(scene, dt=dt)
            history.append(get_effective_substeps(self.cloth_obj))
        self.assertEqual(history, [14, 12, 10, 8, 6, 4, 4, 4])

    def test_rapid_cloth_motion_boosts_substeps(self):
        """ピン無し自由落下による実頂点変位および外部変位通知でサブステップが上限以内で増加すること"""
        bpy.data.objects.remove(self.cloth_obj, do_unlink=True)
        clear_simulators()
        self.cloth_obj = _create_cloth_grid(name="FreeCloth", pin_all=False)
        self.cloth_obj.taremin_cloth.gravity = 2.0  # 10数ステップで cfl_margin を十分超える落下速度にする

        scene = bpy.context.scene
        dt = 1.0 / 60.0

        observed = []
        for _ in range(16):
            step_cloth_scene(scene, dt=dt)
            observed.append(get_effective_substeps(self.cloth_obj))

        # 非同期変位フィードバックの進行により min_substeps (4) を超えて上昇する
        self.assertGreater(max(observed), 4, f"実頂点運動でのブーストが確認できませんでした: {observed}")
        self.assertLessEqual(max(observed), 20)

        # 巨大な外部変位を与えても adaptive_max_substeps (20) を超えない
        sim = _get_sim(self.cloth_obj)
        sim.note_external_displacement(1.0)
        step_cloth_scene(scene, dt=dt)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 20)

    def test_collider_rapid_motion_triggers_adaptive_boost(self):
        """sync_colliders が検出したコライダーの急激な移動で即座に適応サブステップがブーストされること"""
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(0.0, 0.0, 0.0))
        col_obj = bpy.context.active_object
        col_obj.name = "MovingSphere"
        col_settings = col_obj.taremin_cloth_collider
        col_settings.is_collider = True
        col_settings.collider_type = 'SPHERE'
        col_settings.radius = 0.2

        scene = bpy.context.scene
        dt = 1.0 / 60.0

        # 初回フレーム同期（初回は前フレーム履歴がないため静止扱い = min_substeps 4）
        step_cloth_scene(scene, dt=dt)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 4)

        # 静止の次フレームでも低いまま維持される
        step_cloth_scene(scene, dt=dt)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 4)

        # コライダーを 0.25m 急移動してステップ実行
        col_obj.location.x += 0.25
        bpy.context.view_layer.update()
        step_cloth_scene(scene, dt=dt)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 20)

    def test_gravity_freefall_floor_and_stationary_behavior(self):
        """通常重力・静止時は min_substeps へ抑えられ、大 dt 下では自由落下フロアが下限を保つこと"""
        scene = bpy.context.scene

        # 通常重力 (1.0)・dt=1/60 では自由落下変位 < cfl_margin のため min_substeps=4 まで下がる
        step_cloth_scene(scene, dt=1.0 / 60.0)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 4)

        # 大時間刻み (dt=0.25s) では 0.5*g*dt^2 ≈ 0.307m (cfl_margin=0.005 -> grav_steps = 8)
        # 頂点がピン固定で実変位ゼロでも重力フロアにより下限が引き上げられる（非緊急時は +2/step で 4 -> 6 -> 8 へ追従）
        step_cloth_scene(scene, dt=0.25)
        eff_step1 = get_effective_substeps(self.cloth_obj)
        self.assertEqual(eff_step1, 6)

        step_cloth_scene(scene, dt=0.25)
        eff_step2 = get_effective_substeps(self.cloth_obj)
        self.assertEqual(eff_step2, 8)

    def test_bone_sdf_pose_motion_triggers_adaptive_boost(self):
        """オブジェクト原点不変でも BONE_SDF のポーズボーン単体移動で適応ブーストが発火すること"""
        bpy.ops.object.armature_add()
        arm_obj = bpy.context.active_object
        arm_obj.name = "AdaptiveBoneArmature"
        bpy.ops.object.mode_set(mode='EDIT')
        bone1 = arm_obj.data.edit_bones[0]
        bone1.name = "Bone1"
        bone1.head = (0.0, 0.0, 0.0)
        bone1.tail = (0.0, 0.0, 1.0)
        bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=1.0, location=(0.0, 0.0, 0.5))
        body_obj = bpy.context.active_object
        body_obj.name = "AdaptiveBoneBody"
        vg = body_obj.vertex_groups.new(name="Bone1")
        vg.add(list(range(len(body_obj.data.vertices))), 1.0, 'REPLACE')
        mod = body_obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = arm_obj

        col_settings = body_obj.taremin_cloth_collider
        col_settings.is_collider = True
        col_settings.enabled = True
        col_settings.collider_type = 'BONE_SDF'
        col_settings.sdf_resolution = '32'
        col_settings.sdf_cache_enabled = False
        col_settings.enable_joint_mesh = False

        scene = bpy.context.scene
        dt = 1.0 / 60.0

        step_cloth_scene(scene, dt=dt)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 4)
        step_cloth_scene(scene, dt=dt)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 4)

        # コライダー/アーマチュア本体のオブジェクト座標は動かさず、ポーズボーンのみを 0.25m 移動
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='POSE')
        arm_obj.pose.bones["Bone1"].location = (0.25, 0.0, 0.0)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()

        step_cloth_scene(scene, dt=dt)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 20)

    def test_collider_disable_reenable_clears_stale_motion(self):
        """コライダー無効化中の位置変更は再有効化時に前フレーム変位として誤爆しないこと"""
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(0.0, 0.0, 0.0))
        col_obj = bpy.context.active_object
        col_obj.name = "ToggleSphere"
        col_settings = col_obj.taremin_cloth_collider
        col_settings.is_collider = True
        col_settings.enabled = True
        col_settings.collider_type = 'SPHERE'
        col_settings.radius = 0.2

        scene = bpy.context.scene
        dt = 1.0 / 60.0

        # 初期同期・静止安定
        step_cloth_scene(scene, dt=dt)
        step_cloth_scene(scene, dt=dt)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 4)

        # 無効化 -> 追跡キャッシュから除去される
        col_settings.enabled = False
        step_cloth_scene(scene, dt=dt)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 4)

        # 無効中に別位置へ大きく移動しても、再有効化初フレームでは誤爆しない
        col_obj.location.x = 5.0
        bpy.context.view_layer.update()
        col_settings.enabled = True
        step_cloth_scene(scene, dt=dt)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 4)

        # 再有効化後の実移動では即座にブーストすること
        col_obj.location.x += 0.25
        bpy.context.view_layer.update()
        step_cloth_scene(scene, dt=dt)
        self.assertEqual(get_effective_substeps(self.cloth_obj), 20)


if __name__ == "__main__":
    unittest.main()
