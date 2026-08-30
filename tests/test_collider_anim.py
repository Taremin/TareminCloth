"""
コライダーアニメーション機能（シェイプキー、ポーズブレンド、アクション再生、再生制御）のユニットテスト
"""

import unittest
import bpy
import math
import mathutils
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import taremin_cloth
from taremin_cloth.utils import anim_driver


class TestColliderAnimation(unittest.TestCase):
    def setUp(self):
        taremin_cloth.register()
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def tearDown(self):
        taremin_cloth.unregister()

    def test_properties_initialization(self):
        """コライダーアニメーション設定プロパティが正しく初期化されるかテスト"""
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        col_settings = obj.taremin_collider
        col_settings.is_collider = True

        self.assertTrue(hasattr(col_settings, "anim"))
        anim = col_settings.anim
        self.assertFalse(anim.enabled)
        self.assertEqual(anim.target_type, 'POSE_BLEND')
        self.assertEqual(anim.play_mode, 'ONCE')
        self.assertEqual(anim.cycle_frames, 60)
        self.assertEqual(anim.loop_count, 1)
        self.assertFalse(anim.infinite_loop)
        self.assertEqual(anim.easing, 'SMOOTH')
        self.assertAlmostEqual(anim.progress, 0.0)

    def test_cycle_progress_computation(self):
        """再生モードごとの進行度計算（ONCE, REPEAT, PINGPONG）のテスト"""
        # 1. ONCE
        t, finished = anim_driver.compute_cycle_progress(0, 60, play_mode='ONCE')
        self.assertAlmostEqual(t, 0.0)
        self.assertFalse(finished)

        t, finished = anim_driver.compute_cycle_progress(30, 60, play_mode='ONCE')
        self.assertAlmostEqual(t, 0.5)
        self.assertFalse(finished)

        t, finished = anim_driver.compute_cycle_progress(60, 60, play_mode='ONCE')
        self.assertAlmostEqual(t, 1.0)
        self.assertTrue(finished)

        t, finished = anim_driver.compute_cycle_progress(90, 60, play_mode='ONCE')
        self.assertAlmostEqual(t, 1.0)
        self.assertTrue(finished)

        # 2. REPEAT (loop_count=2)
        t, finished = anim_driver.compute_cycle_progress(30, 60, play_mode='REPEAT', loop_count=2)
        self.assertAlmostEqual(t, 0.5)
        self.assertFalse(finished)

        t, finished = anim_driver.compute_cycle_progress(60, 60, play_mode='REPEAT', loop_count=2)
        self.assertAlmostEqual(t, 0.0)
        self.assertFalse(finished)

        t, finished = anim_driver.compute_cycle_progress(90, 60, play_mode='REPEAT', loop_count=2)
        self.assertAlmostEqual(t, 0.5)
        self.assertFalse(finished)

        t, finished = anim_driver.compute_cycle_progress(120, 60, play_mode='REPEAT', loop_count=2)
        self.assertAlmostEqual(t, 1.0)
        self.assertTrue(finished)

        # 3. PINGPONG (loop_count=1 往復 = 2半サイクル)
        # 0 -> 60: 0.0 -> 1.0
        t, finished = anim_driver.compute_cycle_progress(30, 60, play_mode='PINGPONG', loop_count=1)
        self.assertAlmostEqual(t, 0.5)
        self.assertFalse(finished)

        t, finished = anim_driver.compute_cycle_progress(60, 60, play_mode='PINGPONG', loop_count=1)
        self.assertAlmostEqual(t, 1.0)
        self.assertFalse(finished)

        # 60 -> 120: 1.0 -> 0.0 (逆再生)
        t, finished = anim_driver.compute_cycle_progress(90, 60, play_mode='PINGPONG', loop_count=1)
        self.assertAlmostEqual(t, 0.5)
        self.assertFalse(finished)

        # 120: 往復完了
        t, finished = anim_driver.compute_cycle_progress(120, 60, play_mode='PINGPONG', loop_count=1)
        self.assertAlmostEqual(t, 0.0)
        self.assertTrue(finished)

    def test_easing_computation(self):
        """イージング計算（Linear, Smoothstep）のテスト"""
        # Linear
        self.assertAlmostEqual(anim_driver.apply_easing(0.0, 'LINEAR'), 0.0)
        self.assertAlmostEqual(anim_driver.apply_easing(0.5, 'LINEAR'), 0.5)
        self.assertAlmostEqual(anim_driver.apply_easing(1.0, 'LINEAR'), 1.0)

        # Smoothstep: 3*t^2 - 2*t^3
        self.assertAlmostEqual(anim_driver.apply_easing(0.0, 'SMOOTH'), 0.0)
        self.assertAlmostEqual(anim_driver.apply_easing(0.5, 'SMOOTH'), 0.5)
        self.assertAlmostEqual(anim_driver.apply_easing(1.0, 'SMOOTH'), 1.0)

        # Smoothstepは0.25時点で Linear (0.25) より緩やか (速度が低い)
        t_smooth = anim_driver.apply_easing(0.25, 'SMOOTH')
        self.assertLess(t_smooth, 0.25)

    def test_shape_key_animation(self):
        """シェイプキーのアニメーション補間テスト"""
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        col_settings = obj.taremin_collider
        col_settings.is_collider = True
        anim = col_settings.anim
        anim.enabled = True
        anim.target_type = 'SHAPE_KEY'

        # シェイプキーを追加
        basis = obj.shape_key_add(name="Basis")
        key1 = obj.shape_key_add(name="Bend")
        anim.shape_key_name = "Bend"
        anim.start_value = 0.0
        anim.end_value = 1.0
        anim.cycle_frames = 60
        anim.play_mode = 'ONCE'
        anim.easing = 'LINEAR'

        # frame 0
        progress, deformed = anim_driver.step_collider_animation(obj, 0)
        self.assertTrue(deformed)
        self.assertAlmostEqual(key1.value, 0.0)

        # frame 30 (中間)
        progress, deformed = anim_driver.step_collider_animation(obj, 30)
        self.assertTrue(deformed)
        self.assertAlmostEqual(key1.value, 0.5)

        # frame 60 (終了)
        progress, deformed = anim_driver.step_collider_animation(obj, 60)
        self.assertTrue(deformed)
        self.assertAlmostEqual(key1.value, 1.0)

    def test_pose_blend_animation(self):
        """2ポーズスナップショット間のブレンドテスト"""
        # 単純なアーマチュアの作成
        bpy.ops.object.armature_add()
        armature = bpy.context.active_object
        pbone = armature.pose.bones[0]
        pbone.rotation_mode = 'XYZ'

        # ポーズA (0.0): 回転 0度
        pbone.rotation_euler = (0.0, 0.0, 0.0)
        snapshot_a = anim_driver.capture_pose_snapshot(armature)

        # ポーズB (1.0): X軸に 90度 (pi/2) 回転
        pbone.rotation_euler = (math.pi / 2, 0.0, 0.0)
        snapshot_b = anim_driver.capture_pose_snapshot(armature)

        # ブレンド (t = 0.5)
        success = anim_driver.apply_pose_blend(armature, snapshot_a, snapshot_b, 0.5)
        self.assertTrue(success)
        # X軸回転が約45度 (pi/4) になっているか検証
        self.assertAlmostEqual(pbone.rotation_euler.x, math.pi / 4, places=4)

        # コライダー設定との連携テスト
        col_settings = armature.taremin_collider
        col_settings.is_collider = True
        anim = col_settings.anim
        anim.enabled = True
        anim.target_type = 'POSE_BLEND'
        anim.armature_obj = armature
        anim.start_pose_data = snapshot_a
        anim.target_pose_data = snapshot_b
        anim.cycle_frames = 60
        anim.play_mode = 'ONCE'
        anim.easing = 'LINEAR'

        # オペレータ taremin_cloth.record_pose のテスト
        pbone.rotation_euler = (0.0, 0.0, 0.0)
        res = bpy.ops.taremin_cloth.record_pose(slot='START')
        self.assertEqual(res, {'FINISHED'})

        # オペレータ taremin_cloth.apply_pose_preview のテスト
        res = bpy.ops.taremin_cloth.apply_pose_preview(slot='TARGET')
        self.assertEqual(res, {'FINISHED'})
        self.assertAlmostEqual(pbone.rotation_euler.x, math.pi / 2, places=4)

        res = bpy.ops.taremin_cloth.apply_pose_preview(slot='REST')
        self.assertEqual(res, {'FINISHED'})
        self.assertAlmostEqual(pbone.rotation_euler.x, 0.0, places=4)

    def test_action_keyframe_evaluation(self):
        """アクションのFCurve直接評価によるポーズ適用のテスト"""
        bpy.ops.object.armature_add()
        armature = bpy.context.active_object
        pbone = armature.pose.bones[0]
        pbone.rotation_mode = 'XYZ'

        # キーフレーム挿入によりActionを作成
        pbone.rotation_euler = (0.0, 0.0, 0.0)
        pbone.keyframe_insert(data_path="rotation_euler", index=0, frame=1.0)

        pbone.rotation_euler = (math.pi / 2, 0.0, 0.0)
        pbone.keyframe_insert(data_path="rotation_euler", index=0, frame=60.0)

        action = armature.animation_data.action
        self.assertIsNotNone(action)

        # ポーズを一旦リセット
        pbone.rotation_euler = (0.0, 0.0, 0.0)

        # FCurve直接評価でフレーム30.5（中間）を適用
        success = anim_driver.apply_action_frame(armature, action, 30.5)
        self.assertTrue(success)
        expected = (math.pi / 2) * (29.5 / 59.0)
        self.assertAlmostEqual(pbone.rotation_euler.x, expected, places=3)


if __name__ == '__main__':
    unittest.main()
