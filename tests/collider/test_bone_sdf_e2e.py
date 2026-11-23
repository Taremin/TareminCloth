"""
taremin_cloth ボーンSDFコライダー E2E統合テスト (Blender API依存)
Armatureモディファイア付き素体メッシュに対するBONE_SDFコライダーのベイク、
布シミュレーションにおける接触・押し出し、ボーンアニメーション追従の検証
"""

import unittest
import bpy
import numpy as np
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests
import taremin_cloth
from taremin_cloth.engine.collider import sync_colliders
from taremin_cloth.engine.sdf_baker import clear_all_cached_sdf


@unittest.skipUnless(getattr(tests, "_IS_REAL_BLENDER", False), "Blender実行環境が必要です")
class TestBoneSdfE2E(unittest.TestCase):
    def setUp(self):
        clear_all_cached_sdf()
        taremin_cloth.register()
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def tearDown(self):
        taremin_cloth.unregister()
        clear_all_cached_sdf()

    def create_skinned_character(self):
        """2本のボーンを持つアーマチュアとスキニング済み円柱素体を作成する"""
        # 1. アーマチュア作成
        bpy.ops.object.armature_add()
        arm_obj = bpy.context.active_object
        arm_obj.name = "Character_Armature"

        bpy.ops.object.mode_set(mode='EDIT')
        edit_bones = arm_obj.data.edit_bones
        bone1 = edit_bones[0]
        bone1.name = "Bone1"
        bone1.head = (0.0, 0.0, 0.0)
        bone1.tail = (0.0, 0.0, 1.0)

        bone2 = edit_bones.new("Bone2")
        bone2.parent = bone1
        bone2.head = (0.0, 0.0, 1.0)
        bone2.tail = (0.0, 0.0, 2.0)
        bpy.ops.object.mode_set(mode='OBJECT')

        # 2. 素体メッシュ作成 (Z=0〜2 に伸びる円柱)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=2.0, location=(0.0, 0.0, 1.0))
        body_obj = bpy.context.active_object
        body_obj.name = "Character_Body"

        # 高さ方向に細分化して胴体頂点を生成
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=10)
        bpy.ops.object.mode_set(mode='OBJECT')

        # 頂点グループ作成とウェイト設定
        vg1 = body_obj.vertex_groups.new(name="Bone1")
        vg2 = body_obj.vertex_groups.new(name="Bone2")

        for v in body_obj.data.vertices:
            z = v.co.z + 1.0  # メッシュ中心が原点なので Z=0〜2
            if z <= 1.0:
                vg1.add([v.index], 1.0, 'REPLACE')
            else:
                vg2.add([v.index], 1.0, 'REPLACE')

        # Armature モディファイア追加
        mod = body_obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = arm_obj

        # コライダー設定
        body_obj.taremin_cloth_collider.is_collider = True
        body_obj.taremin_cloth_collider.enabled = True
        body_obj.taremin_cloth_collider.collider_type = 'BONE_SDF'
        body_obj.taremin_cloth_collider.sdf_resolution = '32'
        body_obj.taremin_cloth_collider.thickness = 0.02
        body_obj.taremin_cloth_collider.sdf_cache_enabled = True

        return arm_obj, body_obj

    def test_bone_sdf_collision_and_animation(self):
        """BONE_SDFコライダーを用いた step_cloth_scene 経由の布シミュレーションとボーンアニメーション追従の検証"""
        arm_obj, body_obj = self.create_skinned_character()

        # 布メッシュ作成 (素体の上部 Z=2.3 付近に水平グリッドを配置)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=5, y_subdivisions=5, size=0.8, location=(0.0, 0.0, 2.3))
        cloth_obj = bpy.context.active_object
        cloth_obj.name = "Cloth_Grid"
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        cloth_settings = cloth_obj.taremin_cloth
        cloth_settings.is_cloth = True
        cloth_settings.thickness = 0.02
        cloth_settings.tension_stiffness = 5000.0
        cloth_settings.bending_stiffness = 20.0
        cloth_settings.substeps = 5

        n_verts = len(cloth_obj.data.vertices)
        scene = bpy.context.scene

        # 1. 自由落下シミュレーション (15ステップ: step_cloth_scene 内でBONE_SDF自動ベイク＆同期)
        res = {}
        for _ in range(15):
            res = taremin_cloth.step_cloth_scene(scene, dt=0.016, update_mesh=True)

        _, out_coords = res[cloth_obj.name]
        res_pos = out_coords.reshape((n_verts, 3))
        min_z = np.min(res_pos[:, 2])

        # 素体頂上は Z=2.0、厚み 0.02 + 0.02 なので、布は Z >= 2.0 付近で停止し貫通しないこと
        self.assertGreaterEqual(min_z, 1.95, f"布がBONE_SDF素体コライダーを貫通していないこと: min_z={min_z}")

        # 2. ボーンアニメーション追従テスト
        center_idx = n_verts // 2
        center_z_initial = float(res_pos[center_idx, 2])

        # アーマチュアの Bone2 をボーン長軸（+Zワールド方向＝ボーンローカル+Y方向）に 0.3m 上昇させるポーズ
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='POSE')
        p_bone2 = arm_obj.pose.bones["Bone2"]
        p_bone2.location = (0.0, 0.3, 0.0)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()

        # 追加でシミュレーション (15ステップ: 毎フレーム step_cloth_scene 内でボーン行列更新同期)
        for _ in range(15):
            res = taremin_cloth.step_cloth_scene(scene, dt=0.016, update_mesh=True)

        _, new_coords = res[cloth_obj.name]
        new_res_pos = new_coords.reshape((n_verts, 3))
        new_center_z = float(new_res_pos[center_idx, 2])

        # ボーンの上昇に伴って布中央が押し上げられていること (Z >= initial + 0.15)
        self.assertGreater(new_center_z, center_z_initial + 0.15, f"ボーン上昇に伴い布中央が押し上げられていること: {new_center_z} > {center_z_initial + 0.15}")

    def test_operators_clear_and_rebake(self):
        """オペレータ clear_bone_sdf_cache および rebake_bone_sdf の正常動作を検証"""
        arm_obj, body_obj = self.create_skinned_character()
        bpy.context.view_layer.objects.active = body_obj

        # 1. 再ベイクオペレータ
        res = bpy.ops.taremin_cloth.rebake_bone_sdf()
        self.assertIn('FINISHED', res)

        # 2. キャッシュ消去オペレータ
        res_clear = bpy.ops.taremin_cloth.clear_bone_sdf_cache()
        self.assertIn('FINISHED', res_clear)


if __name__ == "__main__":
    unittest.main()
