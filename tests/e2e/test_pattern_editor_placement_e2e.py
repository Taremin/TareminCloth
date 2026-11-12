# -*- coding: utf-8 -*-
"""型紙エディタのボーン追従配置・Pick Bone・3Dプレビュー・E2Eテスト."""

import unittest
import bpy
import numpy as np


class TestPatternEditorPlacementE2E(unittest.TestCase):
    """Blender環境下での型紙ボーン配置および3Dプレビュー生成の結合検証."""

    def setUp(self):
        import taremin_cloth
        taremin_cloth.register()
        for obj in list(bpy.context.scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh, do_unlink=True)
        for arm in list(bpy.data.armatures):
            bpy.data.armatures.remove(arm, do_unlink=True)

    def tearDown(self):
        import taremin_cloth
        for obj in list(bpy.context.scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh, do_unlink=True)
        for arm in list(bpy.data.armatures):
            bpy.data.armatures.remove(arm, do_unlink=True)
        taremin_cloth.unregister()

    def test_bone_placement_and_tubular_build_e2e(self):
        """アーマチュアが存在するシーンでボーン追従型紙と筒迂回縫合メッシュが正しくビルドされること."""
        scene = bpy.context.scene

        # 1. アーマチュアおよびボーンの作成 (Spine: Z=0.0 -> Z=1.0)
        arm_data = bpy.data.armatures.new("TestArmatureData")
        arm_obj = bpy.data.objects.new("TestArmature", arm_data)
        scene.collection.objects.link(arm_obj)
        bpy.context.view_layer.objects.active = arm_obj

        bpy.ops.object.mode_set(mode='EDIT')
        bone = arm_data.edit_bones.new("Spine")
        bone.head = (0.0, 0.0, 0.0)
        bone.tail = (0.0, 0.0, 1.0)
        bpy.ops.object.mode_set(mode='OBJECT')

        # 2. ボーン抽出ユーティリティの検証
        from taremin_cloth.ops.wrinkle import _find_target_armature, _extract_all_armature_bones
        found_arm = _find_target_armature(bpy.context)
        self.assertIsNotNone(found_arm)
        self.assertEqual(found_arm.name, "TestArmature")

        all_bones = _extract_all_armature_bones(found_arm)
        self.assertEqual(len(all_bones), 1)
        self.assertEqual(all_bones[0][0], "Spine")

        # 3. 型紙ドキュメントおよび配置設定の作成
        from taremin_cloth.pattern.document import Document, EdgeRange, Placement, Seam, SeamRouting
        from taremin_cloth.pattern.build import build_garment_mesh_in_place
        from taremin_cloth.pattern.bridge import save_document_to_scene

        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.3, 0.2), name="Sleeve")
        pl = Placement(
            piece_id=p.id,
            target_bone="Spine",
            bone_t=0.6,
            bone_angle_deg=90.0,
            surface_offset=0.03,
            pattern_up_angle_deg=90.0,
        )
        doc.placements[p.id] = pl
        nodes = p.get_outer_path().nodes
        doc.seams["wrap_seam"] = Seam(
            id="wrap_seam",
            a=EdgeRange(piece_id=p.id, path_id=p.outer_path_id, start_node_id=nodes[0].id, end_node_id=nodes[1].id),
            b=EdgeRange(piece_id=p.id, path_id=p.outer_path_id, start_node_id=nodes[3].id, end_node_id=nodes[2].id),
            routing=SeamRouting.BONE_WRAP,
        )
        save_document_to_scene(doc, scene=scene)

        # 4. 非破壊3D衣服メッシュのビルド実行
        cloth_obj = build_garment_mesh_in_place(doc, scene=scene)
        self.assertIsNotNone(cloth_obj)
        self.assertEqual(cloth_obj.name, "Garment_Cloth")

        # 頂点が存在し、筒モード迂回中継点 (Waypoints) および Loose Edge が追加されていること
        verts = cloth_obj.data.vertices
        self.assertGreater(len(verts), 4)

        # 面を構成しない縫合 Loose Edge が生成されていること (各ステップ3本ずつ)
        cloth_obj.data.calc_loop_triangles()
        face_edges = set()
        for tri in cloth_obj.data.loop_triangles:
            v0, v1, v2 = tri.vertices
            face_edges.add((min(v0, v1), max(v0, v1)))
            face_edges.add((min(v1, v2), max(v1, v2)))
            face_edges.add((min(v2, v0), max(v2, v0)))
        loose_edges = [
            e for e in cloth_obj.data.edges
            if (min(e.vertices[0], e.vertices[1]), max(e.vertices[0], e.vertices[1])) not in face_edges
        ]
        self.assertGreaterEqual(len(loose_edges), 6)
        self.assertTrue(cloth_obj.taremin_cloth.enable_sewing)

        # 頂点座標がワールド原点ではなくボーン位置 (Z=0.6 付近) に配置されていること
        zs = [v.co.z for v in verts]
        mean_z = float(np.mean(zs))
        self.assertAlmostEqual(mean_z, 0.6, delta=0.25)

    def test_draw_callback_3d_preview_execution(self):
        """PatternCanvasRenderer.draw_callback_3d が例外なく3Dプレビューおよび筒迂回ルートを描画すること."""
        from unittest.mock import MagicMock, patch
        from taremin_cloth.pattern.document import Document, EdgeRange, Placement, Seam, SeamRouting
        from taremin_cloth.pattern.editor import EditorController
        from taremin_cloth.pattern.canvas import PatternCanvasRenderer

        doc = Document.new()
        p = doc.add_rectangle_piece((0.5, 0.5), (0.8, 0.7), name="Sleeve")
        pl = Placement(
            piece_id=p.id,
            target_bone="Spine",
            bone_t=0.5,
            bone_angle_deg=45.0,
            surface_offset=0.05,
            pattern_up_angle_deg=90.0,
        )
        doc.placements[p.id] = pl
        nodes = p.get_outer_path().nodes
        doc.seams["wrap_seam"] = Seam(
            id="wrap_seam",
            a=EdgeRange(piece_id=p.id, path_id=p.outer_path_id, start_node_id=nodes[0].id, end_node_id=nodes[1].id),
            b=EdgeRange(piece_id=p.id, path_id=p.outer_path_id, start_node_id=nodes[3].id, end_node_id=nodes[2].id),
            routing=SeamRouting.BONE_WRAP,
        )

        ctrl = EditorController(doc=doc)
        ctrl.selection.select_piece(p.id)

        canvas = PatternCanvasRenderer(ctrl)
        canvas.all_bones_data = [
            ("Spine", [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.05),
        ]

        mock_shader = MagicMock()
        mock_batch = MagicMock()

        with patch("taremin_cloth.utils.drawing.get_3d_uniform_color_shader", return_value=mock_shader), \
             patch("taremin_cloth.utils.drawing.get_point_shader", return_value=mock_shader), \
             patch("taremin_cloth.utils.drawing.batch_for_shader", return_value=mock_batch), \
             patch("taremin_cloth.pattern.canvas.batch_for_shader", return_value=mock_batch), \
             patch("gpu.state.depth_test_set", create=True), \
             patch("gpu.state.depth_test_get", return_value='LESS_EQUAL', create=True), \
             patch("gpu.state.line_width_set", create=True), \
             patch("gpu.state.line_width_get", return_value=1.0, create=True), \
             patch("gpu.state.blend_set", create=True), \
             patch("gpu.state.blend_get", return_value='NONE', create=True), \
             patch("gpu.state.point_size_set", create=True), \
             patch("gpu.state.point_size_get", return_value=1.0, create=True):
            canvas.draw_callback_3d(bpy.context)
            # ボーン八面体 + ボーン軸 + オフセット線 + 3D平面TRIS + 外枠線 + 筒迂回ルートがすべて描画されたこと
            self.assertGreaterEqual(mock_batch.draw.call_count, 5)


if __name__ == "__main__":
    unittest.main()
