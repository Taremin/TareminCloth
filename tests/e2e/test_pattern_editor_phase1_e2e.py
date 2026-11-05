# -*- coding: utf-8 -*-
"""Phase 1: 2D型紙エディタのBlender統合・非破壊ビルド・E2Eテスト."""

import unittest
import bpy

import taremin_cloth
from taremin_cloth.pattern import Document, EdgeRange, Seam, Placement
from taremin_cloth.pattern.bridge import load_document_from_scene, save_document_to_scene
from taremin_cloth.pattern.build import build_garment_mesh_in_place
from taremin_cloth.pattern.editor import EditorController, EditorEvent
from taremin_cloth.pattern.session import ViewTransform2D


class TestPatternEditorPhase1E2E(unittest.TestCase):
    """Blender環境下における型紙エディタの結合・E2Eテスト."""

    def setUp(self):
        taremin_cloth.register()
        for obj in list(bpy.context.scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh, do_unlink=True)
        self.scene = bpy.context.scene

    def tearDown(self):
        for obj in list(bpy.context.scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh, do_unlink=True)
        taremin_cloth.unregister()

    def test_document_scene_persistence(self):
        # 1. Document 作成とシーン保存
        doc = Document.new(name="E2E Pattern")
        p1 = doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.3), name="Front")
        p2 = doc.add_rectangle_piece((0.4, 0.0), (0.6, 0.3), name="Back")

        save_document_to_scene(doc, scene=self.scene)
        self.assertTrue(len(self.scene.taremin_pattern_document) > 100)

        # 2. シーンからの復元
        restored = load_document_from_scene(self.scene)
        self.assertEqual(restored.name, "E2E Pattern")
        self.assertEqual(len(restored.pieces), 2)
        self.assertIn(p1.id, restored.pieces)
        self.assertIn(p2.id, restored.pieces)

    def test_build_garment_in_place_non_destructive(self):
        doc = Document.new(name="NonDestructive Test")
        p1 = doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.3), name="Front")
        p2 = doc.add_rectangle_piece((0.4, 0.0), (0.6, 0.3), name="Back")
        p1.target_edge_length = 0.05
        p2.target_edge_length = 0.05

        doc.placements[p2.id] = Placement(piece_id=p2.id, offset_3d=(0.0, 0.1, 0.0))

        segs_a = doc.evaluate_path_segments(p1.id, p1.outer_path_id)
        segs_b = doc.evaluate_path_segments(p2.id, p2.outer_path_id)
        doc.seams["s1"] = Seam(
            id="s1",
            a=EdgeRange(piece_id=p1.id, path_id=p1.outer_path_id, start_node_id=segs_a[1][0], end_node_id=segs_a[1][1]),
            b=EdgeRange(piece_id=p2.id, path_id=p2.outer_path_id, start_node_id=segs_b[3][0], end_node_id=segs_b[3][1]),
        )

        # 1回目のビルド
        obj = build_garment_mesh_in_place(doc, scene=self.scene)
        self.assertIsNotNone(obj)
        self.assertEqual(obj.name, "Garment_Cloth")
        mesh = obj.data
        self.assertGreater(len(mesh.vertices), 10)
        self.assertGreater(len(mesh.polygons), 5)
        self.assertTrue(obj.taremin_cloth.is_cloth)

        # UVMapの確認
        self.assertIn("UVMap", mesh.uv_layers)
        # 面属性の確認
        self.assertIn("pattern_piece_index", mesh.attributes)
        self.assertIn("cloth_layer", mesh.attributes)
        # 頂点グループの確認
        self.assertIn("Piece_Front", obj.vertex_groups)
        self.assertIn("Piece_Back", obj.vertex_groups)

        # オブジェクトにマテリアルを割り当てて非破壊性を検証
        mat = bpy.data.materials.new(name="CustomGarmentMaterial")
        obj.data.materials.append(mat)
        orig_obj_ptr = obj.as_pointer()

        # 2回目のビルド（ノード移動やパラメータ変更後の再ビルドを想定）
        p1.target_edge_length = 0.04
        obj2 = build_garment_mesh_in_place(doc, scene=self.scene)

        # オブジェクト参照が同一（再生成・破棄されていない）こと
        self.assertEqual(obj2.as_pointer(), orig_obj_ptr)
        # マテリアルが保持されていること
        self.assertEqual(len(obj2.data.materials), 1)
        self.assertEqual(obj2.data.materials[0].name, "CustomGarmentMaterial")

    def test_operator_build_garment(self):
        doc = Document.new(name="Op Test")
        p = doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="Square")
        save_document_to_scene(doc, scene=self.scene)

        res = bpy.ops.taremin_cloth.build_garment()
        self.assertEqual(res, {'FINISHED'})

        garment = self.scene.objects.get("Garment_Cloth")
        self.assertIsNotNone(garment)
        self.assertGreater(len(garment.data.vertices), 4)

    def test_editor_controller_full_workflow(self):
        view = ViewTransform2D(pan_x=0.0, pan_y=0.0, zoom=1000.0)
        ctrl = EditorController(doc=Document.new(), view=view)

        # 矩形ツールで前身頃を作成
        ctrl.set_active_tool("RECTANGLE")
        ctrl.handle_event(EditorEvent(screen_pos=(0.0, 0.0), is_press=True))
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 300.0), is_release=True))

        # 矩形ツールで後身頃を作成
        ctrl.handle_event(EditorEvent(screen_pos=(400.0, 0.0), is_press=True))
        ctrl.handle_event(EditorEvent(screen_pos=(600.0, 300.0), is_release=True))

        self.assertEqual(len(ctrl.doc.pieces), 2)

        # 縫合ツールで縫合
        ctrl.set_active_tool("SEW")
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 150.0), is_press=True))
        ctrl.handle_event(EditorEvent(screen_pos=(400.0, 150.0), is_press=True))
        self.assertEqual(len(ctrl.doc.seams), 1)

        # シーン保存とメッシュビルド
        save_document_to_scene(ctrl.doc, scene=self.scene)
        obj = build_garment_mesh_in_place(ctrl.doc, scene=self.scene)
        self.assertIsNotNone(obj)
        self.assertGreater(len(obj.data.vertices), 10)


if __name__ == "__main__":
    unittest.main()
