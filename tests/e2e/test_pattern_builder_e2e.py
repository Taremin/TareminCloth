# -*- coding: utf-8 -*-
"""Document ベース 2D型紙統合ビルド・3Dプロキシ同期・シミュレーション連携 E2Eテスト."""
import sys
import unittest
from pathlib import Path
import bpy
import numpy as np

root_dir = Path(__file__).resolve().parents[2]
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import taremin_cloth
from taremin_cloth import step_cloth_scene
from taremin_cloth.pattern.bridge import load_document_from_scene, save_document_to_scene
from taremin_cloth.pattern.build import build_garment_mesh_in_place
from taremin_cloth.pattern.document import Document, EdgeRange, Placement, Seam
from taremin_cloth.utils.mesh_extract import extract_cloth_mesh_data


class TestPatternBuilderE2E(unittest.TestCase):
    def setUp(self):
        taremin_cloth.register()
        for obj in list(bpy.context.scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh, do_unlink=True)
        for curve in list(bpy.data.curves):
            bpy.data.curves.remove(curve, do_unlink=True)
        bpy.context.scene.taremin_pattern_document = ""
        from taremin_cloth.engine.cache import _timeline_frame_cache, clear_simulators
        clear_simulators()
        _timeline_frame_cache.clear()

    def tearDown(self):
        for obj in list(bpy.context.scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh, do_unlink=True)
        for curve in list(bpy.data.curves):
            bpy.data.curves.remove(curve, do_unlink=True)
        bpy.context.scene.taremin_pattern_document = ""
        from taremin_cloth.engine.cache import _timeline_frame_cache, clear_simulators
        clear_simulators()
        _timeline_frame_cache.clear()
        taremin_cloth.unregister()

    def _build_two_piece_document(self) -> tuple[Document, str, str]:
        """前後身頃2ピースと縫合を持つサンプルDocumentを作成する."""
        doc = Document.new(name="E2E_Garment")
        p_front = doc.add_rectangle_piece((-0.10, -0.10), (0.10, 0.15), name="FrontPiece")
        p_back = doc.add_rectangle_piece((0.20, -0.10), (0.40, 0.15), name="BackPiece")
        p_front.target_edge_length = 0.025
        p_back.target_edge_length = 0.025

        doc.placements[p_front.id] = Placement(piece_id=p_front.id, offset_3d=(0.0, -0.10, 0.50))
        doc.placements[p_back.id] = Placement(piece_id=p_back.id, offset_3d=(0.0, 0.10, 0.50))

        outer_f = p_front.paths[p_front.outer_path_id]
        outer_b = p_back.paths[p_back.outer_path_id]
        doc.seams["seam_side"] = Seam(
            id="seam_side",
            a=EdgeRange(
                piece_id=p_front.id,
                path_id=p_front.outer_path_id,
                start_node_id=outer_f.nodes[1].id,
                end_node_id=outer_f.nodes[2].id,
            ),
            b=EdgeRange(
                piece_id=p_back.id,
                path_id=p_back.outer_path_id,
                start_node_id=outer_b.nodes[0].id,
                end_node_id=outer_b.nodes[3].id,
            ),
        )
        return doc, p_front.id, p_back.id

    def test_document_build_and_simulate(self):
        """Documentから単一3Dメッシュ・UV・頂点グループ・縫合が生成されステップ実行できること."""
        doc, _, _ = self._build_two_piece_document()
        garment_obj = build_garment_mesh_in_place(
            doc,
            target_object_name="TestGarment",
            scene=bpy.context.scene,
        )

        self.assertIsNotNone(garment_obj)
        self.assertEqual(garment_obj.type, 'MESH')
        self.assertTrue(garment_obj.taremin_cloth.is_cloth)
        self.assertTrue(garment_obj.taremin_cloth.enable_sewing)

        self.assertGreater(len(garment_obj.data.uv_layers), 0)
        self.assertIn("Piece_FrontPiece", garment_obj.vertex_groups)
        self.assertIn("Piece_BackPiece", garment_obj.vertex_groups)

        # 共通パイプライン step_cloth_scene で10ステップ物理計算が破綻なく回ること
        for frame in range(1, 11):
            step_cloth_scene(bpy.context.scene, dt=1.0 / 60.0, anim_frame=frame, update_mesh=True)

        data = extract_cloth_mesh_data(garment_obj)
        self.assertTrue(np.all(np.isfinite(data.positions)))

    def test_edit_and_sync_pattern_proxies_3d_roundtrip(self):
        """edit_pattern_proxies_3d と sync_pattern_proxies_3d で3DプロキシとDocumentが往復同期されること."""
        doc, front_id, _ = self._build_two_piece_document()
        save_document_to_scene(doc, scene=bpy.context.scene, push_undo=False)

        res = bpy.ops.taremin_cloth.edit_pattern_proxies_3d()
        self.assertEqual(res, {'FINISHED'})

        proxy_obj = bpy.context.scene.objects.get("Proxy_FrontPiece")
        self.assertIsNotNone(proxy_obj)
        self.assertEqual(proxy_obj.display_type, 'WIRE')
        self.assertTrue(proxy_obj.show_in_front)
        self.assertAlmostEqual(proxy_obj.location.y, -0.10, places=3)
        self.assertAlmostEqual(proxy_obj.location.z, 0.50, places=3)

        # プロキシを3D移動して同期
        proxy_obj.location = (0.15, -0.22, 0.85)
        res_sync = bpy.ops.taremin_cloth.sync_pattern_proxies_3d()
        self.assertEqual(res_sync, {'FINISHED'})

        reloaded = load_document_from_scene(bpy.context.scene)
        pl_front = reloaded.placements[front_id]
        self.assertAlmostEqual(pl_front.offset_3d[0], 0.15, places=3)
        self.assertAlmostEqual(pl_front.offset_3d[1], -0.22, places=3)
        self.assertAlmostEqual(pl_front.offset_3d[2], 0.85, places=3)


if __name__ == "__main__":
    unittest.main()

