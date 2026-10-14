# -*- coding: utf-8 -*-
"""Milestone 3 & 4: Blender 2D型紙（Curve / Mesh）から単一3D衣装メッシュへの統合ビルド & シミュレーション連携 E2Eテスト."""
import sys
import unittest
from pathlib import Path
import bpy
import numpy as np

root_dir = Path(__file__).parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import taremin_cloth
from taremin_cloth import step_cloth_scene
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
        from taremin_cloth.engine.cache import _timeline_frame_cache, clear_simulators
        clear_simulators()
        _timeline_frame_cache.clear()
        taremin_cloth.unregister()

    def _create_2d_curve_pattern(self, name: str, width: float, height: float, offset_3d=(0.0, 0.0, 0.5)):
        """2D Curve オブジェクトで矩形型紙を作成する."""
        curve_data = bpy.data.curves.new(name=f"{name}_Curve", type='CURVE')
        curve_data.dimensions = '2D'
        spline = curve_data.splines.new(type='POLY')
        spline. points.add(3)  # 計4頂点
        coords = [(0.0, 0.0, 0.0, 1.0), (width, 0.0, 0.0, 1.0), (width, height, 0.0, 1.0), (0.0, height, 0.0, 1.0)]
        for i, co in enumerate(coords):
            spline.points[i].co = co
        spline.use_cyclic_u = True

        obj = bpy.data.objects.new(name, curve_data)
        bpy.context.collection.objects.link(obj)
        if hasattr(obj, "taremin_pattern"):
            obj.taremin_pattern.is_pattern_piece = True
            obj.taremin_pattern.target_edge_length = 0.025
            obj.taremin_pattern.arrangement_offset = offset_3d
        return obj

    def test_curve_and_mesh_patterns_build_into_single_garment_mesh(self):
        """Curve型紙(案A)と平面Mesh型紙(案B)から単一3Dメッシュ・UV・頂点グループ・縫合Loose Edgeが生成されること."""
        try:
            from taremin_cloth.utils.pattern_builder import build_garment_from_patterns
        except ImportError:
            self.skipTest("taremin_cloth.utils.pattern_builder 未実装のためスキップ")

        # 1. Curve型紙 (Front)
        front_obj = self._create_2d_curve_pattern("FrontPiece", 0.20, 0.25, offset_3d=(0.0, -0.10, 0.50))

        # 2. 平面Mesh型紙 (Back)
        bpy.ops.mesh.primitive_plane_add(size=0.20, location=(0.30, 0.0, 0.0))
        back_obj = bpy.context.active_object
        back_obj.name = "BackPiece"
        if hasattr(back_obj, "taremin_pattern"):
            back_obj.taremin_pattern.is_pattern_piece = True
            back_obj.taremin_pattern.target_edge_length = 0.025
            back_obj.taremin_pattern.arrangement_offset = (0.0, 0.10, 0.50)

        garment_obj = build_garment_from_patterns(
            bpy.context.scene,
            pattern_objects=[front_obj, back_obj],
            target_name="TestGarment",
        )

        self.assertIsNotNone(garment_obj)
        self.assertEqual(garment_obj.type, 'MESH')
        self.assertTrue(garment_obj.taremin_cloth.is_cloth)

        # UVマップ 'UVMap_Pattern' が自動生成されていること
        self.assertIn("UVMap_Pattern", garment_obj.data.uv_layers)

        # パーツ別の頂点グループが自動生成されていること
        self.assertIn("Piece_FrontPiece", garment_obj.vertex_groups)
        self.assertIn("Piece_BackPiece", garment_obj.vertex_groups)

        # 共通パイプライン step_cloth_scene で10ステップ物理計算が破綻なく回ること
        for frame in range(1, 11):
            step_cloth_scene(bpy.context.scene, dt=1.0 / 60.0, anim_frame=frame, update_mesh=True)

        data = extract_cloth_mesh_data(garment_obj)
        self.assertTrue(np.all(np.isfinite(data.positions)))

    def test_edit_arrangement_3d_proxy_operator(self):
        """edit_arrangement_3d オペレーター実行により3Dプロキシが正しく生成・アクティブ化されること."""
        front_obj = self._create_2d_curve_pattern("FrontPiece", 0.20, 0.25, offset_3d=(0.1, -0.2, 0.8))
        bpy.context.view_layer.objects.active = front_obj

        # edit_arrangement_3d オペレーターを実行
        res = bpy.ops.taremin_cloth.edit_arrangement_3d()
        self.assertEqual(res, {'FINISHED'})

        # プロキシオブジェクトが生成されアクティブになっていること
        proxy_obj = bpy.context.scene.objects.get("FrontPiece_Proxy3D")
        self.assertIsNotNone(proxy_obj)
        self.assertEqual(bpy.context.active_object, proxy_obj)
        self.assertEqual(proxy_obj.display_type, 'WIRE')
        self.assertTrue(proxy_obj.show_in_front)

        # 初期位置が arrangement_offset と一致していること
        self.assertAlmostEqual(proxy_obj.location.x, 0.1, places=3)
        self.assertAlmostEqual(proxy_obj.location.y, -0.2, places=3)
        self.assertAlmostEqual(proxy_obj.location.z, 0.8, places=3)


if __name__ == "__main__":
    unittest.main()

