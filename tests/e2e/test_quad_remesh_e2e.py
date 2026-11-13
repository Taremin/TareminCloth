# -*- coding: utf-8 -*-
"""Blender E2Eテスト: Quad リメッシュ (Static Merged / Keep Pattern / 属性引き継ぎ / シミュレーション連携)"""

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
from taremin_cloth.preferences import get_preferences
from taremin_cloth.utils.mesh_extract import extract_cloth_mesh_data


class TestQuadRemeshE2E(unittest.TestCase):
    """Blender 上での Remesh to Quad E2E テスト"""

    def setUp(self):
        taremin_cloth.register()
        for obj in list(bpy.context.scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh, do_unlink=True)
        for mat in list(bpy.data.materials):
            bpy.data.materials.remove(mat, do_unlink=True)
        from taremin_cloth.engine.cache import _timeline_frame_cache, clear_simulators

        clear_simulators()
        _timeline_frame_cache.clear()
        prefs = get_preferences(bpy.context)
        if prefs is not None:
            prefs.enable_quad_remesh = True

    def tearDown(self):
        prefs = get_preferences(bpy.context)
        if prefs is not None:
            prefs.enable_quad_remesh = False
        for obj in list(bpy.context.scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh, do_unlink=True)
        for mat in list(bpy.data.materials):
            bpy.data.materials.remove(mat, do_unlink=True)
        from taremin_cloth.engine.cache import _timeline_frame_cache, clear_simulators

        clear_simulators()
        _timeline_frame_cache.clear()
        taremin_cloth.unregister()

    def _build_two_piece_garment(self):
        """2枚のアイランドと縫合Loose Edge・Pin・マテリアル・レイヤー属性を持つ布オブジェクトを構築"""
        mesh = bpy.data.meshes.new("PatternCloth_Mesh")
        # Front (y=0, nx=4, ny=4 -> 25 verts) + Back (y=0.25, 25 verts)
        verts = []
        uvs_front = []
        uvs_back = []
        nx, ny = 4, 4
        for y_3d, uv_offset_u in ((0.0, 0.02), (0.25, 0.52)):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    x = (i / nx) * 0.40
                    z = 1.0 + (j / ny) * 0.40
                    verts.append((x, y_3d, z))
                    u = uv_offset_u + (i / nx) * 0.46
                    v = 0.02 + (j / ny) * 0.96
                    if uv_offset_u < 0.5:
                        uvs_front.append((u, v))
                    else:
                        uvs_back.append((u, v))

        faces = []
        face_pieces = []
        face_mats = []
        face_uvs = []
        stride = nx + 1
        for piece_id, (v_base, uv_arr) in enumerate(((0, uvs_front), (25, uvs_back))):
            for j in range(ny):
                for i in range(nx):
                    a = v_base + j * stride + i
                    b = a + 1
                    c = a + stride + 1
                    d = a + stride
                    faces.append((a, b, c))
                    faces.append((a, c, d))
                    face_pieces.extend([piece_id, piece_id])
                    face_mats.extend([piece_id, piece_id])
                    face_uvs.append([uv_arr[a - v_base], uv_arr[b - v_base], uv_arr[c - v_base]])
                    face_uvs.append([uv_arr[a - v_base], uv_arr[c - v_base], uv_arr[d - v_base]])

        # 左右辺に縫合 Loose Edge を設定
        loose_edges = []
        for j in range(ny + 1):
            loose_edges.append((j * stride, 25 + j * stride))
            loose_edges.append((j * stride + nx, 25 + j * stride + nx))

        mesh.from_pydata(verts, loose_edges, faces)
        mesh.update()

        mat0 = bpy.data.materials.new("Mat_Front")
        mat1 = bpy.data.materials.new("Mat_Back")
        mesh.materials.append(mat0)
        mesh.materials.append(mat1)
        mesh.polygons.foreach_set("material_index", face_mats)

        uv_layer = mesh.uv_layers.new(name="UVMap")
        for poly, corners in zip(mesh.polygons, face_uvs):
            for li, uv in zip(poly.loop_indices, corners):
                uv_layer.data[li].uv = uv

        attr_p = mesh.attributes.new(name="pattern_piece_index", type='INT', domain='FACE')
        attr_p.data.foreach_set("value", face_pieces)
        attr_l = mesh.attributes.new(name="cloth_layer", type='INT', domain='FACE')
        attr_l.data.foreach_set("value", [1 if p == 1 else 0 for p in face_pieces])

        obj = bpy.data.objects.new("PatternCloth", mesh)
        bpy.context.collection.objects.link(obj)
        obj.taremin_cloth.is_cloth = True
        obj.taremin_cloth.enable_sewing = True
        obj.taremin_cloth.pin_vertex_group = "Cloth_Pin"

        vg_f = obj.vertex_groups.new(name="Piece_Front")
        vg_f.add(list(range(0, 25)), 1.0, 'REPLACE')
        vg_b = obj.vertex_groups.new(name="Piece_Back")
        vg_b.add(list(range(25, 50)), 1.0, 'REPLACE')
        vg_pin = obj.vertex_groups.new(name="Cloth_Pin")
        top_verts = [4 * stride + i for i in range(nx + 1)] + [25 + 4 * stride + i for i in range(nx + 1)]
        vg_pin.add(top_verts, 1.0, 'REPLACE')

        bpy.context.view_layer.objects.active = obj
        return obj

    def test_experimental_flag_gates_poll(self):
        from types import SimpleNamespace
        from unittest import mock

        obj = self._build_two_piece_garment()
        fake_prefs = SimpleNamespace(enable_quad_remesh=False)
        with mock.patch("taremin_cloth.preferences.get_preferences", return_value=fake_prefs):
            self.assertFalse(bpy.ops.taremin_cloth.remesh_to_quad.poll())
            fake_prefs.enable_quad_remesh = True
            self.assertTrue(bpy.ops.taremin_cloth.remesh_to_quad.poll())
        self.assertIsNotNone(obj)

    def test_static_merged_keep_pattern_and_replace_original(self):
        src_obj = self._build_two_piece_garment()

        # 1. STATIC_MERGED
        res = bpy.ops.taremin_cloth.remesh_to_quad(
            target_quad_size=0.10,
            remesh_mode='STATIC_MERGED',
            keep_structured_grid=True,
            replace_original=False,
        )
        self.assertEqual(res, {'FINISHED'})
        static_obj = bpy.context.active_object
        self.assertNotEqual(static_obj, src_obj)

        static_mesh = static_obj.data
        self.assertEqual(len(static_mesh.materials), 2)
        self.assertIn("UVMap", static_mesh.uv_layers)
        self.assertIn("pattern_piece_index", static_mesh.attributes)
        self.assertIn("cloth_layer", static_mesh.attributes)
        self.assertIn("Piece_Front", static_obj.vertex_groups)
        self.assertIn("Piece_Back", static_obj.vertex_groups)
        self.assertIn("Cloth_Pin", static_obj.vertex_groups)
        self.assertTrue(static_obj.taremin_cloth.is_cloth)

        # Loose Edge がゼロかつ全面 Quad
        face_edges = {e.key for p in static_mesh.polygons for e in static_mesh.edges if set(e.vertices).issubset(set(p.vertices))}
        self.assertGreater(len(static_mesh.polygons), 0)
        quad_count = sum(1 for p in static_mesh.polygons if len(p.vertices) == 4)
        self.assertEqual(quad_count, len(static_mesh.polygons))
        # STATIC_MERGED では loose edge ゼロ (Blender のエッジは全て面に属する)
        for p in static_mesh.polygons:
            self.assertEqual(len(p.vertices), 4)
        poly_edge_keys = set()
        for p in static_mesh.polygons:
            vs = p.vertices
            for k in range(len(vs)):
                a, b = vs[k], vs[(k + 1) % len(vs)]
                poly_edge_keys.add((a, b) if a < b else (b, a))
        loose_cnt = sum(1 for e in static_mesh.edges if (min(e.vertices), max(e.vertices)) not in poly_edge_keys)
        self.assertEqual(loose_cnt, 0)
        self.assertGreater(len(face_edges), 0)

        # 2. KEEP_PATTERN + step_cloth_scene シミュレーション疎通
        bpy.data.objects.remove(static_obj, do_unlink=True)
        bpy.context.view_layer.objects.active = src_obj
        res2 = bpy.ops.taremin_cloth.remesh_to_quad(
            target_quad_size=0.10,
            remesh_mode='KEEP_PATTERN',
            keep_structured_grid=True,
            replace_original=False,
        )
        self.assertEqual(res2, {'FINISHED'})
        keep_obj = bpy.context.active_object
        keep_mesh = keep_obj.data

        # 縫合 Loose Edge が再構築されていること
        keep_data = extract_cloth_mesh_data(keep_obj, settings=keep_obj.taremin_cloth)
        self.assertIsNotNone(keep_data.sewing_edges)
        self.assertGreater(len(keep_data.sewing_edges), 0)

        # 元の src_obj は sim 対象外にして KEEP_PATTERN 出力だけ 5 ステップ回す
        src_obj.taremin_cloth.is_cloth = False
        for _ in range(5):
            step_cloth_scene(bpy.context.scene, dt=1.0 / 60.0)

        co = np.empty(len(keep_mesh.vertices) * 3, dtype=np.float32)
        keep_mesh.vertices.foreach_get("co", co)
        self.assertTrue(np.all(np.isfinite(co)))

        # 3. replace_original=True で元オブジェクト上書き
        bpy.context.view_layer.objects.active = src_obj
        src_obj.taremin_cloth.is_cloth = True
        res3 = bpy.ops.taremin_cloth.remesh_to_quad(
            target_quad_size=0.10,
            remesh_mode='STATIC_MERGED',
            replace_original=True,
        )
        self.assertEqual(res3, {'FINISHED'})
        self.assertGreater(len(src_obj.data.polygons), 0)
        self.assertEqual(len(src_obj.data.materials), 2)


if __name__ == "__main__":
    unittest.main()
