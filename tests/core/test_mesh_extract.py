"""
mesh_extract ユーティリティの単体テスト (tests/core/test_mesh_extract.py)
Blender非依存でモックオブジェクトを用いてメッシュ抽出・ピン質量計算の動作を検証する。
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from taremin_cloth.utils.mesh_extract import (
    ClothMeshData,
    extract_cloth_mesh_data,
    extract_mesh_vertices_and_triangles,
    extract_sewing_topology,
    get_pin_inv_masses,
    get_cloth_layer_ids,
    get_mesh_face_layers_summary,
)


class TestMeshExtract(unittest.TestCase):
    def setUp(self):
        # 4頂点、2三角形の平面メッシュを模擬
        self.mock_mesh = MagicMock()
        
        # 頂点座標: 4頂点 (4, 3)
        self.verts_co = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        def mock_verts_foreach_get(attr, target):
            if attr == "co":
                target[:] = self.verts_co.flatten()

        mock_verts = MagicMock()
        mock_verts.__len__.return_value = 4
        def mock_verts_foreach_get(attr, target):
            if attr == "co":
                target[:] = self.verts_co.flatten()
        mock_verts.foreach_get = mock_verts_foreach_get
        self.mock_mesh.vertices = mock_verts

        # 三角形: 2面 (0, 1, 2) と (0, 2, 3)
        self.tri_indices = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.uint32)

        tri0 = MagicMock()
        tri0.vertices = (0, 1, 2)
        tri1 = MagicMock()
        tri1.vertices = (0, 2, 3)

        mock_loop_tris = MagicMock()
        mock_loop_tris.__len__.return_value = 2
        mock_loop_tris.__iter__.return_value = iter([tri0, tri1])
        def mock_tris_foreach_get(attr, target):
            if attr == "vertices":
                target[:] = self.tri_indices.flatten()
        mock_loop_tris.foreach_get = mock_tris_foreach_get
        self.mock_mesh.loop_triangles = mock_loop_tris

        # エッジ: 5本の通常エッジ + 1本の縫合エッジ (合計6本)
        self.all_edges = np.array([
            [0, 1],
            [1, 2],
            [0, 2],
            [2, 3],
            [0, 3],
            [1, 3],  # 縫合エッジ候補
        ], dtype=np.uint32)

        mock_edges = MagicMock()
        mock_edges.__len__.return_value = 6
        def mock_edges_foreach_get(attr, target):
            if attr == "vertices":
                target[:] = self.all_edges.flatten()
        mock_edges.foreach_get = mock_edges_foreach_get
        self.mock_mesh.edges = mock_edges

        # ポリゴン: 2面
        poly0 = MagicMock()
        poly0.vertices = (0, 1, 2)
        poly1 = MagicMock()
        poly1.vertices = (0, 2, 3)
        mock_polys = MagicMock()
        mock_polys.__len__.return_value = 2
        mock_polys.__iter__.return_value = iter([poly0, poly1])
        self.mock_mesh.polygons = mock_polys

        # オブジェクト模擬
        self.mock_obj = MagicMock()
        self.mock_obj.data = self.mock_mesh
        self.mock_obj.vertex_groups = {}

    def test_extract_mesh_vertices_and_triangles(self):
        """汎用頂点・面抽出の検証"""
        pos, faces = extract_mesh_vertices_and_triangles(self.mock_mesh)
        self.assertEqual(pos.shape, (4, 3))
        np.testing.assert_allclose(pos, self.verts_co)
        self.assertEqual(faces.shape, (2, 3))
        np.testing.assert_array_equal(faces, self.tri_indices)

    def test_get_pin_inv_masses_default(self):
        """ピン未設定時は全頂点の inv_mass が 1.0 であること"""
        inv_m = get_pin_inv_masses(self.mock_obj, None, 4)
        self.assertEqual(len(inv_m), 4)
        np.testing.assert_array_equal(inv_m, np.ones(4, dtype=np.float32))

    def test_get_pin_inv_masses_with_weights(self):
        """ピン頂点グループのウェイトから正しい inv_mass が計算されること"""
        pin_group = MagicMock()
        # 頂点0はウェイト1.0 (固定), 頂点1はウェイト0.5, 頂点2,3はウェイト0.0
        weights = {0: 1.0, 1: 0.5, 2: 0.0}
        def mock_weight(idx):
            if idx in weights:
                return weights[idx]
            raise RuntimeError("not in group")
        pin_group.weight = mock_weight

        self.mock_obj.vertex_groups = {"Cloth_Pin": pin_group}
        settings = MagicMock()
        settings.pin_vertex_group = "Cloth_Pin"

        inv_m = get_pin_inv_masses(self.mock_obj, settings, 4)
        self.assertAlmostEqual(inv_m[0], 0.0)  # 1.0 - 1.0 = 0.0 (固定)
        self.assertAlmostEqual(inv_m[1], 0.5)  # 1.0 - 0.5 = 0.5
        self.assertAlmostEqual(inv_m[2], 1.0)  # 1.0 - 0.0 = 1.0
        self.assertAlmostEqual(inv_m[3], 1.0)  # 例外時は 1.0

    def test_extract_cloth_mesh_data_sewing_enabled(self):
        """縫合有効時に面に含まれないエッジが sewing_edges として分離されること"""
        settings = MagicMock()
        settings.enable_sewing = True
        settings.pin_vertex_group = ""

        data = extract_cloth_mesh_data(self.mock_obj, settings)
        self.assertIsInstance(data, ClothMeshData)
        self.assertEqual(data.positions.shape, (4, 3))
        self.assertEqual(data.faces.shape, (2, 3))
        # 通常エッジは5本
        self.assertEqual(len(data.normal_edges), 5)
        # 縫合エッジは1本 [1, 3]
        self.assertIsNotNone(data.sewing_edges)
        self.assertEqual(len(data.sewing_edges), 1)
        np.testing.assert_array_equal(data.sewing_edges[0], [1, 3])

    def test_extract_cloth_mesh_data_sewing_disabled(self):
        """縫合無効時は全エッジが normal_edges に分類されること"""
        settings = MagicMock()
        settings.enable_sewing = False
        settings.pin_vertex_group = ""

        data = extract_cloth_mesh_data(self.mock_obj, settings)
        self.assertEqual(len(data.normal_edges), 6)
        self.assertIsNone(data.sewing_edges)

    def test_extract_cloth_mesh_data_without_cloth_layer(self):
        """cloth_layer 属性が存在しない場合、layer_ids は None であること"""
        self.mock_mesh.attributes.get.return_value = None
        data = extract_cloth_mesh_data(self.mock_obj)
        self.assertIsNone(data.layer_ids)

    def test_get_cloth_layer_ids_face_domain(self):
        """FACEドメインの属性から頂点レイヤーIDが正しく抽出され、共有頂点はMax則に従うこと"""
        attr = MagicMock()
        attr.domain = 'FACE'
        # 2ポリゴン: ポリゴン0=0, ポリゴン1=1
        poly_layers = np.array([0, 1], dtype=np.int32)
        attr.data.foreach_get = lambda prop, arr: np.copyto(arr, poly_layers)

        self.mock_mesh.attributes.get = lambda name: attr if name == "cloth_layer" else None

        # ポリゴン所属インデックス設定: 三角形0 -> ポリゴン0, 三角形1 -> ポリゴン1
        poly_indices = np.array([0, 1], dtype=np.int32)
        orig_foreach_get = self.mock_mesh.loop_triangles.foreach_get
        def mock_lt_foreach(prop, arr):
            if prop == "polygon_index":
                np.copyto(arr, poly_indices)
            else:
                orig_foreach_get(prop, arr)
        self.mock_mesh.loop_triangles.foreach_get = mock_lt_foreach

        layer_ids = get_cloth_layer_ids(self.mock_obj, n_verts=4)
        self.assertIsNotNone(layer_ids)
        self.assertEqual(len(layer_ids), 4)
        # 頂点0: 面0と面1で共有 -> max(0, 1) = 1
        # 頂点1: 面0のみ -> 0
        # 頂点2: 面0と面1で共有 -> max(0, 1) = 1
        # 頂点3: 面1のみ -> 1
        np.testing.assert_array_equal(layer_ids, [1, 0, 1, 1])

    def test_get_cloth_layer_ids_point_domain(self):
        """POINTドメインの属性から頂点ごとの値が直接抽出されること"""
        attr = MagicMock()
        attr.domain = 'POINT'
        point_layers = np.array([0, 1, 2, 0], dtype=np.int32)
        attr.data.foreach_get = lambda prop, arr: np.copyto(arr, point_layers)

        self.mock_mesh.attributes.get = lambda name: attr if name == "cloth_layer" else None

        layer_ids = get_cloth_layer_ids(self.mock_obj, n_verts=4)
        self.assertIsNotNone(layer_ids)
        np.testing.assert_array_equal(layer_ids, [0, 1, 2, 0])

    def test_get_mesh_face_layers_summary(self):
        """属性のサマリー情報が正しく取得できること"""
        attr = MagicMock()
        attr.domain = 'FACE'
        poly_layers = np.array([0, 2], dtype=np.int32)
        mock_data = MagicMock()
        mock_data.__len__.return_value = 2
        mock_data.foreach_get = lambda prop, arr: np.copyto(arr, poly_layers)
        attr.data = mock_data

        self.mock_mesh.attributes.get = lambda name: attr if name == "cloth_layer" else None

        summary = get_mesh_face_layers_summary(self.mock_mesh)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["domain"], 'FACE')
        self.assertEqual(summary["unique_layers"], [0, 2])
        self.assertEqual(summary["count"], 2)

    def test_extract_cloth_mesh_data_excludes_degenerate_faces(self):
        """面積0の縮退三角形がシミュレーション用面データから安全に除外されること"""
        lt0 = MagicMock(); lt0.vertices = (0, 1, 2)
        lt1 = MagicMock(); lt1.vertices = (0, 1, 0)  # 同一頂点を含む面積0の面
        self.mock_mesh.loop_triangles = [lt0, lt1]

        data = extract_cloth_mesh_data(self.mock_obj)
        self.assertIsNotNone(data.faces)
        self.assertEqual(len(data.faces), 1)
        np.testing.assert_array_equal(data.faces[0], [0, 1, 2])

    def test_extract_sewing_topology_excludes_branching(self):
        """枝分かれした縫合線が単一経路保証のために除外されること"""
        loose_edges = [(0, 1), (1, 2), (1, 3)]
        faces_2d = np.array([[0, 10, 11], [2, 12, 13], [3, 14, 15]], dtype=np.uint32)
        positions = np.zeros((16, 3), dtype=np.float32)

        simple, waypoints = extract_sewing_topology(loose_edges, faces_2d, positions)
        self.assertEqual(len(simple), 0)
        self.assertEqual(len(waypoints), 0)


if __name__ == "__main__":
    unittest.main()
