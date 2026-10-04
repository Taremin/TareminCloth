# -*- coding: utf-8 -*-
"""布メッシュのワールド座標系同期および逆変換の単体テスト."""
import unittest
import numpy as np


class MockCollection:
    def __init__(self, items, foreach_get_fn=None):
        self.items = items
        self._foreach_get_fn = foreach_get_fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def foreach_get(self, attr, arr):
        if self._foreach_get_fn:
            self._foreach_get_fn(attr, arr)


class MockMesh:
    def __init__(self, vertices_co, edges=None, polygons=None):
        def vert_foreach_get(attr, arr):
            if attr == "co":
                arr[:] = np.array(vertices_co, dtype=np.float32).flatten()
        self.vertices = MockCollection([None] * len(vertices_co), vert_foreach_get)

        edges_list = edges or []
        def edge_foreach_get(attr, arr):
            if attr == "vertices":
                arr[:] = np.array(edges_list, dtype=np.uint32).flatten()
        self.edges = MockCollection([None] * len(edges_list), edge_foreach_get)

        self.polygons = polygons or []
        self.loop_triangles = []

    def calc_loop_triangles(self):
        pass


class MockObject:
    def __init__(self, mesh, matrix_world):
        self.type = 'MESH'
        self.data = mesh
        self.matrix_world = matrix_world
        self.vertex_groups = {}


class TestWorldSpaceExtract(unittest.TestCase):
    def test_world_space_coordinate_transform(self):
        """obj.matrix_world が適用されてワールド座標に正しく変換されることを検証."""
        from taremin_cloth.utils.mesh_extract import extract_cloth_mesh_data

        local_verts = [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ]
        mock_mesh = MockMesh(local_verts)

        # トランスフォーム: Location=(10, 20, 30), Scale=(2, 2, 2)
        mat = np.array([
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 2.0, 30.0],
            [0.0, 0.0, 0.0,  1.0],
        ], dtype=np.float32)

        mock_obj = MockObject(mock_mesh, mat)

        # ワールド座標系抽出
        cloth_data_world = extract_cloth_mesh_data(mock_obj, apply_world_matrix=True)
        pos_w = cloth_data_world.positions

        # 期待値: [0, 0, 0] -> [10, 20, 30]
        #         [1, 2, 3] -> [1*2+10, 2*2+20, 3*2+30] = [12, 24, 36]
        np.testing.assert_allclose(pos_w[0], [10.0, 20.0, 30.0], rtol=1e-5)
        np.testing.assert_allclose(pos_w[1], [12.0, 24.0, 36.0], rtol=1e-5)

        # ローカル座標系抽出 (apply_world_matrix=False)
        cloth_data_local = extract_cloth_mesh_data(mock_obj, apply_world_matrix=False)
        pos_l = cloth_data_local.positions
        np.testing.assert_allclose(pos_l[0], [0.0, 0.0, 0.0], rtol=1e-5)
        np.testing.assert_allclose(pos_l[1], [1.0, 2.0, 3.0], rtol=1e-5)

    def test_world_to_local_inverse_transform(self):
        """ワールド座標から逆行列でローカル座標へ完全に復元できることを検証."""
        pos_local_orig = np.array([
            [-1.5,  2.0,  0.5],
            [ 3.0, -4.0,  1.2],
            [ 0.0,  0.0, -2.5],
        ], dtype=np.float32)

        # 任意の回転・平行移動・均等スケール行列
        angle = np.pi / 4.0
        c, s = np.cos(angle), np.sin(angle)
        scale = 1.5
        mat = np.array([
            [scale * c, -scale * s, 0.0,  5.0],
            [scale * s,  scale * c, 0.0, -3.0],
            [0.0,        0.0,       scale, 7.0],
            [0.0,        0.0,       0.0,   1.0],
        ], dtype=np.float32)

        # ワールド変換
        pos_world = (pos_local_orig @ mat[:3, :3].T) + mat[:3, 3]

        # 逆変換
        inv_mat = np.linalg.inv(mat)
        pos_local_restored = (pos_world @ inv_mat[:3, :3].T) + inv_mat[:3, 3]

        np.testing.assert_allclose(pos_local_restored, pos_local_orig, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
