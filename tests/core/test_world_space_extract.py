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

    def test_attachment_pins_preserve_multi_vertex_offsets_in_world_space(self):
        """非単位matrix_worldかつ複数頂点ピン(N>=2)で各頂点の相対距離が保たれ1点に潰れないことを検証."""
        from types import SimpleNamespace
        from taremin_cloth.engine.params import sync_attachment_pins
        from taremin_cloth.engine.cache import _attachment_pin_indices_cache, _prev_attachment_pin_targets
        from taremin_cloth.engine.gui_client import ClothGuiClient

        _attachment_pin_indices_cache.clear()
        _prev_attachment_pin_targets.clear()

        # スタンドアロンテスト環境 (fake-bpy-moduleスタブ等) でも動作する Vector / Matrix ヘルパー
        class Vector(np.ndarray):
            def __new__(cls, values):
                return np.asarray(values, dtype=np.float32).view(cls)
            @property
            def x(self): return float(self[0])
            @property
            def y(self): return float(self[1])
            @property
            def z(self): return float(self[2])

        class Matrix:
            def __init__(self, mat=None):
                self._mat = np.eye(4, dtype=np.float32) if mat is None else np.array(mat, dtype=np.float32)
            @classmethod
            def Translation(cls, vec):
                m = np.eye(4, dtype=np.float32)
                m[0, 3] = vec[0]; m[1, 3] = vec[1]; m[2, 3] = vec[2]
                return cls(m)
            @property
            def translation(self):
                return Vector(self._mat[:3, 3])
            def inverted(self):
                return Matrix(np.linalg.inv(self._mat))
            def __matmul__(self, other):
                if isinstance(other, Matrix):
                    return Matrix(self._mat @ other._mat)
                v = np.array([other[0], other[1], other[2], 1.0], dtype=np.float32)
                res = self._mat @ v
                return Vector(res[:3])
            def __iter__(self):
                return iter(self._mat)
            def __array__(self, dtype=None):
                return np.asarray(self._mat, dtype=dtype)

        # 布オブジェクト: ローカル座標では (0,0,0) と (0.4,0,0) だが、ワールド位置は (10, 20, 30)
        v0 = SimpleNamespace(index=0, co=Vector((0.0, 0.0, 0.0)), groups=[SimpleNamespace(group=0, weight=1.0)])
        v1 = SimpleNamespace(index=1, co=Vector((0.4, 0.0, 0.0)), groups=[SimpleNamespace(group=0, weight=0.8)])
        mesh = SimpleNamespace(vertices=[v0, v1])
        vg = SimpleNamespace(index=0, name="Pin")
        cloth_mat = Matrix.Translation((10.0, 20.0, 30.0))

        # ターゲットオブジェクト: 初期ワールド位置 (10, 20, 30) -> 移動後 (11, 22, 33)
        target_obj = SimpleNamespace(
            name="TargetEmpty",
            type="EMPTY",
            matrix_world=Matrix.Translation((10.0, 20.0, 30.0)),
            pose=None,
        )
        settings = SimpleNamespace(
            pin_target_object=target_obj,
            pin_vertex_group="Pin",
            pin_target_bone="",
        )
        cloth_obj = SimpleNamespace(
            name="OffsetCloth",
            data=mesh,
            matrix_world=cloth_mat,
            taremin_cloth=settings,
            vertex_groups={"Pin": vg},
        )

        recorded_pins = {}

        class DummySim:
            def set_pin(self, v_idx, target_xyz, weight):
                recorded_pins[v_idx] = (np.array(target_xyz, dtype=np.float32), weight)

        sim = DummySim()
        # 1. In-process 初回同期 (ターゲット初期位置)
        sync_attachment_pins(sim, cloth_obj, scene=None)
        self.assertEqual(len(recorded_pins), 2)
        dist_init = float(np.linalg.norm(recorded_pins[1][0] - recorded_pins[0][0]))
        self.assertAlmostEqual(dist_init, 0.4, places=4, msg="複数頂点ピンが1点に収束せず初期距離0.4mを維持すること")
        np.testing.assert_allclose(recorded_pins[0][0], [10.0, 20.0, 30.0], atol=1e-4)
        np.testing.assert_allclose(recorded_pins[1][0], [10.4, 20.0, 30.0], atol=1e-4)

        # 2. ターゲットを (+1, +2, +3) 平行移動して In-process 再同期
        target_obj.matrix_world = Matrix.Translation((11.0, 22.0, 33.0))
        sync_attachment_pins(sim, cloth_obj, scene=None)
        np.testing.assert_allclose(recorded_pins[0][0], [11.0, 22.0, 33.0], atol=1e-4)
        np.testing.assert_allclose(recorded_pins[1][0], [11.4, 22.0, 33.0], atol=1e-4)

        # 3. GUI IPC (ClothGuiClient.send_update_pins) でも同一のワールド座標・相対距離が送信されること
        sent_commands = []
        client = ClothGuiClient()
        client.send_command = lambda cmd_type, data=None, timeout=1.0: (sent_commands.append((cmd_type, data)) or True)

        ok = client.send_update_pins(cloth_obj, scene=None)
        self.assertTrue(ok)
        self.assertEqual(len(sent_commands), 1)
        cmd_type, payload = sent_commands[0]
        self.assertEqual(cmd_type, "UpdatePins")
        self.assertEqual(payload["vertex_indices"], [0, 1])
        self.assertAlmostEqual(payload["weights"][0], 1.0, places=5)
        self.assertAlmostEqual(payload["weights"][1], 0.8, places=5)
        np.testing.assert_allclose(payload["positions"][0], [11.0, 22.0, 33.0], atol=1e-4)
        np.testing.assert_allclose(payload["positions"][1], [11.4, 22.0, 33.0], atol=1e-4)


if __name__ == "__main__":
    unittest.main()

