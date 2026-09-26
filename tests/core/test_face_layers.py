"""
Face Layer 機能の単体テスト (tests/core/test_face_layers.py)
Blender非依存でモックを用いて以下を検証する:
1. Face Attribute から頂点レイヤーIDへのMax則マッピング
2. 属性未指定時のデフォルトフォールバック
3. runner.py による ClothSimulator への layer_ids 連携
4. レイヤー操作オペレーター (Assign, Select, Clear) の実行ロジック
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from taremin_cloth.utils.mesh_extract import (
    ClothMeshData,
    extract_cloth_mesh_data,
    get_cloth_layer_ids,
    get_mesh_face_layers_summary,
)
from taremin_cloth.ops.tools import (
    TAREMIN_CLOTH_OT_assign_face_layer,
    TAREMIN_CLOTH_OT_select_face_layer,
    TAREMIN_CLOTH_OT_clear_face_layers,
)


class TestFaceLayers(unittest.TestCase):
    def setUp(self):
        # 4頂点、2面のメッシュを模擬
        self.mock_mesh = MagicMock()
        self.mock_mesh.vertices = [MagicMock() for _ in range(4)]
        self.mock_mesh.polygons = [MagicMock(), MagicMock()]
        self.mock_mesh.polygons[0].index = 0
        self.mock_mesh.polygons[1].index = 1
        self.mock_mesh.polygons[0].vertices = (0, 1, 2)
        self.mock_mesh.polygons[1].vertices = (0, 2, 3)
        self.mock_mesh.polygons[0].select = True
        self.mock_mesh.polygons[1].select = False

        self.mock_obj = MagicMock()
        self.mock_obj.type = 'MESH'
        self.mock_obj.mode = 'OBJECT'
        self.mock_obj.data = self.mock_mesh

        self.settings = MagicMock()
        self.settings.layer_id = 0
        self.settings.target_face_layer_id = 2
        self.mock_obj.taremin_cloth = self.settings

    def test_assign_face_layer_object_mode(self):
        """オブジェクトモードで選択ポリゴンに cloth_layer 属性が設定されること"""
        attr_data = [MagicMock(value=0), MagicMock(value=0)]
        mock_attr = MagicMock()
        mock_attr.data = attr_data
        self.mock_mesh.attributes.get.return_value = None
        self.mock_mesh.attributes.new.return_value = mock_attr

        context = MagicMock()
        context.active_object = self.mock_obj

        op = TAREMIN_CLOTH_OT_assign_face_layer()
        op.report = MagicMock()
        res = op.execute(context)
        self.assertEqual(res, {'FINISHED'})

        # 選択されていたポリゴン0に target_face_layer_id (2) が割り当てられていること
        self.assertEqual(attr_data[0].value, 2)
        # 非選択のポリゴン1は変更されないこと
        self.assertEqual(attr_data[1].value, 0)

    @patch("bmesh.from_edit_mesh")
    @patch("bmesh.update_edit_mesh")
    def test_assign_face_layer_edit_mode(self, mock_update, mock_from_edit):
        """編集モードで選択面に cloth_layer カスタムデータレイヤーが設定されること"""
        self.mock_obj.mode = 'EDIT'
        bm = MagicMock()
        mock_from_edit.return_value = bm

        face0 = MagicMock(select=True)
        face1 = MagicMock(select=False)
        face_storage = {face0: 0, face1: 0}
        layer_ref = MagicMock()

        def setitem(face, key, val):
            if key == layer_ref:
                face_storage[face] = val

        face0.__setitem__ = lambda self, key, val: setitem(face0, key, val)
        face1.__setitem__ = lambda self, key, val: setitem(face1, key, val)

        mock_faces = MagicMock()
        mock_faces.__iter__.return_value = iter([face0, face1])
        mock_faces.layers.int.get.return_value = layer_ref
        bm.faces = mock_faces

        context = MagicMock()
        context.active_object = self.mock_obj

        op = TAREMIN_CLOTH_OT_assign_face_layer()
        op.report = MagicMock()
        res = op.execute(context)
        self.assertEqual(res, {'FINISHED'})

        self.assertEqual(face_storage[face0], 2)
        self.assertEqual(face_storage[face1], 0)
        mock_update.assert_called_once_with(self.mock_mesh)

    def test_clear_face_layers(self):
        """clear_face_layers オペレーターで属性が削除されること"""
        mock_attr = MagicMock()
        mock_attrs = MagicMock()
        mock_attrs.__contains__ = lambda self, k: k == "cloth_layer"
        mock_attrs.__getitem__ = lambda self, k: mock_attr
        mock_attrs.remove = MagicMock()
        self.mock_mesh.attributes = mock_attrs

        context = MagicMock()
        context.active_object = self.mock_obj

        op = TAREMIN_CLOTH_OT_clear_face_layers()
        op.report = MagicMock()
        res = op.execute(context)
        self.assertEqual(res, {'FINISHED'})
        mock_attrs.remove.assert_called_once_with(mock_attr)

    def test_layered_skirt_joint_max_rule(self):
        """多層構造（レイヤードスカート等）の接合部で共有頂点が上位レイヤー値になること"""
        # 6頂点、3ポリゴン (内層: poly 0, 1 / 外層: poly 2)
        # 頂点 2, 3 がウエスト接合部で内層と外層に共有されている
        n_verts = 6
        mock_mesh = MagicMock()
        mock_mesh.vertices = [MagicMock() for _ in range(n_verts)]
        poly0 = MagicMock(vertices=(0, 1, 2))
        poly1 = MagicMock(vertices=(1, 2, 3))
        poly2 = MagicMock(vertices=(2, 3, 4, 5))
        mock_mesh.polygons = [poly0, poly1, poly2]

        attr = MagicMock()
        attr.domain = 'FACE'
        # poly0=0, poly1=0 (内層), poly2=1 (外層)
        poly_layers = np.array([0, 0, 1], dtype=np.int32)
        mock_data = MagicMock()
        mock_data.__len__.return_value = 3
        mock_data.foreach_get = lambda prop, arr: np.copyto(arr, poly_layers)
        attr.data = mock_data

        mock_mesh.attributes.get = lambda name: attr if name == "cloth_layer" else None
        # loop_triangles が存在しないフォールバック走査を検証
        mock_mesh.calc_loop_triangles = MagicMock(side_effect=AttributeError)

        obj = MagicMock(data=mock_mesh)
        layer_ids = get_cloth_layer_ids(obj, n_verts=n_verts)
        self.assertIsNotNone(layer_ids)

        # 頂点 0, 1 は内層のみ -> 0
        self.assertEqual(layer_ids[0], 0)
        self.assertEqual(layer_ids[1], 0)
        # 頂点 2, 3 は内層と外層の共有接合部 -> Max則により外層の 1
        self.assertEqual(layer_ids[2], 1)
        self.assertEqual(layer_ids[3], 1)
        # 頂点 4, 5 は外層のみ -> 1
        self.assertEqual(layer_ids[4], 1)
        self.assertEqual(layer_ids[5], 1)

    def test_select_face_layer_object_mode(self):
        """オブジェクトモードで指定レイヤーIDのポリゴンが選択されること"""
        attr_data = [MagicMock(value=2), MagicMock(value=0)]
        mock_attr = MagicMock()
        mock_attr.data = attr_data
        self.mock_mesh.attributes.get.return_value = mock_attr

        context = MagicMock()
        context.active_object = self.mock_obj

        op = TAREMIN_CLOTH_OT_select_face_layer()
        op.report = MagicMock()
        res = op.execute(context)
        self.assertEqual(res, {'FINISHED'})

        # target_face_layer_id == 2 のポリゴン0が選択され、ポリゴン1は非選択
        self.assertTrue(self.mock_mesh.polygons[0].select)
        self.assertFalse(self.mock_mesh.polygons[1].select)

    @patch("taremin_cloth.engine.runner.extract_cloth_mesh_data")
    @patch("taremin_cloth_core.ClothSimulator")
    def test_runner_propagates_layer_ids_to_simulator(self, mock_sim_cls, mock_extract):
        """runner.py が extract_cloth_mesh_data から得た layer_ids を ClothSimulator に渡すこと"""
        from taremin_cloth.engine.runner import begin_simulator_init, create_simulator_from_state

        dummy_layer_ids = np.array([0, 1, 0, 1], dtype=np.uint32)
        mock_cloth_data = ClothMeshData(
            positions=np.zeros((4, 3), dtype=np.float32),
            normal_edges=np.array([[0, 1], [2, 3]], dtype=np.uint32),
            faces=None,
            sewing_edges=None,
            inv_masses=np.ones(4, dtype=np.float32),
            layer_ids=dummy_layer_ids,
        )
        mock_extract.return_value = mock_cloth_data

        state = begin_simulator_init(self.mock_obj)
        self.assertFalse(state["cached"])
        self.assertIn("layer_ids", state)
        np.testing.assert_array_equal(state["layer_ids"], dummy_layer_ids)

        # create_simulator_from_state の呼び出し検証
        sim = create_simulator_from_state(self.mock_obj, state)
        mock_sim_cls.assert_called_once()
        _, kwargs = mock_sim_cls.call_args
        self.assertIn("layer_ids", kwargs)
        np.testing.assert_array_equal(kwargs["layer_ids"], dummy_layer_ids)


if __name__ == "__main__":
    unittest.main()
