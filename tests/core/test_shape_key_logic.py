"""
シミュレーション形状のシェイプキー保存（Save as Shape Key）オペレーターの単体テスト
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# プロジェクトルートおよび python/ ディレクトリをモジュール検索パスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from taremin_cloth.ops import basic
from taremin_cloth import i18n


class TestShapeKeyLogic(unittest.TestCase):
    """save_as_shape_key のロジック・プロパティ・ガードの単体テスト"""

    def test_operator_registered_and_properties(self):
        """オペレーターの bl_idname およびプロパティが正しく定義されていることを確認"""
        op_cls = basic.TAREMIN_CLOTH_OT_save_as_shape_key
        self.assertEqual(op_cls.bl_idname, "taremin_cloth.save_as_shape_key")
        self.assertEqual(op_cls.bl_label, "Save as Shape Key")

        # RNA アノテーション
        annotations = getattr(op_cls, "__annotations__", {})
        self.assertIn("shape_key_name", annotations)
        self.assertIn("target_mode", annotations)

    def test_i18n_translations_exist(self):
        """シェイプキー保存関連の翻訳が日英辞書に登録されていることを確認"""
        raw_dict = i18n.get_raw_translations()
        self.assertIn("ja_JP", raw_dict)
        self.assertIn("en_US", raw_dict)

        key = "Save as Shape Key"
        self.assertEqual(raw_dict["ja_JP"].get(key), "シェイプキーとして保存")
        self.assertEqual(raw_dict["en_US"].get(key), key)

        key_msg = "Saved shape key '%s' to '%s'"
        self.assertIn("シェイプキー '%s' を '%s' に保存しました", raw_dict["ja_JP"].get(key_msg, ""))

    @patch("taremin_cloth.ops.basic.topology.is_cross_subdivided")
    def test_cross_subdivided_guard(self, mock_is_cross):
        """十字分割中のオブジェクトに対して警告を出して安全にキャンセルされることを検証"""
        mock_is_cross.return_value = True

        op = basic.TAREMIN_CLOTH_OT_save_as_shape_key()
        op.report = MagicMock()

        mock_context = MagicMock()
        mock_obj = MagicMock()
        mock_obj.type = 'MESH'
        mock_context.active_object = mock_obj

        res = op.execute(mock_context)
        self.assertEqual(res, {'CANCELLED'})
        self.assertTrue(op.report.called)
        call_args = op.report.call_args[0]
        self.assertEqual(call_args[0], {'WARNING'})
        self.assertTrue(len(call_args) > 1 and call_args[1] is not None)
        self.assertIn("cross-subdivided", str(call_args[1]))

    @patch("taremin_cloth.ops.basic.topology.is_cross_subdivided")
    @patch("taremin_cloth.ops.basic.bpy")
    def test_execute_auto_target_creation_and_key_addition(self, mock_bpy, mock_is_cross):
        """初回実行時にターゲットオブジェクトとBasisキーが作成され、新キーが追加されるロジックを検証"""
        mock_is_cross.return_value = False

        op = basic.TAREMIN_CLOTH_OT_save_as_shape_key()
        op.shape_key_name = "Custom_Pose"
        op.target_mode = 'AUTO_TARGET'
        op.report = MagicMock()

        # ソースオブジェクト（布メッシュ）の設定 (4頂点)
        cloth_obj = MagicMock()
        cloth_obj.name = "TestCloth"
        cloth_obj.type = 'MESH'
        cloth_obj.users_collection = [MagicMock()]
        cloth_obj.matrix_world = MagicMock()

        # 初期レスト座標と変形後座標
        initial_coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        deformed_coords = np.array([0.1, 0.0, 0.5, 0.9, 0.0, 0.4, 0.0, 0.8, 0.3, 1.1, 0.9, 0.2], dtype=np.float32)

        mock_verts = MagicMock()
        mock_verts.__len__.return_value = 4

        def mock_foreach_get(attr, arr):
            if attr == "co":
                arr[:] = deformed_coords

        mock_verts.foreach_get = mock_foreach_get
        cloth_obj.data.vertices = mock_verts

        cloth_obj.__contains__ = lambda self, key: key == "_taremin_cloth_rest_positions"
        cloth_obj.__getitem__ = lambda self, key: initial_coords.tobytes() if key == "_taremin_cloth_rest_positions" else None

        # シーン内のオブジェクト（初回は TestCloth_Poses は存在しない）
        mock_scene = MagicMock()
        mock_scene.objects.get.return_value = None
        mock_scene.frame_current = 10

        # 新規作成されるターゲットメッシュおよびオブジェクトのモック
        target_mesh = MagicMock()
        target_mesh.shape_keys = None
        cloth_obj.data.copy.return_value = target_mesh

        target_obj = MagicMock()
        target_obj.name = "TestCloth_Shapes"
        target_obj.data = target_mesh

        basis_key = MagicMock()
        basis_key.data = MagicMock()
        pose_key = MagicMock()
        pose_key.name = "Custom_Pose"
        pose_key.data = MagicMock()

        def mock_shape_key_add(name, from_mix=False):
            if name == "Basis":
                target_mesh.shape_keys = MagicMock()
                return basis_key
            else:
                return pose_key

        target_obj.shape_key_add = mock_shape_key_add
        mock_bpy.data.objects.new.return_value = target_obj

        mock_context = MagicMock()
        mock_context.active_object = cloth_obj
        mock_context.scene = mock_scene

        # 実行
        res = op.execute(mock_context)
        self.assertEqual(res, {'FINISHED'})

        # Basis キーに初期座標が設定されたことの検証
        self.assertTrue(basis_key.data.foreach_set.called)
        basis_call_args = basis_key.data.foreach_set.call_args[0]
        self.assertEqual(basis_call_args[0], "co")
        np.testing.assert_array_almost_equal(basis_call_args[1], initial_coords)

        # 新規シェイプキーに現在の変形座標が設定され、value=1.0 になったことの検証
        self.assertTrue(pose_key.data.foreach_set.called)
        pose_call_args = pose_key.data.foreach_set.call_args[0]
        self.assertEqual(pose_call_args[0], "co")
        np.testing.assert_array_almost_equal(pose_call_args[1], deformed_coords)
        self.assertEqual(pose_key.value, 1.0)

    @patch("taremin_cloth.ops.basic.topology.is_cross_subdivided")
    def test_execute_existing_target_append_key(self, mock_is_cross):
        """2回目実行時に既存のターゲットオブジェクトに新しいシェイプキーが追加されることを検証"""
        mock_is_cross.return_value = False

        op = basic.TAREMIN_CLOTH_OT_save_as_shape_key()
        op.shape_key_name = ""  # デフォルト名（Cloth_Shape）の検証
        op.target_mode = 'AUTO_TARGET'
        op.report = MagicMock()

        cloth_obj = MagicMock()
        cloth_obj.name = "MyCloth"
        cloth_obj.type = 'MESH'
        cloth_obj.users_collection = [MagicMock()]
        cloth_obj.matrix_world = MagicMock()

        coords2 = np.array([0.2, 0.1, 0.4, 0.8, 0.1, 0.3, 0.1, 0.7, 0.2, 1.0, 0.8, 0.1], dtype=np.float32)
        mock_verts = MagicMock()
        mock_verts.__len__.return_value = 4
        mock_verts.foreach_get = lambda attr, arr: arr.__setitem__(slice(None), coords2)
        cloth_obj.data.vertices = mock_verts

        # 既存ターゲットオブジェクト
        existing_target = MagicMock()
        existing_target.name = "MyCloth_Shapes"
        existing_target.type = 'MESH'
        existing_target.data.vertices = [MagicMock() for _ in range(4)]

        old_basis = MagicMock()
        old_basis.name = "Basis"
        old_basis.value = 0.0

        prev_shape_key = MagicMock()
        prev_shape_key.name = "Cloth_Shape"
        prev_shape_key.value = 1.0  # 前回の保存で 1.0 だったキー

        key_blocks = [old_basis, prev_shape_key]
        mock_shape_keys = MagicMock()
        mock_shape_keys.key_blocks = key_blocks
        existing_target.data.shape_keys = mock_shape_keys

        new_key = MagicMock()
        new_key.name = "Cloth_Shape.001"
        new_key.data = MagicMock()

        def mock_shape_key_add(name, from_mix=False):
            key_blocks.append(new_key)
            return new_key

        existing_target.shape_key_add = mock_shape_key_add

        mock_scene = MagicMock()
        mock_scene.objects.get.side_effect = lambda name: existing_target if name == "MyCloth_Shapes" else None
        mock_scene.frame_current = 25

        mock_context = MagicMock()
        mock_context.active_object = cloth_obj
        mock_context.scene = mock_scene

        res = op.execute(mock_context)
        self.assertEqual(res, {'FINISHED'})

        # 既存キーのウェイトが 0.0 にリセットされたこと
        self.assertEqual(prev_shape_key.value, 0.0)
        # 新規キーのみ 1.0 に設定されたこと
        self.assertEqual(new_key.value, 1.0)
        # 新規キーがアクティブシェイプキーとして選択されたこと (インデックス 2)
        self.assertEqual(existing_target.active_shape_key_index, 2)
        self.assertTrue(new_key.data.foreach_set.called)


if __name__ == "__main__":
    unittest.main()
