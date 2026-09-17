"""
ピン留め頂点グループの自動作成・ウェイトペイント連携 (Quick Pinning) オペレーターの単体テスト
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# プロジェクトルートおよび python/ ディレクトリをモジュール検索パスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from taremin_cloth.ops import basic
from taremin_cloth import i18n


class TestQuickPinning(unittest.TestCase):
    """create_pin_group および toggle_weight_paint の単体テスト"""

    def test_operators_registered(self):
        """オペレーターの bl_idname および設定が正しく定義されていることを確認"""
        self.assertEqual(basic.TAREMIN_CLOTH_OT_create_pin_group.bl_idname, "taremin_cloth.create_pin_group")
        self.assertEqual(basic.TAREMIN_CLOTH_OT_toggle_weight_paint.bl_idname, "taremin_cloth.toggle_weight_paint")
        self.assertIn('REGISTER', basic.TAREMIN_CLOTH_OT_create_pin_group.bl_options)
        self.assertIn('UNDO', basic.TAREMIN_CLOTH_OT_create_pin_group.bl_options)
        self.assertIn('REGISTER', basic.TAREMIN_CLOTH_OT_toggle_weight_paint.bl_options)
        self.assertIn('UNDO', basic.TAREMIN_CLOTH_OT_toggle_weight_paint.bl_options)

    def test_i18n_translations_exist(self):
        """ピン留め関連の新規文言が日英辞書に登録されていることを確認"""
        raw_dict = i18n.get_raw_translations()
        self.assertIn("ja_JP", raw_dict)
        self.assertIn("en_US", raw_dict)

        self.assertEqual(raw_dict["ja_JP"].get("Create Pin Group"), "ピン頂点グループを作成")
        self.assertEqual(raw_dict["ja_JP"].get("Toggle Weight Paint"), "ウェイトペイント切り替え")
        self.assertEqual(raw_dict["en_US"].get("Create Pin Group"), "Create Pin Group")

    @patch("taremin_cloth.ops.basic.bpy")
    def test_create_pin_group_execution(self, mock_bpy):
        """新規頂点グループが作成され、設定に反映され、ウェイトペイントモードへ移行することを検証"""
        op = MagicMock()

        mock_context = MagicMock()
        mock_context.mode = 'OBJECT'

        mock_obj = MagicMock()
        mock_obj.type = 'MESH'
        mock_settings = MagicMock()
        mock_settings.pin_vertex_group = "Old_Pin"
        mock_obj.taremin_cloth = mock_settings

        mock_vg = MagicMock()
        mock_vg.name = "Cloth_Pin"
        mock_vg.index = 2
        mock_obj.vertex_groups.new.return_value = mock_vg

        mock_context.active_object = mock_obj

        # mode_set のモック
        mock_bpy.ops.object.mode_set.poll.return_value = True

        res = basic.TAREMIN_CLOTH_OT_create_pin_group.execute(op, mock_context)
        self.assertEqual(res, {'FINISHED'})

        # 頂点グループが作成されたこと
        mock_obj.vertex_groups.new.assert_called_with(name="Cloth_Pin")
        # 設定にグループ名が反映されたこと
        self.assertEqual(mock_settings.pin_vertex_group, "Cloth_Pin")
        # アクティブインデックスが更新されたこと
        self.assertEqual(mock_obj.vertex_groups.active_index, 2)
        # ウェイトペイントモードへ移行したこと
        mock_bpy.ops.object.mode_set.assert_called_with(mode='WEIGHT_PAINT')
        self.assertTrue(op.report.called)

    @patch("taremin_cloth.ops.basic.bpy")
    def test_toggle_weight_paint_from_object_to_paint(self, mock_bpy):
        """オブジェクトモードからウェイトペイントモードへのトグル移行を検証"""
        op = MagicMock()

        mock_context = MagicMock()
        mock_context.mode = 'OBJECT'

        mock_obj = MagicMock()
        mock_obj.type = 'MESH'
        mock_settings = MagicMock()
        mock_settings.pin_vertex_group = "Existing_Pin"
        mock_obj.taremin_cloth = mock_settings

        mock_existing_vg = MagicMock()
        mock_existing_vg.name = "Existing_Pin"
        mock_existing_vg.index = 1
        mock_obj.vertex_groups.get.side_effect = lambda name: mock_existing_vg if name == "Existing_Pin" else None

        mock_context.active_object = mock_obj
        mock_bpy.ops.object.mode_set.poll.return_value = True

        res = basic.TAREMIN_CLOTH_OT_toggle_weight_paint.execute(op, mock_context)
        self.assertEqual(res, {'FINISHED'})

        # 既存グループがアクティブに設定されたこと
        self.assertEqual(mock_obj.vertex_groups.active_index, 1)
        # ウェイトペイントモードに移行したこと
        mock_bpy.ops.object.mode_set.assert_called_with(mode='WEIGHT_PAINT')

    @patch("taremin_cloth.ops.basic.bpy")
    def test_toggle_weight_paint_from_paint_to_object(self, mock_bpy):
        """ウェイトペイントモードからオブジェクトモードへの復帰トグルを検証"""
        op = MagicMock()

        mock_context = MagicMock()
        mock_context.mode = 'PAINT_WEIGHT'

        mock_obj = MagicMock()
        mock_obj.type = 'MESH'
        mock_context.active_object = mock_obj
        mock_bpy.ops.object.mode_set.poll.return_value = True

        res = basic.TAREMIN_CLOTH_OT_toggle_weight_paint.execute(op, mock_context)
        self.assertEqual(res, {'FINISHED'})

        # オブジェクトモードに復帰したこと
        mock_bpy.ops.object.mode_set.assert_called_with(mode='OBJECT')

    @patch("taremin_cloth.ops.basic.bpy")
    def test_toggle_weight_paint_creates_group_if_missing(self, mock_bpy):
        """グループ未存在時に自動作成してウェイトペイントモードへ移行することを検証"""
        op = MagicMock()

        mock_context = MagicMock()
        mock_context.mode = 'OBJECT'

        mock_obj = MagicMock()
        mock_obj.type = 'MESH'
        mock_settings = MagicMock()
        mock_settings.pin_vertex_group = ""  # 未指定
        mock_obj.taremin_cloth = mock_settings
        mock_obj.vertex_groups.get.return_value = None

        new_vg = MagicMock()
        new_vg.name = "Cloth_Pin"
        new_vg.index = 0
        mock_obj.vertex_groups.new.return_value = new_vg

        mock_context.active_object = mock_obj
        mock_bpy.ops.object.mode_set.poll.return_value = True

        res = basic.TAREMIN_CLOTH_OT_toggle_weight_paint.execute(op, mock_context)
        self.assertEqual(res, {'FINISHED'})

        # Cloth_Pin が新規作成されたこと
        mock_obj.vertex_groups.new.assert_called_with(name="Cloth_Pin")
        self.assertEqual(mock_settings.pin_vertex_group, "Cloth_Pin")
        self.assertEqual(mock_obj.vertex_groups.active_index, 0)
        mock_bpy.ops.object.mode_set.assert_called_with(mode='WEIGHT_PAINT')


if __name__ == "__main__":
    unittest.main()
