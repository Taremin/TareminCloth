# -*- coding: utf-8 -*-
"""2D製図モードと他オブジェクト選択時の自動解除ハンドラーの検証テスト."""

import unittest
from unittest.mock import MagicMock, patch
from taremin_cloth.utils import pattern_view


class TestPatternViewHandler(unittest.TestCase):
    """型紙2D製図モード解除ハンドラーの動作テスト"""

    def setUp(self):
        pattern_view._pending_restore_timer = False



    def test_handler_registration(self):
        """register/unregister で depsgraph_update_post に正しく登録・解除されることの検証"""
        mock_handlers = []
        with patch("bpy.app.handlers.depsgraph_update_post", mock_handlers, create=True):
            pattern_view.register()
            self.assertIn(pattern_view.on_depsgraph_update_pattern_view, mock_handlers)

            pattern_view.unregister()
            self.assertNotIn(pattern_view.on_depsgraph_update_pattern_view, mock_handlers)

    def test_handler_ignores_pattern_piece(self):
        """型紙オブジェクトがアクティブな場合は解除処理をトリガーしないことの検証"""
        mock_obj = MagicMock()
        mock_obj.taremin_pattern.is_pattern_piece = True

        mock_context = MagicMock()
        mock_context.view_layer.objects.active = mock_obj

        with patch("bpy.context", mock_context):
            pattern_view.on_depsgraph_update_pattern_view(None, None)
            self.assertFalse(pattern_view._pending_restore_timer)

    def test_handler_triggers_when_other_object_active(self):
        """型紙以外のオブジェクト（素体、カメラ等）がアクティブかつ2Dモード中の場合、解除タイマーを登録することの検証"""
        # 型紙フラグが False の一般オブジェクト
        mock_obj = MagicMock()
        mock_obj.taremin_pattern.is_pattern_piece = False

        # 2Dモード（ORTHO かつ lock_rotation=True）の3Dビュー
        mock_rv3d = MagicMock()
        mock_rv3d.lock_rotation = True
        mock_rv3d.view_perspective = 'ORTHO'

        mock_space = MagicMock()
        mock_space.region_3d = mock_rv3d

        mock_area = MagicMock()
        mock_area.type = 'VIEW_3D'
        mock_area.spaces.active = mock_space

        mock_screen = MagicMock()
        mock_screen.areas = [mock_area]

        mock_window = MagicMock()
        mock_window.screen = mock_screen

        mock_context = MagicMock()
        mock_context.view_layer.objects.active = mock_obj
        mock_context.window_manager.windows = [mock_window]

        mock_app = MagicMock()
        mock_timer_register = MagicMock()
        mock_app.timers.register = mock_timer_register

        with patch("bpy.context", mock_context), \
             patch("bpy.app", mock_app):
            pattern_view.on_depsgraph_update_pattern_view(None, None)
            self.assertTrue(pattern_view._pending_restore_timer)
            mock_timer_register.assert_called_once()

    def test_enter_pattern_local_view_with_target_objects(self):
        """target_objects を指定した場合に指定オブジェクトのみが選択されローカルビューに入ることの検証"""
        target_obj = MagicMock()
        target_obj.name = "NewPattern"

        other_obj = MagicMock()
        other_obj.name = "OtherObj"

        mock_space = MagicMock()
        mock_space.type = 'VIEW_3D'
        mock_space.local_view = None

        mock_area = MagicMock()
        mock_area.type = 'VIEW_3D'
        mock_area.spaces.active = mock_space

        mock_context = MagicMock()
        mock_context.space_data = mock_space
        mock_context.area = mock_area
        mock_context.scene.objects = {"NewPattern": target_obj, "OtherObj": other_obj}
        mock_context.selected_objects = [other_obj]

        mock_localview = MagicMock()
        with patch("bpy.ops.view3d.localview", mock_localview):
            pattern_view.enter_pattern_local_view(mock_context, target_objects=[target_obj])

            # 他のオブジェクトは非選択、ターゲットオブジェクトのみ選択・アクティブ化
            other_obj.select_set.assert_called_with(False)
            target_obj.select_set.assert_called_with(True)
            self.assertEqual(mock_context.view_layer.objects.active, target_obj)
            mock_localview.assert_called_once_with(frame_selected=True)


