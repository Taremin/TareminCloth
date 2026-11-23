"""
3Dビューポート・オーバーレイ描画のシミュレーション中限定化およびキャッシュ動作の単体テスト
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# pythonディレクトリをsys.pathに追加
root_dir = Path(__file__).parent.parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from taremin_cloth.utils import drawing


class TestDrawingOverlayOptimization(unittest.TestCase):
    def setUp(self):
        drawing.set_interactive_active(False)
        drawing.clear_overlay_caches()
        self._saved_shaders = (
            drawing._cached_point_shader,
            drawing._cached_line_shader,
            drawing._cached_2d_shader,
            drawing._cached_smooth_line_shader,
        )

    def tearDown(self):
        drawing.set_interactive_active(False)
        drawing.clear_overlay_caches()
        (drawing._cached_point_shader,
         drawing._cached_line_shader,
         drawing._cached_2d_shader,
         drawing._cached_smooth_line_shader) = self._saved_shaders

    def test_interactive_state_toggle_and_cache_clear(self):
        """インタラクティブ状態の切り替えとキャッシュクリアの連動テスト"""
        self.assertFalse(drawing.is_interactive_active())

        # キャッシュにダミーデータを注入
        drawing._pin_indices_cache[("TestObj", "Pin")] = ([0, 1, 2], 3)
        drawing._sewing_edge_indices_cache["TestObj"] = ([(0, 1)], 2, 1)
        self.assertEqual(len(drawing._pin_indices_cache), 1)
        self.assertEqual(len(drawing._sewing_edge_indices_cache), 1)

        # アクティブ化
        drawing.set_interactive_active(True)
        self.assertTrue(drawing.is_interactive_active())
        # アクティブ化時はキャッシュ維持
        self.assertEqual(len(drawing._pin_indices_cache), 1)
        self.assertEqual(len(drawing._sewing_edge_indices_cache), 1)

        # 非アクティブ化（シミュレーション終了時）
        drawing.set_interactive_active(False)
        self.assertFalse(drawing.is_interactive_active())
        # 非アクティブ化時に全オーバーレイキャッシュが自動クリアされること
        self.assertEqual(len(drawing._pin_indices_cache), 0)
        self.assertEqual(len(drawing._sewing_edge_indices_cache), 0)

    def test_draw_callback_3d_early_return_when_not_interactive(self):
        """非インタラクティブ実行時かつ常時表示対象布がない場合は GPU 描画処理に入らず早期リターンすること"""
        drawing.set_interactive_active(False)

        mock_obj = MagicMock()
        mock_obj.type = "MESH"
        mock_obj.taremin_cloth.is_cloth = True
        mock_obj.taremin_cloth.pin_overlay_interactive_only = True
        mock_obj.taremin_cloth.elastic_overlay_interactive_only = True
        mock_obj.visible_get = MagicMock(return_value=True)

        for objs in ([], [mock_obj]):
            mock_scene = MagicMock()
            mock_scene.objects = objs
            with (
                patch.object(drawing, "bpy") as mock_bpy,
                patch.object(drawing, "get_point_shader", side_effect=AssertionError("should not fetch point shader")),
                patch.object(drawing, "get_line_shader", side_effect=AssertionError("should not fetch line shader")),
                patch.object(drawing, "get_smooth_color_line_shader", side_effect=AssertionError("should not fetch smooth shader")),
                patch.object(drawing, "batch_for_shader", side_effect=AssertionError("should not build batch")),
            ):
                mock_bpy.context.scene = mock_scene
                drawing.draw_callback_3d()

    def test_shader_caching(self):
        """シェーダー取得関数がキャッシュを保持し再利用すること"""
        drawing._cached_point_shader = "mock_point_shader"
        drawing._cached_line_shader = "mock_line_shader"
        drawing._cached_2d_shader = "mock_2d_shader"
        drawing._cached_smooth_line_shader = "mock_smooth_line_shader"

        self.assertEqual(drawing.get_point_shader(), "mock_point_shader")
        self.assertEqual(drawing.get_line_shader(), "mock_line_shader")
        self.assertEqual(drawing.get_2d_uniform_color_shader(), "mock_2d_shader")
        self.assertEqual(drawing.get_smooth_color_line_shader(), "mock_smooth_line_shader")


if __name__ == "__main__":
    unittest.main()
