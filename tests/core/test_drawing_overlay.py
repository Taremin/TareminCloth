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

from taremin_cloth.utils import drawing


class TestDrawingOverlayOptimization(unittest.TestCase):
    def setUp(self):
        drawing.set_interactive_active(False)
        drawing.clear_overlay_caches()

    def tearDown(self):
        drawing.set_interactive_active(False)
        drawing.clear_overlay_caches()

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
        """非インタラクティブ実行時は draw_callback_3d が即座に早期リターンすること"""
        drawing.set_interactive_active(False)

        # bpy.context.scene がアクセスされても例外が起きない・オブジェクト走査が行われないことを確認
        with patch.object(drawing, "bpy") as mock_bpy:
            # scene.objects が参照されたら例外を発生させるモックを設定
            mock_scene = MagicMock()
            type(mock_scene).objects = property(lambda self: (_ for _ in ()).throw(RuntimeError("objects should not be accessed")))
            mock_bpy.context.scene = mock_scene

            # 非アクティブ時は objects に触れず即 return するためエラーにならない
            try:
                drawing.draw_callback_3d()
            except RuntimeError:
                self.fail("draw_callback_3d should return immediately when not interactive")

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
