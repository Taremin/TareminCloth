"""
HUD操作ガイド（Viewport 2D Overlay）および描画ステート管理の単体テスト
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
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from taremin_cloth.utils import drawing
from taremin_cloth import properties, i18n


class TestDrawingHud(unittest.TestCase):
    """HUD操作ガイドおよびFPS情報管理のテスト"""

    def setUp(self):
        drawing.clear_interactive_fps_info()
        drawing.clear_bake_overlay_info()
        drawing.clear_init_overlay_info()
        drawing.clear_toast_overlay_info()
        drawing.set_interactive_active(False)

    def tearDown(self):
        drawing.clear_interactive_fps_info()
        drawing.clear_bake_overlay_info()
        drawing.clear_init_overlay_info()
        drawing.clear_toast_overlay_info()
        drawing.set_interactive_active(False)

    def test_set_and_get_interactive_fps_info_with_help(self):
        """show_help を含むFPS/HUD情報の更新と取得テスト"""
        drawing.set_interactive_fps_info(
            fps=59.9,
            frame_ms=16.7,
            show_overlay=True,
            position='TOP_CENTER',
            show_help=True,
        )
        info = drawing.get_interactive_fps_info()
        self.assertIsNotNone(info)
        self.assertAlmostEqual(info["fps"], 59.9, places=1)
        self.assertAlmostEqual(info["frame_ms"], 16.7, places=1)
        self.assertTrue(info["show_overlay"])
        self.assertEqual(info["position"], 'TOP_CENTER')
        self.assertTrue(info["show_help"])

    def test_show_help_toggle(self):
        """show_help が False の場合の設定保持テスト"""
        drawing.set_interactive_fps_info(
            fps=60.0,
            frame_ms=16.6,
            show_overlay=False,
            position='BOTTOM_RIGHT',
            show_help=False,
        )
        info = drawing.get_interactive_fps_info()
        self.assertIsNotNone(info)
        self.assertFalse(info["show_overlay"])
        self.assertFalse(info["show_help"])
        self.assertEqual(info["position"], 'BOTTOM_RIGHT')

    def test_properties_has_show_hud_help(self):
        """TareminClothObjectSettings に show_hud_help プロパティが定義されていることを確認"""
        settings_cls = properties.TareminClothObjectSettings
        has_attr = (
            hasattr(settings_cls, "show_hud_help")
            or ("__annotations__" in dir(settings_cls) and "show_hud_help" in settings_cls.__annotations__)
        )
        self.assertTrue(has_attr, "TareminClothObjectSettings に show_hud_help プロパティが存在すること")

    def test_i18n_translation_keys_exist(self):
        """操作ガイド用の文言が日英辞書に登録されていることを確認"""
        raw_dict = i18n.get_raw_translations()
        self.assertIn("ja_JP", raw_dict)
        self.assertIn("en_US", raw_dict)

        key = "[LMB Drag] Move  |  [P] Pin/Unpin  |  [Esc / RMB] Exit"
        self.assertIn(key, raw_dict["ja_JP"])
        self.assertIn("左ドラッグ", raw_dict["ja_JP"][key])
        self.assertEqual(raw_dict["en_US"][key], key)

        guide_key = "Show Key Guide"
        self.assertIn(guide_key, raw_dict["ja_JP"])
        self.assertEqual(raw_dict["ja_JP"][guide_key], "キー操作ガイドを表示")

        hud_key = "Show HUD Help"
        self.assertIn(hud_key, raw_dict["ja_JP"])
        self.assertEqual(raw_dict["ja_JP"][hud_key], "HUD操作ガイドを表示")

    @patch.dict("sys.modules", {"blf": MagicMock()})
    @patch("taremin_cloth.utils.drawing.batch_for_shader")
    @patch("taremin_cloth.utils.drawing.gpu")
    @patch("taremin_cloth.utils.drawing.bpy")
    def test_draw_callback_2d_execution(self, mock_bpy, mock_gpu, mock_batch_for_shader):
        """draw_callback_2d がエラーなく実行され、各モックが適切に呼ばれるか確認"""
        mock_region = MagicMock()
        mock_region.width = 1920
        mock_region.height = 1080
        mock_bpy.context.region = mock_region

        mock_batch = MagicMock()
        mock_batch_for_shader.return_value = mock_batch

        mock_shader = MagicMock()
        mock_gpu.shader.from_builtin.return_value = mock_shader

        # インタラクティブ有効化 + 情報セット
        drawing.set_interactive_active(True)
        drawing.set_interactive_fps_info(
            fps=60.0,
            frame_ms=16.6,
            show_overlay=True,
            position='TOP_CENTER',
            show_help=True,
        )

        # 描画コールバック呼び出し（例外が発生しないこと）
        try:
            drawing.draw_callback_2d()
        except Exception as e:
            self.fail(f"draw_callback_2d raised exception: {e}")

        self.assertTrue(mock_batch.draw.called)

        # 両方非表示の場合に即座に戻ること
        mock_batch.reset_mock()
        drawing.set_interactive_fps_info(
            fps=60.0,
            frame_ms=16.6,
            show_overlay=False,
            position='TOP_CENTER',
            show_help=False,
        )
        try:
            drawing.draw_callback_2d()
        except Exception as e:
            self.fail(f"draw_callback_2d with both disabled raised exception: {e}")
        self.assertFalse(mock_batch.draw.called)

    @patch.dict("sys.modules", {"blf": MagicMock()})
    @patch("taremin_cloth.utils.drawing.batch_for_shader")
    @patch("taremin_cloth.utils.drawing.gpu")
    @patch("taremin_cloth.utils.drawing.bpy")
    def test_draw_callback_2d_sewing_badge(self, mock_bpy, mock_gpu, mock_batch_for_shader):
        """縫合フェーズバッジ付きでも2Dオーバーレイ描画が正常に実行されるか確認"""
        mock_region = MagicMock()
        mock_region.width = 1920
        mock_region.height = 1080
        mock_bpy.context.region = mock_region

        mock_batch = MagicMock()
        mock_batch_for_shader.return_value = mock_batch

        mock_shader = MagicMock()
        mock_gpu.shader.from_builtin.return_value = mock_shader

        drawing.set_interactive_active(True)
        drawing.set_interactive_fps_info(
            fps=60.0,
            frame_ms=16.6,
            show_overlay=True,
            position='TOP_CENTER',
            show_help=False,
            sewing_text="Phase: Sewing 62% (gravity 0.00x)",
            sewing_active=True,
        )
        try:
            drawing.draw_callback_2d()
        except Exception as e:
            self.fail(f"draw_callback_2d with sewing badge raised exception: {e}")
        self.assertTrue(mock_batch.draw.called)

        # FPS非表示時も縫合バッジ単独で描画されること
        mock_batch.reset_mock()
        drawing.set_interactive_fps_info(
            fps=60.0,
            frame_ms=16.6,
            show_overlay=False,
            position='TOP_CENTER',
            show_help=False,
            sewing_text="Phase: Normal (gravity 1.00x)",
            sewing_active=False,
        )
        try:
            drawing.draw_callback_2d()
        except Exception as e:
            self.fail(f"draw_callback_2d with sewing-only badge raised exception: {e}")
        self.assertTrue(mock_batch.draw.called)

    def test_bake_overlay_info_state(self):
        """ベイクオーバーレイ情報の更新・取得・クリアの動作確認"""
        self.assertFalse(drawing.is_bake_overlay_active())
        self.assertIsNone(drawing.get_bake_overlay_info())

        drawing.set_bake_overlay_info(current_frame=15, end_frame=250, pct=6)
        self.assertTrue(drawing.is_bake_overlay_active())
        info = drawing.get_bake_overlay_info()
        self.assertIsNotNone(info)
        self.assertEqual(info["current_frame"], 15)
        self.assertEqual(info["end_frame"], 250)
        self.assertEqual(info["pct"], 6)
        self.assertIsNone(info["sewing_text"])

        drawing.set_bake_overlay_info(current_frame=15, end_frame=250, pct=6,
                                      sewing_text="Phase: Sewing 62% (gravity 0.00x)")
        info = drawing.get_bake_overlay_info()
        self.assertEqual(info["sewing_text"], "Phase: Sewing 62% (gravity 0.00x)")

        drawing.clear_bake_overlay_info()
        self.assertFalse(drawing.is_bake_overlay_active())
        self.assertIsNone(drawing.get_bake_overlay_info())

    def test_interactive_fps_info_sewing_keys(self):
        """FPS情報に縫合フェーズ表示キーが既定保持されること"""
        drawing.set_interactive_fps_info(fps=60.0, frame_ms=16.6)
        info = drawing.get_interactive_fps_info()
        self.assertIsNone(info["sewing_text"])
        self.assertFalse(info["sewing_active"])

        drawing.set_interactive_fps_info(fps=60.0, frame_ms=16.6,
                                         sewing_text="Phase: Sewing 62% (gravity 0.00x)",
                                         sewing_active=True)
        info = drawing.get_interactive_fps_info()
        self.assertEqual(info["sewing_text"], "Phase: Sewing 62% (gravity 0.00x)")
        self.assertTrue(info["sewing_active"])

    def test_interactive_fps_info_guide_lines(self):
        """Guidance 2行表示キー（guide_line1_suffix/guide_line2）の保持確認"""
        drawing.set_interactive_fps_info(
            fps=60.0, frame_ms=16.6,
            guide_line1_suffix="[E] switch [F]radius",
            guide_line2="Tool: Range Grab r=50mm s=0.80",
        )
        info = drawing.get_interactive_fps_info()
        self.assertEqual(info["guide_line1_suffix"], "[E] switch [F]radius")
        self.assertEqual(info["guide_line2"], "Tool: Range Grab r=50mm s=0.80")
        # 既定はNone（単行フォールバック）
        drawing.set_interactive_fps_info(fps=60.0, frame_ms=16.6)
        info = drawing.get_interactive_fps_info()
        self.assertIsNone(info["guide_line1_suffix"])
        self.assertIsNone(info["guide_line2"])

    @patch.dict("sys.modules", {"blf": MagicMock()})
    @patch("taremin_cloth.utils.drawing.batch_for_shader")
    @patch("taremin_cloth.utils.drawing.gpu")
    @patch("taremin_cloth.utils.drawing.bpy")
    def test_draw_callback_2d_guide_2row(self, mock_bpy, mock_gpu, mock_batch_for_shader):
        """2行ガイダンス指定でも2Dオーバーレイ描画が正常に実行されるか確認"""
        mock_region = MagicMock()
        mock_region.width = 1920
        mock_region.height = 1080
        mock_bpy.context.region = mock_region

        mock_batch = MagicMock()
        mock_batch_for_shader.return_value = mock_batch

        mock_shader = MagicMock()
        mock_gpu.shader.from_builtin.return_value = mock_shader

        drawing.set_interactive_active(True)
        drawing.set_interactive_fps_info(
            fps=60.0,
            frame_ms=16.6,
            show_overlay=False,
            position='TOP_CENTER',
            show_help=True,
            guide_line1_suffix="[E] switch [F]radius",
            guide_line2="Tool: Range Grab r=50mm s=0.80",
        )
        try:
            drawing.draw_callback_2d()
        except Exception as e:
            self.fail(f"draw_callback_2d with 2-row guide raised exception: {e}")
        self.assertTrue(mock_batch.draw.called)

    @patch.dict("sys.modules", {"blf": MagicMock()})
    @patch("taremin_cloth.utils.drawing.batch_for_shader")
    @patch("taremin_cloth.utils.drawing.gpu")
    @patch("taremin_cloth.utils.drawing.bpy")
    def test_draw_callback_2d_bake_overlay(self, mock_bpy, mock_gpu, mock_batch_for_shader):
        """ベイク中の2Dオーバーレイ描画が正常に実行されるか確認"""
        mock_region = MagicMock()
        mock_region.width = 1920
        mock_region.height = 1080
        mock_bpy.context.region = mock_region

        mock_batch = MagicMock()
        mock_batch_for_shader.return_value = mock_batch

        mock_shader = MagicMock()
        mock_gpu.shader.from_builtin.return_value = mock_shader

        # インタラクティブは非アクティブ、ベイク中のみ有効
        drawing.set_interactive_active(False)
        drawing.set_bake_overlay_info(current_frame=50, end_frame=100, pct=50)

        try:
            drawing.draw_callback_2d()
        except Exception as e:
            self.fail(f"draw_callback_2d for bake overlay raised exception: {e}")

        # 背景ボックスと進捗バーの描画が行われたことを確認
        self.assertTrue(mock_batch.draw.called)

    @patch.dict("sys.modules", {"blf": MagicMock()})
    @patch("taremin_cloth.utils.drawing._draw_bottom_center_progress")
    @patch("taremin_cloth.utils.drawing._draw_bottom_badge_2row")
    @patch("taremin_cloth.utils.drawing.gpu")
    @patch("taremin_cloth.utils.drawing.bpy")
    def test_draw_callback_2d_bottom_exclusivity(self, mock_bpy, mock_gpu, mock_draw_guide, mock_draw_progress):
        """下部中央の Guidance / Progress / Toast が排他表示され重複描画しないことを確認"""
        mock_region = MagicMock()
        mock_region.width = 1920
        mock_region.height = 1080
        mock_bpy.context.region = mock_region
        mock_gpu.shader.from_builtin.return_value = MagicMock()

        drawing.set_interactive_active(True)
        drawing.set_interactive_fps_info(
            fps=60.0,
            frame_ms=16.6,
            show_overlay=False,
            position='TOP_CENTER',
            show_help=True,
        )
        drawing.set_bake_overlay_info(current_frame=10, end_frame=100, pct=10)
        drawing.set_init_overlay_info(message="Init", current=1, total=3)
        drawing.set_toast_overlay_info(message="Toast", duration=0.0)

        drawing.draw_callback_2d()

        mock_draw_guide.assert_not_called()
        mock_draw_progress.assert_called_once()
        # 最優先の Bake が描画されたことを確認
        self.assertIn("10 / 100", mock_draw_progress.call_args.args[2])

        # Bake / Init / Toast がなければ Guidance が描画される
        mock_draw_guide.reset_mock()
        mock_draw_progress.reset_mock()
        drawing.clear_bake_overlay_info()
        drawing.clear_init_overlay_info()
        drawing.clear_toast_overlay_info()
        drawing.draw_callback_2d()
        mock_draw_guide.assert_called_once()
        mock_draw_progress.assert_not_called()


if __name__ == "__main__":
    unittest.main()
