"""2Dパターンエディタ下絵参照機能のBlender E2E・実GPUピクセル描画テスト。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import bpy
import gpu
from mathutils import Matrix

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PYTHON_DIR = os.path.join(_ROOT, "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

import taremin_cloth
from taremin_cloth.ops.pattern import (
    clear_active_pattern_editor_op,
    get_active_pattern_canvas,
    get_active_pattern_editor_controller,
    set_active_pattern_editor_op,
)
from taremin_cloth.pattern.canvas import PatternCanvasRenderer
from taremin_cloth.pattern.document import Document, ReferenceImage
from taremin_cloth.pattern.editor import EditorController
from taremin_cloth.pattern.renderer import DrawCommandKind, build_render_commands
from taremin_cloth.pattern.session import ViewTransform2D
from taremin_cloth.utils.view3d import is_mouse_over_viewport_chrome


def _pixel_ortho_matrix(width: float, height: float) -> Matrix:
    """スクリーン座標 [0, width] x [0, height] -> NDC [-1, 1] の正射影行列を返す。"""
    return Matrix((
        (2.0 / width, 0.0, 0.0, -1.0),
        (0.0, 2.0 / height, 0.0, -1.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))


def _create_quadrant_png(filepath: str) -> None:
    """UV向きと色合成を検証するための4象限カラーPNGを作成する。

    左下: 赤 (1,0,0,1), 右下: 緑 (0,1,0,1),
    左上: 青 (0,0,1,1), 右上: 白 (1,1,1,1)
    """
    w, h = 8, 8
    img = bpy.data.images.new("__QuadTestSrc", width=w, height=h, alpha=True)
    try:
        try:
            img.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        pixels = [0.0] * (w * h * 4)
        for y in range(h):
            for x in range(w):
                idx = (y * w + x) * 4
                is_right = x >= (w // 2)
                is_top = y >= (h // 2)
                if not is_top and not is_right:
                    rgba = (1.0, 0.0, 0.0, 1.0)
                elif not is_top and is_right:
                    rgba = (0.0, 1.0, 0.0, 1.0)
                elif is_top and not is_right:
                    rgba = (0.0, 0.0, 1.0, 1.0)
                else:
                    rgba = (1.0, 1.0, 1.0, 1.0)
                pixels[idx:idx + 4] = rgba
        img.pixels.foreach_set(pixels)
        img.filepath_raw = filepath
        img.file_format = "PNG"
        img.save()
    finally:
        bpy.data.images.remove(img, do_unlink=True)


class TestPatternReferenceImageE2E(unittest.TestCase):
    """下絵参照機能のBlender結合およびGPU描画ピクセル検証テスト。"""

    def setUp(self) -> None:
        taremin_cloth.register()
        clear_active_pattern_editor_op()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.scene = bpy.context.scene

    def tearDown(self) -> None:
        clear_active_pattern_editor_op()
        for img in list(bpy.data.images):
            if img.name.startswith("__QuadTest") or img.name.startswith("__EmptyTest") or img.filepath.startswith(self._tmpdir.name):
                bpy.data.images.remove(img, do_unlink=True)
        self._tmpdir.cleanup()
        taremin_cloth.unregister()

    def test_gpu_offscreen_pixel_draw_and_opacity_blend(self):
        png_path = os.path.join(self._tmpdir.name, "quad_ref.png")
        _create_quadrant_png(png_path)

        doc = Document()
        doc.reference_image = ReferenceImage(
            image_path=png_path,
            offset=(0.0, 0.0),
            width=1.0,
            aspect_ratio=1.0,
            opacity=1.0,
            visible=True,
        )
        ctrl = EditorController(
            doc=doc,
            view=ViewTransform2D(zoom=128.0, pan_x=0.0, pan_y=0.0),
        )
        renderer = PatternCanvasRenderer(ctrl)

        if not hasattr(gpu, "init"):
            # Blender 4.2 などバックグラウンド環境で gpu.init が利用不可の場合はモックで描画コマンド発行経路を検証
            mock_shader = MagicMock()
            mock_batch = MagicMock()
            with patch.object(renderer, "_get_image_shader", return_value=mock_shader), \
                 patch.object(renderer, "_resolve_image_texture", return_value=MagicMock()), \
                 patch("taremin_cloth.pattern.canvas.batch_for_shader", return_value=mock_batch):
                renderer._draw_image(
                    [(0.0, 0.0), (128.0, 0.0), (128.0, 128.0), (0.0, 128.0)],
                    color=(1.0, 1.0, 1.0, 0.5),
                    image_path=png_path,
                )
            mock_shader.uniform_sampler.assert_called_once()
            mock_shader.uniform_float.assert_called_with("color", (1.0, 1.0, 1.0, 0.5))
            mock_batch.draw.assert_called_once_with(mock_shader)
            return

        gpu.init()
        offscreen = gpu.types.GPUOffScreen(128, 128)
        try:
            cmds = build_render_commands(doc, ctrl, screen_size=(128, 128))
            img_cmd = next(c for c in cmds if c.kind == DrawCommandKind.IMAGE)

            def render_once(opacity_val: float):
                with offscreen.bind():
                    fb = gpu.state.active_framebuffer_get()
                    fb.clear(color=(0.0, 0.0, 0.0, 1.0))
                    gpu.state.blend_set("ALPHA")
                    gpu.matrix.push_projection()
                    gpu.matrix.load_projection_matrix(_pixel_ortho_matrix(128.0, 128.0))
                    gpu.matrix.push()
                    gpu.matrix.load_identity()
                    try:
                        col = (img_cmd.color[0], img_cmd.color[1], img_cmd.color[2], opacity_val)
                        renderer._draw_image(img_cmd.points, color=col, image_path=img_cmd.image_path)
                    finally:
                        gpu.matrix.pop()
                        gpu.matrix.pop_projection()
                        gpu.state.blend_set("NONE")
                    buf = fb.read_color(0, 0, 128, 128, 4, 0, "FLOAT")
                    buf.dimensions = 128 * 128 * 4
                    return buf

            # opacity=1.0: 四象限がそれぞれ 赤/緑/青/白 で描画されること（UV上下反転なし）
            full_buf = render_once(1.0)

            def sample(buf, x: int, y: int) -> tuple[float, float, float, float]:
                idx = (y * 128 + x) * 4
                return tuple(buf[idx:idx + 4])

            bl = sample(full_buf, 32, 32)
            br = sample(full_buf, 96, 32)
            tl = sample(full_buf, 32, 96)
            tr = sample(full_buf, 96, 96)

            # 左下=赤
            self.assertGreater(bl[0], 0.8)
            self.assertLess(bl[1], 0.2)
            self.assertLess(bl[2], 0.2)
            # 右下=緑
            self.assertLess(br[0], 0.2)
            self.assertGreater(br[1], 0.8)
            self.assertLess(br[2], 0.2)
            # 左上=青
            self.assertLess(tl[0], 0.2)
            self.assertLess(tl[1], 0.2)
            self.assertGreater(tl[2], 0.8)
            # 右上=白
            self.assertGreater(tr[0], 0.8)
            self.assertGreater(tr[1], 0.8)
            self.assertGreater(tr[2], 0.8)

            # opacity=0.5: 黒背景上で輝度が約半分にブレンドされること
            half_buf = render_once(0.5)
            bl_half = sample(half_buf, 32, 32)
            self.assertAlmostEqual(bl_half[0], bl[0] * 0.5, delta=0.12)
        finally:
            offscreen.free()

    def test_missing_path_and_zero_size_image_guard_and_cache_clear(self):
        ctrl = EditorController(doc=Document())
        renderer = PatternCanvasRenderer(ctrl)

        # 存在しないファイルは負キャッシュに入り例外にならないこと
        missing = os.path.join(self._tmpdir.name, "not_found.png")
        self.assertIsNone(renderer._resolve_image_texture(missing))
        self.assertGreater(len(renderer._failed_image_paths), 0)

        renderer.clear_image_cache(image_path=missing)
        self.assertEqual(len(renderer._failed_image_paths), 0)

        # size == (0, 0) / has_data == False の空画像はクラッシュせず None を返すこと
        empty_path = os.path.join(self._tmpdir.name, "empty_placeholder.png")
        empty_img = SimpleNamespace(
            name="__EmptyTestImg",
            filepath=empty_path,
            size=(0, 0),
            has_data=False,
            packed_file=None,
        )
        fake_images = MagicMock()
        fake_images.__iter__.return_value = iter([empty_img])
        fake_images.get.return_value = empty_img
        fake_bpy = SimpleNamespace(data=SimpleNamespace(images=fake_images), path=bpy.path)
        renderer._image_name_cache[renderer._normalize_image_path(empty_path)] = empty_img.name
        with patch.dict(sys.modules, {"bpy": fake_bpy}):
            self.assertIsNone(renderer._resolve_image_texture(empty_path))

    def test_operators_sync_closed_and_active_editor_session(self):
        png_path = os.path.join(self._tmpdir.name, "op_ref.png")
        _create_quadrant_png(png_path)

        # 1. エディタ非起動時: JSONプロパティへ設定・トグル・クリアが保存されること
        res_set = bpy.ops.taremin_cloth.set_reference_image(filepath=png_path)
        self.assertIn("FINISHED", res_set)
        raw = json.loads(self.scene.taremin_pattern_document)
        self.assertIsNotNone(raw.get("reference_image"))
        self.assertTrue(raw["reference_image"]["visible"])

        res_tog = bpy.ops.taremin_cloth.toggle_reference_image()
        self.assertIn("FINISHED", res_tog)
        raw_tog = json.loads(self.scene.taremin_pattern_document)
        self.assertFalse(raw_tog["reference_image"]["visible"])

        res_clr = bpy.ops.taremin_cloth.clear_reference_image()
        self.assertIn("FINISHED", res_clr)
        raw_clr = json.loads(self.scene.taremin_pattern_document)
        self.assertIsNone(raw_clr.get("reference_image"))

        # 2. エディタ起動中: アクティブコントローラーへ即時反映され Undo 可能であること
        ctrl = EditorController(doc=Document())
        canvas = PatternCanvasRenderer(ctrl)
        mock_op = SimpleNamespace(controller=ctrl, canvas=canvas)
        set_active_pattern_editor_op(mock_op)
        self.assertIs(get_active_pattern_editor_controller(), ctrl)
        self.assertIs(get_active_pattern_canvas(), canvas)

        from taremin_cloth.panels import _get_panel_reference_image_info

        bpy.ops.taremin_cloth.set_reference_image(filepath=png_path)
        self.assertIsNotNone(ctrl.doc.reference_image)
        self.assertTrue(ctrl.doc.reference_image.visible)
        self.assertEqual(_get_panel_reference_image_info(self.scene), ("op_ref.png", True))

        bpy.ops.taremin_cloth.toggle_reference_image()
        self.assertFalse(ctrl.doc.reference_image.visible)
        self.assertEqual(_get_panel_reference_image_info(self.scene), ("op_ref.png", False))

        bpy.ops.taremin_cloth.clear_reference_image()
        self.assertIsNone(ctrl.doc.reference_image)
        self.assertIsNone(_get_panel_reference_image_info(self.scene))

        # Undoで戻り、パネル表示名も復元されること
        self.assertTrue(ctrl.undo())
        self.assertIsNotNone(ctrl.doc.reference_image)
        self.assertFalse(ctrl.doc.reference_image.visible)
        self.assertEqual(_get_panel_reference_image_info(self.scene), ("op_ref.png", False))

        # アクティブエディタ解除後はシーンJSONプロパティを参照すること
        clear_active_pattern_editor_op()
        self.assertIsNone(_get_panel_reference_image_info(self.scene))
        bpy.ops.taremin_cloth.set_reference_image(filepath=png_path)
        self.assertEqual(_get_panel_reference_image_info(self.scene), ("op_ref.png", True))

    def test_viewport_chrome_hit_detection_boundary(self):
        region_window = SimpleNamespace(type="WINDOW", x=0, y=0, width=1000, height=800)
        region_header = SimpleNamespace(type="HEADER", x=0, y=770, width=1000, height=30)
        region_ui = SimpleNamespace(type="UI", x=760, y=0, width=240, height=770)
        region_tools = SimpleNamespace(type="TOOLS", x=0, y=0, width=50, height=770)
        area = SimpleNamespace(type="VIEW_3D", regions=[region_window, region_header, region_ui, region_tools])

        self.assertTrue(is_mouse_over_viewport_chrome(area, mouse_x=800, mouse_y=400))  # Nパネル上
        self.assertTrue(is_mouse_over_viewport_chrome(area, mouse_x=500, mouse_y=785))  # ヘッダー上
        self.assertTrue(is_mouse_over_viewport_chrome(area, mouse_x=20, mouse_y=400))   # ツールバー上
        self.assertFalse(is_mouse_over_viewport_chrome(area, mouse_x=400, mouse_y=400)) # キャンバス内


if __name__ == "__main__":
    unittest.main()
