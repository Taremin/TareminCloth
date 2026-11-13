"""2Dパターンエディタの下絵参照画像 (ReferenceImage) に関する単体・結合テスト。"""

from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PYTHON_DIR = os.path.join(_ROOT, "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

from taremin_cloth.pattern.cards import format_selection_card_lines
from taremin_cloth.pattern.document import Document, ReferenceImage
from taremin_cloth.pattern.editor import EditorController, EditorEvent
from taremin_cloth.pattern.layout import Rect2D, compute_hud_layout
from taremin_cloth.pattern.renderer import DrawCommandKind, build_render_commands
from taremin_cloth.pattern.session import ViewTransform2D


def _rect_center(rect: Rect2D) -> tuple[float, float]:
    return (rect.x + rect.w * 0.5, rect.y + rect.h * 0.5)


class TestReferenceImageRendererAndCards(unittest.TestCase):
    """下絵の描画コマンド順序・クランプ・非選択時情報カード生成の検証。"""

    def test_image_command_drawn_before_grid_and_border_after_grid(self):
        doc = Document()
        doc.reference_image = ReferenceImage(
            image_path="ref.png",
            offset=(0.1, 0.2),
            width=1.2,
            aspect_ratio=0.8,
            opacity=0.6,
            visible=True,
        )
        ctrl = EditorController(
            doc=doc,
            view=ViewTransform2D(zoom=200.0, pan_x=500.0, pan_y=400.0),
        )

        cmds = build_render_commands(doc, ctrl, screen_size=(1000, 800))
        image_indices = [i for i, c in enumerate(cmds) if c.kind == DrawCommandKind.IMAGE]
        grid_line_indices = [i for i, c in enumerate(cmds) if c.kind == DrawCommandKind.LINE]

        self.assertEqual(len(image_indices), 1)
        self.assertTrue(len(grid_line_indices) > 0)
        # 下絵はグリッド線より先（背面）に出力されること
        self.assertLess(image_indices[0], grid_line_indices[0])

        img_cmd = cmds[image_indices[0]]
        self.assertEqual(img_cmd.image_path, "ref.png")
        self.assertAlmostEqual(img_cmd.color[3], 0.6)
        self.assertEqual(len(img_cmd.points), 4)

        # 下絵枠線のポリラインはグリッド線より後に出力されること
        after_grid_polylines = [
            i for i, c in enumerate(cmds)
            if c.kind == DrawCommandKind.POLYLINE and i > grid_line_indices[-1]
        ]
        self.assertTrue(len(after_grid_polylines) > 0)

    def test_hidden_or_empty_reference_emits_no_image_command(self):
        doc = Document()
        doc.reference_image = ReferenceImage(image_path="ref.png", visible=False)
        ctrl = EditorController(doc=doc)
        cmds_hidden = build_render_commands(doc, ctrl, screen_size=(800, 600))
        self.assertFalse(any(c.kind == DrawCommandKind.IMAGE for c in cmds_hidden))

        doc.reference_image = ReferenceImage(image_path="", visible=True)
        cmds_empty = build_render_commands(doc, ctrl, screen_size=(800, 600))
        self.assertFalse(any(c.kind == DrawCommandKind.IMAGE for c in cmds_empty))

    def test_opacity_and_aspect_ratio_clamped_safely(self):
        doc = Document()
        doc.reference_image = ReferenceImage(
            image_path="ref.png",
            width=1.0,
            aspect_ratio=-2.0,
            opacity=2.5,
            visible=True,
        )
        ctrl = EditorController(
            doc=doc,
            view=ViewTransform2D(zoom=100.0, pan_x=400.0, pan_y=300.0),
        )
        cmds = build_render_commands(doc, ctrl, screen_size=(800, 600))
        img_cmds = [c for c in cmds if c.kind == DrawCommandKind.IMAGE]
        self.assertEqual(len(img_cmds), 1)
        self.assertLessEqual(img_cmds[0].color[3], 1.0)
        self.assertGreaterEqual(img_cmds[0].color[3], 0.0)
        self.assertEqual(len(img_cmds[0].points), 4)
        xs = [p[0] for p in img_cmds[0].points]
        ys = [p[1] for p in img_cmds[0].points]
        self.assertGreater(max(xs) - min(xs), 0.0)
        self.assertGreater(max(ys) - min(ys), 0.0)

    def test_unselected_info_card_displays_reference_controls(self):
        doc = Document()
        doc.reference_image = ReferenceImage(
            image_path="dir/front_bodice.png",
            offset=(0.12, -0.05),
            width=1.5,
            opacity=0.4,
            visible=True,
        )
        ctrl = EditorController(doc=doc)
        lines = format_selection_card_lines(doc, ctrl.selection)
        self.assertTrue(any("Ref Image" in line or "下絵" in line for line in lines))
        self.assertTrue(any("Visible" in line or "表示" in line for line in lines))
        self.assertTrue(any("Opacity" in line or "不透明度" in line for line in lines))
        self.assertTrue(any("Offset X" in line or "オフセット X" in line for line in lines))
        self.assertTrue(any("Offset Y" in line or "オフセット Y" in line for line in lines))
        self.assertTrue(any("Clear Image" in line or "画像をクリア" in line for line in lines))


class TestReferenceImageEditorWorkflow(unittest.TestCase):
    """EditorController 上での下絵操作・スライダー・数値編集・Undo/Redo 検証。"""

    def _make_controller(self) -> EditorController:
        doc = Document()
        doc.reference_image = ReferenceImage(
            image_path="shirt_sketch.png",
            offset=(0.0, 0.0),
            width=1.0,
            aspect_ratio=1.0,
            opacity=0.5,
            visible=True,
        )
        ctrl = EditorController(
            doc=doc,
            view=ViewTransform2D(zoom=200.0, pan_x=500.0, pan_y=400.0),
        )
        ctrl.screen_size = (1200, 900)
        return ctrl

    def test_set_toggle_clear_and_undo_redo(self):
        ctrl = EditorController(doc=Document())
        ref = ReferenceImage(image_path="pattern.png", opacity=0.35, width=1.2)
        self.assertTrue(ctrl.set_reference_image(ref))
        self.assertIsNotNone(ctrl.doc.reference_image)

        self.assertTrue(ctrl.toggle_reference_image_visible())
        self.assertFalse(ctrl.doc.reference_image.visible)

        self.assertTrue(ctrl.clear_reference_image())
        self.assertIsNone(ctrl.doc.reference_image)

        # Undoで復元
        self.assertTrue(ctrl.undo())
        self.assertIsNotNone(ctrl.doc.reference_image)
        self.assertFalse(ctrl.doc.reference_image.visible)

        self.assertTrue(ctrl.undo())
        self.assertTrue(ctrl.doc.reference_image.visible)

        self.assertTrue(ctrl.undo())
        self.assertIsNone(ctrl.doc.reference_image)

        # Redoで再適用
        self.assertTrue(ctrl.redo())
        self.assertIsNotNone(ctrl.doc.reference_image)

    def test_hud_card_toggle_and_clear_clicks(self):
        ctrl = self._make_controller()
        hud = compute_hud_layout(1200, 900)

        vis_rect = hud.card_fields["ref_visible"]
        vx, vy = _rect_center(vis_rect)
        consumed = ctrl.handle_event(EditorEvent(screen_pos=(vx, vy), is_press=True, button="LEFT"))
        self.assertTrue(consumed)
        self.assertFalse(ctrl.doc.reference_image.visible)

        clear_rect = hud.card_fields["ref_clear"]
        cx, cy = _rect_center(clear_rect)
        consumed_clear = ctrl.handle_event(EditorEvent(screen_pos=(cx, cy), is_press=True, button="LEFT"))
        self.assertTrue(consumed_clear)
        self.assertIsNone(ctrl.doc.reference_image)

    def test_hud_opacity_slider_drag_and_undo(self):
        ctrl = self._make_controller()
        hud = compute_hud_layout(1200, 900)

        op_rect = hud.card_fields["ref_opacity"]
        mid_y = op_rect.y + op_rect.h * 0.5
        start_x = op_rect.x + op_rect.w * 0.5
        end_x = op_rect.x + op_rect.w * 0.9

        ctrl.handle_event(EditorEvent(screen_pos=(start_x, mid_y), is_press=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(end_x, mid_y), is_move=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(end_x, mid_y), is_release=True, button="LEFT"))

        self.assertGreater(ctrl.doc.reference_image.opacity, 0.7)
        self.assertTrue(ctrl.undo())
        self.assertAlmostEqual(ctrl.doc.reference_image.opacity, 0.5)

    def test_hud_inline_numeric_edit_width_and_offsets(self):
        ctrl = self._make_controller()
        hud = compute_hud_layout(1200, 900)

        # Width フィールドをクリックしてインライン編集 ("180cm" -> 1.8m)
        wx, wy = _rect_center(hud.card_fields["ref_width"])
        ctrl.handle_event(EditorEvent(screen_pos=(wx, wy), is_press=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(wx, wy), is_release=True, button="LEFT"))
        self.assertEqual(ctrl.editing_field, "ref_width")
        ctrl.editing_buffer = "180cm"
        ctrl.handle_event(EditorEvent(screen_pos=(wx, wy), is_press=True, key="ENTER"))
        self.assertIsNone(ctrl.editing_field)
        self.assertAlmostEqual(ctrl.doc.reference_image.width, 1.8)

        # Offset X フィールドをクリックしてインライン編集 ("25cm" -> 0.25m)
        ox_x, ox_y = _rect_center(hud.card_fields["ref_offset_x"])
        ctrl.handle_event(EditorEvent(screen_pos=(ox_x, ox_y), is_press=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(ox_x, ox_y), is_release=True, button="LEFT"))
        self.assertEqual(ctrl.editing_field, "ref_offset_x")
        ctrl.editing_buffer = "25cm"
        ctrl.handle_event(EditorEvent(screen_pos=(ox_x, ox_y), is_press=True, key="ENTER"))
        self.assertAlmostEqual(ctrl.doc.reference_image.offset[0], 0.25)

        # Undoで元の値へ戻ること
        self.assertTrue(ctrl.undo())
        self.assertAlmostEqual(ctrl.doc.reference_image.offset[0], 0.0)


if __name__ == "__main__":
    unittest.main()
