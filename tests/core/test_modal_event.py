"""
モーダル操作イベント判定ヘルパーの単体テスト (test_modal_event.py)
Blender非起動で実行できる (bpy非依存)。
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

# プロジェクトルートおよび python/ ディレクトリをモジュール検索パスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from taremin_cloth.utils.modal_event import (
    CLICK_DRAG_THRESHOLD_PX,
    PressDragTracker,
    event_pos,
    is_cancel,
    is_confirm,
    is_left_press,
    is_left_release,
)


def _event(typ=None, val=None, **coords):
    ev = MagicMock()
    ev.type = typ
    ev.value = val
    for key in ("mouse_x", "mouse_y", "mouse_region_x", "mouse_region_y"):
        setattr(ev, key, coords.get(key))
    return ev


class TestModalEventPredicates(unittest.TestCase):
    """イベント種別判定の検証"""

    def test_left_press_release(self):
        self.assertTrue(is_left_press(_event('LEFTMOUSE', 'PRESS', mouse_x=1, mouse_y=2)))
        self.assertFalse(is_left_press(_event('LEFTMOUSE', 'RELEASE', mouse_x=1, mouse_y=2)))
        self.assertTrue(is_left_release(_event('LEFTMOUSE', 'RELEASE', mouse_x=1, mouse_y=2)))
        self.assertFalse(is_left_release(_event('LEFTMOUSE', 'PRESS', mouse_x=1, mouse_y=2)))

    def test_confirm_cancel(self):
        self.assertTrue(is_confirm(_event('LEFTMOUSE', 'PRESS')))
        self.assertTrue(is_confirm(_event('RET', 'PRESS')))
        self.assertTrue(is_confirm(_event('NUMPAD_ENTER', 'PRESS')))
        self.assertFalse(is_confirm(_event('RET', 'RELEASE')))
        self.assertTrue(is_cancel(_event('RIGHTMOUSE', 'PRESS')))
        self.assertTrue(is_cancel(_event('ESC', 'PRESS')))
        self.assertFalse(is_cancel(_event('ESC', 'RELEASE')))

    def test_invalid_event_never_raises(self):
        self.assertFalse(is_left_press(None))
        self.assertFalse(is_left_release(None))
        self.assertFalse(is_confirm(None))
        self.assertFalse(is_cancel(None))
        self.assertFalse(is_left_press(MagicMock()))
        self.assertIsNone(event_pos(None))

    def test_event_pos_prefers_window_coords(self):
        pos = event_pos(_event('MOUSEMOVE', None, mouse_x=10, mouse_y=20,
                               mouse_region_x=30, mouse_region_y=40))
        self.assertEqual(pos, (10.0, 20.0))

    def test_event_pos_falls_back_to_region(self):
        pos = event_pos(_event('MOUSEMOVE', None, mouse_region_x=30, mouse_region_y=40))
        self.assertEqual(pos, (30.0, 40.0))


class TestPressDragTracker(unittest.TestCase):
    """クリック / ドラッグ識別の検証"""

    def test_click_within_threshold(self):
        tracker = PressDragTracker(threshold_px=5.0)
        tracker.on_press(0.0, 0.0)
        self.assertTrue(tracker.pressed)
        self.assertFalse(tracker.on_move(2.0, 2.0))
        self.assertEqual(tracker.on_release(2.0, 2.0), "click")
        self.assertFalse(tracker.pressed)

    def test_drag_beyond_threshold(self):
        tracker = PressDragTracker(threshold_px=5.0)
        tracker.on_press(0.0, 0.0)
        self.assertTrue(tracker.on_move(10.0, 0.0))
        self.assertTrue(tracker.dragging)
        self.assertEqual(tracker.on_release(10.0, 0.0), "drag")
        self.assertFalse(tracker.pressed)

    def test_boundary_distance_is_click(self):
        tracker = PressDragTracker(threshold_px=5.0)
        tracker.on_press(0.0, 0.0)
        # 3-4-5直角三角形: 距離ちょうど5はクリック扱い (超過のみドラッグ)
        self.assertEqual(tracker.on_release(3.0, 4.0), "click")

    def test_release_without_press_returns_none(self):
        tracker = PressDragTracker()
        self.assertIsNone(tracker.on_release(0.0, 0.0))
        self.assertFalse(tracker.on_move(10.0, 10.0))

    def test_phase_guard_pattern(self):
        tracker = PressDragTracker()
        tracker.on_press_event(_event('LEFTMOUSE', 'PRESS', mouse_x=100, mouse_y=50))
        self.assertTrue(tracker.pressed)
        # フェーズ遷移直後の同一押下は指が離れるまで抑止できる
        self.assertTrue(tracker.pressed)
        tracker.on_release_event(_event('LEFTMOUSE', 'RELEASE', mouse_x=101, mouse_y=50))
        self.assertFalse(tracker.pressed)

    def test_default_threshold_constant(self):
        self.assertGreater(CLICK_DRAG_THRESHOLD_PX, 0.0)


if __name__ == "__main__":
    unittest.main()
