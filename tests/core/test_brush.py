"""範囲グラブブラシ純粋ロジックの単体テスト (Blender非依存)"""
import os
import sys
import unittest

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from taremin_cloth.brush.math import (
    falloff_weights,
    verts_in_brush,
    depth_keep_mask,
    radial_adjust,
    select_grab_pins,
)


class TestBrushFalloff(unittest.TestCase):
    def test_boundaries(self):
        for shape in ('SMOOTH', 'SPHERE', 'SHARP', 'LINEAR', 'CONSTANT'):
            w = falloff_weights([0.0, 0.05, 0.1, 0.2], 0.1, shape)
            self.assertAlmostEqual(float(w[0]), 1.0, msg=shape)
            self.assertAlmostEqual(float(w[3]), 0.0, msg=shape)
        # 境界ちょうど (d == r): CONSTANTのみ1.0、他は0.0
        for shape in ('SMOOTH', 'SPHERE', 'SHARP', 'LINEAR'):
            w = falloff_weights([0.1], 0.1, shape)
            self.assertAlmostEqual(float(w[0]), 0.0, msg=shape)
        w = falloff_weights([0.1], 0.1, 'CONSTANT')
        self.assertAlmostEqual(float(w[0]), 1.0)

    def test_monotonic_and_range(self):
        d = np.linspace(0.0, 0.099, 20)
        for shape in ('SMOOTH', 'SPHERE', 'SHARP', 'LINEAR'):
            w = falloff_weights(d, 0.1, shape)
            self.assertTrue(bool((w >= 0.0).all() and (w <= 1.0).all()), shape)
            self.assertTrue(bool((np.diff(w) <= 1e-6).all()), shape)

    def test_smooth_mid_value(self):
        # smoothstep(0.5) = 0.5
        w = falloff_weights([0.05], 0.1, 'SMOOTH')
        self.assertAlmostEqual(float(w[0]), 0.5, places=5)

    def test_constant(self):
        w = falloff_weights([0.0, 0.09, 0.1, 0.11], 0.1, 'CONSTANT')
        self.assertEqual([float(x) for x in w], [1.0, 1.0, 1.0, 0.0])

class TestVertsInBrush(unittest.TestCase):
    def test_square(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        found, w = verts_in_brush(pos, np.array([0.0, 0.0, 0.0]), 0.6)
        self.assertEqual(found.tolist(), [0])
        self.assertAlmostEqual(float(w[0]), 1.0)
        found2, _ = verts_in_brush(pos, np.array([5.0, 5.0, 5.0]), 0.1)
        self.assertEqual(len(found2), 0)


class TestDepthMask(unittest.TestCase):
    def test_behind_removed(self):
        # カメラ(0,0,5)から-z方向。z=1は手前(残す)、z=-1は命中面の奥(除く)
        pts = np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32)
        origin = np.array([0, 0, 5], dtype=np.float32)
        direction = np.array([0, 0, -1], dtype=np.float32)
        mask = depth_keep_mask(pts, origin, direction, 5.0, 0.1)
        self.assertEqual(mask.tolist(), [True, True, False])

    def test_no_hit_keeps_all(self):
        pts = np.array([[0, 0, 0], [0, 0, 9]], dtype=np.float32)
        mask = depth_keep_mask(pts, np.zeros(3), np.array([0, 0, -1]), None, 0.1)
        self.assertEqual(mask.tolist(), [True, True])


class TestSelectGrabPins(unittest.TestCase):
    def test_threshold_cuts_wall(self):
        # 裾野 (w<0.01) はピン留めしない = 運動学的壁を作らない
        idx, w = select_grab_pins([0, 1, 2, 3], [1.0, 0.6, 0.05, 0.005])
        self.assertEqual(idx, [0, 1, 2])
        self.assertAlmostEqual(w[0], 1.0)
        self.assertAlmostEqual(w[2], 0.05)

    def test_weight_passthrough(self):
        # 減衰値をそのままウェイトに (中心固定・裾野ソフト)
        idx, w = select_grab_pins([5], [0.7])
        self.assertEqual(idx, [5])
        self.assertAlmostEqual(w[0], 0.7)

    def test_empty(self):
        self.assertEqual(select_grab_pins([], []), ([], []))


class TestBrushToolInterface(unittest.TestCase):
    """全ブラシツールのフック実引数とモーダル呼び出し側の一致検証 (Eキー不発の再発防止)"""

    def test_hook_arities(self):
        import inspect
        from inspect import Parameter
        from taremin_cloth import brush as brush_pkg

        calls = {
            'on_press': 1,
            'on_move': 1,
            'on_release': 2,  # (ctx, pinned_verts)
            'on_hover': 1,
            'on_activate': 1,
            'on_deactivate': 2,  # (ctx, pinned_verts)
        }
        self.assertIn('GRAB', brush_pkg.TOOLS)
        self.assertIn('RANGE_GRAB', brush_pkg.TOOLS)
        for name, cls in brush_pkg.TOOLS.items():
            inst = cls()
            for hook, nargs in calls.items():
                fn = getattr(inst, hook, None)
                self.assertIsNotNone(fn, f'{name}.{hook} missing')
                #束縛メソッドのため self は除外して数える
                params = [p for p in inspect.signature(fn).parameters.values()
                          if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)]
                self.assertEqual(len(params), nargs, f'{name}.{hook} arity')

    def test_active_tool_fallback(self):
        from taremin_cloth.brush import active_tool_name

        class FakeBrush:
            tool_mode = 'RANGE_GRAB'

        self.assertEqual(active_tool_name(FakeBrush()), 'RANGE_GRAB')
        self.assertEqual(active_tool_name(None), 'GRAB')

        class FakeUnknown:
            tool_mode = 'NO_SUCH_BRUSH'

        self.assertEqual(active_tool_name(FakeUnknown()), 'GRAB')


class TestRadialAdjust(unittest.TestCase):
    def test_up_down_clamp(self):
        self.assertAlmostEqual(radial_adjust(0.05, 200.0, 0.005, 0.5), 0.1)
        self.assertAlmostEqual(radial_adjust(0.05, -100.0, 0.005, 0.5), 0.025)
        self.assertAlmostEqual(radial_adjust(0.5, 10000.0, 0.005, 0.5), 0.5)
        self.assertAlmostEqual(radial_adjust(0.005, -10000.0, 0.005, 0.5), 0.005)


class TestBrushOverlayState(unittest.TestCase):
    def setUp(self):
        from taremin_cloth.utils import drawing

        drawing.clear_brush_info()
        drawing.clear_interactive_fps_info()

    def tearDown(self):
        from taremin_cloth.utils import drawing

        drawing.clear_brush_info()
        drawing.clear_interactive_fps_info()

    def test_brush_info_set_clear(self):
        from taremin_cloth.utils import drawing

        self.assertIsNone(drawing.get_brush_info())
        drawing.set_brush_info(100.0, 200.0, 40.0)
        info = drawing.get_brush_info()
        self.assertAlmostEqual(info["x"], 100.0)
        self.assertAlmostEqual(info["radius_px"], 40.0)
        drawing.set_brush_info(0.0, 0.0, 0.5)
        self.assertGreaterEqual(drawing.get_brush_info()["radius_px"], 2.0)
        drawing.clear_brush_info()
        self.assertIsNone(drawing.get_brush_info())

    def test_fps_info_tool_text(self):
        from taremin_cloth.utils import drawing

        drawing.set_interactive_fps_info(60.0, 16.6, tool_text="Tool: Range Grab")
        self.assertEqual(drawing.get_interactive_fps_info()["tool_text"], "Tool: Range Grab")
        drawing.set_interactive_fps_info(60.0, 16.6)
        self.assertIsNone(drawing.get_interactive_fps_info()["tool_text"])


if __name__ == "__main__":
    unittest.main()
