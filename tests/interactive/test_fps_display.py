import unittest
import time
import sys
from pathlib import Path

# pythonディレクトリをsys.pathに追加
repo_root = Path(__file__).resolve().parent.parent
python_dir = repo_root / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))

from taremin_cloth.operators import FPSCounter
from taremin_cloth.utils import drawing


class TestFPSCounter(unittest.TestCase):
    """FPSCounterの計算ロジックの単体テスト"""

    def test_initial_state(self):
        counter = FPSCounter()
        self.assertEqual(counter.fps, 0.0)
        self.assertEqual(counter.frame_ms, 0.0)

    def test_first_tick(self):
        counter = FPSCounter()
        fps, ms = counter.tick(100.0)
        self.assertEqual(fps, 0.0)
        self.assertEqual(ms, 0.0)

    def test_stable_60fps_simulation(self):
        counter = FPSCounter(ema_alpha=0.2)
        counter.tick(0.0)
        dt = 1.0 / 60.0  # 約 16.666 ms
        t = 0.0
        for i in range(1, 60):
            t += dt
            fps, ms = counter.tick(t)

        self.assertAlmostEqual(counter.fps, 60.0, delta=1.0)
        self.assertAlmostEqual(counter.frame_ms, 16.67, delta=1.0)

    def test_zero_or_negative_dt_resilience(self):
        counter = FPSCounter()
        counter.tick(1.0)
        counter.tick(1.0 + 1.0 / 30.0)
        fps_before = counter.fps

        # 同時刻 (dt=0) のtickでゼロ除算エラーが発生せず、前回の値が維持されること
        counter.tick(1.0 + 1.0 / 30.0)
        self.assertEqual(counter.fps, fps_before)

    def test_reset(self):
        counter = FPSCounter()
        counter.tick(0.0)
        counter.tick(0.05)
        self.assertGreater(counter.fps, 0.0)
        counter.reset()
        self.assertEqual(counter.fps, 0.0)
        self.assertEqual(counter.frame_ms, 0.0)

    def test_tick_default_time(self):
        # operators.py 内で import された time.perf_counter() を使用した tick() の動作テスト
        counter = FPSCounter()
        counter.tick()
        time.sleep(0.01)
        fps, ms = counter.tick()
        self.assertGreater(fps, 0.0)
        self.assertGreater(ms, 0.0)


class TestDrawingFPSState(unittest.TestCase):
    """drawingモジュールのFPS情報管理の単体テスト"""

    def test_set_and_clear_fps_info(self):
        drawing.clear_interactive_fps_info()
        self.assertIsNone(drawing.get_interactive_fps_info())

        drawing.set_interactive_fps_info(59.5, 16.8, show_overlay=True, position='TOP_CENTER')
        info = drawing.get_interactive_fps_info()
        self.assertIsNotNone(info)
        self.assertAlmostEqual(info["fps"], 59.5)
        self.assertAlmostEqual(info["frame_ms"], 16.8)
        self.assertTrue(info["show_overlay"])
        self.assertEqual(info["position"], 'TOP_CENTER')

        drawing.clear_interactive_fps_info()
        self.assertIsNone(drawing.get_interactive_fps_info())


if __name__ == "__main__":
    unittest.main()
