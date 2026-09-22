"""縫合優先モードのスタンドアロン検証 (Blender不要)。

- 結合率ヘルパーの純粋試験 (GPU不要)
- Rustコアのラッチ・ランプ・重力抑制の結合試験
"""
import unittest

import numpy as np

from taremin_cloth.engine.sewing_priority import closure_ratio, default_merge_dist, sim_phase, phase_text


class TestClosureRatioHelper(unittest.TestCase):
    def test_empty_is_identity(self):
        pos = np.zeros((2, 3), dtype=np.float32)
        self.assertEqual(closure_ratio(pos, None, 0.005), 1.0)
        self.assertEqual(closure_ratio(pos, np.zeros((0, 2), dtype=np.uint32), 0.005), 1.0)

    def test_half_closed(self):
        pos = np.array([
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        edges = np.array([[0, 1], [2, 3]], dtype=np.uint32)
        self.assertAlmostEqual(closure_ratio(pos, edges, 0.005), 0.5)

    def test_out_of_range_skipped(self):
        pos = np.zeros((2, 3), dtype=np.float32)
        edges = np.array([[0, 99]], dtype=np.uint32)
        self.assertEqual(closure_ratio(pos, edges, 0.005), 1.0)

    def test_default_merge_dist(self):
        self.assertAlmostEqual(default_merge_dist(0.005), 0.01)
        self.assertAlmostEqual(default_merge_dist(0.0001), 0.005)


class TestPhaseHelpers(unittest.TestCase):
    def test_phase_text_format(self):
        trans = lambda s: s
        self.assertEqual(
            phase_text(True, 0.625, 0.0, trans),
            "Phase: Sewing 62% (gravity 0.00x)",
        )
        self.assertEqual(
            phase_text(False, 1.0, 1.0, trans),
            "Phase: Normal (gravity 1.00x)",
        )

    def test_sim_phase_with_fake_sim(self):
        class FakeSim:
            def is_sewing_priority_active(self):
                return True

            def get_sewing_closure_ratio(self):
                return 0.5

            def get_sewing_priority_scale(self):
                return 0.0

        self.assertEqual(sim_phase(FakeSim()), (True, 0.5, 0.0))

    def test_sim_phase_without_api_returns_none(self):
        self.assertIsNone(sim_phase(object()))


class TestSewingPriorityCore(unittest.TestCase):
    def _make_sim(self, gap=0.5):
        import taremin_cloth_core
        positions = np.array([
            [-gap / 2.0, 0.0, 1.0],
            [gap / 2.0, 0.0, 1.0],
        ], dtype=np.float32)
        edges = np.empty((0, 2), dtype=np.uint32)
        sewing = np.array([[0, 1]], dtype=np.uint32)
        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            sewing_springs=sewing,
            stiffness=10000.0,
            sewing_shrink_speed=5.0,
        )
        return sim

    def test_gravity_suppressed_then_restored(self):
        sim = self._make_sim()
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_sewing_priority_options(True, 0.9, 0.05, 2, 0)
        self.assertTrue(sim.is_sewing_priority_active())
        self.assertAlmostEqual(sim.get_sewing_priority_scale(), 0.0)

        out = np.zeros(2 * 3, dtype=np.float32)
        # 初回測定: 離れているのでスケール0
        sim.get_positions(out)
        ratio, scale, latched = sim.update_sewing_priority(out.reshape(-1, 3))
        self.assertEqual(ratio, 0.0)
        self.assertEqual(scale, 0.0)
        self.assertFalse(latched)

        z_first = None
        latched_at = None
        for i in range(60):
            sim.get_positions(out)
            sim.update_sewing_priority(out.reshape(-1, 3))
            sim.step(dt=1.0 / 60.0, substeps=10)
            sim.get_positions(out)
            pos = out.reshape((2, 3))
            if z_first is None:
                z_first = float(pos[0, 2])
            if sim.get_sewing_closure_ratio() >= 0.9 and latched_at is None:
                latched_at = i
                break

        self.assertIsNotNone(latched_at, "60フレーム以内に縫合が結合すること")
        # 抑制期間中はほとんど落下していないこと (無重力と同等)
        sim.get_positions(out)
        pos = out.reshape((2, 3))
        dist = float(np.linalg.norm(pos[0] - pos[1]))
        self.assertLess(dist, 0.05, f"縫合が閉じていること (実測: {dist})")
        # ランプ完了後は重力が効いて落下すること
        for _ in range(30):
            sim.get_positions(out)
            sim.update_sewing_priority(out.reshape(-1, 3))
            sim.step(dt=1.0 / 60.0, substeps=10)
        sim.get_positions(out)
        pos = out.reshape((2, 3))
        self.assertLess(float(pos[0, 2]), 0.9, "重力復帰後に落下すること")
        self.assertAlmostEqual(sim.get_sewing_priority_scale(), 1.0)

    def test_one_way_latch(self):
        sim = self._make_sim(gap=0.01)
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_sewing_priority_options(True, 0.9, 0.05, 0, 0)
        out = np.zeros(2 * 3, dtype=np.float32)
        sim.get_positions(out)
        ratio, scale, latched = sim.update_sewing_priority(out.reshape(-1, 3))
        self.assertEqual(ratio, 1.0)
        self.assertTrue(latched)
        self.assertAlmostEqual(scale, 1.0)
        # 再び離れてもラッチは戻らない
        far = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        ratio2, scale2, latched2 = sim.update_sewing_priority(far)
        self.assertTrue(latched2)
        self.assertAlmostEqual(scale2, 1.0)

    def test_max_frames_fallback(self):
        sim = self._make_sim()
        sim.set_gravity(0.0, 0.0, -9.81)
        # 結合判定距離を極小にして到達不能にし、上限3フレームで強制復帰
        sim.set_sewing_priority_options(True, 1.0, 0.0005, 0, 3)
        out = np.zeros(2 * 3, dtype=np.float32)
        latched = False
        for _ in range(5):
            sim.get_positions(out)
            _, _, latched = sim.update_sewing_priority(out.reshape(-1, 3))
            sim.step(dt=1.0 / 60.0, substeps=5)
            if latched:
                break
        self.assertTrue(latched, "max_framesで強制ラッチすること")
        self.assertAlmostEqual(sim.get_sewing_priority_scale(), 1.0)

    def test_disabled_identity(self):
        sim = self._make_sim()
        out = np.zeros(2 * 3, dtype=np.float32)
        sim.get_positions(out)
        ratio, scale, latched = sim.update_sewing_priority(out.reshape(-1, 3))
        self.assertEqual((ratio, scale, latched), (1.0, 1.0, True))
        self.assertFalse(sim.is_sewing_priority_active())


if __name__ == "__main__":
    unittest.main()
