"""Active-Region Sculpt Mode (部分アクティブ・スカルプト) の単体テスト。"""

import unittest
import numpy as np

try:
    import taremin_cloth_core as core
    HAS_CORE = True
except ImportError:
    try:
        from taremin_cloth import taremin_cloth_core as core
        HAS_CORE = True
    except ImportError:
        HAS_CORE = False

from taremin_cloth.brush.base import ActiveRegionManager


@unittest.skipUnless(HAS_CORE, "taremin_cloth_core not available")
class TestActiveRegion(unittest.TestCase):
    def test_active_region_freeze_and_release(self):
        # 10x10 グリッド布 (100頂点, 0.0〜0.9m)
        positions = []
        for y in range(10):
            for x in range(10):
                positions.append([x * 0.1, y * 0.1, 0.0])
        positions = np.array(positions, dtype=np.float32)

        edges = []
        for y in range(10):
            for x in range(10):
                idx = y * 10 + x
                if x + 1 < 10:
                    edges.append([idx, idx + 1])
                if y + 1 < 10:
                    edges.append([idx, idx + 10])
        edges = np.array(edges, dtype=np.uint32)
        faces = np.zeros((0, 3), dtype=np.uint32)
        inv_masses = np.ones(len(positions), dtype=np.float32)

        sim = core.ClothSimulator(positions, edges, faces, inv_masses)

        mgr = ActiveRegionManager()
        self.assertFalse(mgr.is_active)

        # 中心 (0.45, 0.45, 0.0), 半径 0.1, scale 2.0 -> active radius 0.2m
        center = [0.45, 0.45, 0.0]
        radius = 0.1
        mgr.freeze_inactive_region(sim, positions, center, radius, active_scale=2.0)

        self.assertTrue(mgr.is_active)
        frozen_count = len(mgr._temp_frozen_indices)
        self.assertGreater(frozen_count, 50, "Majority of vertices far from center should be frozen")
        self.assertLess(frozen_count, 100, "Center vertices must remain active")

        # 中心近傍 (x=4, y=4 -> 44) は凍結されていないこと
        center_idx = 4 * 10 + 4
        self.assertNotIn(center_idx, mgr._temp_frozen_indices)

        # 四隅 (0, 9, 90, 99) は確実に凍結されていること
        self.assertIn(0, mgr._temp_frozen_indices)
        self.assertIn(9, mgr._temp_frozen_indices)
        self.assertIn(90, mgr._temp_frozen_indices)
        self.assertIn(99, mgr._temp_frozen_indices)

        # 全ピン解放
        mgr.release_temp_pins(sim)
        self.assertFalse(mgr.is_active)
        self.assertEqual(len(mgr._temp_frozen_indices), 0)


if __name__ == "__main__":
    unittest.main()
