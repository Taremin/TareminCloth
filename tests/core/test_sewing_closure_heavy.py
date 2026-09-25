"""
実衣装データでの縫合閉鎖テスト (heavy_test.blend 由来フィクスチャ)
身体コライダー上で縫合が開いたままにならず、自己衝突OFFでも全閉鎖することを検証する。
"""

import os
import unittest
import numpy as np
import taremin_cloth_core

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sewing_heavy")


def load_fixture():
    cloth = np.load(os.path.join(FIXTURE_DIR, "cloth.npz"))
    body = np.load(os.path.join(FIXTURE_DIR, "body.npz"))["triangles"]
    return cloth, body


class TestSewingClosureHeavy(unittest.TestCase):
    """身体上で縫合が膠着せず閉鎖すること (OFF条件)"""

    def _require_physical_gpu(self):
        """11k頂点×10kコライダーのため物理GPU必須。ソフトウェアGPUではスキップ。
        (WARP実測で60フレームが制限時間内に終わらないため)"""
        try:
            name = taremin_cloth_core.get_gpu_device_name().lower()
        except Exception:
            return
        if any(k in name for k in ("basic render", "warp", "llvmpipe", "lavapipe", "software", "cpu")):
            self.skipTest(f"物理GPU必須のためスキップ (device={name})")

    def test_closure_without_self_collision(self):
        self._require_physical_gpu()
        cloth, body = load_fixture()
        pos, edges, faces = cloth["positions"], cloth["edges"], cloth["faces"]
        sew, inv = cloth["sewing"], cloth["inv_masses"]
        sim = taremin_cloth_core.ClothSimulator(
            positions=pos, edges=edges, faces=faces, inv_masses=inv,
            sewing_springs=sew, stiffness=1000.0, compression_stiffness=50.0,
            shear_stiffness=50.0, bending_stiffness=5.0,
            sewing_shrink_speed=0.2, sewing_stiffness=10000.0, areal_density=0.15)
        sim.set_mesh_collider_triangles(body, friction=0.5, thickness=0.005,
                                        restitution=0.0, single_sided=True)
        sim.set_gravity(0, 0, 0)
        sim.set_enable_self_collision(False)
        sim.set_damping(1.0)
        sim.set_damping_all(5.0, 5.0, 5.0, 0.5)
        out = np.empty(len(pos) * 3, dtype=np.float32)
        for _ in range(60):
            sim.step(dt=1.0 / 24.0, substeps=20, solver_iterations=2)
            sim.get_positions(out)
        gaps = np.linalg.norm(out.reshape(-1, 3)[sew[:, 0]] - out.reshape(-1, 3)[sew[:, 1]], axis=1)
        print(f"\n[Heavy Sewing Closure OFF] max_gap={gaps.max()*1000:.2f}mm "
              f"open={(gaps > 0.005).sum()}/{len(gaps)}")
        self.assertLess(float(gaps.max()), 0.001,
                        f"OFFでも全縫合が1mm未満に閉鎖すること (実測max={gaps.max()*1000:.2f}mm)")
        self.assertEqual(int((gaps <= 0.005).sum()), len(gaps),
                         "OFFでも結合率100% (5mm判定) に到達すること")


if __name__ == "__main__":
    unittest.main()
