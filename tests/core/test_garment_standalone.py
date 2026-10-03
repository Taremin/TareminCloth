# -*- coding: utf-8 -*-
"""実寸Tシャツ型紙 + SiroinoSotai素体の独立再現テスト (Blender不要).

型紙はすべて平面パネル: 前後身頃 (衿ぐり・肩傾斜・アームホール整形) と
袖の前後パネル (天側・地側の縫合で筒化)。立体化は縫合とドレープに任せる。
"""
import os
import sys
import unittest

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

from taremin_cloth.utils.fixture_io import audit_seams, load_scene_manifest

MANIFEST = os.path.join(project_root, "tests", "fixtures", "garments",
                        "tshirt", "tshirt_scene.json")


def run_drape(m, steps=150):
    import taremin_cloth_core as core
    cfg = m["config"]
    sim = core.ClothSimulator(
        m["positions"], m["edges"], m["faces"],
        sewing_springs=m["sewing_springs"],
        sewing_shrink_speed=float(cfg.get("sewing_shrink_speed", 0.8)),
        stiffness=float(cfg.get("tension_stiffness", 10000.0)),
        bending_stiffness=float(cfg.get("bending_stiffness", 20.0)))
    sim.set_sewing_priority_options(True, 0.9, 0.005, 3, 600)
    cp, cf = m["collider_positions"], m["collider_faces"]
    tris = np.stack([cp[cf[:, 0]], cp[cf[:, 1]], cp[cf[:, 2]]], axis=1).astype(np.float32)
    sim.set_mesh_collider_triangles(
        tris, float(m.get("collider_friction", 0.8)),
        float(m.get("collider_thickness", 0.008)), 0.0,
        bool(m.get("collider_single_sided", False)))
    coords = m["positions"].copy()
    for _ in range(steps):
        sim.update_sewing_priority(np.ascontiguousarray(coords))
        sim.step(1.0 / 60.0, 10)
        out = np.zeros(len(coords) * 3, dtype=np.float32)
        sim.get_positions(out)
        coords = out.reshape(-1, 3).copy()
    return coords


class TestTShirtOnSiroino(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import taremin_cloth_core  # noqa
        except ImportError:
            raise unittest.SkipTest("taremin_cloth_core が利用できないためスキップします")
        cls.m = load_scene_manifest(MANIFEST)

    def test_manifest_resolves(self):
        m = self.m
        self.assertEqual(m["positions"].shape, (3084, 3))
        self.assertEqual(m["faces"].shape, (5736, 3))
        self.assertEqual(m["sewing_springs"].shape[1], 2)
        self.assertEqual(m["collider_positions"].shape[0], 1790)

    def test_seam_lengths_match(self):
        rep = audit_seams(self.m["positions"], self.m["faces"], self.m["sewing_springs"])
        self.assertEqual(rep["num_pairs"], 156)
        for c in rep["chains"]:
            self.assertLessEqual(c["diff_m"], 0.005)

    def test_sewing_closure_no_gravity(self):
        import taremin_cloth_core as core
        m = self.m
        sim = core.ClothSimulator(
            m["positions"], m["edges"], m["faces"],
            sewing_springs=m["sewing_springs"], sewing_shrink_speed=0.8)
        sim.set_gravity(0.0, 0.0, 0.0)
        for _ in range(60):
            sim.step(1.0 / 60.0, 10)
        out = np.zeros(len(m["positions"]) * 3, dtype=np.float32)
        sim.get_positions(out)
        p = out.reshape(-1, 3)
        self.assertTrue(np.all(np.isfinite(p)))
        for a, b in m["sewing_springs"]:
            self.assertLess(float(np.linalg.norm(p[a] - p[b])), 0.03)

    def test_drape_on_shoulders_with_sleeves(self):
        m = self.m
        coords = run_drape(m)
        self.assertTrue(np.all(np.isfinite(coords)))
        # 肩掛かり: 上端は肩線(約1.04m)近傍に留まること
        self.assertGreater(float(coords[:, 2].max()), 1.00)
        # 裾は腰上に留まること (ずり落ち防止)
        self.assertGreater(float(coords[:, 2].min()), 0.30)
        # 袖は腕軸周りを全周覆うこと (筒化の成立: 最大角度間隙で判定)
        init_x = m["positions"][:, 0]
        for xa, xb in ((0.14, 0.22), (-0.22, -0.14)):
            idx = np.where((init_x >= xa) & (init_x <= xb))[0]
            self.assertGreater(len(idx), 20)
            pts = coords[idx]
            ang = np.sort(np.arctan2(pts[:, 2] - 1.015, pts[:, 1] - 0.005))
            gaps = np.diff(np.concatenate([ang, ang[:1] + 2.0 * np.pi]))
            self.assertLess(float(gaps.max()), np.deg2rad(100.0))


if __name__ == "__main__":
    unittest.main()
