# -*- coding: utf-8 -*-
"""中継点（Waypoint）誘導による複数セグメント縫合エッジの単体・統合テスト."""
import unittest
import numpy as np

import taremin_cloth_core as core
from taremin_cloth.utils.mesh_extract import extract_sewing_topology, WaypointSeam


class TestWaypointSewing(unittest.TestCase):
    def setUp(self):
        # 腕コライダー (X軸沿い): Y=0, Z=0.50, 半径 0.05 (底面 Z=0.45)
        self.arm_pa = [0.05, 0.0, 0.50]
        self.arm_pb = [0.35, 0.0, 0.50]
        self.arm_radius = 0.05

    def _create_draped_sleeve_mesh(self, num_slices=3):
        """腕の上に二つ折りで跨ぐ袖ストリップメッシュを作成する."""
        xs = np.linspace(0.15, 0.25, num_slices)
        positions = []
        faces = []
        edges = []
        inv_masses = []

        for i, x in enumerate(xs):
            base = i * 3
            positions.extend([
                [x,  0.00, 0.56], # base+0: 上端 (Pin固定)
                [x, -0.06, 0.48], # base+1: 前布端 A (可動)
                [x,  0.06, 0.48], # base+2: 後布端 B (可動)
            ])
            inv_masses.extend([0.0, 1.0, 1.0])
            edges.append([base+0, base+1])
            edges.append([base+0, base+2])

            if i > 0:
                pbase = (i - 1) * 3
                faces.append([pbase+0, pbase+1, base+1])
                faces.append([pbase+0, base+1, base+0])
                edges.append([pbase+0, base+0])
                edges.append([pbase+1, base+1])
                edges.append([pbase+0, base+1])

                faces.append([pbase+0, base+2, pbase+2])
                faces.append([pbase+0, base+0, base+2])
                edges.append([pbase+2, base+2])
                edges.append([pbase+0, base+2])

        return xs, positions, faces, edges, inv_masses

    def test_direct_sewing_pinches_arm(self):
        """直線縫合スプリングは腕コライダーを挟み込み、腕の下に回り込めないことを確認."""
        xs, positions, faces, edges, inv_masses = self._create_draped_sleeve_mesh()
        sewing = []
        for i in range(len(xs)):
            base = i * 3
            sewing.append([base+1, base+2])

        pos = np.array(positions, dtype=np.float32)
        f = np.array(faces, dtype=np.uint32)
        e = np.array(edges, dtype=np.uint32)
        sew = np.array(sewing, dtype=np.uint32)
        im = np.array(inv_masses, dtype=np.float32)

        sim = core.ClothSimulator(
            pos, e, f, inv_masses=im, sewing_springs=sew,
            sewing_shrink_speed=0.6, stiffness=10000.0, bending_stiffness=20.0
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.add_capsule_collider(self.arm_pa, self.arm_pb, self.arm_radius, 0.5, 0.0)

        for _ in range(100):
            sim.step(1.0 / 60.0, 10)

        out = np.zeros(len(pos) * 3, dtype=np.float32)
        sim.get_positions(out)
        final_pos = out.reshape(-1, 3)

        # 直線スプリングだと、すべてのスライスで腕の横（Z > 0.50）に挟み込まれる
        for i in range(len(xs)):
            base = i * 3
            pA = final_pos[base+1]
            pB = final_pos[base+2]
            # 腕の底面(0.45)より下に降りられない
            self.assertGreater(float(pA[2]), 0.48, f"Slice {i} should be pinched on top/side")
            self.assertGreater(float(pB[2]), 0.48)

    def test_single_waypoint_sewing_under_arm(self):
        """中継点を腕の下に配置した場合、腕を安全に迂回して腕の下で合流・結合することを検証."""
        xs, positions, faces, edges, inv_masses = self._create_draped_sleeve_mesh()

        # Loose Edges として、各スライスの前端 A と後端 B の間に中継点 W (Y=0, Z=0.42) を挟む
        # A - W - B
        loose_edges = []
        for i, x in enumerate(xs):
            base = i * 3
            w_idx = len(positions)
            positions.append([x, 0.0, 0.42]) # 腕の底面(0.45)より下
            inv_masses.append(0.0)           # 中継点は固定アンカー
            loose_edges.append((base+1, w_idx))
            loose_edges.append((w_idx, base+2))

        pos = np.array(positions, dtype=np.float32)
        f = np.array(faces, dtype=np.uint32)
        e = np.array(edges, dtype=np.uint32)

        # トポロジー抽出のテスト
        simple, waypoints = extract_sewing_topology(loose_edges, f, pos)
        self.assertEqual(len(simple), 0)
        self.assertEqual(len(waypoints), len(xs))

        # 中継点スプリングの適用: A -> W, B -> W
        sewing = []
        for ws in waypoints:
            w_idx = ws.waypoint_indices[0]
            sewing.append([ws.vert_a, w_idx])
            sewing.append([ws.vert_b, w_idx])

        sew = np.array(sewing, dtype=np.uint32)
        im = np.array(inv_masses, dtype=np.float32)

        sim = core.ClothSimulator(
            pos, e, f, inv_masses=im, sewing_springs=sew,
            sewing_shrink_speed=0.6, stiffness=10000.0, bending_stiffness=20.0
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.add_capsule_collider(self.arm_pa, self.arm_pb, self.arm_radius, 0.5, 0.0)

        for _ in range(120):
            sim.step(1.0 / 60.0, 10)

        out = np.zeros(len(pos) * 3, dtype=np.float32)
        sim.get_positions(out)
        final_pos = out.reshape(-1, 3)

        # すべてのスライスで、A と B が腕の下（Z <= 0.45）に到達し、距離が数mm以内で合流していること
        for i in range(len(xs)):
            base = i * 3
            pA = final_pos[base+1]
            pB = final_pos[base+2]
            dist_AB = float(np.linalg.norm(pA - pB))
            self.assertLessEqual(dist_AB, 0.005, f"Slice {i} should merge at waypoint within 5mm")
            self.assertLessEqual(float(pA[2]), 0.45, f"Slice {i} A should be under arm (Z <= 0.45)")
            self.assertLessEqual(float(pB[2]), 0.45, f"Slice {i} B should be under arm (Z <= 0.45)")

    def test_multi_waypoint_sewing_under_arm(self):
        """2つの中継点 (W1, W2) を腕の下両脇に配置した場合の段階的合流検証."""
        xs, positions, faces, edges, inv_masses = self._create_draped_sleeve_mesh(num_slices=2)

        # 各スライス: A - W1 - W2 - B
        # W1 = (X, -0.03, 0.42), W2 = (X, +0.03, 0.42)
        phase1_sewing = []
        for i, x in enumerate(xs):
            base = i * 3
            w1 = len(positions)
            w2 = len(positions) + 1
            positions.append([x, -0.03, 0.42])
            positions.append([x,  0.03, 0.42])
            inv_masses.extend([0.0, 0.0])
            phase1_sewing.append([base+1, w1])
            phase1_sewing.append([base+2, w2])

        pos = np.array(positions, dtype=np.float32)
        f = np.array(faces, dtype=np.uint32)
        e = np.array(edges, dtype=np.uint32)
        sew1 = np.array(phase1_sewing, dtype=np.uint32)
        im = np.array(inv_masses, dtype=np.float32)

        sim = core.ClothSimulator(
            pos, e, f, inv_masses=im, sewing_springs=sew1,
            sewing_shrink_speed=0.8, stiffness=10000.0, bending_stiffness=20.0
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.add_capsule_collider(self.arm_pa, self.arm_pb, self.arm_radius, 0.5, 0.0)

        # フェーズ1: 腕の下に引き込む (40ステップ)
        for _ in range(40):
            sim.step(1.0 / 60.0, 10)

        out = np.zeros(len(pos) * 3, dtype=np.float32)
        sim.get_positions(out)
        mid_pos = out.reshape(-1, 3)

        # フェーズ2: A と B を直接引き合わせる (40ステップ)
        phase2_sewing = []
        for i in range(len(xs)):
            base = i * 3
            phase2_sewing.append([base+1, base+2])

        sew2 = np.array(phase2_sewing, dtype=np.uint32)
        sim2 = core.ClothSimulator(
            mid_pos, e, f, inv_masses=im, sewing_springs=sew2,
            sewing_shrink_speed=0.8, stiffness=10000.0, bending_stiffness=20.0
        )
        sim2.set_gravity(0.0, 0.0, -9.81)
        sim2.add_capsule_collider(self.arm_pa, self.arm_pb, self.arm_radius, 0.5, 0.0)

        for _ in range(40):
            sim2.step(1.0 / 60.0, 10)

        sim2.get_positions(out)
        final_pos = out.reshape(-1, 3)

        for i in range(len(xs)):
            base = i * 3
            pA = final_pos[base+1]
            pB = final_pos[base+2]
            dist_AB = float(np.linalg.norm(pA - pB))
            self.assertLessEqual(dist_AB, 0.005, f"Multi-waypoint slice {i} should merge within 5mm")
            self.assertLessEqual(float(pA[2]), 0.45, f"Multi-waypoint slice {i} should be under arm")


if __name__ == "__main__":
    unittest.main()
