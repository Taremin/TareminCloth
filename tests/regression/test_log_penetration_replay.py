"""
実ログデータ (cloth_debug_20260829_185924_Plane.json.gz) の貫通状態 (Frame 125 / Frame 135) を用いた
貫通解消不全の再現・検証テスト (TDD)
"""

import gzip
import json
import os
import sys
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import taremin_cloth_core

def check_tri_intersection(p0, p1, p2, q0, q1, q2):
    n1 = np.cross(p1 - p0, p2 - p0)
    d_q = [np.dot(n1, q0 - p0), np.dot(n1, q1 - p0), np.dot(n1, q2 - p0)]
    if (d_q[0] > 1e-5 and d_q[1] > 1e-5 and d_q[2] > 1e-5) or (d_q[0] < -1e-5 and d_q[1] < -1e-5 and d_q[2] < -1e-5):
        return False
    n2 = np.cross(q1 - q0, q2 - q0)
    d_p = [np.dot(n2, p0 - q0), np.dot(n2, p1 - q0), np.dot(n2, p2 - q0)]
    if (d_p[0] > 1e-5 and d_p[1] > 1e-5 and d_p[2] > 1e-5) or (d_p[0] < -1e-5 and d_p[1] < -1e-5 and d_p[2] < -1e-5):
        return False
    edges1 = [p1 - p0, p2 - p1, p0 - p2]
    edges2 = [q1 - q0, q2 - q1, q0 - q2]
    for e1 in edges1:
        for e2 in edges2:
            axis = np.cross(e1, e2)
            norm = np.linalg.norm(axis)
            if norm < 1e-6:
                continue
            axis /= norm
            p_proj = [np.dot(axis, p0), np.dot(axis, p1), np.dot(axis, p2)]
            q_proj = [np.dot(axis, q0), np.dot(axis, q1), np.dot(axis, q2)]
            if min(p_proj) > max(q_proj) + 1e-5 or min(q_proj) > max(p_proj) + 1e-5:
                return False
    return True

def count_inter_cloth_intersections(pos, faces_lower, faces_upper):
    tris1 = pos[faces_lower]
    tris2 = pos[faces_upper]
    min1 = np.min(tris1, axis=1)
    max1 = np.max(tris1, axis=1)
    min2 = np.min(tris2, axis=1)
    max2 = np.max(tris2, axis=1)
    grid = {}
    cell_size = 0.5
    for idx, (bmin, bmax) in enumerate(zip(min1, max1)):
        cmin = np.floor(bmin / cell_size).astype(int)
        cmax = np.floor(bmax / cell_size).astype(int)
        for cx in range(cmin[0], cmax[0] + 1):
            for cy in range(cmin[1], cmax[1] + 1):
                for cz in range(cmin[2], cmax[2] + 1):
                    key = (cx, cy, cz)
                    if key not in grid:
                        grid[key] = []
                    grid[key].append(idx)
    cnt = 0
    tested = set()
    for idx2, (bmin2, bmax2) in enumerate(zip(min2, max2)):
        cmin = np.floor(bmin2 / cell_size).astype(int)
        cmax = np.floor(bmax2 / cell_size).astype(int)
        nearby = set()
        for cx in range(cmin[0], cmax[0] + 1):
            for cy in range(cmin[1], cmax[1] + 1):
                for cz in range(cmin[2], cmax[2] + 1):
                    key = (cx, cy, cz)
                    if key in grid:
                        nearby.update(grid[key])
        for idx1 in nearby:
            pair = (idx1, idx2)
            if pair in tested:
                continue
            tested.add(pair)
            if np.any(min1[idx1] > max2[idx2]) or np.any(min2[idx2] > max1[idx1]):
                continue
            f1 = faces_lower[idx1]
            f2 = faces_upper[idx2]
            p0, p1, p2 = pos[f1[0]], pos[f1[1]], pos[f1[2]]
            q0, q1, q2 = pos[f2[0]], pos[f2[1]], pos[f2[2]]
            if check_tri_intersection(p0, p1, p2, q0, q1, q2):
                cnt += 1
    return cnt


class TestLogPenetrationState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log_path = os.path.join(project_root, "frame_logs", "cloth_debug_20260829_185924_Plane.json.gz")
        if not os.path.exists(cls.log_path):
            raise unittest.SkipTest(f"Log file not found: {cls.log_path}")
        
        with gzip.open(cls.log_path, "rt", encoding="utf-8") as f:
            cls.log_data = json.load(f)
        
        cls.meta = cls.log_data["metadata"]
        cls.edges = np.array(cls.meta["edges"], dtype=np.uint32)
        cls.faces = np.array(cls.meta["faces"], dtype=np.uint32)
        cls.num_vertices = cls.meta["num_vertices"]
        
        f0_pos = np.array(cls.log_data["frames"][0]["positions"])
        cls.group_lower = set(np.where(f0_pos[:, 2] < 0)[0])
        cls.group_upper = set(np.where(f0_pos[:, 2] > 0)[0])
        
        cls.faces_lower = cls.faces[[i for i, f in enumerate(cls.faces) if f[0] in cls.group_lower]]
        cls.faces_upper = cls.faces[[i for i, f in enumerate(cls.faces) if f[0] in cls.group_upper]]

    def test_penetration_persists_from_frame135(self):
        """Frame 135 (貫通発生状態) からシミュレーションを継続した際、現行コードでは貫通が固定化・解消されないことを検証"""
        # Frame 135 (322面の交差が発生している状態) の座標をセット
        f135_pos = np.array(self.log_data["frames"][135]["positions"], dtype=np.float32)
        init_inter = count_inter_cloth_intersections(f135_pos, self.faces_lower, self.faces_upper)
        print(f"Frame 135 初期交差面数: {init_inter}")
        self.assertGreater(init_inter, 50, "Frame 135 には多数の貫通が存在するはず")
        
        sim = taremin_cloth_core.ClothSimulator(
            positions=f135_pos,
            edges=self.edges,
            faces=self.faces,
            thickness=self.meta["thickness"],
            stiffness=self.meta["stiffness"],
            bending_stiffness=self.meta["bending_stiffness"],
            workgroup_size=self.meta["workgroup_size"],
            solver_mode=self.meta["solver_mode"],
        )
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=0.2,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
        )
        sim.set_solver_iterations(4)
        
        out_pos = np.zeros(self.num_vertices * 3, dtype=np.float32)
        dt = 0.016666668
        substeps = 40
        
        # 15フレーム（約0.25秒）進行
        for step in range(1, 16):
            sim.step(dt=dt, substeps=substeps)
            if step in [5, 10, 15]:
                sim.get_positions(out_pos)
                pos_3d = out_pos.reshape((-1, 3))
                inter_cnt = count_inter_cloth_intersections(pos_3d, self.faces_lower, self.faces_upper)
                print(f"Step {step:2d}: 交差面数 = {inter_cnt}")

        sim.get_positions(out_pos)
        pos_final = out_pos.reshape((-1, 3))
        final_inter = count_inter_cloth_intersections(pos_final, self.faces_lower, self.faces_upper)
        print(f"最終結果 (Step 15): 交差面数 = {final_inter} (初期: {init_inter})")

        # 期待値: 貫通が自律解消されること（初期の 1/5 以下、または 30面以下）
        self.assertLess(final_inter, 30, f"貫通が解消されず残存・固定化しています: 初期 {init_inter} -> 最終 {final_inter}")

    def test_collision_from_frame110_no_explosion(self):
        """Frame 110 (衝突直前) から20フレームシミュレーションを進めた際、速度爆発 (max_vel > 50m/s) や宇宙射出が起きないことを検証"""
        f110_pos = np.array(self.log_data["frames"][110]["positions"], dtype=np.float32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=f110_pos,
            edges=self.edges,
            faces=self.faces,
            thickness=self.meta["thickness"],
            stiffness=self.meta["stiffness"],
            bending_stiffness=self.meta["bending_stiffness"],
            workgroup_size=self.meta["workgroup_size"],
            solver_mode=self.meta["solver_mode"],
        )
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=0.2,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
        )
        sim.set_solver_iterations(4)

        out_pos = np.zeros(self.num_vertices * 3, dtype=np.float32)
        prev_pos = f110_pos.copy()
        dt = 0.016666668
        substeps = 40

        max_vel = 0.0
        max_z_overall = -1e9

        for step in range(1, 21):
            sim.step(dt=dt, substeps=substeps)
            sim.get_positions(out_pos)
            pos_3d = out_pos.reshape((-1, 3))

            vel = np.linalg.norm(pos_3d - prev_pos, axis=1) / dt
            step_max_v = float(np.max(vel))
            step_max_z = float(np.max(pos_3d[:, 2]))
            prev_pos = pos_3d.copy()

            if step_max_v > max_vel:
                max_vel = step_max_v
            if step_max_z > max_z_overall:
                max_z_overall = step_max_z

            if step in [5, 10, 15, 20]:
                print(f"Step {step:2d} (Frame {110+step}): max_vel={step_max_v:.2f} m/s, max_z={step_max_z:.2f}m")

        print(f"Frame 110 リプレイ結果: Max Velocity={max_vel:.2f} m/s, Max Z={max_z_overall:.2f}m")
        # 期待値: 衝突瞬間に速度爆発（50m/s超）や宇宙への射出（Z > 15m）が起きないこと
        self.assertLess(max_vel, 50.0, f"衝突の瞬間に速度爆発が発生しています: max_vel={max_vel:.2f} m/s >= 50.0")
        self.assertLess(max_z_overall, 15.0, f"布が上空へ跳ね上がっています: max_z={max_z_overall:.2f}m >= 15.0")


if __name__ == "__main__":
    unittest.main()
