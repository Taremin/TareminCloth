"""
Grab操作（ピンによる強制引っ張り・めり込み）をシミュレートし、
ピン解除後に布の自己貫通が自律的に解消されるかを検証するTDDテスト
"""

import os
import sys
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import taremin_cloth_core


class TestGrabSelfPenetration(unittest.TestCase):
    def test_grab_drag_and_release_untangling(self):
        """Grab操作で布を相手の布の裏側にめり込ませた後、ピン解除で自律脱出できるかを検証"""
        # 2枚の対向する布グリッド (8x8, 各64頂点)
        # 下の布 (z = 0.0, 頂点 0..63)
        # 上の布 (z = 0.05, 頂点 64..127)
        nx, ny = 8, 8
        dx = 0.04
        positions = []
        inv_masses = []
        
        # 下の布 (四隅を固定)
        for y in range(ny):
            for x in range(nx):
                px = (x - nx / 2.0) * dx
                py = (y - ny / 2.0) * dx
                pz = 0.0
                positions.append([px, py, pz])
                if (x == 0 or x == nx - 1) and (y == 0 or y == ny - 1):
                    inv_masses.append(0.0)
                else:
                    inv_masses.append(1.0)
                    
        # 上の布 (外枠固定なし)
        for y in range(ny):
            for x in range(nx):
                px = (x - nx / 2.0) * dx
                py = (y - ny / 2.0) * dx
                pz = 0.05
                positions.append([px, py, pz])
                inv_masses.append(1.0)

        positions = np.array(positions, dtype=np.float32)
        inv_masses = np.array(inv_masses, dtype=np.float32)

        edges = []
        faces = []
        for base in [0, nx * ny]:
            for y in range(ny):
                for x in range(nx):
                    idx = base + y * nx + x
                    if x + 1 < nx:
                        edges.append([idx, idx + 1])
                    if y + 1 < ny:
                        edges.append([idx, idx + nx])
                    if x + 1 < nx and y + 1 < ny:
                        faces.append([idx, idx + 1, idx + nx + 1])
                        faces.append([idx, idx + nx + 1, idx + nx])

        edges = np.array(edges, dtype=np.uint32)
        faces = np.array(faces, dtype=np.uint32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions.copy(),
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.005, # 5mm
            stiffness=2000.0,
        )
        sim.set_gravity(0.0, 0.0, 0.0) # 重力なしで純粋な衝突・Untangling復元力を観察
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=0.3,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
        )

        grab_vert = 64 + 27 # 上の布の中央頂点
        out_pos = np.zeros(len(positions) * 3, dtype=np.float32)

        # 1. Grabフェーズ: 中央頂点をピン留めし、下の布の裏側 (Z = -0.03m / -3cm) まで徐々に引っ張る
        for step in range(1, 11):
            target_z = 0.05 - (0.08 * (step / 10.0)) # 0.05 -> -0.03
            sim.set_pin(grab_vert, [0.0, 0.0, target_z], weight=1.0)
            sim.step(dt=1.0/60.0, substeps=10)

        sim.get_positions(out_pos)
        pos_grabbed = out_pos.reshape((-1, 3))
        print(f"Grab完了時: 頂点{grab_vert}のZ座標 = {pos_grabbed[grab_vert, 2]:.4f}m (下の布 Z=0 を突き抜けて裏抜け)")
        self.assertLess(pos_grabbed[grab_vert, 2], -0.01, "Grab操作で頂点が裏抜けしているはず")

        # 2. Releaseフェーズ: ピンを解除し、自律的に表側 (Z > 0) へ脱出するか追跡
        sim.release_pin(grab_vert)
        for step in range(1, 31):
            sim.step(dt=1.0/60.0, substeps=15)
            if step in [10, 20, 30]:
                sim.get_positions(out_pos)
                pos_curr = out_pos.reshape((-1, 3))
                print(f"Release後 Step {step:2d}: 頂点{grab_vert}のZ座標 = {pos_curr[grab_vert, 2]:+.4f}m")

        sim.get_positions(out_pos)
        pos_final = out_pos.reshape((-1, 3))
        final_z = pos_final[grab_vert, 2]

        print(f"最終結果: 頂点{grab_vert}のZ = {final_z:+.4f}m")
        # 期待値: 表側 (Z > 0.0) へ脱出・復元していること
        self.assertGreater(final_z, 0.0, f"Grab解除後に裏抜け頂点が表側 (Z > 0) に脱出できていません: 実測 {final_z:.4f}m")


if __name__ == "__main__":
    unittest.main()
