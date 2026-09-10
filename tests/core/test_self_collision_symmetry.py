import unittest
import numpy as np


class TestSelfCollisionSymmetry(unittest.TestCase):
    """自己衝突における作用・反作用（運動量保存則）の対称性検証テスト"""

    def create_two_plane_cloth(self, nx=3, ny=3, z_top=0.01, z_bottom=-0.01, inv_m_top=1.0, inv_m_bottom=1.0):
        """上下に対向する2枚の正方形メッシュを生成する"""
        dx = 0.05
        positions = []
        inv_masses = []

        # 1. 下の布 (Bottom Plane)
        for y in range(ny):
            for x in range(nx):
                positions.append([x * dx, y * dx, z_bottom])
                inv_masses.append(inv_m_bottom)

        offset_top = nx * ny

        # 2. 上の布 (Top Plane)
        for y in range(ny):
            for x in range(nx):
                positions.append([x * dx, y * dx, z_top])
                inv_masses.append(inv_m_top)

        edges = []
        faces = []

        # 下の布のエッジと面
        for y in range(ny):
            for x in range(nx):
                idx = y * nx + x
                if x + 1 < nx:
                    edges.append([idx, idx + 1])
                if y + 1 < ny:
                    edges.append([idx, idx + nx])
                if x + 1 < nx and y + 1 < ny:
                    faces.append([idx, idx + 1, idx + nx + 1])
                    faces.append([idx, idx + nx + 1, idx + nx])

        # 上の布のエッジと面
        for y in range(ny):
            for x in range(nx):
                idx = offset_top + y * nx + x
                if x + 1 < nx:
                    edges.append([idx, idx + 1])
                if y + 1 < ny:
                    edges.append([idx, idx + nx])
                if x + 1 < nx and y + 1 < ny:
                    faces.append([idx, idx + 1, idx + nx + 1])
                    faces.append([idx, idx + nx + 1, idx + nx])

        return (
            np.array(positions, dtype=np.float32),
            np.array(edges, dtype=np.uint32),
            np.array(faces, dtype=np.uint32),
            np.array(inv_masses, dtype=np.float32),
            offset_top,
        )

    def test_equal_mass_symmetry(self):
        """等質量の2枚布が接触した際、作用・反作用により上下の変位が対称になることを確認"""
        import taremin_cloth_core

        nx, ny = 3, 3
        thickness = 0.015  # 合計厚み 0.030m > 距離 0.020m なので接触反発が発生
        pos, edges, faces, inv_m, offset_top = self.create_two_plane_cloth(
            nx=nx, ny=ny, z_top=0.01, z_bottom=-0.01, inv_m_top=1.0, inv_m_bottom=1.0
        )

        sim = taremin_cloth_core.ClothSimulator(
            positions=pos,
            edges=edges,
            faces=faces,
            inv_masses=inv_m,
            layer_id=0,
            thickness=thickness,
            stiffness=10000.0,
        )
        # 重力をゼロにして純粋な自己衝突のみを評価
        sim.set_gravity(0.0, 0.0, 0.0)
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=1.0,
            max_displacement_ratio=1.0,
            exclude_neighbors=True,
            enable_normal_untangling=False,
            max_iterations=128,
        )

        # 1サブステップ実行
        sim.step_single_substep(dt_sub=0.005)

        out_coords = np.zeros(len(pos) * 3, dtype=np.float32)
        sim.get_positions(out_coords)
        out_pos = out_coords.reshape((-1, 3))

        disp = out_pos - pos
        bottom_z_disp = disp[:offset_top, 2]
        top_z_disp = disp[offset_top:, 2]

        total_bottom = np.sum(bottom_z_disp)
        total_top = np.sum(top_z_disp)

        print(f"\n[Test Equal Mass Symmetry] Bottom Z Sum: {total_bottom:.6f}m, Top Z Sum: {total_top:.6f}m")

        # 1. 接触により上下両方に反発変位が発生していること
        self.assertLess(total_bottom, 0.0, "下の布は下向きに変位するべき")
        self.assertGreater(total_top, 0.0, "上の布は上向きに変位するべき")

        # 2. 作用・反作用（運動量保存）により、上下の変位の絶対値がほぼ等しいこと
        net_momentum = total_bottom + total_top
        max_abs = max(abs(total_bottom), abs(total_top))
        relative_error = abs(net_momentum) / max(max_abs, 1e-6)
        print(f"[Test Equal Mass Symmetry] Net Z Sum: {net_momentum:.6f}m (Relative Error: {relative_error * 100:.2f}%)")

        self.assertLess(relative_error, 0.05, f"等質量布の自己衝突において運動量が保存されるべき (誤差 {relative_error * 100:.2f}%)")

    def test_unequal_mass_momentum_conservation(self):
        """異質量の2枚布において、質量比に応じた変位分配（運動量保存）が行われることを確認"""
        import taremin_cloth_core

        nx, ny = 3, 3
        thickness = 0.015
        # 上の布: 質量 2.0 (inv_m = 0.5), 下の布: 質量 1.0 (inv_m = 1.0)
        pos, edges, faces, inv_m, offset_top = self.create_two_plane_cloth(
            nx=nx, ny=ny, z_top=0.01, z_bottom=-0.01, inv_m_top=0.5, inv_m_bottom=1.0
        )

        sim = taremin_cloth_core.ClothSimulator(
            positions=pos,
            edges=edges,
            faces=faces,
            inv_masses=inv_m,
            layer_id=0,
            thickness=thickness,
            stiffness=10000.0,
        )
        sim.set_gravity(0.0, 0.0, 0.0)
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=1.0,
            max_displacement_ratio=1.0,
            exclude_neighbors=True,
            enable_normal_untangling=False,
            max_iterations=128,
        )

        # 1サブステップ実行
        sim.step_single_substep(dt_sub=0.005)

        out_coords = np.zeros(len(pos) * 3, dtype=np.float32)
        sim.get_positions(out_coords)
        out_pos = out_coords.reshape((-1, 3))

        disp = out_pos - pos
        bottom_z_disp = disp[:offset_top, 2]
        top_z_disp = disp[offset_top:, 2]

        total_bottom = np.sum(bottom_z_disp)
        total_top = np.sum(top_z_disp)

        # 運動量変化: Delta P = sum(m_i * Delta x_i) = sum(Delta x_i / inv_m_i)
        bottom_momentum = total_bottom / 1.0  # m_bottom = 1.0
        top_momentum = total_top / 0.5        # m_top = 2.0

        print(f"\n[Test Unequal Mass] Bottom Momentum: {bottom_momentum:.6f}, Top Momentum: {top_momentum:.6f}")
        net_momentum = bottom_momentum + top_momentum
        max_p = max(abs(bottom_momentum), abs(top_momentum))
        relative_error = abs(net_momentum) / max(max_p, 1e-6)
        print(f"[Test Unequal Mass] Net Momentum: {net_momentum:.6f} (Relative Error: {relative_error * 100:.2f}%)")

        # 軽い下の布の方が重い上の布よりも大きく動くこと
        self.assertGreater(abs(total_bottom), abs(total_top), "軽い布の方が大きな変位を受けるべき")
        self.assertLess(relative_error, 0.10, f"質量比に応じた運動量が保存されるべき (誤差 {relative_error * 100:.2f}%)")


if __name__ == "__main__":
    unittest.main()
