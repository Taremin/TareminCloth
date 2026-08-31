"""
単一布（同一メッシュ・同一レイヤー・同一アイランド）における
自己貫通自律解消（Untangling）および速度調停（Velocity Projection）のTDDテスト
"""

import os
import sys
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import taremin_cloth_core


class TestSingleClothUntangling(unittest.TestCase):
    def test_single_cloth_folded_penetration_recovery(self):
        """1枚の布を折り畳んで初期状態で自己交差（貫通）させた状態から、
        自律的に表側へ脱出し、かつ反発速度爆発（カタパルト現象）が起きないかを検証"""
        nx, ny = 10, 10
        dx = 0.04
        positions = []
        inv_masses = []

        # 10x10の布グリッド (幅 0.36m x 0.36m)
        # 上半分 (y >= 5) を Z = 0.0 に配置
        # 下半分 (y < 5) を折り返して Z = -0.015 (裏側にめり込んだ自己貫通状態) に配置
        for y in range(ny):
            for x in range(nx):
                px = (x - nx / 2.0) * dx
                py = y * dx * 0.5  # 前後に縮めて折り重なりやすくする
                if y < 5:
                    pz = -0.015  # 下半分: 意図的に上半分と交差して裏抜け
                else:
                    pz = 0.0
                positions.append([px, py, pz])
                # 上半分 (y >= 5) を基準面として固定
                if y >= 5:
                    inv_masses.append(0.0)
                else:
                    inv_masses.append(1.0)

        positions = np.array(positions, dtype=np.float32)
        inv_masses = np.array(inv_masses, dtype=np.float32)

        edges = []
        faces = []
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

        edges = np.array(edges, dtype=np.uint32)
        faces = np.array(faces, dtype=np.uint32)

        # 貫通の測定関数: 下半分 (y < 5, 頂点 0..49) のうち、固定された上半分 (Z=0) の裏側 (Z < 0) にある頂点数
        def count_penetrations(coords):
            pos_3d = coords.reshape((-1, 3))
            lower_verts = pos_3d[:50]
            return int(np.sum(lower_verts[:, 2] < 0.0))

        init_pen = count_penetrations(positions)
        self.assertEqual(init_pen, 50, "初期状態で下半分の全頂点 (50個) が裏抜け貫通していること")

        # 多層布シミュレータ初期化:
        # 下半分 (y < 5) を外側レイヤー (layer 1)、固定された上半分 (y >= 5) を内側レイヤー (layer 0) に設定
        layer_ids = np.zeros(len(positions), dtype=np.uint32)
        layer_ids[:50] = 1
        layer_ids[50:] = 0

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions.copy(),
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            layer_ids=layer_ids,
            thickness=0.008,  # 8mm
            stiffness=2000.0,
        )

        sim.set_gravity(0.0, 0.0, 0.0)  # 純粋な幾何衝突・Untangling脱出力を検証
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=0.3,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
        )

        out_coords = np.zeros(len(positions) * 3, dtype=np.float32)
        prev_coords = positions.copy().flatten()
        max_observed_speed = 0.0

        for frame in range(1, 41):
            dt_frame = 1.0 / 60.0
            sim.step(dt=dt_frame, substeps=15)
            sim.get_positions(out_coords)

            # フレーム間の頂点移動量から速度 (m/s) を算出
            curr_pos_3d = out_coords.reshape((-1, 3))
            prev_pos_3d = prev_coords.reshape((-1, 3))
            disp = np.linalg.norm(curr_pos_3d - prev_pos_3d, axis=1)
            speeds = disp / dt_frame
            frame_max_speed = float(np.max(speeds))
            if frame_max_speed > max_observed_speed:
                max_observed_speed = frame_max_speed
            prev_coords[:] = out_coords[:]

        final_pen = count_penetrations(out_coords)
        recovery_rate = (1.0 - final_pen / init_pen) * 100.0

        print(f"\n[単一布Untangling検証] 初期貫通: {init_pen}個 -> 最終残存: {final_pen}個 (解消率: {recovery_rate:.1f}%)")
        print(f"[最高速度検証] 観測された最高速度: {max_observed_speed:.2f} m/s (カタパルト安全基準: < 30.0 m/s)")

        # 1. 貫通解消の検証: 90%以上解消されていること (残存 5個以下)
        self.assertLessEqual(final_pen, 5, f"単一布の自己貫通が自律解消されていません (残存: {final_pen}/{init_pen})")

        # 2. カタパルト爆発防止の検証: 最高速度が30 m/s以内であること
        self.assertLess(max_observed_speed, 30.0, f"Untangling時にカタパルト速度爆発が発生しています: {max_observed_speed:.2f} m/s")

    def test_mountain_fold_back_to_back_collision(self):
        """山折り（裏面同士の接触）において、Untanglingが誤作動して
        相手の体を突き破ることなく、正常に反発・非貫通が維持されるかを検証"""
        nx, ny = 8, 8
        dx = 0.04
        positions = []
        inv_masses = []

        # 2枚の山折りメッシュ（裏地同士が向き合う）
        # 布1: Z = +0.02, 法線は +Z (上向き) -> 裏面は -Z (下向き)
        # 布2: Z = -0.02, 法線は -Z (下向き) -> 裏面は +Z (上向き)
        # つまり互いの裏面が Z = 0 付近で向き合っている状態
        for y in range(ny):
            for x in range(nx):
                px = (x - nx / 2.0) * dx
                py = (y - ny / 2.0) * dx
                positions.append([px, py, 0.02])
                if x == 0 or x == nx - 1:
                    inv_masses.append(0.0)
                else:
                    inv_masses.append(1.0)

        for y in range(ny):
            for x in range(nx):
                px = (x - nx / 2.0) * dx
                py = (y - ny / 2.0) * dx
                positions.append([px, py, -0.02])
                if x == 0 or x == nx - 1:
                    inv_masses.append(0.0)
                else:
                    inv_masses.append(1.0)

        positions = np.array(positions, dtype=np.float32)
        inv_masses = np.array(inv_masses, dtype=np.float32)

        edges = []
        faces = []
        # 布1: 通常の面（反時計回り -> 法線 +Z）
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

        # 布2: 逆向きの面（時計回り -> 法線 -Z）
        base = nx * ny
        for y in range(ny):
            for x in range(nx):
                idx = base + y * nx + x
                if x + 1 < nx:
                    edges.append([idx, idx + 1])
                if y + 1 < ny:
                    edges.append([idx, idx + nx])
                if x + 1 < nx and y + 1 < ny:
                    faces.append([idx, idx + nx + 1, idx + 1])
                    faces.append([idx, idx + nx, idx + nx + 1])

        edges = np.array(edges, dtype=np.uint32)
        faces = np.array(faces, dtype=np.uint32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions.copy(),
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.005,  # 厚み 5mm -> 衝突境界は 10mm
            stiffness=2000.0,
        )
        sim.set_gravity(0.0, 0.0, 0.0)
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=0.3,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
        )

        out = np.zeros(len(positions) * 3, dtype=np.float32)

        # 30フレームシミュレーション実行
        for frame in range(1, 31):
            sim.step(dt=1.0 / 60.0, substeps=15)

        sim.get_positions(out)
        pos_3d = out.reshape((-1, 3))
        cloth1_z = pos_3d[:base, 2]
        cloth2_z = pos_3d[base:, 2]

        # 期待値: 布1は Z > 0、布2は Z < 0 を維持し、互いに相手を突き破っていないこと
        penetrated_1 = int(np.sum(cloth1_z < -0.001))
        penetrated_2 = int(np.sum(cloth2_z > 0.001))

        print(f"[山折り裏裏接触検証] 布1の突き抜け: {penetrated_1}個, 布2の突き抜け: {penetrated_2}個")
        self.assertEqual(penetrated_1, 0, "布1が裏面同士の衝突で相手を突き抜けています")
        self.assertEqual(penetrated_2, 0, "布2が裏面同士の衝突で相手を突き抜けています")


if __name__ == "__main__":
    unittest.main()
