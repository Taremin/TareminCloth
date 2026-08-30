"""
シワの谷底がおちょこ状に裏返らない（傘の反転防止）ことを保証する回帰ガードテスト
球コライダーの上に布を被せてドレープさせ、表裏の整合性を自動検証する。
"""

import os
import sys
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import taremin_cloth_core


class TestWrinkleInversionGuard(unittest.TestCase):
    def test_wrinkle_does_not_invert_on_sphere(self):
        """球コライダーの上に布を落下させ、シワが裏返らず（法線が外向きを維持し）美しくドレープすることを検証"""
        # 30x30 グリッド (0.6m x 0.6m)
        nx, ny = 20, 20
        dx = 0.03
        positions = []
        inv_masses = []

        for y in range(ny):
            for x in range(nx):
                px = (x - nx / 2.0) * dx
                py = (y - ny / 2.0) * dx
                pz = 0.25 # 球の上に配置
                positions.append([px, py, pz])
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

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.005,
            stiffness=1000.0,
            bending_stiffness=10.0,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=0.2,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
        )

        # 半径 0.15m の球コライダーを原点 (0, 0, 0) に配置
        sim.add_sphere_collider(center=[0.0, 0.0, 0.0], radius=0.15, friction=0.3, restitution=0.0)

        out_pos = np.zeros(len(positions) * 3, dtype=np.float32)

        # 40フレームシミュレーション (球に覆いかぶさる)
        for step in range(1, 41):
            sim.step(dt=1.0/60.0, substeps=20)

        sim.get_positions(out_pos)
        pos_3d = out_pos.reshape((-1, 3))

        # 布が球の上に留まっているか (宇宙への射出・爆発が起きていないか)
        max_z = np.max(pos_3d[:, 2])
        min_z = np.min(pos_3d[:, 2])
        print(f"ドレープ完了時: Z範囲 = [{min_z:.4f}m, {max_z:.4f}m]")
        self.assertLess(max_z, 0.5, f"布が上空へ跳ね上がっています (max_z = {max_z})")
        self.assertGreater(max_z, 0.10, f"布が球を突き抜けて下に落ちています (max_z = {max_z})")

        # 面法線の計算: 各面の法線が上向きまたは外向きを向いているか
        # おちょこ状に裏返った場合、法線が反転して下・内側を向く
        tris = pos_3d[faces]
        fnorms = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        fnorm_lens = np.linalg.norm(fnorms, axis=1, keepdims=True) + 1e-7
        unit_fnorms = fnorms / fnorm_lens

        # 面重心から原点（球中心）への方向ベクトル
        centers = np.mean(tris, axis=1)
        center_lens = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-7
        radial_dirs = centers / center_lens

        # 法線と放射方向の内積: 表側を向いていれば正 (dot > 0)
        # ドレープで垂れ下がった部分も、外向き法線であれば radial_dir と鋭角または垂直付近
        dots = np.sum(unit_fnorms * radial_dirs, axis=1)
        inverted_faces = np.sum(dots < -0.3)
        print(f"裏返り面数 (dots < -0.3): {inverted_faces} / {len(faces)}")

        self.assertLessEqual(inverted_faces, int(len(faces) * 0.05), "布の面が広範囲でおちょこ状に裏返っています")


if __name__ == "__main__":
    unittest.main()
