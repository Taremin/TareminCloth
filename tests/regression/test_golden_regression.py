import os
import sys
from pathlib import Path
import unittest
import numpy as np

_PROJECT_ROOT = os.path.abspath(str(Path(__file__).resolve().parents[2]))
_PYTHON_DIR = os.path.join(_PROJECT_ROOT, "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import tests  # noqa: F401
import taremin_cloth_core

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "golden_master")

#: 物理質量の面密度 (再生成スクリプトと同一値)
CASE3_AREAL_DENSITY = 0.15
CASE4_AREAL_DENSITY = 0.10


def areal_inv_masses(pos, faces, density):
    """面積に応じて分配した物理単位の逆質量 (ゴールデン再現用)。"""
    masses = np.zeros(len(pos), dtype=np.float64)
    for f in np.asarray(faces, dtype=np.int64):
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        if a == b or b == c or c == a:
            continue
        ab = pos[b].astype(np.float64) - pos[a].astype(np.float64)
        ac = pos[c].astype(np.float64) - pos[a].astype(np.float64)
        share = 0.5 * float(np.linalg.norm(np.cross(ab, ac))) * density / 3.0
        masses[a] += share
        masses[b] += share
        masses[c] += share
    masses = np.maximum(masses, 1e-9)
    return (1.0 / masses).astype(np.float32)


def _edge_lengths(pos, edges):
    """各エッジのユークリッド長さを返す。"""
    edges_arr = np.asarray(edges, dtype=np.int64)
    diff = pos[edges_arr[:, 0]] - pos[edges_arr[:, 1]]
    return np.linalg.norm(diff, axis=1)


def _max_edge_strain(actual_pos, edges, *, rest_lengths):
    """初期エッジ長に対する現在エッジ長の最大倍率を返す。"""
    current_lengths = _edge_lengths(actual_pos, edges)
    safe_rest = np.maximum(rest_lengths, 1e-6)
    return float((current_lengths / safe_rest).max())


def _min_segment_distance(points, *, seg_a, seg_b):
    """各頂点から線分 (seg_a, seg_b) までの最短距離を返す。"""
    a = np.asarray(seg_a, dtype=np.float64)
    b = np.asarray(seg_b, dtype=np.float64)
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq <= 1e-12:
        return float(np.linalg.norm(points.astype(np.float64) - a, axis=1).min())
    ap = points.astype(np.float64) - a
    t = np.clip(np.dot(ap, ab) / ab_len_sq, 0.0, 1.0)
    closest = a + t[:, None] * ab[None, :]
    return float(np.linalg.norm(points.astype(np.float64) - closest, axis=1).min())


class TestGoldenRegression(unittest.TestCase):
    """
    基準となる物理シミュレーション結果（golden_master/*.npz）の全フレーム物理状態と
    短期厳密一致・全フレーム大域一致および物理不変条件を検証する自己完結型ゴールデンマスター・リグレッションテスト。
    外部ログファイルに依存せず、同梱の golden_master/*.npz データのみで決定論的に実行される。
    """

    def test_case1_cloth_body_regression(self):
        """Case 1: 2点ピン垂直クロス自由落下ドレープ"""
        golden_file = os.path.join(GOLDEN_DIR, "case1_cloth_body.npz")
        self.assertTrue(os.path.exists(golden_file), f"Golden file not found: {golden_file}")
        data = np.load(golden_file)
        golden_pos = data["positions"]
        pos_init = data["initial_pos"]
        edges = data["edges"]
        faces = data["faces"]
        inv_masses = data["inv_masses"]
        rest_lengths = _edge_lengths(pos_init, edges)
        pinned_mask = inv_masses == 0.0
        self.assertTrue(np.any(pinned_mask), "Case 1 must contain pinned vertices")

        sim = taremin_cloth_core.ClothSimulator(
            positions=pos_init,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.005,
            stiffness=1500.0,
            bending_stiffness=15.0,
        )
        sim.set_gravity(0.0, 0.0, -9.81)

        out = np.empty(len(pos_init) * 3, dtype=np.float32)
        for fi in range(1, len(golden_pos)):
            sim.step(1.0 / 60.0, 20, 2)
            sim.get_positions(out)
            actual_pos = out.reshape(-1, 3)

            self.assertFalse(np.isnan(actual_pos).any(), f"Case 1 contains NaN at Frame {fi}")

            # ピン固定保持
            pin_drift = float(np.linalg.norm(actual_pos[pinned_mask] - pos_init[pinned_mask], axis=1).max())
            self.assertLess(pin_drift, 1e-5, f"Case 1 pinned vertices moved at Frame {fi}: {pin_drift}")

            # 頂点誤差検証
            diff = float(np.linalg.norm(actual_pos - golden_pos[fi], axis=1).max())
            if fi <= 10:
                self.assertLess(diff, 0.001, f"Case 1 short-term diverged at Frame {fi}: {diff*1000:.4f} mm")
            self.assertLess(diff, 0.010, f"Case 1 max vertex diverged at Frame {fi}: {diff*1000:.4f} mm")

            # 重心一致 & エッジ歪み検証
            center_diff = float(np.linalg.norm(actual_pos.mean(axis=0) - golden_pos[fi].mean(axis=0)))
            self.assertLess(center_diff, 0.010, f"Case 1 center of mass diverged at Frame {fi}: {center_diff*1000:.4f} mm")
            max_strain = _max_edge_strain(actual_pos, edges, rest_lengths=rest_lengths)
            self.assertLess(max_strain, 1.15, f"Case 1 excessive edge strain at Frame {fi}: {max_strain:.4f}")

    def test_case2_plane_sphere_regression(self):
        """Case 2: 水平クロス → 球体衝突ドレープ"""
        golden_file = os.path.join(GOLDEN_DIR, "case2_plane_sphere.npz")
        self.assertTrue(os.path.exists(golden_file), f"Golden file not found: {golden_file}")
        data = np.load(golden_file)
        golden_pos = data["positions"]
        pos_init = data["initial_pos"]
        edges = data["edges"]
        faces = data["faces"]
        inv_masses = data["inv_masses"]
        rest_lengths = _edge_lengths(pos_init, edges)
        sphere_center = np.array([0.0, 0.0, 0.16], dtype=np.float32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=pos_init,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.005,
            stiffness=1200.0,
            bending_stiffness=5.0,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.add_sphere_collider([0.0, 0.0, 0.16], radius=0.14, friction=0.4, restitution=0.0)

        out = np.empty(len(pos_init) * 3, dtype=np.float32)
        for fi in range(1, len(golden_pos)):
            sim.step(1.0 / 60.0, 20, 2)
            sim.get_positions(out)
            actual_pos = out.reshape(-1, 3)

            self.assertFalse(np.isnan(actual_pos).any(), f"Case 2 contains NaN at Frame {fi}")

            # 球体非貫通
            min_sphere_dist = float(np.linalg.norm(actual_pos - sphere_center, axis=1).min())
            self.assertGreaterEqual(
                min_sphere_dist,
                0.14 - 0.002,
                f"Case 2 sphere penetration at Frame {fi}: min_dist={min_sphere_dist:.5f}",
            )

            # 頂点誤差検証
            diff = float(np.linalg.norm(actual_pos - golden_pos[fi], axis=1).max())
            if fi <= 10:
                self.assertLess(diff, 0.001, f"Case 2 short-term diverged at Frame {fi}: {diff*1000:.4f} mm")
            self.assertLess(diff, 0.015, f"Case 2 max vertex diverged at Frame {fi}: {diff*1000:.4f} mm")

            # 重心一致 & エッジ歪み検証
            center_diff = float(np.linalg.norm(actual_pos.mean(axis=0) - golden_pos[fi].mean(axis=0)))
            self.assertLess(center_diff, 0.015, f"Case 2 center of mass diverged at Frame {fi}: {center_diff*1000:.4f} mm")
            max_strain = _max_edge_strain(actual_pos, edges, rest_lengths=rest_lengths)
            self.assertLess(max_strain, 1.15, f"Case 2 excessive edge strain at Frame {fi}: {max_strain:.4f}")

    def test_case3_multi_collider_regression(self):
        """Case 3: カプセル2本 + 球体 + 床面 同時干渉"""
        golden_file = os.path.join(GOLDEN_DIR, "case3_multi_collider.npz")
        self.assertTrue(os.path.exists(golden_file), f"Golden file not found: {golden_file}")
        data = np.load(golden_file)
        golden_pos = data["positions"]
        pos_init = data["initial_pos"]
        edges = data["edges"]
        faces = data["faces"]
        inv_masses = data["inv_masses"] if "inv_masses" in data.files else areal_inv_masses(pos_init, faces, CASE3_AREAL_DENSITY)
        rest_lengths = _edge_lengths(pos_init, edges)
        sphere_center = np.array([0.0, 0.0, 0.3], dtype=np.float32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=pos_init,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.005,
            stiffness=1000.0,
            bending_stiffness=10.0,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.add_capsule_collider([-0.3, -0.3, 0.4], [0.3, -0.3, 0.5], radius=0.08, friction=0.4, restitution=0.0)
        sim.add_capsule_collider([-0.3, 0.3, 0.5], [0.3, 0.3, 0.4], radius=0.08, friction=0.4, restitution=0.0)
        sim.add_sphere_collider([0.0, 0.0, 0.3], radius=0.12, friction=0.5, restitution=0.0)
        sim.add_plane_collider([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], friction=0.6, restitution=0.0)

        out = np.empty(len(pos_init) * 3, dtype=np.float32)
        for fi in range(1, len(golden_pos)):
            sim.step(1.0 / 60.0, 20, 2)
            sim.get_positions(out)
            actual_pos = out.reshape(-1, 3)

            self.assertFalse(np.isnan(actual_pos).any(), f"Case 3 contains NaN at Frame {fi}")

            # 短期厳密一致 (Frame 1..10)
            diff = float(np.linalg.norm(actual_pos - golden_pos[fi], axis=1).max())
            if fi <= 10:
                self.assertLess(diff, 0.001, f"Case 3 short-term diverged at Frame {fi}: {diff*1000:.4f} mm")

            # 大域重心一致 (全フレーム)
            center_diff = float(np.linalg.norm(actual_pos.mean(axis=0) - golden_pos[fi].mean(axis=0)))
            self.assertLess(center_diff, 0.025, f"Case 3 center of mass diverged at Frame {fi}: {center_diff*1000:.4f} mm")

            # 全フレーム非貫通 & エッジ歪み検証
            min_sphere_dist = float(np.linalg.norm(actual_pos - sphere_center, axis=1).min())
            self.assertGreaterEqual(min_sphere_dist, 0.12 - 0.003, f"Case 3 sphere penetration at Frame {fi}: {min_sphere_dist:.5f}")
            min_z = float(actual_pos[:, 2].min())
            self.assertGreaterEqual(min_z, -0.002, f"Case 3 plane penetration at Frame {fi}: min_z={min_z:.5f}")
            min_c1 = _min_segment_distance(actual_pos, seg_a=[-0.3, -0.3, 0.4], seg_b=[0.3, -0.3, 0.5])
            self.assertGreaterEqual(min_c1, 0.08 - 0.003, f"Case 3 capsule 1 penetration at Frame {fi}: {min_c1:.5f}")
            min_c2 = _min_segment_distance(actual_pos, seg_a=[-0.3, 0.3, 0.5], seg_b=[0.3, 0.3, 0.4])
            self.assertGreaterEqual(min_c2, 0.08 - 0.003, f"Case 3 capsule 2 penetration at Frame {fi}: {min_c2:.5f}")
            max_strain = _max_edge_strain(actual_pos, edges, rest_lengths=rest_lengths)
            self.assertLess(max_strain, 1.20, f"Case 3 excessive edge strain at Frame {fi}: {max_strain:.4f}")

    def test_case4_self_collision_drape_regression(self):
        """Case 4: 高密度多重折り畳み・セルフコリジョン"""
        golden_file = os.path.join(GOLDEN_DIR, "case4_self_collision_drape.npz")
        self.assertTrue(os.path.exists(golden_file), f"Golden file not found: {golden_file}")
        data = np.load(golden_file)
        golden_pos = data["positions"]
        pos_init = data["initial_pos"]
        edges = data["edges"]
        faces = data["faces"]
        inv_masses = data["inv_masses"] if "inv_masses" in data.files else areal_inv_masses(pos_init, faces, CASE4_AREAL_DENSITY)
        rest_lengths = _edge_lengths(pos_init, edges)
        sphere_center = np.array([0.0, 0.0, 0.2], dtype=np.float32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=pos_init,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.005,
            stiffness=800.0,
            bending_stiffness=5.0,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=1.0,
            max_displacement_ratio=0.5,
            exclude_neighbors=True,
            enable_normal_untangling=True,
            max_iterations=256,
        )
        sim.add_sphere_collider([0.0, 0.0, 0.2], radius=0.20, friction=0.6, restitution=0.0)

        out = np.empty(len(pos_init) * 3, dtype=np.float32)
        for fi in range(1, len(golden_pos)):
            sim.step(1.0 / 60.0, 20, 2)
            sim.get_positions(out)
            actual_pos = out.reshape(-1, 3)

            self.assertFalse(np.isnan(actual_pos).any(), f"Case 4 contains NaN at Frame {fi}")

            # 短期厳密一致 (Frame 1..5)
            diff = float(np.linalg.norm(actual_pos - golden_pos[fi], axis=1).max())
            if fi <= 5:
                self.assertLess(diff, 0.005, f"Case 4 short-term diverged at Frame {fi}: {diff*1000:.4f} mm")

            # 大域重心一致
            center_diff = float(np.linalg.norm(actual_pos.mean(axis=0) - golden_pos[fi].mean(axis=0)))
            self.assertLess(center_diff, 0.020, f"Case 4 center of mass diverged at Frame {fi}: {center_diff*1000:.4f} mm")

            # 全フレーム非貫通 & エッジ歪み検証
            min_sphere_dist = float(np.linalg.norm(actual_pos - sphere_center, axis=1).min())
            self.assertGreaterEqual(min_sphere_dist, 0.20 - 0.003, f"Case 4 sphere penetration at Frame {fi}: {min_sphere_dist:.5f}")
            max_strain = _max_edge_strain(actual_pos, edges, rest_lengths=rest_lengths)
            self.assertLess(max_strain, 1.25, f"Case 4 excessive edge strain at Frame {fi}: {max_strain:.4f}")
            if fi >= 20:
                intersections = taremin_cloth_core.find_triangle_intersections(actual_pos, faces)
                self.assertLessEqual(
                    len(intersections),
                    12,
                    f"Case 4 excessive self-intersections at Frame {fi}: {len(intersections)}",
                )

        # 最終フレームでは自己交差がほぼ解消されていること
        final_intersections = taremin_cloth_core.find_triangle_intersections(actual_pos, faces)
        self.assertLessEqual(
            len(final_intersections),
            2,
            f"Case 4 remaining self-intersections at final Frame {len(golden_pos) - 1}: {len(final_intersections)}",
        )


if __name__ == '__main__':
    unittest.main()
