"""
縫合の密着距離 (sewing_lock_distance) およびマグネット吸着テスト
一定距離内に接近した縫合ペアが、自然長待ち時間をスキップして目標長(0.0)にラッチされ、
剛体ロックされることを検証する。
"""

import unittest
import numpy as np
import taremin_cloth_core


def make_sewing_pair(gap=0.015):
    """隙間 gap (m) で対向する2頂点・2エッジの最小メッシュを作成"""
    # 頂点: 4頂点 (0-1エッジ, 2-3エッジ), 1と2の間を縫合
    # v0: (-0.05, 0, 0), v1: (0, 0, 0)
    # v2: (gap, 0, 0),   v3: (gap + 0.05, 0, 0)
    pos = np.array([
        [-0.05, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [gap, 0.0, 0.0],
        [gap + 0.05, 0.0, 0.0],
    ], dtype=np.float32)
    edges = np.array([[0, 1], [2, 3]], dtype=np.uint32)
    faces = np.zeros((0, 3), dtype=np.uint32)
    sew = np.array([[1, 2]], dtype=np.uint32)
    inv = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)  # 外側ピン固定
    return pos, edges, faces, sew, inv


class TestSewingLockDistance(unittest.TestCase):
    """密着距離 (sewing_lock_distance) による即時吸着・剛体ロックの検証"""

    def test_getter_setter(self):
        """ゲッター・セッターが正しく動作すること"""
        pos, edges, faces, sew, inv = make_sewing_pair(gap=0.015)
        sim = taremin_cloth_core.ClothSimulator(
            positions=pos, edges=edges, faces=faces, inv_masses=inv,
            sewing_springs=sew, sewing_lock_distance=0.03,
        )
        self.assertAlmostEqual(sim.get_sewing_lock_distance(), 0.03, places=5)
        sim.set_sewing_lock_distance(0.012)
        self.assertAlmostEqual(sim.get_sewing_lock_distance(), 0.012, places=5)

    def test_snap_when_within_lock_distance(self):
        """距離が lock_distance 以下の場合、即座に自然長が0になり結合すること"""
        gap = 0.015  # 15mm
        pos, edges, faces, sew, inv = make_sewing_pair(gap=gap)
        # shrink_speed を 0.001 (非常に遅い) に設定し、通常の収縮では1ステップで縮まないようにする
        sim = taremin_cloth_core.ClothSimulator(
            positions=pos, edges=edges, faces=faces, inv_masses=inv,
            sewing_springs=sew,
            sewing_shrink_speed=0.001,
            sewing_stiffness=10000.0,
            enable_sewing_lock=True,
            sewing_lock_distance=0.02,  # 20mm (gap 15mm より大きい -> 吸着発動)
        )
        sim.set_gravity(0, 0, 0)
        # 1ステップ実行
        sim.step(dt=1.0 / 60.0, substeps=10, solver_iterations=5)

        # 自然長が即座に 0.0 にラッチされていることを確認
        rest_lens = sim.get_sewing_current_rest_lengths()
        self.assertAlmostEqual(rest_lens[0], 0.0, places=5)

        # 頂点1と頂点2の距離がほぼ0に収束していること
        out = np.zeros(len(pos) * 3, dtype=np.float32)
        sim.get_positions(out)
        coords = out.reshape(-1, 3)
        p1 = coords[1]
        p2 = coords[2]
        dist = float(np.linalg.norm(p1 - p2))
        self.assertLess(dist, 0.001, f"吸着後の距離が1mm未満であるべき: {dist * 1000:.2f}mm")

    def test_no_snap_when_outside_lock_distance(self):
        """距離が lock_distance を超える場合、即時吸着は発動せず緩やかに収縮すること"""
        gap = 0.015  # 15mm
        pos, edges, faces, sew, inv = make_sewing_pair(gap=gap)
        # shrink_speed 0.001 m/s (1ステップ dt=1/60s で 0.016mm 程度しか縮まない)
        sim = taremin_cloth_core.ClothSimulator(
            positions=pos, edges=edges, faces=faces, inv_masses=inv,
            sewing_springs=sew,
            sewing_shrink_speed=0.001,
            sewing_stiffness=10000.0,
            enable_sewing_lock=True,
            sewing_lock_distance=0.005,  # 5mm (gap 15mm より小さい -> 吸着発動しない)
        )
        sim.set_gravity(0, 0, 0)
        sim.step(dt=1.0 / 60.0, substeps=10, solver_iterations=5)

        # 自然長は 0.0 になっていないこと
        rest_lens = sim.get_sewing_current_rest_lengths()
        self.assertGreater(rest_lens[0], 0.01)

        # 頂点間距離もまだ離れていること
        out = np.zeros(len(pos) * 3, dtype=np.float32)
        sim.get_positions(out)
        coords = out.reshape(-1, 3)
        dist = float(np.linalg.norm(coords[1] - coords[2]))
        self.assertGreater(dist, 0.005, f"非吸着時の距離はまだ離れているべき: {dist * 1000:.2f}mm")


if __name__ == "__main__":
    unittest.main()
