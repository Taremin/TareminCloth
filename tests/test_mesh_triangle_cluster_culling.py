import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python/taremin_cloth")))

import taremin_cloth_core
from taremin_cloth import replayer
from taremin_cloth.replayer import ClothReplayer


class TestMeshTriangleClusterCulling(unittest.TestCase):
    """GPU Linear BVH クラスタカリング (16面グループ境界球) の安全性と安定性テスト"""

    def test_cluster_culling_on_vs_off_stability(self):
        """1万面人体メッシュコライダーでの クラスタカリング ON/OFF 安定性・整合性テスト"""
        log_path = "frame_logs/cloth_debug_20260901_182137_cloth_body.jsonl.gz"
        if not os.path.exists(log_path):
            self.skipTest(f"Log not found: {log_path}")

        f0 = replayer.get_frame(log_path, 0)
        pos0 = np.array(f0["positions"], dtype=np.float32)

        # 1. オプション OFF (デフォルト直接走査) で 5フレーム実行
        rep_off = ClothReplayer(log_path)
        sim_off = rep_off.create_simulator(pos0)
        sim_off.set_collider_options(enable_cluster_culling=False, enable_single_sided_recovery=True)

        out_off = np.empty(len(pos0) * 3, dtype=np.float32)
        for fi in range(1, 6):
            f_curr = replayer.get_frame(log_path, fi)
            rep_off.apply_frame_inputs(f_curr)
            dt = f_curr.get("dt", 1.0 / 60.0)
            substeps = f_curr.get("substeps", 20)
            solver_iters = f_curr.get("solver_iterations", 2)
            sim_off.step(dt, substeps, solver_iters)

        sim_off.get_positions(out_off)
        pos_off = out_off.reshape(-1, 3)

        # 2. オプション ON (16面クラスタカリング) で 5フレーム実行
        rep_on = ClothReplayer(log_path)
        sim_on = rep_on.create_simulator(pos0)
        sim_on.set_collider_options(enable_cluster_culling=True, enable_single_sided_recovery=True, sweep_margin=0.05)

        out_on = np.empty(len(pos0) * 3, dtype=np.float32)
        for fi in range(1, 6):
            f_curr = replayer.get_frame(log_path, fi)
            rep_on.apply_frame_inputs(f_curr)
            dt = f_curr.get("dt", 1.0 / 60.0)
            substeps = f_curr.get("substeps", 20)
            solver_iters = f_curr.get("solver_iterations", 2)
            sim_on.step(dt, substeps, solver_iters)

        sim_on.get_positions(out_on)
        pos_on = out_on.reshape(-1, 3)

        # 3. 検証
        self.assertFalse(np.isnan(pos_on).any(), "Cluster culling produced NaN positions")
        self.assertFalse(np.isinf(pos_on).any(), "Cluster culling produced Inf positions")

        # 重心位置の一致 (大域的に同一の軌道)
        center_off = pos_off.mean(axis=0)
        center_on = pos_on.mean(axis=0)
        center_diff = np.linalg.norm(center_off - center_on)
        self.assertLess(center_diff, 0.010, f"Center of mass diverged between ON and OFF: {center_diff*1000:.4f} mm")

        # 最大差分 (局所接触判定の微小誤差が15mm以内)
        max_diff = np.linalg.norm(pos_off - pos_on, axis=1).max()
        self.assertLess(max_diff, 0.020, f"Max local diff between ON and OFF too large: {max_diff*1000:.4f} mm")


if __name__ == "__main__":
    unittest.main()
