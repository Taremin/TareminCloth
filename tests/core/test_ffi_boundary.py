# -*- coding: utf-8 -*-
"""PyO3 FFI境界における境界値・異常系入力の防御テスト."""

import unittest
import numpy as np
import tests  # noqa: F401


class TestFfiBoundary(unittest.TestCase):
    def setUp(self):
        try:
            import taremin_cloth_core as core
            self.core = core
        except ImportError:
            try:
                from taremin_cloth import taremin_cloth_core as core
                self.core = core
            except ImportError:
                self.skipTest("taremin_cloth_core が利用できないためスキップします")

    def test_non_contiguous_numpy_slices_handled_safely(self):
        """非連続メモリレイアウト (step slice) のNumPy配列を事前変換なしで渡しても安全かつ正確に処理されること"""
        raw_pos = np.zeros((10, 3), dtype=np.float32)
        raw_pos[:, 0] = np.arange(10, dtype=np.float32) * 0.1
        sliced_pos = raw_pos[::2]  # 非C-contiguous (5頂点, X=[0.0, 0.2, 0.4, 0.6, 0.8])
        self.assertFalse(sliced_pos.flags["C_CONTIGUOUS"])

        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.uint32)
        sim = self.core.ClothSimulator(positions=sliced_pos, edges=edges)
        out = np.zeros(5 * 3, dtype=np.float32)
        sim.get_positions(out)
        np.testing.assert_allclose(out.reshape(-1, 3), sliced_pos, atol=1e-6)

        sim.step(1.0 / 60.0, 5)
        sim.get_positions(out)
        self.assertTrue(np.all(np.isfinite(out)))

        # compute_combined_inv_masses へ非連続 pin_weights を渡してもサイレント空化せずピンが反映されること
        pw_raw = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        pw_sliced = pw_raw[::2]  # [1.0, 0.0, 1.0]
        self.assertFalse(pw_sliced.flags["C_CONTIGUOUS"])
        pos3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        faces3 = np.array([[0, 1, 2]], dtype=np.uint32)
        inv = self.core.compute_combined_inv_masses(
            pos3, faces3, None, pw_sliced, 0.15, 0.0
        )
        self.assertEqual(float(inv[0]), 0.0)
        self.assertGreater(float(inv[1]), 0.0)
        self.assertEqual(float(inv[2]), 0.0)

    def test_zero_or_extreme_substeps_and_dt(self):
        """substeps=1, solver_iterations=1, 極小dt や負の減衰値でNaNやゼロ除算パニックが起きないこと"""
        pos = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32)
        edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.uint32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        sim = self.core.ClothSimulator(positions=pos, edges=edges, faces=faces)

        # 負の減衰・ゼロ剛性を設定してもクランプされNaNにならないこと
        sim.set_damping(-10.0)
        sim.set_damping_all(-1.0, -1.0, -1.0, -1.0)
        sim.set_stiffness_all(0.0, 0.0, 0.0, 0.0)
        sim.step(1e-5, 1, 1)

        out = np.zeros(3 * 3, dtype=np.float32)
        sim.get_positions(out)
        self.assertTrue(np.all(np.isfinite(out)), "極端なパラメータ入力でNaN/Infが発生してはならない")

    def test_brush_functions_mismatch_lengths_raise_value_error(self):
        """ブラシFFI関数に不正次元や長さが不一致の配列を渡した際、パニックせずPyValueErrorを送出すること"""
        pos = np.zeros((4, 3), dtype=np.float32)
        adj_offsets = np.array([0, 1, 2, 3, 4], dtype=np.uint32)
        adj_indices = np.array([1, 2, 3, 0], dtype=np.uint32)
        # brush_idx と brush_weights の長さが不一致
        brush_idx = np.array([0, 1], dtype=np.uint32)
        brush_weights = np.array([1.0], dtype=np.float32)

        with self.assertRaises(ValueError):
            self.core.brush_smooth_targets(
                pos, adj_offsets, adj_indices, brush_idx, brush_weights, 0.5
            )


if __name__ == "__main__":
    unittest.main()
