"""階層メッシュSDFベイクの legacy 等価性テスト（Blender不要）.

近傍場のbit等価、遠方場の符号一致＋差分上限を検証する。
"""

import os
import sys
import unittest

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import taremin_cloth_core
from taremin_cloth.engine.sdf_baker import bake_mesh_sdf_from_data


def _decode(bytes_):
    import struct

    n = len(bytes_) // 4
    vals = np.empty(n, dtype=np.float32)
    for i in range(n):
        (packed,) = struct.unpack_from("<I", bytes_, i * 4)
        half_bits = packed & 0xFFFF
        vals[i] = np.frombuffer(
            np.array([half_bits], dtype=np.uint16).tobytes(), dtype=np.float16
        )[0].astype(np.float32)
    return vals


def _packed_u16_list(bytes_):
    import struct

    n = len(bytes_) // 4
    return [struct.unpack_from("<I", bytes_, i * 4)[0] & 0xFFFF for i in range(n)]


CUBE_V = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float32,
)
CUBE_T = np.array(
    [
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ],
    dtype=np.int32,
)


def _double_wall():
    """2mm厚プレート2枚を3mmギャップで平行配置（候補継承の adversarial ケース）."""

    def plate(z0):
        t = 0.002
        s = 0.15
        v = np.array(
            [
                [-s, -s, z0], [s, -s, z0], [s, s, z0], [-s, s, z0],
                [-s, -s, z0 + t], [s, -s, z0 + t], [s, s, z0 + t], [-s, s, z0 + t],
            ],
            dtype=np.float32,
        )
        t_ = np.array(
            [
                [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
            ],
            dtype=np.int32,
        )
        return v, t_

    v1, t1 = plate(0.0)
    v2, t2 = plate(0.005)
    t2 = t2 + len(v1)
    return np.vstack([v1, v2]), np.vstack([t1, t2])


class TestMeshSdfHierarchical(unittest.TestCase):
    def setUp(self):
        if not taremin_cloth_core.is_gpu_available():
            self.skipTest("GPUが利用できない環境のためスキップします")
        if not hasattr(taremin_cloth_core, "bake_mesh_sdf_hierarchical_gpu"):
            self.skipTest("階層ベイク関数が未公開のpydのためスキップします")

    def _assert_equivalent(self, verts, tris, voxel, near_bound, far_bound):
        legacy = bake_mesh_sdf_from_data(
            verts, tris, voxel_size=voxel, method="legacy"
        )
        hier = bake_mesh_sdf_from_data(
            verts, tris, voxel_size=voxel, method="hierarchical"
        )
        self.assertIsNotNone(legacy)
        self.assertIsNotNone(hier)
        self.assertEqual(
            (hier.width, hier.height, hier.depth),
            (legacy.width, legacy.height, legacy.depth),
        )
        ld = _decode(legacy.texture_bytes)
        hd = _decode(hier.texture_bytes)
        lp = _packed_u16_list(legacy.texture_bytes)
        hp = _packed_u16_list(hier.texture_bytes)
        near_mismatch = 0
        far_sign_mismatch = 0
        far_max_diff = 0.0
        for i in range(len(ld)):
            if abs(float(ld[i])) <= near_bound:
                if lp[i] != hp[i]:
                    near_mismatch += 1
            else:
                if (ld[i] < 0.0) != (hd[i] < 0.0):
                    far_sign_mismatch += 1
                far_max_diff = max(far_max_diff, abs(float(ld[i] - hd[i])))
        print(
            f"[hier-equiv] near_mismatch={near_mismatch} "
            f"far_sign_mismatch={far_sign_mismatch} far_max_diff={far_max_diff:.4f}"
        )
        self.assertEqual(near_mismatch, 0, "近傍場はbit等価であること")
        self.assertEqual(far_sign_mismatch, 0, "遠方場の符号は一致すること")
        self.assertLessEqual(far_max_diff, far_bound, "遠方場の差分は上限以内であること")

    def test_cube_equivalence(self):
        # band = 4 x 密ボクセル幅 (1.46m/30 = 48.7mm -> 0.195)。近傍はbit等価、
        # 遠方は粗半対角 (0.1825mセル -> 0.158) 以下の差分を要求
        self._assert_equivalent(CUBE_V, CUBE_T, 0.05, near_bound=0.21, far_bound=0.18)

    def test_double_wall_equivalence(self):
        # 密ボクセル幅は最大軸で 0.43m/108 = 3.98mm -> band 約16mm。
        # 遠方差分上限は粗半対角 (30.7/30.7/17.1mmセル -> 23.3mm)＋余裕
        v, t = _double_wall()
        self._assert_equivalent(v, t, 0.004, near_bound=0.02, far_bound=0.03)

    def test_auto_routes_small_to_legacy(self):
        import struct

        r = bake_mesh_sdf_from_data(CUBE_V, CUBE_T, voxel_size=0.05, method="auto")
        ref = bake_mesh_sdf_from_data(CUBE_V, CUBE_T, voxel_size=0.05, method="legacy")
        self.assertIsNotNone(r)
        self.assertEqual(r.texture_bytes, ref.texture_bytes)


if __name__ == "__main__":
    unittest.main()
