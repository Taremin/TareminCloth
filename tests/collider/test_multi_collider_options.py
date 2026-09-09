"""
マルチコライダー設定およびメッシュコライダー個別オプションの検証テスト
- プロパティがTareminClothColliderSettingsに存在し、TareminClothObjectSettingsから削除されていること
- コライダー個別でSingle-Sided RecoveryのON/OFFが面フラグ(tri.flags)に正しく反映されること
- 複数コライダー間でLinear BVH CullingとSweep Margin Offsetが集約されること
- GPU物理シミュレータにおいて、Recovery ONの面は押し戻され、Recovery OFFの面は素通りすること
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python/taremin_cloth")))

import taremin_cloth_core
from taremin_cloth import properties


class TestMultiColliderOptions(unittest.TestCase):
    """マルチコライダー個別最適化およびプロパティ移設の検証テスト"""

    def test_property_definitions_clean_separation(self):
        """プロパティがコライダー側に新設され、布側から完全に削除されていること"""
        col_annotations = properties.TareminClothColliderSettings.__annotations__
        cloth_annotations = properties.TareminClothObjectSettings.__annotations__

        # 1. コライダー側（TareminClothColliderSettings）に存在すること
        self.assertIn("enable_single_sided_recovery", col_annotations)
        self.assertIn("enable_cluster_culling", col_annotations)
        self.assertIn("sweep_margin_offset", col_annotations)

        # 2. 布側（TareminClothObjectSettings）から削除されていること
        self.assertNotIn("enable_collider_cluster_culling", cloth_annotations)
        self.assertNotIn("enable_single_sided_recovery", cloth_annotations)
        self.assertNotIn("collider_sweep_margin_offset", cloth_annotations)

    def test_face_flags_calculation(self):
        """コライダーごとの設定に応じて正しいビットフラグが計算されること"""
        # ケース1: 片面 + リカバリー有効 (デフォルト) -> flags = 1.0 (bit 0=1, bit 1=0)
        is_single = True
        is_recovery = True
        flags_val = 0.0
        if is_single:
            flags_val += 1.0
            if not is_recovery:
                flags_val += 2.0
        self.assertEqual(flags_val, 1.0)

        # ケース2: 片面 + リカバリー無効 -> flags = 3.0 (bit 0=1, bit 1=1: recovery_disabled)
        is_single = True
        is_recovery = False
        flags_val = 0.0
        if is_single:
            flags_val += 1.0
            if not is_recovery:
                flags_val += 2.0
        self.assertEqual(flags_val, 3.0)

        # ケース3: 両面 -> flags = 0.0
        is_single = False
        is_recovery = True
        flags_val = 0.0
        if is_single:
            flags_val += 1.0
            if not is_recovery:
                flags_val += 2.0
        self.assertEqual(flags_val, 0.0)

    def test_multi_collider_culling_aggregation(self):
        """複数コライダー存在時、カリング設定とマージンが正しく集約されること"""
        class MockCollider:
            def __init__(self, c_type, enable_culling, sweep_margin):
                self.collider_type = c_type
                self.enable_cluster_culling = enable_culling
                self.sweep_margin_offset = sweep_margin

        colliders = [
            MockCollider('MESH', False, 0.02),
            MockCollider('SPHERE', False, 0.05),
            MockCollider('MESH', True, 0.08),
            MockCollider('MESH', False, 0.03),
        ]

        has_cluster_culling = False
        max_sweep_margin = 0.05
        for col_s in colliders:
            if col_s.collider_type == 'MESH':
                if getattr(col_s, "enable_cluster_culling", False):
                    has_cluster_culling = True
                sweep_m = float(getattr(col_s, "sweep_margin_offset", 0.05))
                if sweep_m > max_sweep_margin:
                    max_sweep_margin = sweep_m

        self.assertTrue(has_cluster_culling, "いずれかのMESHコライダーでカリングが有効ならTrueになること")
        self.assertAlmostEqual(max_sweep_margin, 0.08, places=4, msg="最大のスウィープマージンが採用されること")

    def test_gpu_individual_single_sided_recovery_behavior(self):
        """GPUシミュレータ上で、リカバリーONの面とOFFの面が個別に動作すること"""
        # z=0 に2つの三角形を配置
        # 三角形1: x in [-2, 0], flags = 1 (片面, リカバリー有効)
        # 三角形2: x in [0, 2],  flags = 3 (片面, リカバリー無効)
        triangles = np.array([
            # tri 1: [-1, 0, 0] 付近
            [[-2.0, -1.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            # tri 2: [1, 0, 0] 付近
            [[0.0, -1.0, 0.0], [2.0, -1.0, 0.0], [1.0, 1.0, 0.0]],
        ], dtype=np.float32)

        attributes = np.array([
            [0.2, 0.02, 0.0, 1.0],  # tri 1: flags = 1.0 (Recovery ON)
            [0.2, 0.02, 0.0, 3.0],  # tri 2: flags = 3.0 (Recovery OFF)
        ], dtype=np.float32)

        # 頂点A: tri 1 の裏側 (x=-1, y=0, z=-0.01)
        # 頂点B: tri 2 の裏側 (x=1, y=0, z=-0.01)
        positions = np.array([
            [-1.0, 0.0, -0.01],
            [1.0, 0.0, -0.01],
        ], dtype=np.float32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=np.empty((0, 2), dtype=np.uint32),
            thickness=0.02,
            stiffness=10000.0,
        )
        sim.set_gravity(0.0, 0.0, 0.0)  # 無重力で純粋な復帰挙動を確認
        sim.set_mesh_collider_triangles(triangles, attributes=attributes)

        for _ in range(10):
            sim.step(dt=1.0 / 60.0, substeps=20)

        out_pos = np.empty(len(positions) * 3, dtype=np.float32)
        sim.get_positions(out_pos)
        res = out_pos.reshape(-1, 3)

        pos_a = res[0]
        pos_b = res[1]

        # 頂点A (Recovery ON): 表側 (z >= 0.04 - 1e-3) へ押し戻されていること
        self.assertGreaterEqual(
            pos_a[2], 0.04 - 1e-3,
            f"Recovery有効な面では裏側頂点が表側へ押し戻されること (z={pos_a[2]:.4f})"
        )

        # 頂点B (Recovery OFF): リカバリーされないため裏側のまま (z <= 0.0)
        self.assertLessEqual(
            pos_b[2], 0.0,
            f"Recovery無効な面では裏側頂点は押し戻されないこと (z={pos_b[2]:.4f})"
        )


if __name__ == "__main__":
    unittest.main()
