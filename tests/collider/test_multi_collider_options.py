"""
マルチコライダー設定およびメッシュコライダー個別オプションの検証テスト
- プロパティがTareminClothColliderSettingsに存在し、TareminClothObjectSettingsから削除されていること
- コライダー個別でSingle-Sided RecoveryのON/OFFが面フラグ(tri.flags)に正しく反映されること
- 複数コライダー間でLinear BVH CullingとSweep Margin Offsetが集約されること
- GPU物理シミュレータにおいて、Recovery ONの面は押し戻され、Recovery OFFの面は素通りすること
"""

from pathlib import Path
import sys
import unittest
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
_PKG_DIR = _PYTHON_DIR / "taremin_cloth"
for _p in (str(_REPO_ROOT), str(_PYTHON_DIR), str(_PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tests  # noqa: F401
import taremin_cloth_core
from taremin_cloth import properties
from taremin_cloth.engine.collider import (
    _extract_mesh_collider_triangles,
    extract_all_scene_colliders,
)


class _DummyVerts:
    def __init__(self, coords: np.ndarray):
        self._co = coords.astype(np.float32).ravel()
        self._n = len(coords)

    def __len__(self):
        return self._n

    def foreach_get(self, attr, out):
        assert attr == "co"
        out[:] = self._co


class _DummyLoopTris:
    def __init__(self, tris: np.ndarray):
        self._idx = tris.astype(np.int32).ravel()
        self._n = len(tris)

    def __len__(self):
        return self._n

    def foreach_get(self, attr, out):
        assert attr == "vertices"
        out[:] = self._idx


class _DummyMesh:
    def __init__(self, verts: np.ndarray, tris: np.ndarray):
        self.vertices = _DummyVerts(verts)
        self.loop_triangles = _DummyLoopTris(tris)

    def calc_loop_triangles(self):
        pass


class _DummyObj:
    def __init__(self, name: str, *, col_settings, obj_type: str = 'MESH'):
        self.name = name
        self.type = obj_type
        self.taremin_cloth_collider = col_settings
        self.matrix_world = np.eye(4, dtype=np.float32)
        self.modifiers = []
        self.scale = (1.0, 1.0, 1.0)
        self.data = _DummyMesh(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int32),
        )


class _DummyColliderSettings:
    def __init__(
        self,
        *,
        c_type: str = 'MESH',
        single_sided: bool = True,
        enable_recovery: bool = True,
        enable_culling: bool = False,
        sweep_margin: float = 0.05,
        radius: float = 0.5,
    ):
        self.is_collider = True
        self.enabled = True
        self.collider_type = c_type
        self.friction = 0.25
        self.thickness = 0.015
        self.restitution = 0.1
        self.single_sided = single_sided
        self.enable_single_sided_recovery = enable_recovery
        self.enable_cluster_culling = enable_culling
        self.sweep_margin_offset = sweep_margin
        self.close_open_boundaries = False
        self.radius = radius


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
        """コライダーごとの設定に応じて本番抽出関数が正しいビットフラグ(1/3/0)を返すこと"""
        target_obj = _DummyObj("MeshCol", col_settings=_DummyColliderSettings())

        # ケース1: 片面 + リカバリー有効 (デフォルト) -> flags = 1.0 (bit 0=1, bit 1=0)
        col1 = _DummyColliderSettings(single_sided=True, enable_recovery=True)
        _, attr1, dicts1 = _extract_mesh_collider_triangles(target_obj.data, target_obj, col1)
        self.assertEqual(float(attr1[0, 3]), 1.0)
        self.assertEqual(dicts1[0]["flags"], 1)

        # ケース2: 片面 + リカバリー無効 -> flags = 3.0 (bit 0=1, bit 1=1: recovery_disabled)
        col2 = _DummyColliderSettings(single_sided=True, enable_recovery=False)
        _, attr2, dicts2 = _extract_mesh_collider_triangles(target_obj.data, target_obj, col2)
        self.assertEqual(float(attr2[0, 3]), 3.0)
        self.assertEqual(dicts2[0]["flags"], 3)

        # ケース3: 両面 -> flags = 0.0
        col3 = _DummyColliderSettings(single_sided=False, enable_recovery=True)
        _, attr3, dicts3 = _extract_mesh_collider_triangles(target_obj.data, target_obj, col3)
        self.assertEqual(float(attr3[0, 3]), 0.0)
        self.assertEqual(dicts3[0]["flags"], 0)

    def test_multi_collider_culling_aggregation(self):
        """複数コライダー存在時、本番関数 extract_all_scene_colliders がカリング設定と最大スウィープマージンを集約すること"""
        objs = [
            _DummyObj("M1", col_settings=_DummyColliderSettings(c_type='MESH', enable_culling=False, sweep_margin=0.02)),
            _DummyObj("S1", col_settings=_DummyColliderSettings(c_type='SPHERE', enable_culling=False, sweep_margin=0.20, radius=0.4)),
            _DummyObj("M2", col_settings=_DummyColliderSettings(c_type='MESH', enable_culling=True, sweep_margin=0.08)),
            _DummyObj("M3", col_settings=_DummyColliderSettings(c_type='MESH', enable_culling=False, sweep_margin=0.03)),
        ]
        scene = type("_DummyScene", (), {"objects": objs})()

        extracted = extract_all_scene_colliders(scene=scene, depsgraph=None)
        self.assertEqual(len(extracted["analytic_colliders"]), 1)
        self.assertEqual(len(extracted["mesh_triangles_data"]), 3)
        self.assertTrue(
            extracted["has_cluster_culling"],
            "いずれかのMESHコライダーでカリングが有効ならTrueになること",
        )
        # SPHERE の sweep_margin=0.20 は MESH 専用集約に混入しないことも合わせて検証
        self.assertAlmostEqual(
            extracted["max_sweep_margin"],
            0.08,
            places=4,
            msg="MESHコライダーの最大スウィープマージンが採用されること",
        )

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
