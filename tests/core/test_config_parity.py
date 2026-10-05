# -*- coding: utf-8 -*-
"""SimConfig一本化のパリティテスト.

新規 set_* 追加時に SimulationMetadata / SimConfig への写像漏れを検出する。
"""

import json
import os
import sys
import unittest

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

# 状態系 (物理パラメータではない) で parity 対象外とする setter
STATE_EXCLUDE = {
    "set_pin",
    "set_pins_batch",
    "set_positions_and_velocities",
    "set_mesh_collider_triangles",
    "set_bone_sdf_colliders",
    "set_dynamic_bone_sdf_dirty_bones",
    "set_dynamic_bone_sdf_enabled",
    "set_dynamic_bone_sdf_update_interval",
    "set_edge_rest_length_scales",
    "set_collider_options",
    "set_sewing_priority_options",  # SimConfigには含むが旧metadata個別フィールドにもあるため別枠検証
    "set_solver_iterations",  # FrameRecord側にもあるため別枠
    "set_gravity",
    "set_damping",
    "set_debug_recording_options",  # 記録形式オプション (物理パラメータではない)
    "set_sewing_current_rest_lengths",  # GPU進行状態の復元 (再現用、調整対象ではない)
    "set_sewing_priority_state",  # ラッチ内部状態の復元 (再現用、調整対象ではない)
    "set_enable_wrinkle_field",  # ドレープガイド有効化 (動的アセット連携、SimConfig非依存)
    "set_wrinkle_field_params",   # ドレープガイド基底・剛性パラメータ
    # 注意: "set_wrinkle_field_profile" は旧1Dパスの残骸であり実装が存在しないため除外した
    "set_wrinkle_field_texture_2d",  # ドレープガイド2D-SDFテクスチャデータ
    "set_vertex_inv_masses",  # 動的頂点質量・ピン解放 (状態系、SimConfig非依存)
}


class TestConfigParity(unittest.TestCase):
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

    def _grid(self, nx=3, ny=3, dx=0.05):
        positions = []
        inv = []
        for y in range(ny):
            for x in range(nx):
                positions.append([x * dx, y * dx, 0.0])
                inv.append(1.0)
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
        return (
            np.array(positions, dtype=np.float32),
            np.array(edges, dtype=np.uint32),
            np.array(faces, dtype=np.uint32),
            np.array(inv, dtype=np.float32),
        )

    def test_config_json_roundtrip(self):
        pos, edges, faces, inv = self._grid()
        sim = self.core.ClothSimulator(pos, edges, faces, inv)
        self.assertTrue(hasattr(sim, "get_config_json"))
        self.assertTrue(hasattr(sim, "apply_config_json"))
        js = sim.get_config_json()
        cfg = json.loads(js)
        # 欠落していた項目が含まれていること
        for key in [
            "coupled_mode", "post_relaxation_iters", "substep_interval",
            "sewing_stiffness", "enable_sewing_lock", "sewing_lock_distance",
            "sewing_priority_enabled", "solver_mode", "workgroup_size",
            "enable_pair_cache", "pair_margin_mode",
        ]:
            self.assertIn(key, cfg, f"SimConfigに {key} がありません")

    def test_config_apply_changes_sim(self):
        pos, edges, faces, inv = self._grid()
        sim = self.core.ClothSimulator(pos, edges, faces, inv)
        js = json.loads(sim.get_config_json())
        js["solver_iterations"] = 7
        js["coupled_mode"] = 1
        js["substep_interval"] = 2
        js["sewing_stiffness"] = 12345.0
        sim.apply_config_json(json.dumps(js))
        js2 = json.loads(sim.get_config_json())
        self.assertEqual(js2["solver_iterations"], 7)
        self.assertEqual(js2["coupled_mode"], 1)
        self.assertEqual(js2["substep_interval"], 2)
        self.assertEqual(js2["sewing_stiffness"], 12345.0)

    def test_setter_parity_with_config(self):
        """全 set_* (状態系除く) が SimConfig キーに対応すること"""
        pos, edges, faces, inv = self._grid()
        sim = self.core.ClothSimulator(pos, edges, faces, inv)
        cfg = json.loads(sim.get_config_json())
        setters = [m for m in dir(sim) if m.startswith("set_") and callable(getattr(sim, m))]
        # 物理系のみ抽出
        phys = [s for s in setters if s not in STATE_EXCLUDE]
        # setter名 -> configキー対応表 (完全一致でなくても許容する緩やかな検証)
        mapping = {
            "set_stiffness": ["tension_stiffness", "bending_stiffness"],
            "set_stiffness_all": ["tension_stiffness", "compression_stiffness", "shear_stiffness", "bending_stiffness"],
            "set_damping_all": ["tension_damping"],
            "set_solver_mode": ["solver_mode"],
            "set_workgroup_size": ["workgroup_size"],
            "set_enable_self_collision": ["enable_self_collision"],
            "set_self_collision_options": ["relief_factor", "max_displacement_ratio"],
            "set_coupled_self_collision_options": ["coupled_mode", "post_relaxation_iters"],
            "set_coupled_collider": ["coupled_collider"],
            "set_self_collision_substep_interval": ["substep_interval"],
            "set_self_collision_ee_substep_interval": ["ee_substep_interval"],
            "set_self_collision_algorithm": ["self_collision_algorithm"],
            "set_enable_pair_cache": ["enable_pair_cache"],
            "set_pair_cache_options": ["pair_max_pairs", "pair_margin_mode"],
            "set_enable_pair_cache_final_fallback": ["enable_pair_final_fallback"],
            "set_enable_edge_collision": ["enable_edge_collision"],
            "set_edge_margin_scale": ["edge_margin_scale"],
            "set_edge_margin_offset": ["edge_margin_offset"],
            "set_sewing_stiffness": ["sewing_stiffness"],
            "set_enable_sewing_lock": ["enable_sewing_lock"],
            "set_sewing_lock_distance": ["sewing_lock_distance"],
            "set_areal_density": ["areal_density"],
            "set_enable_compact_readback": [],  # 転送最適化のみで物理に無関係
            "set_profiling_enabled": [],  # 計測プロファイル用で物理に無関係
        }
        missing = []
        for s in phys:
            if s in mapping:
                for k in mapping[s]:
                    if k not in cfg:
                        missing.append(f"{s}->{k}")
            elif s in ("set_sewing_priority_options",):
                pass  # 別テストで検証
            else:
                # 未知の setter が追加されたらテストを赤くして対応表更新を促す
                missing.append(f"unmapped setter: {s}")
        self.assertEqual(missing, [], f"SimConfig対応漏れ: {missing}")

    def test_invalid_config_json_raises(self):
        pos, edges, faces, inv = self._grid()
        sim = self.core.ClothSimulator(pos, edges, faces, inv)
        with self.assertRaises(Exception):
            sim.apply_config_json("{invalid json")


if __name__ == "__main__":
    unittest.main()
