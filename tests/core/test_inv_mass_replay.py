"""
動的 inv_masses (inv_mass_deltas) 記録・リプレイ復元および旧ログ Waypoint フォールバック検証テスト
"""

import gzip
import json
import os
import shutil
import sys
import tempfile
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import taremin_cloth_core
from taremin_cloth.engine.waypoint import WaypointManager
from taremin_cloth.replayer import (
    ClothReplayer,
    inv_masses_at_frame,
    iter_frames,
    read_metadata,
)
from taremin_cloth.utils.mesh_extract import WaypointSeam


class TestInvMassReplay(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @staticmethod
    def _build_two_triangle_strip():
        """0-1 と 2-3 の布面の間を 4(Waypoint) を介して縫合する最小テストメッシュ."""
        positions = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.0, 0.05, 1.0],
                [0.1, 0.05, 1.0],
                [0.0, 0.025, 1.0],
            ],
            dtype=np.float32,
        )
        # 面は 0-1-2 側ではなく 2枚三角形にすると、1が端点同士、4を孤立WPにしやすいよう 0-1-5 とする
        positions = np.array(
            [
                [0.0, 0.0, 1.0],    # 0: cloth A
                [0.1, 0.0, 1.0],    # 1: cloth A helper
                [0.0, -0.1, 1.0],   # 2: cloth A helper
                [0.0, 0.04, 1.0],   # 3: cloth B (縫合相手)
                [0.1, 0.04, 1.0],   # 4: cloth B helper
                [0.0, 0.14, 1.0],   # 5: cloth B helper
                [0.0, 0.02, 1.05],  # 6: Waypoint (faces/edges に含まれない)
            ],
            dtype=np.float32,
        )
        edges = np.array(
            [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]],
            dtype=np.uint32,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
        sewing = np.array([[0, 6], [6, 3]], dtype=np.uint32)
        inv_masses = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        return positions, edges, faces, sewing, inv_masses

    def test_inv_mass_deltas_recording_and_exact_replay_including_mid_start_and_substeps(self):
        """set_vertex_inv_masses による動的質量変化が記録され、全区間・途中開始・trace_substeps で 0.0mm 完全一致すること。"""
        log_path = os.path.join(self.temp_dir, "inv_mass_deltas.jsonl.gz")
        pos, edges, faces, sewing, inv_masses = self._build_two_triangle_strip()

        sim = taremin_cloth_core.ClothSimulator(
            positions=pos,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            sewing_springs=sewing,
            thickness=0.005,
            stiffness=1500.0,
            bending_stiffness=10.0,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_pin(1, [0.1, 0.0, 1.0], 1.0)
        sim.set_pin(4, [0.1, 0.04, 1.0], 1.0)

        sim.start_debug_recording("InvMassReplayTest", max_frames=30)
        logged_positions = []
        out = np.zeros(len(pos) * 3, dtype=np.float32)

        for frame_i in range(6):
            if frame_i == 2:
                # ステップ前に WP 頂点 6 のピン解除 (動的質量設定) + 別頂点質量変更
                sim.set_vertex_inv_masses(
                    np.array([6, 2], dtype=np.uint32),
                    np.array([1.5, 2.0], dtype=np.float32),
                )
            if frame_i == 4:
                # 途中フレームでピン変更と質量再変更が同時に走るケース
                sim.set_pin(5, [0.0, 0.14, 1.0], 0.5)
                sim.set_vertex_inv_masses(
                    np.array([6], dtype=np.uint32),
                    np.array([0.8], dtype=np.float32),
                )
            sim.step(dt=1.0 / 60.0, substeps=6)
            sim.get_positions(out)
            logged_positions.append(out.reshape((-1, 3)).copy())

        sim.save_debug_recording(log_path)
        sim.stop_debug_recording()

        frames = list(iter_frames(log_path))
        self.assertEqual(len(frames), 6)
        self.assertIn("inv_mass_deltas", frames[0])
        self.assertIsNone(frames[0]["inv_mass_deltas"])
        self.assertIsNone(frames[1]["inv_mass_deltas"])
        self.assertEqual(
            {d["vertex_idx"]: d["inv_mass"] for d in frames[2]["inv_mass_deltas"]},
            {6: 1.5, 2: 2.0},
        )
        self.assertIsNone(frames[3]["inv_mass_deltas"])
        self.assertEqual(
            {d["vertex_idx"]: d["inv_mass"] for d in frames[4]["inv_mass_deltas"]},
            {6: 0.8},
        )
        self.assertIsNone(frames[5]["inv_mass_deltas"])

        # 折り畳み関数単体検証
        meta = read_metadata(log_path)
        m3 = inv_masses_at_frame(log_path, meta, 3)
        self.assertAlmostEqual(float(m3[6]), 1.5, places=6)
        self.assertAlmostEqual(float(m3[2]), 2.0, places=6)
        m4 = inv_masses_at_frame(log_path, meta, 4)
        self.assertAlmostEqual(float(m4[6]), 0.8, places=6)

        # 1. F0 -> F5 フル区間リプレイ
        rp = ClothReplayer(log_path)
        res_full = rp.replay_range(0, 5, compare=True)
        for r in res_full:
            self.assertEqual(r["max_diff"], 0.0, f"Full replay diverged at F{r['frame_index']}")

        # 2. F3 -> F5 途中開始リプレイ (F2 の解放済み質量が維持されること)
        res_mid = rp.replay_range(3, 5, compare=True)
        for r in res_mid:
            self.assertEqual(r["max_diff"], 0.0, f"Mid-start replay diverged at F{r['frame_index']}")

        # 3. F4 のサブステップ顕微鏡トレース (F4 ステップ中の質量変更・ピン変更が反映されること)
        traces = rp.trace_substeps(4)
        self.assertEqual(len(traces), 6)
        diff_sub_end = float(np.max(np.abs(traces[-1]["positions"] - logged_positions[4])))
        self.assertEqual(diff_sub_end, 0.0)

        # 4. replay_until でも誤差ゼロで終端まで走ること
        hit = rp.replay_until(0, 5, stop_on=lambda ctx: ctx["frame_index"] == 5)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["max_diff"], 0.0)

    def test_sparse_recording_preserves_inv_mass_deltas_on_stubs(self):
        """スパース記録時にスタブフレーム上で起きた inv_mass_deltas が欠落しないこと。"""
        log_path = os.path.join(self.temp_dir, "sparse_inv_mass.jsonl.gz")
        pos, edges, faces, sewing, inv_masses = self._build_two_triangle_strip()

        sim = taremin_cloth_core.ClothSimulator(
            positions=pos,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            sewing_springs=sewing,
            thickness=0.005,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_debug_recording_options(full_stride=4, ring_size=0, lookahead=0, enable_triggers=False)
        sim.start_debug_recording("SparseInvMass", max_frames=20)

        for frame_i in range(5):
            if frame_i == 2:  # F2 はスタブになる (stride=4)
                sim.set_vertex_inv_masses(
                    np.array([6], dtype=np.uint32),
                    np.array([1.25], dtype=np.float32),
                )
            sim.step(dt=1.0 / 60.0, substeps=4)

        sim.save_debug_recording(log_path)
        sim.stop_debug_recording()

        frames = list(iter_frames(log_path))
        self.assertEqual(frames[2]["positions"], [])
        self.assertEqual(frames[2]["inv_mass_deltas"], [{"vertex_idx": 6, "inv_mass": 1.25}])

        rp = ClothReplayer(log_path)
        results = rp.replay_range(0, 4, compare=True)
        by_idx = {r["frame_index"]: r for r in results}
        self.assertTrue(by_idx[4]["compared"])
        self.assertEqual(by_idx[4]["max_diff"], 0.0)

    def test_legacy_log_without_inv_mass_deltas_uses_waypoint_fallback(self):
        """inv_mass_deltas を持たない旧ログで WaypointManager フォールバックにより WP解放が復元されること。"""
        new_log_path = os.path.join(self.temp_dir, "with_deltas.jsonl.gz")
        legacy_log_path = os.path.join(self.temp_dir, "legacy_no_deltas.jsonl.gz")
        pos, edges, faces, sewing, inv_masses = self._build_two_triangle_strip()

        sim = taremin_cloth_core.ClothSimulator(
            positions=pos,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            sewing_springs=sewing,
            thickness=0.005,
            stiffness=5000.0,
            bending_stiffness=5.0,
            sewing_shrink_speed=3.0,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        # 両布端の背面頂点を緩く保持し、0-3-6が寄りやすくする
        sim.set_pin(1, pos[1].tolist(), 1.0)
        sim.set_pin(4, pos[4].tolist(), 1.0)

        wp_seam = WaypointSeam(
            vert_a=0,
            vert_b=3,
            waypoint_indices=[6],
            waypoint_coords=pos[6:7].copy(),
        )
        wp_mgr = WaypointManager(
            waypoint_seams=[wp_seam],
            initial_inv_masses=inv_masses,
            merge_dist=0.01,
            pass_dist=0.015,
        )

        sim.start_debug_recording("LegacyWpFallback", max_frames=30)
        out = np.zeros(len(pos) * 3, dtype=np.float32)
        released_at = None
        for fi in range(8):
            sim.step(dt=1.0 / 60.0, substeps=8)
            sim.get_positions(out)
            coords = out.reshape((-1, 3)).copy()
            rel = wp_mgr.update(coords, sim=sim)
            if rel and released_at is None:
                released_at = fi
        self.assertIsNotNone(released_at, "テスト条件上で Waypoint が解放されること")
        self.assertLess(released_at, 6)

        sim.save_debug_recording(new_log_path)
        sim.stop_debug_recording()

        # inv_mass_deltas キーを取り除いて旧形式ログを生成
        with gzip.open(new_log_path, "rt", encoding="utf-8") as fin, gzip.open(
            legacy_log_path, "wt", encoding="utf-8"
        ) as fout:
            for line in fin:
                rec = json.loads(line)
                if rec.get("record_type") == "frame":
                    rec.pop("inv_mass_deltas", None)
                fout.write(json.dumps(rec) + "\n")

        rp = ClothReplayer(legacy_log_path)
        res_full = rp.replay_range(0, 7, compare=True)
        for r in res_full:
            self.assertEqual(
                r["max_diff"],
                0.0,
                f"Legacy WP fallback full replay diverged at F{r['frame_index']}",
            )

        # 解放途中のフレーム以降から開始しても正確であること
        mid_start = released_at + 1
        res_mid = rp.replay_range(mid_start, 7, compare=True)
        for r in res_mid:
            self.assertEqual(
                r["max_diff"],
                0.0,
                f"Legacy WP fallback mid replay diverged at F{r['frame_index']}",
            )

    def test_force_pair_cache_override_survives_param_deltas(self):
        """途中に param_deltas (apply_config_json) があっても force_pair_cache 上書きが維持されること。"""
        log_path = os.path.join(self.temp_dir, "pair_override_param_delta.jsonl.gz")
        pos, edges, faces, _, inv_masses = self._build_two_triangle_strip()
        # 縫合なし4+頂点として使う
        sim = taremin_cloth_core.ClothSimulator(
            positions=pos,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.005,
        )
        sim.set_enable_pair_cache(True)
        sim.start_debug_recording("PairOverrideTest", max_frames=10)
        sim.step(dt=1.0 / 60.0, substeps=2)
        sim.set_gravity(0.0, 0.0, -5.0)  # param_deltas を F1 に発生させる (cfg内でenable_pair_cache=True)
        sim.step(dt=1.0 / 60.0, substeps=2)
        sim.save_debug_recording(log_path)
        sim.stop_debug_recording()

        rp = ClothReplayer(log_path)
        rp.replay_range(0, 1, compare=False, force_pair_cache=False)
        cfg_json = json.loads(rp.sim.get_config_json())
        self.assertFalse(cfg_json["enable_pair_cache"])


if __name__ == "__main__":
    unittest.main()
