"""
Pattern_Hood 自己衝突・安定性 E2E 最終チェックテスト (Blender バックグラウンド実行用)。

本番ライフサイクル関数 `taremin_cloth.step_cloth_scene` を通じて
自然ドレープ（異解像度・異レイヤー・縫合・引き紐・素体コライダー）および
背面めくり摂動（動的ピン牽引・後身頃接触・解放後安定化）の回帰を検証する。
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import bpy
import numpy as np

root_dir = Path(__file__).resolve().parents[2]
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import taremin_cloth
from taremin_cloth.engine import clear_collider_cache, clear_simulators
from tests import is_software_gpu
from tests.fixtures.hood_eval_utils import (
    apply_hood_back_flip_perturbation,
    evaluate_hood_frame_metrics,
    extract_hood_eval_geometry,
    slice_mesh_positions,
)
from tools.setup_tailored_patterns import setup_tailored_scene


# 壊滅的速度劣化検知用の安全網閾値（実機詳細計測は benchmarks/benchmark_hood.py 担当）
MAX_AVG_WALL_MS = 250.0
WARMUP_FRAMES = 5
DT = 1.0 / 60.0


class TestHoodSelfCollisionE2E(unittest.TestCase):
    """Pattern_Hood の自己衝突品質および数値安定性 E2E テストスイート。"""

    def setUp(self) -> None:
        if is_software_gpu():
            self.skipTest("物理GPU必須のためスキップ (ソフトウェアGPU検出)")
        taremin_cloth.register()
        bpy.ops.wm.read_factory_settings(use_empty=True)
        clear_simulators()
        clear_collider_cache()

    def tearDown(self) -> None:
        try:
            clear_simulators()
            clear_collider_cache()
            bpy.ops.wm.read_factory_settings(use_empty=True)
        finally:
            taremin_cloth.unregister()

    def _step_hood(self, *, scene, hood_obj):
        """step_cloth_scene 経由で Pattern_Hood を 1 フレーム進めて (sim, coords) を返す。"""
        res = taremin_cloth.step_cloth_scene(
            scene,
            dt=DT,
            target_objs=[hood_obj],
            update_mesh=True,
        )
        self.assertIn(hood_obj.name, res, "step_cloth_scene の結果に Pattern_Hood が含まれること")
        return res[hood_obj.name]

    def test_hood_natural_drape_self_collision_and_stability(self) -> None:
        """シナリオA: 自然ドレープ 60F で本体自己貫通ゼロ・左右対称・数値安定・極端な速度劣化なしを検証。"""
        setup_res = setup_tailored_scene(
            active_garment="Pattern_Hood",
            garment_names=("Pattern_Hood",),
        )
        hood_obj = setup_res["garments"]["Pattern_Hood"]
        geom = extract_hood_eval_geometry(hood_obj)
        n_mesh_verts = geom["n_mesh_verts"]
        scene = bpy.context.scene

        total_frames = 60
        check_frames = {29, total_frames - 1}
        wall_ms_list: list[float] = []
        prev_pos: np.ndarray | None = None
        frame_metrics: dict[int, dict] = {}

        max_observed_strain = 0.0
        max_observed_p99_strain = 0.0
        max_observed_speed = 0.0

        for f_idx in range(total_frames):
            t0 = time.perf_counter()
            _sim, coords = self._step_hood(scene=scene, hood_obj=hood_obj)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            wall_ms_list.append(wall_ms)

            pos = slice_mesh_positions(coords, n_mesh_verts=n_mesh_verts)
            m = evaluate_hood_frame_metrics(
                positions=pos,
                initial_positions=geom["initial_positions"],
                edges=geom["edges"],
                faces=geom["faces"],
                hood_mask=geom["hood_mask"],
                bodice_mask=geom["bodice_mask"],
                prev_positions=prev_pos,
                dt=DT,
                left_hood_indices=geom["left_hood_indices"],
                right_hood_indices=geom["right_hood_indices"],
                seam_hop_dist=geom["seam_hop_dist"],
                check_intersections=(f_idx in check_frames),
            )
            self.assertFalse(m["has_non_finite"], f"Frame {f_idx} で NaN/Inf を検出")
            max_observed_strain = max(max_observed_strain, float(m["max_strain"]))
            max_observed_p99_strain = max(max_observed_p99_strain, float(m["p99_strain"]))
            if m["max_speed"] is not None:
                max_observed_speed = max(max_observed_speed, float(m["max_speed"]))
            if f_idx in check_frames:
                frame_metrics[f_idx] = m
            prev_pos = pos

        post_warmup_ms = wall_ms_list[WARMUP_FRAMES:]
        avg_wall_ms = float(np.mean(post_warmup_ms))
        f30 = frame_metrics[29]
        f60 = frame_metrics[total_frames - 1]
        print(
            "\n[Hood E2E Drape] "
            f"avg_wall={avg_wall_ms:.2f}ms (F5..F59) | "
            f"max_strain={max_observed_strain:.4f} (p99_peak={max_observed_p99_strain:.4f}) | "
            f"max_speed={max_observed_speed:.3f}m/s | "
            f"F30(cross={f30['cross_layer_intersections']}, raw_cross={f30['raw_cross_layer_intersections']}, "
            f"sym={f30['symmetry_x_mean_error']*1000:.2f}mm, intrusion={f30['back_bodice_intrusion_count']}) | "
            f"F60(cross={f60['cross_layer_intersections']}, raw_cross={f60['raw_cross_layer_intersections']}, "
            f"sym={f60['symmetry_x_mean_error']*1000:.2f}mm, intrusion={f60['back_bodice_intrusion_count']})"
        )

        self.assertLess(
            avg_wall_ms,
            MAX_AVG_WALL_MS,
            f"ウォームアップ後の平均フレーム時間が壊滅的劣化閾値を超過: {avg_wall_ms:.2f} ms",
        )
        self.assertLess(max_observed_strain, 15.0, f"縫合進行中の過渡歪みが発散しています: {max_observed_strain:.4f}")
        self.assertLess(max_observed_p99_strain, 1.65, f"縫合進行中の全体99%歪みが過大です: {max_observed_p99_strain:.4f}")
        self.assertLess(max_observed_speed, 15.0, f"頂点速度が過大です: {max_observed_speed:.3f} m/s")

        for label, fm in (("F30", f30), ("F60", f60)):
            self.assertEqual(
                fm["cross_layer_intersections"],
                0,
                f"{label} で本体面の異レイヤー自己交差が発生: {fm['cross_layer_intersections']}",
            )
            self.assertEqual(
                fm["back_bodice_intrusion_count"],
                0,
                f"{label} で後身頃裏側への食い込みが発生: {fm['back_bodice_intrusion_count']}",
            )
            self.assertLess(fm["p99_strain"], 1.25, f"{label} の p99 歪みが過大: {fm['p99_strain']:.4f}")
            self.assertLess(fm["max_strain"], 3.0, f"{label} の最大歪みが過大: {fm['max_strain']:.4f}")
            self.assertLess(
                fm["symmetry_x_mean_error"],
                0.025,
                f"{label} の左右フード対称誤差が過大: {fm['symmetry_x_mean_error']:.4f} m",
            )

    def test_hood_back_flip_no_tangle_regression(self) -> None:
        """シナリオB: 自然ドレープ後にフードを背面めくり→接触保持→解放し、後身頃噛み込み・絡まりゼロを検証。"""
        setup_res = setup_tailored_scene(
            active_garment="Pattern_Hood",
            garment_names=("Pattern_Hood",),
        )
        hood_obj = setup_res["garments"]["Pattern_Hood"]
        geom = extract_hood_eval_geometry(hood_obj)
        n_mesh_verts = geom["n_mesh_verts"]
        scene = bpy.context.scene

        sim = None
        prev_pos: np.ndarray | None = None
        drape_frames = 30
        drape_end_metrics: dict | None = None

        for f_idx in range(drape_frames):
            sim, coords = self._step_hood(scene=scene, hood_obj=hood_obj)
            pos = slice_mesh_positions(coords, n_mesh_verts=n_mesh_verts)
            m = evaluate_hood_frame_metrics(
                positions=pos,
                initial_positions=geom["initial_positions"],
                edges=geom["edges"],
                faces=geom["faces"],
                hood_mask=geom["hood_mask"],
                bodice_mask=geom["bodice_mask"],
                prev_positions=prev_pos,
                dt=DT,
                left_hood_indices=geom["left_hood_indices"],
                right_hood_indices=geom["right_hood_indices"],
                seam_hop_dist=geom["seam_hop_dist"],
                check_intersections=(f_idx == drape_frames - 1),
            )
            self.assertFalse(m["has_non_finite"], f"事前ドレープ Frame {f_idx} で NaN/Inf を検出")
            if f_idx == drape_frames - 1:
                drape_end_metrics = m
            prev_pos = pos

        self.assertIsNotNone(sim)
        self.assertIsNotNone(drape_end_metrics)

        def _step_for_flip() -> np.ndarray:
            _s, c = self._step_hood(scene=scene, hood_obj=hood_obj)
            return slice_mesh_positions(c, n_mesh_verts=n_mesh_verts)

        flip_records = apply_hood_back_flip_perturbation(
            sim=sim,
            hood_obj=hood_obj,
            scene=scene,
            step_fn=_step_for_flip,
            hood_mask=geom["hood_mask"],
            initial_positions=geom["initial_positions"],
            bodice_mask=geom["bodice_mask"],
            edges=geom["edges"],
            faces=geom["faces"],
            left_hood_indices=geom["left_hood_indices"],
            right_hood_indices=geom["right_hood_indices"],
            seam_hop_dist=geom["seam_hop_dist"],
            flip_frames=20,
            hold_frames=10,
            settle_frames=30,
            dt=DT,
        )

        max_flip_strain = 0.0
        max_flip_p99_strain = 0.0
        max_flip_speed = 0.0
        for rec in flip_records:
            m = rec["metrics"]
            self.assertIsNotNone(m)
            self.assertFalse(
                m["has_non_finite"],
                f"摂動 {rec['phase']}[{rec['phase_step']}] で NaN/Inf を検出",
            )
            max_flip_strain = max(max_flip_strain, float(m["max_strain"]))
            max_flip_p99_strain = max(max_flip_p99_strain, float(m["p99_strain"]))
            if m["max_speed"] is not None:
                max_flip_speed = max(max_flip_speed, float(m["max_speed"]))

        hold_end = [r for r in flip_records if r["phase"] == "hold"][-1]["metrics"]
        settle_end = [r for r in flip_records if r["phase"] == "settle"][-1]["metrics"]

        print(
            "\n[Hood E2E BackFlip] "
            f"drape_cy={drape_end_metrics['hood_centroid_y']:.4f} -> hold_cy={hold_end['hood_centroid_y']:.4f} "
            f"(contact={hold_end['hood_back_contact_count']}) -> settle_cy={settle_end['hood_centroid_y']:.4f} | "
            f"peak_strain={max_flip_strain:.4f} (p99_peak={max_flip_p99_strain:.4f}) | "
            f"peak_speed={max_flip_speed:.3f}m/s | "
            f"settle(cross={settle_end['cross_layer_intersections']}, raw_cross={settle_end['raw_cross_layer_intersections']}, "
            f"max_strain={settle_end['max_strain']:.4f}, speed={settle_end['max_speed']:.3f}m/s, "
            f"intrusion={settle_end['back_bodice_intrusion_count']})"
        )

        # めくり＆保持により実際に後身頃方向 (+Y) へ移動し後身頃近傍へ接触したこと
        self.assertGreater(
            hold_end["hood_centroid_y"],
            drape_end_metrics["hood_centroid_y"] + 0.04,
            "フード重心が後身頃方向 (+Y) へ十分に移動していません",
        )
        self.assertGreater(
            hold_end["hood_back_contact_count"],
            0,
            "めくり保持時にフードが後身頃表面へ接触していません",
        )

        # 摂動中および解放後の安定状態で破綻・異レイヤー貫通が残らないこと
        self.assertLess(max_flip_strain, 15.0, f"摂動中の最大歪みが過大です: {max_flip_strain:.4f}")
        self.assertLess(max_flip_p99_strain, 1.65, f"摂動中の p99 歪みが過大です: {max_flip_p99_strain:.4f}")
        self.assertLess(max_flip_speed, 15.0, f"摂動中の頂点速度が過大です: {max_flip_speed:.3f} m/s")
        self.assertEqual(
            settle_end["cross_layer_intersections"],
            0,
            f"解放後の安定状態で本体面の異レイヤー交差が残存: {settle_end['cross_layer_intersections']}",
        )
        self.assertEqual(
            settle_end["back_bodice_intrusion_count"],
            0,
            f"解放後の安定状態で後身頃裏側への食い込みが残存: {settle_end['back_bodice_intrusion_count']}",
        )
        self.assertLess(settle_end["p99_strain"], 1.25, f"解放後の p99 歪みが過大: {settle_end['p99_strain']:.4f}")
        self.assertLess(settle_end["max_strain"], 3.5, f"解放後の最大歪みが過大: {settle_end['max_strain']:.4f}")
        if settle_end["max_speed"] is not None:
            self.assertLess(settle_end["max_speed"], 3.0, f"解放後の速度が収束していません: {settle_end['max_speed']:.3f} m/s")


if __name__ == "__main__":
    unittest.main()
