#!/usr/bin/env python3
"""
Pattern_Hood 専用性能・品質定点観測ベンチマーク CLI。

通常の Python から実行した場合は `tools.blender_manager.resolve_blender()` で
ヘッドレス Blender を自動起動し、Blender 内部からは直接計測を実行する。
全シミュレーション進行は本番経路 `taremin_cloth.step_cloth_scene` を経由する。

使用例:
    python benchmarks/benchmark_hood.py --scenario both --json scratch/hood_base.json
    python benchmarks/benchmark_hood.py --scenario both --compare scratch/hood_base.json
    python benchmarks/benchmark_hood.py --scenario drape --blender 5.2 --trace-dir scratch/traces
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _resolve_repo_root() -> Path:
    """スクリプト位置または作業ディレクトリからリポジトリルートを解決する。"""
    candidates = [
        Path(__file__).resolve().parent.parent,
        Path.cwd().resolve(),
    ]
    for cand in candidates:
        if (cand / "python" / "taremin_cloth").is_dir():
            return cand
    return Path.cwd().resolve()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _is_running_inside_blender_app() -> bool:
    """実際の Blender プロセス内部で実行されているかを判定する。"""
    try:
        import bpy  # type: ignore

        app = getattr(bpy, "app", None)
        binary_path = getattr(app, "binary_path", "") if app is not None else ""
        return bool(binary_path) and getattr(bpy, "data", None) is not None
    except ImportError:
        return False


def _load_and_validate_baseline(compare_path: Path) -> dict[str, Any]:
    """比較元 JSON を読み込み、形式と必須スキーマを検証する。"""
    if not compare_path.exists():
        raise FileNotFoundError(f"比較対象 JSON が見つかりません: {compare_path}")
    try:
        raw_text = compare_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"比較対象 JSON の読み込みに失敗しました ({compare_path}): {exc}") from exc

    if not isinstance(data, dict) or "scenarios" not in data or not isinstance(data["scenarios"], dict):
        raise ValueError(
            f"比較対象 JSON の形式が不正です（'scenarios' オブジェクトが必要です）: {compare_path}"
        )
    return data


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pattern_Hood 専用 性能・自己衝突品質ベンチマーク CLI"
    )
    parser.add_argument(
        "--scenario",
        choices=["drape", "flip", "both"],
        default="both",
        help="実行するシナリオ (drape / flip / both、既定: both)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="自然ドレープシナリオのフレーム数 (既定: 60)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="壁時計・GPU集計から除外するウォームアップフレーム数 (既定: 5)",
    )
    parser.add_argument(
        "--blender",
        type=str,
        default=None,
        help="使用する Blender バージョン (例: 5.2, latest-lts)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="計測結果 JSON の書き出し先パス",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="比較対象とするベースライン JSON パス",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="指定時のみ Chrome Tracing JSON を保存するディレクトリ",
    )
    return parser.parse_args(list(argv))


def _get_git_sha() -> str:
    """現在の Git コミット SHA (短縮形) を取得する。"""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _compute_wall_stats(wall_ms: Sequence[float]) -> dict[str, Any]:
    """壁時計フレーム時間配列から代表統計量を算出する。"""
    if not wall_ms:
        return {
            "sample_frames": 0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "fps": 0.0,
        }
    arr = np.asarray(wall_ms, dtype=np.float64)
    avg_ms = float(np.mean(arr))
    return {
        "sample_frames": int(len(arr)),
        "avg_ms": avg_ms,
        "p50_ms": float(np.percentile(arr, 50.0)),
        "p95_ms": float(np.percentile(arr, 95.0)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "fps": (1000.0 / avg_ms) if avg_ms > 0.0 else 0.0,
    }


def _run_scenario_in_blender(
    *,
    scenario_name: str,
    frames: int,
    warmup: int,
    trace_dir: Path | None,
) -> dict[str, Any]:
    """単一シナリオをクリーンな空シーンから step_cloth_scene 経由で実行・計測する。"""
    import bpy
    import taremin_cloth
    from taremin_cloth.engine import clear_collider_cache, clear_simulators
    from taremin_cloth.engine.runner import (
        save_profile_traces,
        set_profiling_enabled_all,
        take_profiles,
    )
    from tests.fixtures.hood_eval_utils import (
        apply_hood_back_flip_perturbation,
        collect_gpu_profile_summary,
        evaluate_hood_frame_metrics,
        extract_hood_eval_geometry,
        slice_mesh_positions,
    )
    from tools.setup_tailored_patterns import setup_tailored_scene

    dt = 1.0 / 60.0
    taremin_cloth.register()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    clear_simulators()
    clear_collider_cache()

    try:
        setup_res = setup_tailored_scene(
            active_garment="Pattern_Hood",
            garment_names=("Pattern_Hood",),
        )
        hood_obj = setup_res["garments"]["Pattern_Hood"]
        geom = extract_hood_eval_geometry(hood_obj)
        n_mesh_verts = geom["n_mesh_verts"]
        scene = bpy.context.scene

        prof_enabled = False
        prof_checked = False
        last_raw_entries: list[tuple[str, float]] = []
        raw_profile_frames: list[list[tuple[str, float]]] = []
        unique_profile_frames: list[list[tuple[str, float]]] = []

        def _capture_profile(*, frame_idx: int) -> None:
            nonlocal prof_enabled, prof_checked, last_raw_entries
            if not prof_checked:
                prof_map = set_profiling_enabled_all(enable=True)
                prof_enabled = bool(prof_map.get(hood_obj.name, False))
                prof_checked = True
                return
            if not prof_enabled:
                return
            entries = [(str(k), float(v)) for k, v in take_profiles().get(hood_obj.name, [])]
            if frame_idx >= warmup and entries:
                raw_profile_frames.append(entries)
                if entries != last_raw_entries:
                    unique_profile_frames.append(entries)
            if entries:
                last_raw_entries = entries

        def _step_hood() -> tuple[Any, np.ndarray, float]:
            t0 = time.perf_counter()
            res = taremin_cloth.step_cloth_scene(
                scene,
                dt=dt,
                target_objs=[hood_obj],
                update_mesh=True,
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            sim, coords = res[hood_obj.name]
            return sim, coords, wall_ms

        wall_ms_all: list[float] = []
        prev_pos: np.ndarray | None = None
        max_strain = 0.0
        max_speed = 0.0
        checkpoints: dict[str, dict[str, Any]] = {}

        if scenario_name == "drape":
            drape_count = max(1, int(frames))
            check_set = {drape_count - 1}
            if drape_count >= 30:
                check_set.add(29)

            for f_idx in range(drape_count):
                sim, coords, wall_ms = _step_hood()
                wall_ms_all.append(wall_ms)
                _capture_profile(frame_idx=f_idx)

                pos = slice_mesh_positions(coords, n_mesh_verts=n_mesh_verts)
                m = evaluate_hood_frame_metrics(
                    positions=pos,
                    initial_positions=geom["initial_positions"],
                    edges=geom["edges"],
                    faces=geom["faces"],
                    hood_mask=geom["hood_mask"],
                    bodice_mask=geom["bodice_mask"],
                    prev_positions=prev_pos,
                    dt=dt,
                    left_hood_indices=geom["left_hood_indices"],
                    right_hood_indices=geom["right_hood_indices"],
                    seam_hop_dist=geom["seam_hop_dist"],
                    check_intersections=(f_idx in check_set),
                )
                max_strain = max(max_strain, float(m["max_strain"]))
                if m["max_speed"] is not None and np.isfinite(m["max_speed"]):
                    max_speed = max(max_speed, float(m["max_speed"]))
                if f_idx == 29:
                    checkpoints["F30"] = m
                if f_idx == drape_count - 1:
                    checkpoints["final"] = m
                prev_pos = pos

        elif scenario_name == "flip":
            pre_frames = 30
            sim = None
            global_frame = 0
            for f_idx in range(pre_frames):
                sim, coords, wall_ms = _step_hood()
                wall_ms_all.append(wall_ms)
                _capture_profile(frame_idx=global_frame)
                global_frame += 1

                pos = slice_mesh_positions(coords, n_mesh_verts=n_mesh_verts)
                m = evaluate_hood_frame_metrics(
                    positions=pos,
                    initial_positions=geom["initial_positions"],
                    edges=geom["edges"],
                    faces=geom["faces"],
                    hood_mask=geom["hood_mask"],
                    bodice_mask=geom["bodice_mask"],
                    prev_positions=prev_pos,
                    dt=dt,
                    left_hood_indices=geom["left_hood_indices"],
                    right_hood_indices=geom["right_hood_indices"],
                    seam_hop_dist=geom["seam_hop_dist"],
                    check_intersections=(f_idx == pre_frames - 1),
                )
                max_strain = max(max_strain, float(m["max_strain"]))
                if m["max_speed"] is not None and np.isfinite(m["max_speed"]):
                    max_speed = max(max_speed, float(m["max_speed"]))
                if f_idx == pre_frames - 1:
                    checkpoints["pre_drape"] = m
                prev_pos = pos

            def _step_for_flip() -> np.ndarray:
                nonlocal global_frame
                _s, c, _w = _step_hood()
                _capture_profile(frame_idx=global_frame)
                global_frame += 1
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
                dt=dt,
            )
            for rec in flip_records:
                wall_ms_all.append(float(rec["wall_ms"]))
                rm = rec["metrics"]
                if rm is not None:
                    max_strain = max(max_strain, float(rm["max_strain"]))
                    if rm["max_speed"] is not None and np.isfinite(rm["max_speed"]):
                        max_speed = max(max_speed, float(rm["max_speed"]))

            checkpoints["hold_end"] = [r for r in flip_records if r["phase"] == "hold"][-1]["metrics"]
            checkpoints["final"] = [r for r in flip_records if r["phase"] == "settle"][-1]["metrics"]

        else:
            raise ValueError(f"未対応のシナリオ名です: {scenario_name!r}")

        # wgpu-profiler 非同期遅延回収のため最終フレーム後に保留結果を回収
        if prof_enabled:
            for _ in range(4):
                tail_entries = [(str(k), float(v)) for k, v in take_profiles().get(hood_obj.name, [])]
                if tail_entries and tail_entries != last_raw_entries:
                    unique_profile_frames.append(tail_entries)
                    raw_profile_frames.append(tail_entries)
                    last_raw_entries = tail_entries

            if trace_dir is not None:
                save_profile_traces(str(trace_dir / scenario_name))

        post_warmup_wall = wall_ms_all[warmup:] if len(wall_ms_all) > warmup else wall_ms_all
        wall_stats = _compute_wall_stats(post_warmup_wall)

        profile_samples = unique_profile_frames if unique_profile_frames else raw_profile_frames
        if prof_enabled:
            gpu_summary = collect_gpu_profile_summary(profile_samples, skip_first_frame=False)
            gpu_summary["note"] = "OK" if gpu_summary["available"] else "GPU profile unavailable (no completed samples)"
        else:
            gpu_summary = {
                "available": False,
                "note": "GPU profile unavailable (TIMESTAMP_QUERY unsupported)",
                "sample_frames": 0,
                "net_ms_avg": 0.0,
                "flat_ms_avg": 0.0,
                "self_collision_outer_ms_avg": 0.0,
                "passes": {},
            }

        final_m = dict(checkpoints["final"])
        final_m["peak_max_strain"] = max_strain
        final_m["peak_max_speed"] = max_speed

        return {
            "total_frames": len(wall_ms_all),
            "warmup_frames": warmup,
            "wall_ms": wall_stats,
            "gpu_profile": gpu_summary,
            "final_metrics": final_m,
            "checkpoints": checkpoints,
        }
    finally:
        clear_simulators()
        clear_collider_cache()
        bpy.ops.wm.read_factory_settings(use_empty=True)


def _print_report(report: dict[str, Any]) -> None:
    """計測結果を人間可読なレポート形式で標準出力へ表示する。"""
    meta = report["metadata"]
    print("=" * 78)
    print("Pattern_Hood 性能・品質定点観測ベンチマーク")
    print("=" * 78)
    print(f"日時 (UTC)       : {meta['timestamp_utc']}")
    print(f"Git コミット     : {meta['git_sha']}")
    print(f"Blender          : {meta['blender_version']}")
    print(f"GPU デバイス     : {meta['gpu_device']} (software_gpu={meta['is_software_gpu']})")
    print(
        f"布メッシュ       : Pattern_Hood ({meta['cloth_mesh_vertices']:,} メッシュ頂点 / "
        f"{meta['cloth_vertices']:,} 総粒子 / {meta['cloth_faces']:,} 三角形面)"
    )

    for s_name, s_data in report["scenarios"].items():
        w = s_data["wall_ms"]
        g = s_data["gpu_profile"]
        fm = s_data["final_metrics"]
        print("\n" + "-" * 78)
        print(f"[シナリオ: {s_name}] total_frames={s_data['total_frames']}, warmup={s_data['warmup_frames']}")
        print(
            f"  壁時計 (warmup除外 {w['sample_frames']}F): "
            f"avg={w['avg_ms']:.2f}ms | p50={w['p50_ms']:.2f}ms | "
            f"p95={w['p95_ms']:.2f}ms | max={w['max_ms']:.2f}ms | FPS={w['fps']:.2f}"
        )
        print(
            f"  最終品質: cross_isect={fm['cross_layer_intersections']} | "
            f"hood_isect={fm['hood_self_intersections']} | total_isect={fm['total_intersections']} | "
            f"peak_strain={fm['peak_max_strain']:.4f} | sym_mean={fm['symmetry_x_mean_error']*1000:.2f}mm | "
            f"peak_speed={fm['peak_max_speed']:.3f}m/s | intrusion={fm['back_bodice_intrusion_count']}"
        )
        if g["available"]:
            print(
                f"  GPU プロファイル ({g['sample_frames']} サンプル): "
                f"net_ms_avg={g['net_ms_avg']:.3f}ms | "
                f"self_collision_Outer={g['self_collision_outer_ms_avg']:.3f}ms"
            )
            print(f"    {'pass':<24} {'avg_ms':>10} {'share':>8}  {'type'}")
            for p_name, p_info in list(g["passes"].items())[:12]:
                p_type = "parent" if p_info["is_parent"] else "leaf"
                print(
                    f"    {p_name:<24} {p_info['avg_ms']:10.3f} {p_info['share_pct']:7.1f}%  {p_type}"
                )
        else:
            print(f"  GPU プロファイル: {g['note']}")
    print("=" * 78)


def _format_delta(cur: float, base: float, *, unit: str = "") -> str:
    diff = cur - base
    if abs(base) > 1e-9:
        pct = (diff / base) * 100.0
        return f"{diff:+.3f}{unit} ({pct:+.1f}%)"
    return f"{diff:+.3f}{unit} (n/a)"


def _print_comparison(*, current: dict[str, Any], baseline: dict[str, Any]) -> None:
    """現在計測結果とベースライン JSON の比較テーブルを出力する。"""
    print("\n" + "=" * 78)
    print("ベースライン比較 (--compare)")
    print("=" * 78)
    base_scenarios = baseline.get("scenarios", {})
    cur_scenarios = current.get("scenarios", {})
    common = [name for name in cur_scenarios if name in base_scenarios]
    if not common:
        print("比較可能な共通シナリオが見つかりませんでした。")
        return

    for name in common:
        b = base_scenarios[name]
        c = cur_scenarios[name]
        bw, cw = b["wall_ms"], c["wall_ms"]
        bg, cg = b["gpu_profile"], c["gpu_profile"]
        bf, cf = b["final_metrics"], c["final_metrics"]
        print(f"\n[シナリオ: {name}]")
        print(f"  {'指標':<28} {'Baseline':>12} {'Current':>12} {'Delta':>22}")
        print("  " + "-" * 74)
        print(
            f"  {'wall avg_ms':<28} {bw['avg_ms']:12.3f} {cw['avg_ms']:12.3f} "
            f"{_format_delta(cw['avg_ms'], bw['avg_ms'], unit='ms'):>22}"
        )
        print(
            f"  {'wall p95_ms':<28} {bw['p95_ms']:12.3f} {cw['p95_ms']:12.3f} "
            f"{_format_delta(cw['p95_ms'], bw['p95_ms'], unit='ms'):>22}"
        )
        print(
            f"  {'FPS':<28} {bw['fps']:12.2f} {cw['fps']:12.2f} "
            f"{_format_delta(cw['fps'], bw['fps']):>22}"
        )
        if bg.get("available") and cg.get("available"):
            print(
                f"  {'GPU net_ms_avg':<28} {bg['net_ms_avg']:12.3f} {cg['net_ms_avg']:12.3f} "
                f"{_format_delta(cg['net_ms_avg'], bg['net_ms_avg'], unit='ms'):>22}"
            )
            print(
                f"  {'self_collision_Outer_ms':<28} {bg['self_collision_outer_ms_avg']:12.3f} "
                f"{cg['self_collision_outer_ms_avg']:12.3f} "
                f"{_format_delta(cg['self_collision_outer_ms_avg'], bg['self_collision_outer_ms_avg'], unit='ms'):>22}"
            )
            all_passes = list(dict.fromkeys([*cg.get("passes", {}).keys(), *bg.get("passes", {}).keys()]))
            for p_name in all_passes[:8]:
                bp = bg.get("passes", {}).get(p_name, {}).get("avg_ms", 0.0)
                cp = cg.get("passes", {}).get(p_name, {}).get("avg_ms", 0.0)
                print(
                    f"  {('pass:' + p_name):<28} {bp:12.3f} {cp:12.3f} "
                    f"{_format_delta(cp, bp, unit='ms'):>22}"
                )
        for q_key in ("cross_layer_intersections", "total_intersections", "peak_max_strain", "symmetry_x_mean_error"):
            bv = float(bf.get(q_key, 0.0) or 0.0)
            cv = float(cf.get(q_key, 0.0) or 0.0)
            print(
                f"  {q_key:<28} {bv:12.4f} {cv:12.4f} "
                f"{_format_delta(cv, bv):>22}"
            )
    print("=" * 78)


def _run_inside_blender(cli_argv: Sequence[str]) -> int:
    """Blender プロセス内部でベンチマークを実行する。"""
    import bpy
    import taremin_cloth_core
    from tests import is_software_gpu
    from tools.setup_tailored_patterns import setup_tailored_scene

    args = _parse_args(cli_argv)
    baseline_data = None
    if args.compare is not None:
        baseline_data = _load_and_validate_baseline(args.compare)

    # メッシュ統計取得用メタ情報
    import taremin_cloth
    from taremin_cloth.engine import clear_collider_cache, clear_simulators
    from taremin_cloth.utils.mesh_extract import extract_cloth_mesh_data

    taremin_cloth.register()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_meta = setup_tailored_scene(
        active_garment="Pattern_Hood",
        garment_names=("Pattern_Hood",),
    )
    hood_obj = setup_meta["garments"]["Pattern_Hood"]
    cloth_data = extract_cloth_mesh_data(hood_obj, apply_world_matrix=True)
    cloth_mesh_vertices = len(hood_obj.data.vertices)
    cloth_vertices = len(cloth_data.positions)
    cloth_faces = len(cloth_data.faces) if cloth_data.faces is not None else 0
    clear_simulators()
    clear_collider_cache()
    bpy.ops.wm.read_factory_settings(use_empty=True)

    scenario_list = ["drape", "flip"] if args.scenario == "both" else [args.scenario]
    scenarios_out: dict[str, Any] = {}
    for s_name in scenario_list:
        scenarios_out[s_name] = _run_scenario_in_blender(
            scenario_name=s_name,
            frames=args.frames,
            warmup=args.warmup,
            trace_dir=args.trace_dir,
        )

    report = {
        "schema_version": 1,
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": _get_git_sha(),
            "blender_version": bpy.app.version_string,
            "gpu_device": taremin_cloth_core.get_gpu_device_name(),
            "is_software_gpu": is_software_gpu(),
            "cloth_mesh_vertices": cloth_mesh_vertices,
            "cloth_vertices": cloth_vertices,
            "cloth_faces": cloth_faces,
            "warmup_frames": args.warmup,
        },
        "scenarios": scenarios_out,
    }

    _print_report(report)

    if args.json is not None:
        out_path = args.json.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[benchmark_hood] JSON を保存しました: {out_path}")

    if baseline_data is not None:
        _print_comparison(current=report, baseline=baseline_data)

    return 0


def _run_from_system_python(cli_argv: Sequence[str]) -> int:
    """通常 Python から resolve_blender() でヘッドレス Blender を起動してベンチマークを実行する。"""
    from tools.blender_manager import find_installed_blenders, resolve_blender

    args = _parse_args(cli_argv)
    if args.compare is not None:
        _load_and_validate_baseline(args.compare)

    if args.blender is not None:
        blender_exe = resolve_blender(args.blender)
    else:
        # SiroinoSotai.blend (BONE_SDF 本番構成) は Blender 5.0+ が必要なため、5.x がキャッシュ済みなら優先使用する
        installed_5x = [path for ver, path in find_installed_blenders() if ver and ver[0] >= 5]
        blender_exe = installed_5x[0] if installed_5x else resolve_blender(None)
    forwarded = [
        "--scenario",
        args.scenario,
        "--frames",
        str(args.frames),
        "--warmup",
        str(args.warmup),
    ]
    if args.json is not None:
        forwarded.extend(["--json", str(args.json.resolve())])
    if args.compare is not None:
        forwarded.extend(["--compare", str(args.compare.resolve())])
    if args.trace_dir is not None:
        forwarded.extend(["--trace-dir", str(args.trace_dir.resolve())])

    cmd = [
        str(blender_exe),
        "-b",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        *forwarded,
    ]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(res.returncode)


def main() -> None:
    if _is_running_inside_blender_app():
        custom_argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
        try:
            code = _run_inside_blender(custom_argv)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[benchmark_hood] エラー: {exc}", file=sys.stderr)
            code = 1
        raise SystemExit(code)
    else:
        try:
            code = _run_from_system_python(sys.argv[1:])
        except (FileNotFoundError, ValueError) as exc:
            print(f"[benchmark_hood] エラー: {exc}", file=sys.stderr)
            code = 1
        raise SystemExit(code)


if __name__ == "__main__":
    main()
