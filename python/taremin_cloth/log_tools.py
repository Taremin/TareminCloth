"""
ログ操作・解析・テストケース自動生成CLIツール (taremin_cloth.log_tools)
使用例:
  python -m taremin_cloth.log_tools inspect cloth_debug.jsonl.gz
  python -m taremin_cloth.log_tools slice cloth_debug.jsonl.gz --start 68 --end 74 --output issue_f71.jsonl.gz
  python -m taremin_cloth.log_tools make-test cloth_debug.jsonl.gz --frame 71 --lookback 3 --output tests/test_issue_f71.py
  python -m taremin_cloth.log_tools render cloth_debug.jsonl.gz --frame 71 --output f71.png
  python -m taremin_cloth.log_tools check-intersections cloth_debug.jsonl.gz --frame 71
  python -m taremin_cloth.log_tools export-obj cloth_debug.jsonl.gz --frame 71 --output f71.obj
"""

import argparse
import gzip
import json
import os
import sys
from typing import Any, Dict, List
import numpy as np

from . import analysis
from . import mesh_renderer
from . import replayer


def cmd_inspect(args: argparse.Namespace) -> None:
    """ログ全体の異常値（速度・変位・NaN・交差）をスキャンしてサマリー表示"""
    log_path = args.log_file
    meta = replayer.read_metadata(log_path)
    faces = np.array(meta.get("faces", []), dtype=np.uint32)

    print(f"==================================================")
    print(f" Log Inspection: {os.path.basename(log_path)}")
    print(f"==================================================")
    print(f" Object: {meta.get('object_name')} | Verts: {meta.get('num_vertices')} | Faces: {meta.get('num_faces')}")
    print(f" Gravity: {meta.get('gravity')} | Thickness: {meta.get('thickness')} | Damping: {meta.get('damping')}")
    print(f" SelfCollision: {meta.get('enable_self_collision')} (relief={meta.get('self_collision_relief_factor')})")
    print(f"--------------------------------------------------")

    prev_pos = None
    frame_count = 0
    anomalies = []
    max_overall_disp = 0.0
    max_disp_frame = 0

    for frame in replayer.iter_frames(log_path):
        frame_count += 1
        f_idx = frame["frame_index"]
        pos = np.array(frame["positions"], dtype=np.float32).reshape((-1, 3))
        stats = frame.get("stats", {})

        has_nan = stats.get("has_nan_or_inf", False)
        max_vel = stats.get("max_velocity", 0.0)

        # 変位の計算
        disp_mm = 0.0
        if prev_pos is not None:
            disps = np.linalg.norm(pos - prev_pos, axis=1) * 1000.0  # mm
            disp_mm = float(np.max(disps))
            if disp_mm > max_overall_disp:
                max_overall_disp = disp_mm
                max_disp_frame = f_idx

        # 異常フラグ
        is_spike = disp_mm > args.disp_threshold
        is_high_vel = max_vel > args.vel_threshold
        num_pins = len(frame.get("pins", []))
        num_cols = len(frame.get("colliders", []))

        if has_nan or is_spike or is_high_vel:
            flags = []
            if has_nan: flags.append("NaN/Inf")
            if is_spike: flags.append(f"DispSpike({disp_mm:.1f}mm)")
            if is_high_vel: flags.append(f"HighVel({max_vel:.2f}m/s)")
            anomalies.append((f_idx, ", ".join(flags), num_pins, num_cols))

        prev_pos = pos

    print(f" Total Frames: {frame_count}")
    print(f" Max Displacement: {max_overall_disp:.1f} mm (at Frame {max_disp_frame})")

    if anomalies:
        print(f"\n [!] Anomalies Detected in {len(anomalies)} frames:")
        for f_idx, flag_str, pins, cols in anomalies[:30]:
            print(f"   - Frame {f_idx:4d}: {flag_str} (Pins: {pins}, Colliders: {cols})")
        if len(anomalies) > 30:
            print(f"   ... and {len(anomalies) - 30} more frames")
    else:
        print(" [OK] No major anomalies (NaN or threshold spikes) detected.")


def cmd_slice(args: argparse.Namespace) -> None:
    """指定フレーム区間を切り出して新しい極小ログを作成"""
    log_path = args.log_file
    out_path = args.output
    start_f = args.start
    end_f = args.end

    meta = replayer.read_metadata(log_path)
    sliced_frames = []

    for f in replayer.iter_frames(log_path):
        idx = f["frame_index"]
        if start_f <= idx <= end_f:
            sliced_frames.append(f)

    if not sliced_frames:
        print(f"エラー: 指定された範囲 (Frame {start_f}〜{end_f}) にフレームが見つかりませんでした。")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as out_f:
        meta_rec = {
            "record_type": "metadata",
            "version": 2,
            "created_at": "sliced",
            "metadata": meta,
        }
        out_f.write(json.dumps(meta_rec) + "\n")
        for f in sliced_frames:
            out_f.write(json.dumps(f) + "\n")

    print(f"スライス完了: Frame {start_f}〜{end_f} ({len(sliced_frames)} frames) -> {out_path}")


def cmd_make_test(args: argparse.Namespace) -> None:
    """問題直前フレームから始まる自己完結型 unittest スクリプトを自動生成"""
    log_path = args.log_file
    target_f = args.frame
    lookback = args.lookback
    start_f = max(0, target_f - lookback)
    out_path = args.output

    # まず対象範囲のフレームをスライスしたログを生成（テストファイルと同じディレクトリに保存）
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(out_path))[0]
    slice_log_name = f"{base_name}_data.jsonl.gz"
    slice_log_path = os.path.join(out_dir, slice_log_name)

    # スライス作成
    args_slice = argparse.Namespace(
        log_file=log_path,
        start=start_f,
        end=target_f + 2,
        output=slice_log_path
    )
    cmd_slice(args_slice)

    # unittest スクリプトの生成
    template = f'''"""
Auto-generated standalone reproduction test for Frame {target_f}
Generated by: python -m taremin_cloth.log_tools make-test
"""

import os
import unittest
import numpy as np

import taremin_cloth_core
from taremin_cloth.replayer import ClothReplayer
from taremin_cloth.analysis import find_triangle_intersections, find_displacement_spikes
from taremin_cloth.mesh_renderer import render_mesh_to_image, count_red_pixels

class TestIssueFrame{target_f}(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(os.path.dirname(__file__), "{slice_log_name}")
        cls.replayer = ClothReplayer(cls.data_path)

    def test_reproduce_and_verify(self):
        """Frame {start_f} から {target_f} を再実行し、挙動と貫通を検証"""
        results = self.replayer.replay_range(
            start_frame_idx={start_f},
            end_frame_idx={target_f},
            compare=True
        )
        self.assertTrue(len(results) > 0)

        # ターゲットフレームの結果
        target_res = results[-1]
        sim_pos = target_res["positions"]
        faces = self.replayer.faces

        # 1. ログ実測値との完全一致検証 (同一マシン・決定論的)
        print(f"Frame {target_f}: Max difference from log = {{target_res['max_diff']:.6f}} m")
        self.assertLess(target_res["max_diff"], 1e-3, "シミュレーション再現値が実ログと乖離しています")

        # 2. 幾何交差判定
        intersections = find_triangle_intersections(sim_pos, faces)
        print(f"Frame {target_f}: Self-intersections = {{len(intersections)}}")

        # 3. ソフトウェアレンダリングによる裏面（赤）判定
        img = render_mesh_to_image(sim_pos, faces, width=400, height=300)
        red_pixels = count_red_pixels(img)
        print(f"Frame {target_f}: Backface (red) pixels = {{red_pixels}}")

if __name__ == "__main__":
    unittest.main()
'''
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(template)

    print(f"テストケース生成完了: {out_path}")
    print(f"実行コマンド: python -m unittest {out_path}")


def cmd_render(args: argparse.Namespace) -> None:
    """指定フレームの貫通可視化画像をレンダリング（Blender不要）"""
    log_path = args.log_file
    target_f = args.frame
    out_path = args.output

    meta = replayer.read_metadata(log_path)
    faces = np.array(meta.get("faces", []), dtype=np.uint32)
    frame = replayer.get_frame(log_path, target_f)

    if frame is None:
        print(f"エラー: Frame {target_f} がログに見つかりません。")
        sys.exit(1)

    pos = np.array(frame["positions"], dtype=np.float32).reshape((-1, 3))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    _, red_count = mesh_renderer.render_mesh_to_file(
        out_path, pos, faces, width=args.width, height=args.height
    )

    print(f"レンダリング完了: Frame {target_f} -> {out_path}")
    print(f"裏面（赤）ピクセル数: {red_count} (貫通・裏返り指標)")


def cmd_check_intersections(args: argparse.Namespace) -> None:
    """指定フレームの三角形交差ペアを検出"""
    log_path = args.log_file
    target_f = args.frame

    meta = replayer.read_metadata(log_path)
    faces = np.array(meta.get("faces", []), dtype=np.uint32)
    frame = replayer.get_frame(log_path, target_f)

    if frame is None:
        print(f"エラー: Frame {target_f} がログに見つかりません。")
        sys.exit(1)

    pos = np.array(frame["positions"], dtype=np.float32).reshape((-1, 3))
    pairs = analysis.find_triangle_intersections(pos, faces)

    print(f"Frame {target_f}: 自己交差三角形ペア数 = {len(pairs)}")
    if pairs:
        for p in pairs[:20]:
            print(f"   Intersection: Face {p[0]} <--> Face {p[1]}")
        if len(pairs) > 20:
            print(f"   ... and {len(pairs) - 20} more pairs")


def cmd_export_obj(args: argparse.Namespace) -> None:
    """指定フレームをOBJエクスポート"""
    log_path = args.log_file
    target_f = args.frame
    out_path = args.output

    meta = replayer.read_metadata(log_path)
    faces = np.array(meta.get("faces", []), dtype=np.uint32)
    frame = replayer.get_frame(log_path, target_f)

    if frame is None:
        print(f"エラー: Frame {target_f} がログに見つかりません。")
        sys.exit(1)

    pos = np.array(frame["positions"], dtype=np.float32).reshape((-1, 3))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    analysis.export_obj(out_path, pos, faces)
    print(f"OBJエクスポート完了: Frame {target_f} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m taremin_cloth.log_tools",
        description="taremin_cloth デバッグログ操作・解析・可視化CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. inspect
    p_inspect = subparsers.add_parser("inspect", help="ログ全体の異常値を走査")
    p_inspect.add_argument("log_file", help="対象ログファイル (.jsonl.gz)")
    p_inspect.add_argument("--disp-threshold", type=float, default=5.0, help="変位スパイク閾値 (mm)")
    p_inspect.add_argument("--vel-threshold", type=float, default=10.0, help="速度閾値 (m/s)")
    p_inspect.set_defaults(func=cmd_inspect)

    # 2. slice
    p_slice = subparsers.add_parser("slice", help="指定フレーム区間を切り出して新しい極小ログを作成")
    p_slice.add_argument("log_file", help="元ログファイル (.jsonl.gz)")
    p_slice.add_argument("--start", type=int, required=True, help="開始フレーム番号")
    p_slice.add_argument("--end", type=int, required=True, help="終了フレーム番号")
    p_slice.add_argument("--output", "-o", required=True, help="出力先ファイル (.jsonl.gz)")
    p_slice.set_defaults(func=cmd_slice)

    # 3. make-test
    p_make_test = subparsers.add_parser("make-test", help="自己完結型 unittest テストスクリプトを自動生成")
    p_make_test.add_argument("log_file", help="元ログファイル (.jsonl.gz)")
    p_make_test.add_argument("--frame", "-f", type=int, required=True, help="再現したい問題フレーム番号")
    p_make_test.add_argument("--lookback", type=int, default=3, help="何フレーム前から初期化するか")
    p_make_test.add_argument("--output", "-o", required=True, help="出力先Pythonテストスクリプトパス")
    p_make_test.set_defaults(func=cmd_make_test)

    # 4. render
    p_render = subparsers.add_parser("render", help="指定フレームの貫通可視化画像をレンダリング（Blender不要）")
    p_render.add_argument("log_file", help="対象ログファイル (.jsonl.gz)")
    p_render.add_argument("--frame", "-f", type=int, required=True, help="描画フレーム番号")
    p_render.add_argument("--output", "-o", required=True, help="出力先画像パス (.png)")
    p_render.add_argument("--width", type=int, default=800, help="画像幅")
    p_render.add_argument("--height", type=int, default=600, help="画像高さ")
    p_render.set_defaults(func=cmd_render)

    # 5. check-intersections
    p_check = subparsers.add_parser("check-intersections", help="指定フレームの三角形交差ペアを検出")
    p_check.add_argument("log_file", help="対象ログファイル (.jsonl.gz)")
    p_check.add_argument("--frame", "-f", type=int, required=True, help="検証フレーム番号")
    p_check.set_defaults(func=cmd_check_intersections)

    # 6. export-obj
    p_obj = subparsers.add_parser("export-obj", help="指定フレームをOBJ形式でエクスポート")
    p_obj.add_argument("log_file", help="対象ログファイル (.jsonl.gz)")
    p_obj.add_argument("--frame", "-f", type=int, required=True, help="エクスポートフレーム番号")
    p_obj.add_argument("--output", "-o", required=True, help="出力OBJパス")
    p_obj.set_defaults(func=cmd_export_obj)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
