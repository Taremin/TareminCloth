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
from typing import Any, Dict, List, Optional
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from . import analysis
    from . import mesh_renderer
    from . import replayer
except ImportError:
    from taremin_cloth import analysis
    from taremin_cloth import mesh_renderer
    from taremin_cloth import replayer


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


def parse_frame_spec(spec: str, max_frame: Optional[int] = None) -> List[int]:
    """
    フレーム指定文字列をフレーム番号リストにパースします。
    例: "71", "0-100", "1,5,10-20", "all"
    """
    spec = spec.strip().lower()
    if spec == "all":
        if max_frame is None:
            return []
        return list(range(max_frame + 1))

    frames = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            parts = part.split("-", 1)
            try:
                s_val = int(parts[0])
                e_val = int(parts[1])
                frames.update(range(s_val, e_val + 1))
            except ValueError:
                pass
        else:
            try:
                frames.add(int(part))
            except ValueError:
                pass
    return sorted(frames)


def cmd_render(args: argparse.Namespace) -> None:
    """指定フレームまたはフレーム範囲の画像をレンダリング（単一/連番/APNG/GIF対応、Blender不要）"""
    log_path = args.log_file
    out_path = args.output

    meta = replayer.read_metadata(log_path)
    faces = np.array(meta.get("faces", []), dtype=np.uint32)

    sewing_springs = None
    if not getattr(args, "no_sewing", False):
        sew_raw = meta.get("sewing_springs")
        if sew_raw:
            sewing_springs = np.array(sew_raw, dtype=np.uint32)

    # ログ内の全フレーム情報をストリーミング走査してインデックス化
    available_frames = {}
    for f in replayer.iter_frames(log_path):
        available_frames[f["frame_index"]] = f

    if not available_frames:
        print("エラー: ログ内にフレームデータが存在しません。")
        sys.exit(1)

    max_avail = max(available_frames.keys())

    # 対象フレームの解決
    frames_arg = getattr(args, "frames", None)
    frame_arg = getattr(args, "frame", None)

    if frames_arg:
        target_frames = parse_frame_spec(frames_arg, max_frame=max_avail)
    elif frame_arg is not None:
        target_frames = [frame_arg]
    else:
        target_frames = [0]

    step = getattr(args, "step", 1) or 1
    if step > 1:
        target_frames = target_frames[::step]

    valid_frames = [f_idx for f_idx in target_frames if f_idx in available_frames]
    if not valid_frames:
        print(f"エラー: 指定されたフレーム {target_frames[:5]}... がログに見つかりません（利用可能: 0〜{max_avail}）。")
        sys.exit(1)

    ext = os.path.splitext(out_path)[1].lower()
    fmt = (getattr(args, "format", "") or "").lower()
    is_animation = fmt in ("apng", "gif") or ext in (".gif",) or (fmt == "png" and len(valid_frames) > 1 and "%" not in out_path)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    wire_width = getattr(args, "wire_width", 1.0)
    draw_wire = not getattr(args, "no_wireframe", False)
    no_colliders = getattr(args, "no_colliders", False)
    color_by = getattr(args, "color_by", "none")
    vel_max = getattr(args, "vel_max", None)
    show_sdf_bbox = getattr(args, "show_sdf_bbox", False)

    def prepare_frame_data(frame_obj):
        p = np.array(frame_obj["positions"], dtype=np.float32).reshape((-1, 3))

        vc = None
        if color_by == "velocity":
            vels = frame_obj.get("velocities")
            if vels:
                vc = mesh_renderer.compute_velocity_colors(vels, vmax=vel_max)
        elif color_by == "strain":
            edg = meta.get("edges")
            rl = meta.get("initial_rest_lengths")
            if edg and rl:
                vc = mesh_renderer.compute_strain_colors(p, edg, rl)
        elif color_by == "normal":
            vc = mesh_renderer.compute_normal_colors(p, faces)

        c_tris = None
        if not no_colliders:
            c_tris = mesh_renderer.convert_colliders_to_mesh(frame_obj.get("colliders", []))

        lines = None
        if show_sdf_bbox:
            b_infos = meta.get("bone_infos", [])
            b_trans = frame_obj.get("bone_transforms", [])
            if b_infos and b_trans:
                lines = mesh_renderer.generate_sdf_bbox_lines(b_infos, b_trans)

        return p, c_tris, vc, lines

    if is_animation:
        # アニメーション (APNG / GIF) 生成
        fps = getattr(args, "fps", 30) or 30
        print(f"アニメーションレンダリング開始: {len(valid_frames)} フレーム -> {out_path} (FPS: {fps})")
        rendered_images = []
        for i, f_idx in enumerate(valid_frames):
            frame = available_frames[f_idx]
            pos, cols, vc, lines = prepare_frame_data(frame)

            img = mesh_renderer.render_mesh_to_image(
                pos, faces,
                sewing_springs=sewing_springs,
                mesh_colliders=cols,
                vertex_colors=vc,
                extra_lines=lines,
                width=args.width, height=args.height,
                draw_wireframe=draw_wire,
                wire_width=wire_width,
            )
            rendered_images.append(img)
            if (i + 1) % 10 == 0 or i + 1 == len(valid_frames):
                print(f"  進捗: {i + 1}/{len(valid_frames)} フレーム完了...")

        mesh_renderer.create_animation_file(rendered_images, out_path, fps=fps)
        print(f"アニメーション保存完了: {out_path}")
    elif len(valid_frames) == 1:
        # 単一フレーム保存
        f_idx = valid_frames[0]
        frame = available_frames[f_idx]
        pos, cols, vc, lines = prepare_frame_data(frame)

        _, red_count = mesh_renderer.render_scene_to_file(
            out_path, pos, faces,
            sewing_springs=sewing_springs,
            mesh_colliders=cols,
            vertex_colors=vc,
            extra_lines=lines,
            width=args.width, height=args.height,
            draw_wireframe=draw_wire,
            wire_width=wire_width,
        )
        print(f"レンダリング完了: Frame {f_idx} -> {out_path}")
        print(f"裏面（赤）ピクセル数: {red_count} (貫通・裏返り指標)")
    else:
        # 連番PNG保存
        pattern = out_path if "%" in out_path else os.path.join(out_path, "f_%04d.png")
        print(f"連番画像レンダリング開始: {len(valid_frames)} フレーム -> {pattern}")
        for i, f_idx in enumerate(valid_frames):
            frame = available_frames[f_idx]
            pos, cols, vc, lines = prepare_frame_data(frame)

            cur_path = pattern % f_idx if "%" in pattern else pattern.format(f_idx)
            os.makedirs(os.path.dirname(os.path.abspath(cur_path)) or ".", exist_ok=True)
            mesh_renderer.render_scene_to_file(
                cur_path, pos, faces,
                sewing_springs=sewing_springs,
                mesh_colliders=cols,
                vertex_colors=vc,
                extra_lines=lines,
                width=args.width, height=args.height,
                draw_wireframe=draw_wire,
                wire_width=wire_width,
            )
        print(f"連番画像保存完了: {len(valid_frames)} 枚")


def cmd_render_coloring(args: argparse.Namespace) -> None:
    """制約カラーリング（Welsh-Powellグラフ彩色）を可視化レンダリング"""
    log_path = args.log_file
    out_path = args.output
    c_type = args.type.lower()

    meta = replayer.read_metadata(log_path)
    num_vertices = meta.get("num_vertices", 0)
    faces = np.array(meta.get("faces", []), dtype=np.uint32)

    frame = replayer.get_frame(log_path, args.frame)
    if frame is None:
        # 見つからない場合は最初のフレームを使用
        for f in replayer.iter_frames(log_path):
            frame = f
            break

    if frame is None:
        print("エラー: ログ内に有効なフレームが見つかりません。")
        sys.exit(1)

    positions = np.array(frame["positions"], dtype=np.float32).reshape((-1, 3))

    if c_type == "sewing":
        edges = meta.get("sewing_springs", [])
        name = "Sewing Constraints"
    else:
        edges = meta.get("edges", [])
        name = "Distance Constraints"

    if not edges:
        print(f"警告: ログ内に {name} が存在しません。")
        sys.exit(1)

    edges_arr = np.array(edges, dtype=np.uint32)
    num_constraints = len(edges_arr)

    # Welsh-Powell グラフ彩色
    adj = [[] for _ in range(num_vertices)]
    for idx, (v0, v1) in enumerate(edges_arr):
        if v0 < num_vertices:
            adj[v0].append(idx)
        if v1 < num_vertices:
            adj[v1].append(idx)

    order = list(range(num_constraints))
    order.sort(key=lambda i: len(adj[edges_arr[i][0]]) + len(adj[edges_arr[i][1]]), reverse=True)

    colors = [-1] * num_constraints
    num_colors = 0
    for i in order:
        v0, v1 = edges_arr[i]
        used = set()
        for adj_i in adj[v0]:
            if adj_i != i and colors[adj_i] != -1:
                used.add(colors[adj_i])
        for adj_i in adj[v1]:
            if adj_i != i and colors[adj_i] != -1:
                used.add(colors[adj_i])
        c = 0
        while c in used:
            c += 1
        colors[i] = c
        if c >= num_colors:
            num_colors = c + 1

    print(f"=== {name} Welsh-Powell 彩色統計 ===")
    print(f"  総拘束数: {num_constraints}")
    print(f"  使用色数 (カラーグループ数): {num_colors}")

    # 各色の拘束数
    color_counts = [colors.count(c) for c in range(num_colors)]
    for c, cnt in enumerate(color_counts):
        print(f"    Color {c:2d}: {cnt:5d} constraints ({cnt / num_constraints * 100:5.1f}%)")

    # 鮮やかなカラーパレット (16色)
    palette = [
        [230, 25, 75],    # 赤
        [60, 180, 75],    # 緑
        [255, 225, 25],   # 黄
        [0, 130, 200],    # 青
        [245, 130, 48],   # オレンジ
        [145, 30, 180],   # 紫
        [70, 240, 240],   # シアン
        [240, 50, 230],   # マゼンタ
        [210, 245, 60],   # ライム
        [250, 190, 212],  # ピンク
        [0, 128, 128],    # ティール
        [220, 190, 255],  # ラベンダー
        [170, 110, 40],   # ブラウン
        [255, 250, 200],  # ベージュ
        [128, 0, 0],      # マルーン
        [170, 255, 195],  # ミント
    ]

    # メッシュを暗めのグレーで下地描画し、彩色エッジを描画
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    mesh_renderer.render_scene_to_file(
        out_path,
        positions,
        faces,
        sewing_springs=edges_arr if c_type == "sewing" else None,
        width=args.width,
        height=args.height,
        draw_wireframe=True,
        wire_width=1.2,
    )
    print(f"カラーリング可視化保存完了: {out_path}")


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


def cmd_diff(args: argparse.Namespace) -> None:
    """2つのシミュレーションログ間で頂点位置の差分・誤差を比較"""
    log1 = args.log1
    log2 = args.log2
    tol = args.tolerance  # mm

    meta1 = replayer.read_metadata(log1)
    meta2 = replayer.read_metadata(log2)

    print(f"==================================================")
    print(f" Simulation Log Comparison")
    print(f" Log 1: {os.path.basename(log1)} (Verts: {meta1.get('num_vertices')})")
    print(f" Log 2: {os.path.basename(log2)} (Verts: {meta2.get('num_vertices')})")
    print(f" Tolerance: {tol:.3f} mm")
    print(f"==================================================")

    frames1 = {f["frame_index"]: f for f in replayer.iter_frames(log1)}
    frames2 = {f["frame_index"]: f for f in replayer.iter_frames(log2)}

    common_frames = sorted(set(frames1.keys()) & set(frames2.keys()))
    if not common_frames:
        print("エラー: 共通するフレームが存在しません。")
        sys.exit(1)

    print(f"{'Frame':<8} | {'Max Diff (mm)':<15} | {'Mean Diff (mm)':<15} | {'Status':<10}")
    print("-" * 55)

    max_overall_diff = 0.0
    exceeded_count = 0

    for f_idx in common_frames:
        p1 = np.array(frames1[f_idx]["positions"], dtype=np.float32).reshape((-1, 3))
        p2 = np.array(frames2[f_idx]["positions"], dtype=np.float32).reshape((-1, 3))

        if p1.shape != p2.shape:
            print(f"エラー: 頂点形状が不一致です: {p1.shape} vs {p2.shape}")
            sys.exit(1)

        diffs = np.linalg.norm(p1 - p2, axis=1) * 1000.0  # mm
        max_d = float(np.max(diffs))
        mean_d = float(np.mean(diffs))

        status = "OK"
        if max_d > tol:
            status = "EXCEEDED"
            exceeded_count += 1

        if max_d > max_overall_diff:
            max_overall_diff = max_d

        print(f"F{f_idx:<7} | {max_d:12.4f} mm | {mean_d:12.4f} mm | {status}")

    print("-" * 55)
    print(f"比較結果: 全 {len(common_frames)} フレーム中 {exceeded_count} フレームで許容誤差 ({tol} mm) を超過")
    print(f"最大誤差: {max_overall_diff:.4f} mm")
    if exceeded_count == 0:
        print("[+] 両ログは完全に許容誤差内で一致しています。")
    else:
        print("[!] 挙動に有意な差分が検出されました。")


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
    p_render = subparsers.add_parser("render", help="指定フレームまたはフレーム範囲の画像をレンダリング（連番/APNG/GIF対応、Blender不要）")
    p_render.add_argument("log_file", help="対象ログファイル (.jsonl.gz)")
    p_render.add_argument("--frame", "-f", type=int, default=None, help="単一描画フレーム番号")
    p_render.add_argument("--frames", type=str, default=None, help="複数フレーム指定 (例: '0-50', '1,5,10', 'all')")
    p_render.add_argument("--step", type=int, default=1, help="フレーム間引きステップ")
    p_render.add_argument("--output", "-o", required=True, help="出力先ファイルパス (単一画像/APNG/GIF、または連番用パターン 'f_%%04d.png')")
    p_render.add_argument("--format", type=str, default=None, choices=["png", "apng", "gif"], help="出力フォーマット強制指定")
    p_render.add_argument("--fps", type=int, default=30, help="アニメーション再生FPS")
    p_render.add_argument("--width", type=int, default=800, help="画像幅")
    p_render.add_argument("--height", type=int, default=600, help="画像高さ")
    p_render.add_argument("--wire-width", type=float, default=1.0, help="ワイヤーフレーム線幅(px)")
    p_render.add_argument("--no-wireframe", action="store_true", help="ワイヤーフレームを非表示")
    p_render.add_argument("--no-sewing", action="store_true", help="縫合エッジを非表示")
    p_render.add_argument("--no-colliders", action="store_true", help="コライダーを非表示")
    p_render.add_argument("--color-by", type=str, default="none", choices=["none", "velocity", "strain", "normal"], help="頂点カラー表示モード (none: 白/赤, velocity: 速度ヒートマップ, strain: 歪みヒートマップ, normal: 法線カラーマップ)")
    p_render.add_argument("--show-sdf-bbox", action="store_true", help="ボーンSDFの有効範囲AABBボックスを3D表示")
    p_render.add_argument("--vel-max", type=float, default=None, help="速度ヒートマップの最大速度 (m/s)")
    p_render.set_defaults(func=cmd_render)

    # 5. render-coloring
    p_coloring = subparsers.add_parser("render-coloring", help="制約カラーリング（Welsh-Powellグラフ彩色）を可視化レンダリング")
    p_coloring.add_argument("log_file", help="対象ログファイル (.jsonl.gz)")
    p_coloring.add_argument("--type", type=str, default="distance", choices=["distance", "sewing"], help="彩色対象の拘束タイプ")
    p_coloring.add_argument("--frame", "-f", type=int, default=0, help="描画に使用するフレーム番号")
    p_coloring.add_argument("--output", "-o", required=True, help="出力先画像パス (.png)")
    p_coloring.add_argument("--width", type=int, default=800, help="画像幅")
    p_coloring.add_argument("--height", type=int, default=600, help="画像高さ")
    p_coloring.set_defaults(func=cmd_render_coloring)

    # 6. check-intersections
    p_check = subparsers.add_parser("check-intersections", help="指定フレームの三角形交差ペアを検出")
    p_check.add_argument("log_file", help="対象ログファイル (.jsonl.gz)")
    p_check.add_argument("--frame", "-f", type=int, required=True, help="検証フレーム番号")
    p_check.set_defaults(func=cmd_check_intersections)

    # 7. export-obj
    p_obj = subparsers.add_parser("export-obj", help="指定フレームをOBJ形式でエクスポート")
    p_obj.add_argument("log_file", help="対象ログファイル (.jsonl.gz)")
    p_obj.add_argument("--frame", "-f", type=int, required=True, help="エクスポートフレーム番号")
    p_obj.add_argument("--output", "-o", required=True, help="出力OBJパス")
    p_obj.set_defaults(func=cmd_export_obj)

    # 8. diff
    p_diff = subparsers.add_parser("diff", help="2つのログファイル間で頂点位置の差分・誤差を比較")
    p_diff.add_argument("log1", help="基準ログファイル (.jsonl.gz)")
    p_diff.add_argument("log2", help="比較対象ログファイル (.jsonl.gz)")
    p_diff.add_argument("--tolerance", "-t", type=float, default=1.0, help="許容変位閾値 (mm, デフォルト: 1.0)")
    p_diff.set_defaults(func=cmd_diff)


    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
