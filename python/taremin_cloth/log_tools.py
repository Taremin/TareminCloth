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
    cfg = meta.get("config") if isinstance(meta.get("config"), dict) else None
    if cfg is not None:
        print(f" SimConfig: coupled={cfg.get('coupled_mode')} interval={cfg.get('substep_interval')} "
              f"pair_cache={cfg.get('enable_pair_cache')} max_pairs={cfg.get('pair_max_pairs')}")
    else:
        print(f" SimConfig: (旧ログ: configなし、個別フィールドのみ)")
    print(f"--------------------------------------------------")

    prev_pos = None
    prev_hash = None
    frame_count = 0
    full_count = 0
    anomalies = []
    max_overall_disp = 0.0
    max_disp_frame = 0

    for frame in replayer.iter_frames(log_path):
        frame_count += 1
        f_idx = frame["frame_index"]
        stats = frame.get("stats", {})
        has_full = replayer.frame_has_positions(frame)
        if has_full:
            full_count += 1
            pos = np.array(frame["positions"], dtype=np.float32).reshape((-1, 3))
        else:
            pos = None

        has_nan = stats.get("has_nan_or_inf", False)
        max_vel = stats.get("max_velocity", 0.0)
        vt_sat = bool(stats.get("vt_saturated", False))
        vt_c = int(stats.get("vt_count", 0))
        ee_c = int(stats.get("ee_count", 0))
        strain = float(stats.get("max_strain", 0.0))
        chash = stats.get("config_hash", 0)
        hash_changed = prev_hash is not None and chash != 0 and prev_hash != 0 and chash != prev_hash

        # 変位の計算 (スタブはstatsの記録値を採用)
        disp_mm = 0.0
        if pos is not None and prev_pos is not None and pos.shape == prev_pos.shape:
            disps = np.linalg.norm(pos - prev_pos, axis=1) * 1000.0  # mm
            disp_mm = float(np.max(disps))
            if disp_mm > max_overall_disp:
                max_overall_disp = disp_mm
                max_disp_frame = f_idx
        elif pos is None:
            disp_mm = float(stats.get("max_displacement", 0.0)) * 1000.0
            if disp_mm > max_overall_disp:
                max_overall_disp = disp_mm
                max_disp_frame = f_idx

        # 異常フラグ
        is_spike = disp_mm > args.disp_threshold
        is_high_vel = max_vel > args.vel_threshold
        num_pins = len(frame.get("pins", []))
        num_cols = len(frame.get("colliders", []))

        if has_nan or is_spike or is_high_vel or vt_sat or hash_changed:
            flags = []
            if has_nan: flags.append("NaN/Inf")
            if is_spike: flags.append(f"DispSpike({disp_mm:.1f}mm)")
            if is_high_vel: flags.append(f"HighVel({max_vel:.2f}m/s)")
            if vt_sat: flags.append(f"PairSat(vt={vt_c},ee={ee_c})")
            if strain > 0.5: flags.append(f"Strain({strain:.2f})")
            if hash_changed: flags.append("ConfigChanged")
            if not has_full: flags.append("stub")
            anomalies.append((f_idx, ", ".join(flags), num_pins, num_cols))

        if pos is not None:
            prev_pos = pos
        if chash != 0:
            prev_hash = chash

    print(f" Total Frames: {frame_count} (full: {full_count}, stubs: {frame_count - full_count})")
    if frame_count != full_count:
        print(f" Sparse log: スタブ区間は座標なし (render/check/audit時は近傍フルを使用してください)")
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

    # スタブ警告
    try:
        total, full = replayer.count_full_frames(slice_log_path)
        if total != full:
            print(f"注意: スライス内にスタブ {total - full}/{total} 件が含まれます。"
                  f"再現はフル区間のみ比較します。Target Frame {target_f} がスタブの場合は"
                  f"近傍フル {replayer.nearest_full_frame(slice_log_path, target_f)} を使用してください。")
    except Exception:
        pass

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

    # スタブ除外 (座標なしフレームは描画不可)
    render_frames = [f_idx for f_idx in valid_frames if replayer.frame_has_positions(available_frames[f_idx])]
    skipped = [f_idx for f_idx in valid_frames if f_idx not in render_frames]
    if skipped:
        near = replayer.nearest_full_frame(log_path, skipped[0])
        print(f"注意: スタブ {len(skipped)} 件を除外します (例: Frame {skipped[0]}、近傍フル: Frame {near})。")
    if not render_frames:
        near = replayer.nearest_full_frame(log_path, valid_frames[0])
        print(f"エラー: 指定フレームはすべてスタブ (座標なし) です。近傍フル: Frame {near}。full_stride=1で再記録してください。")
        sys.exit(1)
    valid_frames = render_frames

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
    if frame is None or not replayer.frame_has_positions(frame):
        # スタブ時は近傍フルにフォールバック
        near = replayer.nearest_full_frame(log_path, args.frame)
        if near is not None:
            print(f"注意: Frame {args.frame} はスタブのため近傍フル Frame {near} を使用します。")
            frame = replayer.get_frame(log_path, near)
        if frame is None or not replayer.frame_has_positions(frame):
            for f in replayer.iter_frames(log_path):
                if replayer.frame_has_positions(f):
                    frame = f
                    break

    if frame is None or not replayer.frame_has_positions(frame):
        print("エラー: ログ内にフル座標フレームが見つかりません。")
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

    # Welsh-Powell グラフ彩色 (Rust核を正本とし、利用可能な場合は優先)
    colors = None
    try:
        import taremin_cloth_core as _core

        if hasattr(_core, "color_edge_pairs"):
            colors = list(
                _core.color_edge_pairs(
                    int(num_vertices), np.ascontiguousarray(edges_arr, dtype=np.uint32)
                )
            )
    except Exception:
        colors = None
    if colors is None:
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
    else:
        num_colors = (max(colors) + 1) if colors else 0

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
    if not replayer.frame_has_positions(frame):
        near = replayer.nearest_full_frame(log_path, target_f)
        print(f"エラー: Frame {target_f} はスタブ (座標なし) です。近傍フル: Frame {near}。")
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
    if not replayer.frame_has_positions(frame):
        near = replayer.nearest_full_frame(log_path, target_f)
        print(f"エラー: Frame {target_f} はスタブ (座標なし) です。近傍フル: Frame {near}。")
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

    # スタブ除外 (両方フルなフレームのみ比較)
    cmp_frames = [i for i in common_frames
                  if replayer.frame_has_positions(frames1[i]) and replayer.frame_has_positions(frames2[i])]
    skipped = len(common_frames) - len(cmp_frames)
    if skipped:
        print(f"注意: スタブ {skipped} 件を除外して比較します。")
    if not cmp_frames:
        print("エラー: 比較可能なフルフレームが存在しません。")
        sys.exit(1)

    print(f"{'Frame':<8} | {'Max Diff (mm)':<15} | {'Mean Diff (mm)':<15} | {'Status':<10}")
    print("-" * 55)

    max_overall_diff = 0.0
    exceeded_count = 0

    for f_idx in cmp_frames:
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
    print(f"比較結果: 全 {len(cmp_frames)} フレーム中 {exceeded_count} フレームで許容誤差 ({tol} mm) を超過")
    print(f"最大誤差: {max_overall_diff:.4f} mm")
    if exceeded_count == 0:
        print("[+] 両ログは完全に許容誤差内で一致しています。")
    else:
        print("[!] 挙動に有意な差分が検出されました。")


def cmd_watch(args: argparse.Namespace) -> None:
    """条件付きブレークポイント付きリプレイで最初のHitフレームを特定する"""
    log_path = args.log_file
    meta = replayer.read_metadata(log_path)
    faces = np.array(meta.get("faces", []), dtype=np.uint32)
    rp = replayer.ClothReplayer(log_path)

    # 停止条件の解釈: "disp>5,intersections,vt_sat,vert:123" 形式
    spec = (args.stop_on or "disp>5").strip()
    disp_thr = None
    want_isect = False
    want_sat = False
    want_config = False
    want_release = False
    persist_spec = None
    watch_verts: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if tok.startswith("disp>"):
            try:
                disp_thr = float(tok.split(">", 1)[1])
            except ValueError:
                pass
        elif tok in ("intersections", "isect", "penetration"):
            want_isect = True
        elif tok in ("vt_sat", "saturated", "pair_sat"):
            want_sat = True
        elif tok in ("config", "param", "params", "config_changed"):
            want_config = True
        elif tok in ("release", "released", "after-release"):
            want_release = True
        elif tok.startswith("persist:") or tok.startswith("persist>"):
            try:
                persist_spec = int(tok.split(":", 1)[1] if ":" in tok else tok.split(">", 1)[1])
            except ValueError:
                pass
        elif tok.startswith("vert:"):
            try:
                watch_verts.extend(int(v) for v in tok.split(":", 1)[1].split("+") if v.strip() != "")
            except ValueError:
                pass
    if args.watch_verts:
        watch_verts.extend(int(v) for v in args.watch_verts.split(",") if v.strip() != "")
    if args.vt_sat:
        want_sat = True
    if args.intersections:
        want_isect = True
    if getattr(args, "config_changed", False):
        want_config = True
    if getattr(args, "require_release", False):
        want_release = True
    if args.disp is not None:
        disp_thr = float(args.disp)

    pred = replayer.build_stop_predicate(
        disp_mm=disp_thr,
        intersections=want_isect,
        faces=faces if want_isect else None,
        vt_saturated=want_sat,
        watch_verts=watch_verts or None,
        watch_vert_disp_mm=float(args.watch_disp_mm),
        max_diff_m=float(args.max_diff_mm) / 1000.0 if args.max_diff_mm is not None else None,
        config_changed=want_config,
    )
    # リリースゲート＋持続条件でラップ (既定 persist=1・ゲートなしは従来動作と同一)
    persist = int(getattr(args, "persist", 1) or 1)
    if persist_spec is not None:
        persist = persist_spec
    persist = max(1, persist)
    require_release = bool(getattr(args, "require_release", False)) or want_release
    pred = replayer.build_release_persist_predicate(
        pred, persist=persist, require_release=require_release)

    # end未指定時はログ末尾まで
    end_f = args.end
    if end_f is None:
        end_f = -1
        for f in replayer.iter_frames(log_path):
            end_f = max(end_f, int(f.get("frame_index", 0)))

    hit = rp.replay_until(
        start_frame_idx=int(args.start),
        end_frame_idx=int(end_f),
        stop_on=pred,
        stride=int(args.stride),
    )
    if hit is None:
        print(f"[watch] No hit: {spec} (frames {args.start}..{end_f})")
        return
    f_idx = int(hit["frame_index"])
    print(f"[watch] HIT Frame {f_idx}: {hit.get('reason', '')}")
    try:
        rel = getattr(pred, "state", {}).get("release_frame")
        if rel is not None:
            print(f"  release detected at Frame {rel}")
    except Exception:
        pass
    print(f"  max_disp={hit.get('max_disp_mm', 0.0):.2f}mm max_diff={hit.get('max_diff', 0.0):.6f}m "
          f"vt={hit.get('vt_count')} ee={hit.get('ee_count')} saturated={hit.get('vt_saturated')}")
    if args.output:
        lookback = int(args.lookback)
        s = max(int(args.start), f_idx - lookback)
        e = f_idx + int(args.lookahead)
        args_slice = argparse.Namespace(log_file=log_path, start=s, end=e, output=args.output)
        cmd_slice(args_slice)


def cmd_audit_pairs(args: argparse.Namespace) -> None:
    """指定フレームのペアカバレッジ監査 (GPU収集 vs CPUグラウンドトゥルース)"""
    log_path = args.log_file
    target_f = args.frame
    try:
        from . import pair_audit as pa
    except ImportError:
        from taremin_cloth import pair_audit as pa  # type: ignore

    meta = replayer.read_metadata(log_path)
    faces = np.array(meta.get("faces", []), dtype=np.uint32)
    # トポロジー系は original_edges (せん断対角なし) を優先
    raw_edges = meta.get("original_edges") or meta.get("edges", [])
    edges = np.array(raw_edges, dtype=np.uint32)
    raw_rest = meta.get("original_rest_lengths") or meta.get("initial_rest_lengths")
    if len(raw_edges) == (len(raw_rest) if raw_rest else -1):
        rest_arr = np.array(raw_rest, dtype=np.float64)
    else:
        rest_arr = None
    if len(faces) == 0 or len(edges) == 0:
        print("エラー: ログに面/辺トポロジーが含まれていません。")
        sys.exit(1)

    f_prev = replayer.get_frame(log_path, target_f - 1)
    f_curr = replayer.get_frame(log_path, target_f)
    if f_prev is None or f_curr is None:
        print(f"エラー: Frame {target_f - 1} または {target_f} がログに見つかりません。")
        sys.exit(1)
    if not replayer.frame_has_positions(f_prev) or not replayer.frame_has_positions(f_curr):
        print("エラー: 対象区間にスタブが含まれます。フル区間で実行してください。")
        sys.exit(1)

    cfg = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    thickness = float(meta.get("thickness", 0.005))
    safety = float(cfg.get("pair_safety_margin", meta.get("pair_cache_safety_margin", 0.005)))
    hscale = float(cfg.get("pair_horizon_scale", meta.get("pair_cache_horizon_scale", 1.3)))
    hmax = float(cfg.get("pair_max_horizon", meta.get("pair_cache_max_horizon", 0.02)))
    margin_mode = int(cfg.get("pair_margin_mode", 1 if meta.get("pair_cache_margin_mode", "AUTO") != "FIXED" else 0))
    if isinstance(meta.get("pair_cache_margin_mode"), int):
        margin_mode = int(meta.get("pair_cache_margin_mode", 1))
    max_pairs = int(cfg.get("pair_max_pairs", meta.get("pair_cache_max_pairs", 65536)))

    print(f"フレーム {target_f - 1} -> {target_f} をリプレイしてペアを読戻しています...")
    rp = replayer.ClothReplayer(log_path)
    rp.replay_range(target_f - 1, target_f, compare=False, force_enable_pair_cache=True)
    pairs = rp.get_active_pairs()

    prev_pos = np.array(f_prev["positions"], dtype=np.float64).reshape((-1, 3))
    prev_vel = np.array(f_prev.get("velocities", []), dtype=np.float64).reshape((-1, 3)) \
        if len(f_prev.get("velocities", [])) > 0 else None
    curr_pos = np.array(f_curr["positions"], dtype=np.float64).reshape((-1, 3))

    rep = pa.audit_frame(
        prev_pos, prev_vel, curr_pos,
        pairs["vt_pairs"], pairs["ee_pairs"],
        pairs["vt_count"], pairs["ee_count"],
        faces, edges, rest_arr, thickness, safety, hscale, hmax,
        margin_mode, float(f_curr.get("dt", 1.0 / 60.0)), max_pairs, max_pairs,
    )
    print(pa.format_report(rep))


def cmd_audit_cache(args: argparse.Namespace) -> None:
    """同一ログをキャッシュON/OFFで再現し、貫通の起因を判定する。

    grab解放後の残存貫通がペアキャッシュ起因かを、2通りの再現の交差数で
    切り分ける (新規ログ記録は不要)。
    """
    log_path = args.log_file
    target_f = args.frame
    lookback = max(1, int(getattr(args, "lookback", 3) or 3))
    start_f = max(0, target_f - lookback)
    try:
        from . import pair_audit as pa
        from . import analysis as ana
    except ImportError:
        from taremin_cloth import pair_audit as pa  # type: ignore
        from taremin_cloth import analysis as ana  # type: ignore

    meta = replayer.read_metadata(log_path)
    faces = np.array(meta.get("faces", []), dtype=np.uint32)
    if len(faces) == 0:
        print("エラー: ログに面トポロジーが含まれていません。")
        sys.exit(1)

    f_target = replayer.get_frame(log_path, target_f)
    if f_target is None or not replayer.frame_has_positions(f_target):
        print(f"エラー: Frame {target_f} がフル座標で存在しません。")
        sys.exit(1)
    log_pos = np.array(f_target["positions"], dtype=np.float32).reshape((-1, 3))
    n_log = len(ana.find_triangle_intersections(log_pos, faces))

    outcomes = {}
    drifts = {}
    for label, force in (("ON", True), ("OFF", False)):
        rp = replayer.ClothReplayer(log_path)
        results = rp.replay_range(start_f, target_f, compare=True,
                                  force_pair_cache=force)
        if not results:
            print(f"エラー: {label} 再現が空でした。")
            sys.exit(1)
        sim_pos = results[-1]["positions"]
        outcomes[label] = len(ana.find_triangle_intersections(sim_pos, faces))
        cmp_diffs = [r["max_diff"] for r in results if r.get("compared")]
        drifts[label] = max(cmp_diffs) if cmp_diffs else float("nan")

    verdict, message = pa.attribute_penetration(n_log, outcomes["ON"], outcomes["OFF"])
    print(f"Frame {target_f} (from F{start_f}): "
          f"log={n_log} cacheON={outcomes['ON']} cacheOFF={outcomes['OFF']}")
    print(f"replay drift: ON={drifts['ON'] * 1000:.1f}mm OFF={drifts['OFF'] * 1000:.1f}mm "
          f"(vs log)")
    worst_drift = max(d for d in drifts.values() if d == d)
    if worst_drift > 0.05:
        print(f"注意: 再現ドリフトが大きいため ({worst_drift * 1000:.0f}mm)、"
              f"本判定の信頼性は低いです。交差の有無・分布の質的比較を併用してください。")
    print(f"verdict: {verdict} ({message})")
    if verdict == "cache-attributable":
        print("次手順: audit-pairs で miss 分類 (horizon/飽和) を特定してください。")


def cmd_profile(args: argparse.Namespace) -> None:
    """指定区間をGPU計測付きでリプレイし、壁時計とGPU内訳を表示する。

    親スコープは包含時間のため、正味評価は子合計または親1件で
    行うこと（平坦合計は二重計上の参考値）。
    """
    import time

    try:
        from .engine import profiling as prof_helper
    except ImportError:
        from taremin_cloth.engine import profiling as prof_helper  # type: ignore

    log_path = args.log_file
    target_f = int(args.frame) if getattr(args, "frame", None) is not None else None
    lookback = max(1, int(getattr(args, "lookback", 2) or 2))
    trace_out = getattr(args, "output", None)

    if target_f is None:
        frames_idx = [f["frame_index"] for f in replayer.iter_frames(log_path)]
        if not frames_idx:
            print("エラー: フレームが存在しません。")
            sys.exit(1)
        target_f = max(frames_idx)
    start_f = max(0, target_f - lookback)

    frames = {}
    for f in replayer.iter_frames(log_path):
        idx = f["frame_index"]
        if start_f <= idx <= target_f:
            frames[idx] = f
    if start_f not in frames or not replayer.frame_has_positions(frames[start_f]):
        near = replayer.nearest_full_frame(log_path, start_f)
        hint = f" (近傍フル: Frame {near})" if near is not None else ""
        print(f"エラー: 開始フレーム {start_f} はスタブのため再現できません{hint}。")
        sys.exit(1)

    rp = replayer.ClothReplayer(log_path)
    init_pos = np.array(frames[start_f]["positions"], dtype=np.float32).reshape((-1, 3))
    init_vel = None
    if "velocities" in frames[start_f] and len(frames[start_f]["velocities"]) > 0:
        init_vel = np.array(frames[start_f]["velocities"], dtype=np.float32).reshape((-1, 3))
    rp.create_simulator(init_pos)
    rp.sim.set_positions_and_velocities(init_pos, init_vel)
    rp.apply_config_at(start_f)
    rp.apply_elastic_at(start_f)
    rp.apply_sewing_at(start_f)
    try:
        rp.apply_frame_bone(frames[start_f])
    except Exception:
        pass

    ok = False
    try:
        ok = bool(rp.sim.set_profiling_enabled(True))
    except Exception:
        ok = False
    print(f"profiling enabled: {ok} (F{start_f} -> F{target_f})")
    if not ok:
        print("TIMESTAMP_QUERY非対応のためGPU内訳は取得できません（壁時計のみ）。")

    out_pos = np.zeros(rp.num_vertices * 3, dtype=np.float32)
    prev_fdata = frames.get(start_f)
    prev_replay_pos = init_pos.copy()
    wall_ms = []
    for f_idx in range(start_f + 1, target_f + 1):
        if f_idx not in frames:
            break
        fdata = frames[f_idx]
        dt = float(fdata.get("dt", 1.0 / 60.0))
        substeps = int(fdata.get("substeps", 10))
        try:
            iters = int(fdata.get("solver_iterations", 0))
            if iters > 0 and hasattr(rp.sim, "set_solver_iterations"):
                rp.sim.set_solver_iterations(iters)
        except Exception:
            pass
        rp.apply_frame_inputs(fdata)
        rp.apply_frame_config(fdata)
        rp.apply_frame_elastic(fdata)
        rp.apply_frame_bone(fdata)
        if prev_fdata is not None:
            rp.apply_frame_sewing(prev_fdata)
        if not rp.apply_frame_priority(fdata):
            rp._update_sewing_priority(prev_replay_pos)
        t0 = time.perf_counter()
        rp.sim.step(dt=dt, substeps=substeps)
        wall_ms.append((time.perf_counter() - t0) * 1000.0)
        rp.sim.get_positions(out_pos)
        prev_replay_pos = out_pos.reshape((-1, 3)).copy()
        prev_fdata = fdata

    if wall_ms:
        print(f"wall: frames={len(wall_ms)} avg={sum(wall_ms)/len(wall_ms):.3f}ms "
              f"min={min(wall_ms):.3f}ms max={max(wall_ms):.3f}ms")

    entries: list = []
    try:
        # 非同期回収の遅延吸収のため枯れるまで繰返す
        for _ in range(6):
            cur = list(rp.sim.take_profile())
            if cur:
                entries = cur
    except Exception:
        entries = []
    print(prof_helper.format_profile_table(entries))
    if trace_out and entries:
        try:
            rp.sim.save_profile_trace(trace_out)
            print(f"saved: {trace_out}")
        except Exception as e:
            print(f"trace保存に失敗: {e}")


def cmd_audit(args: argparse.Namespace) -> None:
    """cacheログとdirectログの差分からPairCacheの見逃し率を定量化する"""
    log_cache = args.log_cache
    log_direct = args.log_direct
    tol = float(args.tolerance)
    meta_c = replayer.read_metadata(log_cache)
    faces = np.array(meta_c.get("faces", []), dtype=np.uint32)
    frames_c = {f["frame_index"]: f for f in replayer.iter_frames(log_cache)}
    frames_d = {f["frame_index"]: f for f in replayer.iter_frames(log_direct)}
    common = sorted(set(frames_c.keys()) & set(frames_d.keys()))
    if not common:
        print("エラー: 共通フレームが存在しません。")
        sys.exit(1)
    cmp_frames = [i for i in common
                  if replayer.frame_has_positions(frames_c[i]) and replayer.frame_has_positions(frames_d[i])]
    skipped = len(common) - len(cmp_frames)
    if skipped:
        print(f"注意: スタブ {skipped} 件を除外して監査します。")
    if not cmp_frames:
        print("エラー: 監査可能なフルフレームが存在しません。")
        sys.exit(1)
    try:
        from . import analysis as ana
    except ImportError:
        from taremin_cloth import analysis as ana  # type: ignore
    print(f"{'Frame':<8} | {'Cache-Direct(mm)':<16} | {'CacheIsect':<10} | {'DirectIsect':<11} | Status")
    print("-" * 70)
    exceeded = 0
    for f_idx in cmp_frames:
        pc = np.array(frames_c[f_idx]["positions"], dtype=np.float32).reshape((-1, 3))
        pd = np.array(frames_d[f_idx]["positions"], dtype=np.float32).reshape((-1, 3))
        max_d = float(np.max(np.linalg.norm(pc - pd, axis=1)) * 1000.0)
        # 交差数は重いためサンプリング: --with-intersections時のみ
        if args.with_intersections:
            ic = len(ana.find_triangle_intersections(pc, faces))
            ide = len(ana.find_triangle_intersections(pd, faces))
        else:
            ic = ide = -1
        st = "OK" if max_d <= tol else "EXCEEDED"
        if st == "EXCEEDED":
            exceeded += 1
        if args.with_intersections:
            print(f"F{f_idx:<7} | {max_d:12.4f} mm | {ic:<10d} | {ide:<11d} | {st}")
        else:
            print(f"F{f_idx:<7} | {max_d:12.4f} mm | {'-':<10} | {'-':<11} | {st}")
    print("-" * 70)
    print(f"audit結果: 全 {len(cmp_frames)} フレーム中 {exceeded} フレームで許容誤差 ({tol} mm) を超過")
    print("使い方: 同一開始状態から pair_cache ON/OFF の2本を記録し、本コマンドでmiss率を測定してください。")


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

    # 9. watch (条件付きブレークポイント)
    p_watch = subparsers.add_parser("watch", help="条件付きブレーク付きリプレイでHitフレームを特定")
    p_watch.add_argument("log_file", help="対象ログファイル (.jsonl.gz)")
    p_watch.add_argument("--stop-on", default="disp>5", help="停止条件 (例: 'disp>5,intersections,vt_sat,vert:123')")
    p_watch.add_argument("--start", type=int, default=0, help="開始フレーム")
    p_watch.add_argument("--end", type=int, default=None, help="終了フレーム (省略時は末尾)")
    p_watch.add_argument("--stride", type=int, default=1, help="述語評価間隔")
    p_watch.add_argument("--disp", type=float, default=None, help="変位閾値mm (stop-onより優先)")
    p_watch.add_argument("--intersections", action="store_true", help="貫通発生で停止")
    p_watch.add_argument("--vt-sat", action="store_true", help="Pair飽和で停止")
    p_watch.add_argument("--config-changed", action="store_true", help="設定変更フレームで停止")
    p_watch.add_argument("--require-release", action="store_true",
                         help="ピン数減少 (grab解放) 検出後にアーム (解放後残存の検出用)")
    p_watch.add_argument("--persist", type=int, default=1,
                         help="アーム後に条件が連続成立すべきフレーム数 (既定1)")
    p_watch.add_argument("--watch-verts", type=str, default=None, help="注目頂点ID (例: '123,456')")
    p_watch.add_argument("--watch-disp-mm", type=float, default=2.0, help="注目頂点の変位閾値mm")
    p_watch.add_argument("--max-diff-mm", type=float, default=None, help="再現乖離が閾値超で停止 (mm)")
    p_watch.add_argument("--lookback", type=int, default=3, help="Hit時に切り出す前フレーム数")
    p_watch.add_argument("--lookahead", type=int, default=2, help="Hit時に切り出す後フレーム数")
    p_watch.add_argument("--output", "-o", default=None, help="Hit前後の極小ログ出力先 (.jsonl.gz)")
    p_watch.set_defaults(func=cmd_watch)

    # 10. audit (PairCache精度監査)
    p_audit = subparsers.add_parser("audit", help="cache/directの2本ログから見逃し率を定量化")
    p_audit.add_argument("log_cache", help="PairCache ONログ (.jsonl.gz)")
    p_audit.add_argument("log_direct", help="PairCache OFF(直進)ログ (.jsonl.gz)")
    p_audit.add_argument("--tolerance", "-t", type=float, default=1.0, help="許容変位閾値 (mm)")
    p_audit.add_argument("--with-intersections", action="store_true", help="交差数も比較 (低速)")
    p_audit.set_defaults(func=cmd_audit)

    # 11. audit-pairs (ペアカバレッジ監査)
    p_audit_pairs = subparsers.add_parser("audit-pairs", help="GPU収集ペアとCPU真値のカバレッジ分類")
    p_audit_pairs.add_argument("log_file", help="対象ログファイル (.jsonl.gz)")
    p_audit_pairs.add_argument("--frame", "-f", type=int, required=True, help="監査フレーム番号 (前フレームからリプレイ)")
    p_audit_pairs.set_defaults(func=cmd_audit_pairs)

    # 12. audit-cache (キャッシュ起因判定)
    p_audit_cache = subparsers.add_parser("audit-cache", help="同一ログのON/OFF再現で貫通起因を判定")
    p_audit_cache.add_argument("log_file", help="対象ログファイル (.jsonl.gz)")
    p_audit_cache.add_argument("--frame", "-f", type=int, required=True, help="判定フレーム番号")
    p_audit_cache.add_argument("--lookback", type=int, default=3, help="何フレーム前から再現するか")
    p_audit_cache.set_defaults(func=cmd_audit_cache)

    # 13. profile (GPU計測付きリプレイ)
    p_profile = subparsers.add_parser("profile", help="指定区間をGPU計測付きでリプレイし内訳表示")
    p_profile.add_argument("log_file", help="対象ログファイル (.jsonl.gz)")
    p_profile.add_argument("--frame", "-f", type=int, default=None, help="計測末尾フレーム (省略時は末尾)")
    p_profile.add_argument("--lookback", type=int, default=2, help="何フレーム前から再現するか")
    p_profile.add_argument("--output", "-o", default=None, help="chrometrace保存先 (.json)")
    p_profile.set_defaults(func=cmd_profile)

    # 14. coverage (エンジンコア機能・クライアントカバレッジ監査)
    p_coverage = subparsers.add_parser("coverage", help="エンジンコア機能とBlender/独立GUIの実装カバレッジを測定・比較")
    p_coverage.add_argument("--format", choices=["console", "markdown", "json"], default="console", help="出力形式")
    p_coverage.add_argument("--check", action="store_true", help="Blender Engineパリティ欠落があれば非ゼロ終了")
    p_coverage.add_argument("--output", "-o", default=None, help="レポート保存先")
    p_coverage.set_defaults(func=cmd_coverage)


    args = parser.parse_args()
    args.func(args)


def cmd_coverage(args):
    """エンジンコア機能・クライアント統合カバレッジ監査"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tools_dir = os.path.join(project_root, "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from audit_feature_coverage import run_full_audit, format_console, format_markdown, METADATA_FIELDS
    report = run_full_audit(project_root)
    if args.format == "markdown":
        out = format_markdown(report)
    elif args.format == "json":
        import json
        out = json.dumps({
            "param_counts": report.param_counts,
            "api_counts": report.api_counts,
            "feature_counts": report.feature_counts,
            "actionable_items": report.actionable_items,
        }, indent=2, ensure_ascii=False)
    else:
        out = format_console(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Report saved to {args.output}")
    else:
        print(out)
    if args.check:
        phys_total = len([p for p in report.params if p["name"] not in METADATA_FIELDS])
        phys_be = len([p for p in report.params if p["name"] not in METADATA_FIELDS and p["blender_engine"]])
        if phys_be < phys_total:
            print(f"[ERROR] Blender Engine 物理パラメータ欠落: {phys_be}/{phys_total}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
