#!/usr/bin/env python3
"""Tシャツシミュレーションのアニメーション生成スクリプト (Blender非依存).

初期配置（浮いた型紙と縫合スプリング）から、縫合引き合わせ・着せ付け・重力ドレープ整定
までのシミュレーション過程を連番レンダリングし、APNGおよびGIFアニメーションを生成する。
"""
import argparse
import math
import os
import shutil
import sys
from pathlib import Path
from PIL import Image
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "python"))

import taremin_cloth_core as core
from taremin_cloth.utils.fixture_io import load_scene_manifest
from taremin_cloth.mesh_renderer import create_animation_file


def render_simulation_sequence(
    manifest_path: str,
    output_dir: Path,
    steps: int = 120,
    stride: int = 2,
    camera_view: str = "quarter",  # "quarter" or "front"
    hold_last_frames: int = 15,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    m = load_scene_manifest(manifest_path)
    cfg = m["config"]
    cp, cf = m["collider_positions"], m["collider_faces"]
    mesh_col = np.hstack([cp[cf[:, 0]], cp[cf[:, 1]], cp[cf[:, 2]]]).astype(np.float32)
    tris = np.stack([cp[cf[:, 0]], cp[cf[:, 1]], cp[cf[:, 2]]], axis=1).astype(np.float32)

    sim = core.ClothSimulator(
        m["positions"], m["edges"], m["faces"],
        sewing_springs=m["sewing_springs"],
        sewing_shrink_speed=float(cfg.get("sewing_shrink_speed", 0.8)),
        stiffness=float(cfg.get("tension_stiffness", 10000.0)),
        bending_stiffness=float(cfg.get("bending_stiffness", 20.0)),
    )
    sim.set_sewing_priority_options(
        True,
        float(cfg.get("sewing_priority_threshold", 0.9)),
        0.005, 3, 600
    )
    sim.set_mesh_collider_triangles(
        tris,
        float(m.get("collider_friction", 0.8)),
        float(m.get("collider_thickness", 0.008)),
        0.0,
        bool(m.get("collider_single_sided", False))
    )

    if camera_view == "quarter":
        cam_pos = [0.8, -1.35, 1.05]
        cam_tgt = [0.0, 0.0, 0.85]
    else:  # front
        cam_pos = [0.0, -1.5, 0.95]
        cam_tgt = [0.0, 0.0, 0.85]

    coords = m["positions"].copy()
    frame_paths = []
    frame_idx = 0

    print(f"[{camera_view}] シミュレーション進行 & レンダリング中 (全 {steps} ステップ)...")

    # Step 0 (初期配置)
    p0 = str(output_dir / f"frame_{frame_idx:04d}.png")
    core.render_scene_to_png(
        p0,
        coords.astype(np.float32),
        m["faces"].astype(np.uint32),
        sewing_springs=m["sewing_springs"].astype(np.uint32),
        mesh_colliders=mesh_col,
        camera_pos=cam_pos,
        camera_target=cam_tgt,
        draw_wireframe=True,
    )
    frame_paths.append(p0)
    frame_idx += 1

    for s in range(1, steps + 1):
        sim.update_sewing_priority(np.ascontiguousarray(coords))
        sim.step(1.0 / 60.0, 10)
        out = np.zeros(len(coords) * 3, dtype=np.float32)
        sim.get_positions(out)
        coords = out.reshape(-1, 3).copy()

        if s % stride == 0 or s == steps:
            p = str(output_dir / f"frame_{frame_idx:04d}.png")
            core.render_scene_to_png(
                p,
                coords.astype(np.float32),
                m["faces"].astype(np.uint32),
                sewing_springs=m["sewing_springs"].astype(np.uint32),
                mesh_colliders=mesh_col,
                camera_pos=cam_pos,
                camera_target=cam_tgt,
                draw_wireframe=True,
            )
            frame_paths.append(p)
            frame_idx += 1

    # 最終フレームのホールド（余韻）
    last_frame = frame_paths[-1]
    for _ in range(hold_last_frames):
        frame_paths.append(last_frame)

    print(f"[{camera_view}] レンダリング完了: {len(frame_paths)} フレーム")
    return frame_paths, coords


def render_turntable(
    manifest_path: str,
    final_coords: np.ndarray,
    output_dir: Path,
    num_frames: int = 48,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    m = load_scene_manifest(manifest_path)
    cp, cf = m["collider_positions"], m["collider_faces"]
    mesh_col = np.hstack([cp[cf[:, 0]], cp[cf[:, 1]], cp[cf[:, 2]]]).astype(np.float32)

    cam_tgt = [0.0, 0.0, 0.85]
    radius = 1.5
    cam_z = 0.95

    frame_paths = []
    print(f"[turntable] 360度ターンテーブル レンダリング中 (全 {num_frames} フレーム)...")

    for i in range(num_frames):
        theta = -math.pi / 2.0 + 2.0 * math.pi * (i / num_frames)
        cam_x = radius * math.cos(theta)
        cam_y = radius * math.sin(theta)

        p = str(output_dir / f"turn_{i:04d}.png")
        core.render_scene_to_png(
            p,
            final_coords.astype(np.float32),
            m["faces"].astype(np.uint32),
            sewing_springs=None,
            mesh_colliders=mesh_col,
            camera_pos=[cam_x, cam_y, cam_z],
            camera_target=cam_tgt,
            draw_wireframe=True,
        )
        frame_paths.append(p)

    print(f"[turntable] 完了: {len(frame_paths)} フレーム")
    return frame_paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    manifest = REPO / "tests" / "fixtures" / "garments" / "tshirt" / "tshirt_scene.json"
    scratch_dir = REPO / "scratch" / "anim_frames"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)

    # 1. クォータービュー（斜め前）のシミュレーションシーケンス
    q_frames, final_coords = render_simulation_sequence(
        str(manifest), scratch_dir / "quarter", steps=args.steps, stride=2, camera_view="quarter"
    )
    # APNG と GIF を生成
    q_apng = REPO / "scratch" / "tshirt_simulation_quarter.png"
    q_gif = REPO / "scratch" / "tshirt_simulation_quarter.gif"
    create_animation_file(q_frames, str(q_apng), fps=args.fps)
    create_animation_file(q_frames, str(q_gif), fps=args.fps)
    print(f"Generated: {q_apng} ({q_apng.stat().st_size // 1024} KB)")
    print(f"Generated: {q_gif} ({q_gif.stat().st_size // 1024} KB)")

    # 2. 正面ビュー（Front view）のシミュレーションシーケンス
    f_frames, _ = render_simulation_sequence(
        str(manifest), scratch_dir / "front", steps=args.steps, stride=2, camera_view="front"
    )
    f_apng = REPO / "scratch" / "tshirt_simulation_front.png"
    f_gif = REPO / "scratch" / "tshirt_simulation_front.gif"
    create_animation_file(f_frames, str(f_apng), fps=args.fps)
    create_animation_file(f_frames, str(f_gif), fps=args.fps)
    print(f"Generated: {f_apng} ({f_apng.stat().st_size // 1024} KB)")
    print(f"Generated: {f_gif} ({f_gif.stat().st_size // 1024} KB)")

    # 3. 360度ターンテーブルアニメーション
    t_frames = render_turntable(
        str(manifest), final_coords, scratch_dir / "turntable", num_frames=48
    )
    t_apng = REPO / "scratch" / "tshirt_turntable.png"
    t_gif = REPO / "scratch" / "tshirt_turntable.gif"
    create_animation_file(t_frames, str(t_apng), fps=args.fps)
    create_animation_file(t_frames, str(t_gif), fps=args.fps)
    print(f"Generated: {t_apng} ({t_apng.stat().st_size // 1024} KB)")
    print(f"Generated: {t_gif} ({t_gif.stat().st_size // 1024} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
