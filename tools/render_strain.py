#!/usr/bin/env python3
"""歪み描画: python tools/render_strain.py tests/fixtures/garments/tshirt/tshirt_scene.json -o scratch/strain

無重力で縫合を閉じ、着せた後の辺伸び率を色で描く。
青が縮み、白が自然長、赤が伸び。開口部の内面は赤く見えるため、
赤画素数は裏返しの指標にならない。数値 (平均・最大) で判定する。
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "python"))

import numpy as np

from taremin_cloth.mesh_renderer import compute_strain_colors
from taremin_cloth.utils.fixture_io import load_scene_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("-o", "--output", default="scratch/strain",
                    help="出力先の接頭辞 (front/back を付けて保存)")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--max-strain", type=float, default=0.2,
                    help="色付けの上限 (この伸びで真っ赤)")
    args = ap.parse_args()

    try:
        import taremin_cloth_core as core
    except ImportError:
        print("taremin_cloth_core が無いため描画できない")
        return 2

    m = load_scene_manifest(str(args.manifest))
    cfg = m["config"]
    cp, cf = m["collider_positions"], m["collider_faces"]
    mesh_col = np.hstack([cp[cf[:, 0]], cp[cf[:, 1]], cp[cf[:, 2]]]).astype(np.float32)
    tris = np.stack([cp[cf[:, 0]], cp[cf[:, 1]], cp[cf[:, 2]]], axis=1).astype(np.float32)
    rest = np.linalg.norm(m["positions"][m["edges"][:, 0]]
                         - m["positions"][m["edges"][:, 1]], axis=1)
    sim = core.ClothSimulator(
        m["positions"], m["edges"], m["faces"],
        sewing_springs=m["sewing_springs"],
        sewing_shrink_speed=float(cfg.get("sewing_shrink_speed", 0.8)),
        stiffness=float(cfg.get("tension_stiffness", 10000.0)),
        bending_stiffness=float(cfg.get("bending_stiffness", 20.0)))
    sim.set_sewing_priority_options(True, float(cfg.get("sewing_priority_threshold", 0.9)),
                                    0.005, 3, 600)
    sim.set_mesh_collider_triangles(
        tris, float(m.get("collider_friction", 0.8)),
        float(m.get("collider_thickness", 0.02)), 0.0,
        bool(m.get("collider_single_sided", False)))
    coords = m["positions"].copy()
    for _ in range(args.steps):
        sim.update_sewing_priority(np.ascontiguousarray(coords))
        sim.step(1.0 / 60.0, 10)
        out = np.zeros(len(coords) * 3, dtype=np.float32)
        sim.get_positions(out)
        coords = out.reshape(-1, 3).copy()
    cur = np.linalg.norm(coords[m["edges"][:, 0]] - coords[m["edges"][:, 1]], axis=1)
    strain = (cur - rest) / np.maximum(rest, 1e-9)
    print(f"strain mean {float(np.abs(strain).mean())*100:.1f}% "
          f"max {float(strain.max())*100:.0f}% min {float(strain.min())*100:.0f}%")
    vc = compute_strain_colors(coords, m["edges"], rest.astype(np.float32),
                               max_strain=args.max_strain)
    for tag, pos, tgt in (("front", [0.0, -1.6, 0.9], [0.0, 0.0, 0.7]),
                          ("back", [0.0, 1.6, 0.9], [0.0, 0.0, 0.7])):
        path = f"{args.output}_{tag}.png"
        core.render_scene_to_png(path, coords, m["faces"], mesh_colliders=mesh_col,
                                 vertex_colors=vc, camera_pos=pos, camera_target=tgt)
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
