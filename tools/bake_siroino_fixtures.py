#!/usr/bin/env python3
"""SiroinoSotai Mobile完全体のコライダーOBJ化 + Tシャツ型紙の生成.

入力: tmp/SiroinoSotai.blend (CC0, 要クレジット)
出力:
  tests/fixtures/bodies/siroino/SiroinoSotai_Mobile_collider.obj
  tests/fixtures/bodies/siroino/provenance.json
  tests/fixtures/garments/tshirt/tshirt_panels.obj
  tests/fixtures/garments/tshirt/tshirt_sewing.json
  tests/fixtures/garments/tshirt/tshirt_scene.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PYTHON_PKG = REPO / "python"
if str(PYTHON_PKG) not in sys.path:
    sys.path.insert(0, str(PYTHON_PKG))

from tools.blender_manager import resolve_blender

BLEND = REPO / "tmp" / "SiroinoSotai.blend"
BODY_DIR = REPO / "tests" / "fixtures" / "bodies" / "siroino"
TSHIRT_DIR = REPO / "tests" / "fixtures" / "garments" / "tshirt"

CONVERT_SCRIPT = r"""
import bpy
import numpy as np

obj = bpy.data.objects.get("SiroinoSotai_Mobile")
assert obj is not None, "SiroinoSotai_Mobile not found"
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
if obj.data.shape_keys:
    for kb in obj.data.shape_keys.key_blocks:
        kb.value = 0.0
# アーマチュア無効化: 素体形状(レスト)を素直に取得する
for m in list(obj.modifiers):
    if m.type == 'ARMATURE':
        obj.modifiers.remove(m)
bpy.context.view_layer.update()
mesh = obj.data
mesh.calc_loop_triangles()
n_verts = len(mesh.vertices)
coords = np.empty(n_verts * 3, dtype=np.float32)
mesh.vertices.foreach_get("co", coords)
pos = coords.reshape(-1, 3).astype(np.float64)
mw = np.array(obj.matrix_world, dtype=np.float64)
pos_w = (np.hstack([pos, np.ones((len(pos), 1))]) @ mw.T)[:, :3]
tris = np.array([t.vertices for t in mesh.loop_triangles], dtype=np.int64)
out = {"positions": pos_w.astype(np.float32).tolist(), "faces": tris.tolist(),
       "name": obj.name, "n_verts": n_verts, "n_tris": len(tris)}
import json
with open(r"__OUT__", "w", encoding="utf-8") as f:
    json.dump(out, f)
print(f"[convert] verts={n_verts} tris={len(tris)}")
"""


def export_obj(path: Path, positions: np.ndarray, faces: np.ndarray) -> None:
    from taremin_cloth.analysis import export_obj as _export

    _export(str(path), np.ascontiguousarray(positions, dtype=np.float32),
            np.ascontiguousarray(faces, dtype=np.int64))


def load_obj(path: Path):
    verts, faces = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:4]]
                if len(idx) == 3:
                    faces.append(idx)
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.uint32)


def _ss(a: float, b: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


# 実寸Tシャツ型紙 (前後身頃・袖前後パネル)。素体計測値に基づく平面型紙。
# 前後身頃: 裾-脇-衿ぐり-肩傾斜-アームホールの輪郭を持つ格子 (armpit以下は平面)。
# 袖: 腕の前後に置く平面パネル。天側・地側の縫合で筒化する。
_NX, _NYU = 13, 14
_Z0, _ZTOP_U = 0.55, 1.007
_Z_ARMPIT = 0.905
_TIP_X, _TIP_Z = 0.155, 1.035
_NECK_X = 0.05


def _half_width(z: float) -> float:
    if z <= _Z_ARMPIT:
        for (z0, w0), (z1, w1) in [((0.55, 0.140), (0.70, 0.135)),
                                   ((0.70, 0.135), (0.85, 0.150)),
                                   ((0.85, 0.150), (0.905, 0.150))]:
            if z <= z1:
                return w0 + (w1 - w0) * (z - z0) / (z1 - z0)
        return 0.150
    t = (z - _Z_ARMPIT) / (_TIP_Z - _Z_ARMPIT)
    return 0.150 + 0.005 * t - 0.028 * np.sin(np.pi * min(max(t, 0.0), 1.0))


def _bodice(is_front: bool):
    nx, nyu = _NX, _NYU
    zrows = np.linspace(_Z0, _ZTOP_U, nyu).tolist()
    pos, faces, grid = [], [], set()
    # 型紙は armpit 以下で平面。衿ぐり寄せ (collar ease) だけ中央上部を体側へ寄せる。
    y0 = -0.145 if is_front else 0.110
    y_neck = -0.075 if is_front else 0.042

    def top_z(x: float) -> float:
        ax = abs(x)
        if ax >= _NECK_X:
            return _TIP_Z + (_TIP_X - ax) / (_TIP_X - _NECK_X) * 0.01
        s = np.sin(ax / _NECK_X * np.pi / 2.0)
        return (1.005 + 0.04 * s) if is_front else (1.03 + 0.015 * s)

    def ease(x: float, z: float) -> float:
        return _ss(0.905, 1.06, z) * (1.0 - _ss(0.05, 0.15, abs(x)))

    for j in range(nyu + 1):
        for i in range(nx):
            if j < nyu:
                z = zrows[j]
                half = _half_width(z)
                x = -half + 2.0 * half * i / (nx - 1)
                if j == nyu - 1 and abs(x) < _NECK_X:
                    s = np.sin(abs(x) / _NECK_X * np.pi / 2.0)
                    z -= (0.4 * 0.04 * (1.0 - s)) if is_front else (0.4 * 0.015 * (1.0 - s))
            else:
                half = _TIP_X + 0.005
                x = -half + 2.0 * half * i / (nx - 1)
                z = top_z(x)
            pos.append([x, y0 + (y_neck - y0) * ease(x, z), z])
    n = nx * (nyu + 1)
    for j in range(nyu):
        for i in range(nx - 1):
            a, b, c, d = j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1
            if is_front:
                faces += [[a, b, d], [a, d, c]]
            else:
                # 後身頃は法線が+Y向きになるよう巻きを反転する
                faces += [[a, d, b], [a, c, d]]
            grid.update([(min(a, b), max(a, b)), (min(a, c), max(a, c))])
    return (np.array(pos, dtype=np.float32), np.array(faces, dtype=np.uint32),
            grid, nx, nyu + 1)


def _sleeve_panel(side: int, is_front: bool):
    """袖の平面パネル (腕の前後に1枚ずつ。天側・地側の2辺で筒にする)。"""
    ns, nz = 8, 6
    xs = np.linspace(0.135, 0.295, ns)
    zs = np.linspace(0.920, 1.110, nz)
    y0 = -0.075 if is_front else 0.075
    pos, faces, grid = [], [], set()
    for s in range(ns):
        for k in range(nz):
            pos.append([side * xs[s], y0, zs[k]])
    for s in range(ns - 1):
        for k in range(nz - 1):
            a, b = s * nz + k, s * nz + k + 1
            c, d = (s + 1) * nz + k, (s + 1) * nz + k + 1
            # X反転 (side<0) で幾何学的法線が裏返るため、表裏条件にsideを含める
            if (side > 0) == is_front:
                faces += [[a, d, b], [a, c, d]]
            else:
                faces += [[a, b, d], [a, d, c]]
            grid.update([(min(a, b), max(a, b)), (min(a, c), max(a, c))])
    return np.array(pos, dtype=np.float32), np.array(faces, dtype=np.uint32), grid


def make_tshirt():
    """6パーツ結合メッシュ (前後身頃 + 袖前後x左右) と縫合ペアを返す。"""
    pf, ff, gf, nx, ny = _bodice(True)
    pb, fb, gb, _, _ = _bodice(False)
    prf, frf, grf = _sleeve_panel(1, True)
    prb, frb, grb = _sleeve_panel(1, False)
    plf, flf, glf = _sleeve_panel(-1, True)
    plb, flb, glb = _sleeve_panel(-1, False)
    parts = [pf, pb, prf, prb, plf, plb]
    fparts = [ff, fb, frf, frb, flf, flb]
    gparts = [gf, gb, grf, grb, glf, glb]
    offs, o = [], 0
    for p in parts:
        offs.append(o)
        o += len(p)
    positions = np.vstack(parts)
    faces = np.vstack([f + off for f, off in zip(fparts, offs)])
    normal_edges = set()
    for g, off in zip(gparts, offs):
        normal_edges.update((a + off, b + off) for a, b in g)
    NS, NZ = 8, 6  # 袖パネル格子

    sew = set()
    top = (ny - 1) * nx
    # 肩: 前後上端行 (首穴 |x|<0.045 を除く)
    for i in range(nx):
        x = pf[top + i][0]
        if abs(float(x)) < 0.045:
            continue
        sew.add((min(offs[0] + top + i, offs[1] + top + i),
                 max(offs[0] + top + i, offs[1] + top + i)))
    # 脇: アームホール下の側端列
    for j in range(ny):
        z = float(pf[j * nx][2])
        if z > _Z_ARMPIT + 1e-6:
            continue
        for c in (0, nx - 1):
            a, b = offs[0] + j * nx + c, offs[1] + j * nx + c
            sew.add((min(a, b), max(a, b)))
    # 袖: 前後パネルの天側・地側どうしで筒にし、帽子辺をアームホールへ
    # offs[2]=右前, [3]=右後, [4]=左前, [5]=左後
    for (of, ob, sgn) in [(offs[2], offs[3], 1), (offs[4], offs[5], -1)]:
        for s in range(NS):
            a, b = of + s * NZ + (NZ - 1), ob + s * NZ + (NZ - 1)
            sew.add((min(a, b), max(a, b)))
            a, b = of + s * NZ + 0, ob + s * NZ + 0
            sew.add((min(a, b), max(a, b)))
        ah = []
        for obb, pbb in [(offs[0], pf), (offs[1], pb)]:
            for j in range(ny):
                z = float(pbb[j * nx][2])
                if z <= _Z_ARMPIT + 1e-6:
                    continue
                for c in (0, nx - 1):
                    if float(pbb[j * nx + c][0]) * sgn > 0:
                        ah.append(obb + j * nx + c)
        for obb, pbb in [(offs[0], pf), (offs[1], pb)]:
            for c in (0, nx - 1):
                if float(pbb[top + c][0]) * sgn > 0:
                    ah.append(obb + top + c)
        cap = [of + 0 * NZ + k for k in range(NZ)] + [ob + 0 * NZ + k for k in range(NZ)]
        used = set()
        for c in cap:
            best, bd = None, 1e9
            for a in ah:
                d = float(np.sum((positions[c] - positions[a]) ** 2))
                if d < bd:
                    best, bd = a, d
            sew.add((min(c, best), max(c, best)))
            used.add(best)
        for a in ah:
            if a not in used:
                best, bd = None, 1e9
                for c in cap:
                    d = float(np.sum((positions[c] - positions[a]) ** 2))
                    if d < bd:
                        best, bd = c, d
                sew.add((min(a, best), max(a, best)))
    return positions, faces, normal_edges, sorted(sew)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-blend", action="store_true")
    args = ap.parse_args()

    BODY_DIR.mkdir(parents=True, exist_ok=True)
    TSHIRT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 素体OBJ化
    body_obj = BODY_DIR / "SiroinoSotai_Mobile_collider.obj"
    if not args.skip_blend:
        if not BLEND.exists():
            raise FileNotFoundError(f"blend not found: {BLEND}")
        bl = resolve_blender("5.2")
        tmp_json = REPO / "tmp" / "_body_dump.json"
        script = CONVERT_SCRIPT.replace("__OUT__", str(tmp_json).replace("\\", "\\\\"))
        r = subprocess.run([str(bl), "-b", "--factory-startup", str(BLEND),
                            "--python-expr", script],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        print([l for l in (r.stdout + r.stderr).splitlines() if "convert" in l or "Error" in l][-5:])
        if not tmp_json.exists():
            raise RuntimeError("body dump failed")
        dump = json.loads(tmp_json.read_text(encoding="utf-8"))
        export_obj(body_obj, np.array(dump["positions"], dtype=np.float32),
                   np.array(dump["faces"], dtype=np.uint32))
        tmp_json.unlink(missing_ok=True)
    else:
        if not body_obj.exists():
            raise FileNotFoundError(f"{body_obj} が無いため --skip-blend 不可")

    prov = {
        "source": "https://booth.pm/ja/items/8268676",
        "asset": "SiroinoSotai v1.0 / SiroinoSotai_Mobile (full body)",
        "license": "CC0 1.0 Universal (FBX/blend/textures). Logo/VRChat SDK/lilToonは対象外のため同梱しない",
        "credit": "素体: しろいの (Siroino Works) 『SiroinoSotai』https://booth.pm/ja/items/8268676 / CC0 1.0",
        "extraction": "shape_keys=0, ARMATURE modifier removed (rest shape), matrix_world applied, loop_triangles",
        "measured_m": {"xmin": -0.588, "xmax": 0.588, "zmin": 0.035, "zmax": 1.143,
                       "chest_x_half": 0.111, "chest_y_front": -0.093, "chest_y_back": 0.057},
    }
    (BODY_DIR / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2. Tシャツ型紙 (前身頃・後身頃・左右袖)
    positions, faces, edge_set, sew_list = make_tshirt()
    sew = np.array(sew_list, dtype=np.uint32)
    normal_edges = sorted(edge_set)

    panels_obj = TSHIRT_DIR / "tshirt_panels.obj"
    export_obj(panels_obj, positions, faces)
    (TSHIRT_DIR / "tshirt_sewing.json").write_text(
        json.dumps({"sewing_springs": sew.tolist(),
                    "normal_edges": [list(e) for e in normal_edges],
                    "pieces": {"bodice": "front+back (neckline/shoulder/armhole shaped)",
                               "sleeves": "L/R tapered tubes around arms",
                               "open": ["neckline", "hem", "sleeve cuffs"]}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    scene = {"object_name": "TShirtOnSiroino",
             "mesh_file": "tshirt_panels.obj", "sewing_file": "tshirt_sewing.json",
             "collider_mesh_file": "../../bodies/siroino/SiroinoSotai_Mobile_collider.obj",
             "collider_type": "MESH", "collider_thickness": 0.02,
             "collider_single_sided": False, "collider_friction": 0.8,
             "config": {"substeps": 20, "solver_iterations": 10,
                        "tension_stiffness": 10000.0, "bending_stiffness": 20.0,
                        "thickness": 0.005, "sewing_shrink_speed": 0.8,
                        "sewing_priority_enabled": True, "sewing_priority_threshold": 0.9,
                        "sewing_priority_merge_dist": 0.005, "sewing_priority_ramp_frames": 3,
                        "sewing_priority_max_frames": 600}}
    (TSHIRT_DIR / "tshirt_scene.json").write_text(
        json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] body={body_obj} verts={len(positions)} garment_tris={len(faces)} sew={len(sew)}")


if __name__ == "__main__":
    main()
