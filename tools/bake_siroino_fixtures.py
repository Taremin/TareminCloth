#!/usr/bin/env python3
"""SiroinoSotai Mobile完全体のコライダーOBJ化 + 1cm正方格子Tシャツ型紙の生成.

服飾造形のTシャツ原型理論およびClothシミュレーションの等方性格子設計に基づき、
1cm間隔（0.01m）の正方形方眼紙ベースのQuad格子から切り出した4パーツ構成（前後身頃・左右1枚袖）
のTシャツ型紙と、弧長完全整合の縫合スプリング群を生成する。

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
import math
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


# ==============================================================================
# 1cm正方形方眼紙ベースのTシャツ型紙設計 (SiroinoSotai Mobile 実寸)
# ==============================================================================
Z_HEM = 0.650          # 裾高さ
Z_ARMPIT = 0.930       # 脇下高さ (差 0.28m = 28cm)
Z_SHOULDER = 1.040     # 肩先高さ (差 0.11m = 11cm)
Z_NECK = 1.060         # ネックサイド高さ (差 0.02m = 2cm)
Z_FNP = 0.990          # 前ネック中心 (深さ 7cm のラウンドネック)
Z_BNP = 1.045          # 後ネック中心 (深さ 1.5cm のバックネック)

W_HEM = 0.140          # 裾半幅 14cm (全幅 28cm)
W_ARMPIT = 0.140       # 脇下半幅 14cm (全幅 28cm: 胴体は 28cm x 28cm の完全正方グリッド)
W_SHOULDER = 0.120     # 肩先半幅 12cm
W_NECK = 0.040         # ネックサイド半幅 4cm (首幅 8cm)

DELTA_AH_FRONT = 0.020 # 前アームホール内側入り込み
DELTA_AH_BACK = 0.010  # 後アームホール内側入り込み

ARM_X_START = 0.122    # 袖付け根X
ARM_X_END = 0.232      # 袖口X (袖丈 0.11m = 11cm)
ARM_Y = 0.008          # 腕中心Y
ARM_Z = 1.017          # 腕中心Z
SLEEVE_RADIUS = 0.044  # 袖口半径 44mm (周長約 276mm: 腕コライダーに対する十分なゆとり)

# 1cm方眼グリッド分割数
N_SH = 8               # 肩線 8cm (8区間, 各1.0cm)
N_NK = 12              # 衿ぐり 12cm (12区間, 各1.0cm)
N_X = 2 * N_SH + N_NK  # 横合計 28区間 (全幅 28cm, 各1.0cm)

N_BODY = 28            # 裾〜脇下 28cm (28区間, 各1.0cm)
N_AH = 13              # アームホール 13区間 (各辺長 ~1.0cm)
N_Y = N_BODY + N_AH    # 縦合計 41区間 (全高 41cm, 各1.0cm)

N_SLEEVE_U = 2 * N_AH  # 袖周り 26区間 (各辺長 ~1.0cm)
N_SLEEVE_V = 11        # 袖丈 11cm (11区間, 各1.0cm)


def coons_bodice(is_front: bool):
    """前身頃または後身頃を1cm正方格子ベースで生成する."""
    u_neck_l = N_SH / N_X
    u_neck_r = (N_SH + N_NK) / N_X
    v_armpit = N_BODY / N_Y
    z_center = Z_FNP if is_front else Z_BNP
    delta_ah = DELTA_AH_FRONT if is_front else DELTA_AH_BACK

    def c_bottom(u: float):
        return np.array([-W_HEM + 2.0 * W_HEM * u, Z_HEM])

    def c_top(u: float):
        if u <= u_neck_l:
            t = u / u_neck_l
            x = -W_SHOULDER + (-W_NECK - (-W_SHOULDER)) * t
            z = Z_SHOULDER + (Z_NECK - Z_SHOULDER) * t
        elif u >= u_neck_r:
            t = (u - u_neck_r) / (1.0 - u_neck_r)
            x = W_NECK + (W_SHOULDER - W_NECK) * t
            z = Z_NECK + (Z_SHOULDER - Z_NECK) * t
        else:
            t = (u - u_neck_l) / (u_neck_r - u_neck_l)
            x = -W_NECK + 2.0 * W_NECK * t
            z = Z_NECK - (Z_NECK - z_center) * (math.sin(math.pi * t) ** 2)
        return np.array([x, z])

    def c_left(v: float):
        if v <= v_armpit:
            t = v / v_armpit
            x = -W_HEM + (-W_ARMPIT - (-W_HEM)) * t
            z = Z_HEM + (Z_ARMPIT - Z_HEM) * t
        else:
            t = (v - v_armpit) / (1.0 - v_armpit)
            x = -W_ARMPIT + (-W_SHOULDER - (-W_ARMPIT)) * t + delta_ah * math.sin(math.pi * t)
            z = Z_ARMPIT + (Z_SHOULDER - Z_ARMPIT) * t
        return np.array([x, z])

    def c_right(v: float):
        pt = c_left(v)
        return np.array([-pt[0], pt[1]])

    p00 = c_bottom(0.0)
    p10 = c_bottom(1.0)
    p01 = c_top(0.0)
    p11 = c_top(1.0)

    u_vals = np.linspace(0.0, 1.0, N_X + 1)
    v_vals = np.linspace(0.0, 1.0, N_Y + 1)
    verts_2d = np.zeros((N_Y + 1, N_X + 1, 2), dtype=np.float32)

    for j, v in enumerate(v_vals):
        cl = c_left(v)
        cr = c_right(v)
        for i, u in enumerate(u_vals):
            cb = c_bottom(u)
            ct = c_top(u)
            p = ((1.0 - v) * cb + v * ct +
                 (1.0 - u) * cl + u * cr -
                 ((1.0 - u) * (1.0 - v) * p00 +
                  u * (1.0 - v) * p10 +
                  (1.0 - u) * v * p01 +
                  u * v * p11))
            verts_2d[j, i] = p

    y_pos = -0.110 if is_front else 0.090
    verts_3d = []
    for j in range(N_Y + 1):
        for i in range(N_X + 1):
            xz = verts_2d[j, i]
            verts_3d.append([float(xz[0]), y_pos, float(xz[1])])
    verts_3d = np.array(verts_3d, dtype=np.float32)

    faces = []
    edges = set()
    for j in range(N_Y):
        for i in range(N_X):
            v00 = j * (N_X + 1) + i
            v10 = j * (N_X + 1) + (i + 1)
            v01 = (j + 1) * (N_X + 1) + i
            v11 = (j + 1) * (N_X + 1) + (i + 1)
            # 交互対角線 (Alternating diagonals) でせん断異方性バイアスを完全排除
            if (i + j) % 2 == 0:
                if is_front:
                    faces.append([v00, v10, v11])
                    faces.append([v00, v11, v01])
                else:
                    faces.append([v00, v11, v10])
                    faces.append([v00, v01, v11])
                diag = (v00, v11)
            else:
                if is_front:
                    faces.append([v00, v10, v01])
                    faces.append([v10, v11, v01])
                else:
                    faces.append([v00, v01, v10])
                    faces.append([v10, v01, v11])
                diag = (v10, v01)
            for a, b in [(v00, v10), (v10, v11), (v11, v01), (v01, v00), diag]:
                edges.add((min(a, b), max(a, b)))

    return verts_3d, np.array(faces, dtype=np.uint32), edges


def make_sleeve(side: int, ah_front_segs: np.ndarray, ah_back_segs: np.ndarray):
    """アームホールの各辺長と整合した袖メッシュを円筒ラップ配置で生成する."""
    tot_front = float(np.sum(ah_front_segs))
    tot_back = float(np.sum(ah_back_segs))
    tot_ah = tot_front + tot_back

    delta_theta = 0.08 # 袖下の隙間
    available_angle = 2.0 * math.pi - delta_theta

    angles = [0.0]
    for seg in ah_front_segs:
        angles.append(angles[-1] + (seg / tot_ah) * available_angle)
    for seg in reversed(ah_back_segs):
        angles.append(angles[-1] + (seg / tot_ah) * available_angle)
    angles = np.array(angles, dtype=np.float32)

    theta_top = angles[N_AH]
    theta_vals = (math.pi / 2.0) - (angles - theta_top)
    v_vals = np.linspace(0.0, 1.0, N_SLEEVE_V + 1)

    r_cap = tot_ah / available_angle # 袖山半径 (アームホール周長と厳密一致)
    r_cuff = SLEEVE_RADIUS           # 袖口半径 (腕コライダーに対する十分なゆとり)

    verts_3d = []
    for j, v in enumerate(v_vals):
        u_ratio = np.linspace(0.0, 1.0, N_SLEEVE_U + 1)
        x_cap = ARM_X_START + 0.012 * (1.0 - np.sin(np.pi * u_ratio))
        x_row = ARM_X_END - (ARM_X_END - x_cap) * v

        # 袖口(r_cuff)から袖山(r_cap)への滑らかなテーパー
        r = r_cuff + (r_cap - r_cuff) * v

        for i in range(N_SLEEVE_U + 1):
            th = float(theta_vals[i])
            y = ARM_Y + r * math.cos(th)
            z = ARM_Z + r * math.sin(th)
            x = side * float(x_row[i])
            verts_3d.append([x, y, z])

    verts_3d = np.array(verts_3d, dtype=np.float32)

    faces = []
    edges = set()
    for j in range(N_SLEEVE_V):
        for i in range(N_SLEEVE_U):
            v00 = j * (N_SLEEVE_U + 1) + i
            v10 = j * (N_SLEEVE_U + 1) + (i + 1)
            v01 = (j + 1) * (N_SLEEVE_U + 1) + i
            v11 = (j + 1) * (N_SLEEVE_U + 1) + (i + 1)
            if (i + j) % 2 == 0:
                if side > 0:
                    faces.append([v00, v10, v11])
                    faces.append([v00, v11, v01])
                else:
                    faces.append([v00, v11, v10])
                    faces.append([v00, v01, v11])
                diag = (v00, v11)
            else:
                if side > 0:
                    faces.append([v00, v10, v01])
                    faces.append([v10, v11, v01])
                else:
                    faces.append([v00, v01, v10])
                    faces.append([v10, v01, v11])
                diag = (v10, v01)
            for a, b in [(v00, v10), (v10, v11), (v11, v01), (v01, v00), diag]:
                edges.add((min(a, b), max(a, b)))

    return verts_3d, np.array(faces, dtype=np.uint32), edges


def make_tshirt():
    """1cm方眼ベースの4パーツ結合メッシュと縫合ペアおよびグループを返す."""
    vf, ff, ef = coons_bodice(True)
    vb, fb, eb = coons_bodice(False)

    ah_front_pts = [vf[(N_BODY + k) * (N_X + 1) + N_X] for k in range(N_AH + 1)]
    ah_front_segs = np.array([np.linalg.norm(ah_front_pts[k+1] - ah_front_pts[k]) for k in range(N_AH)])

    ah_back_pts = [vb[(N_BODY + k) * (N_X + 1) + N_X] for k in range(N_AH + 1)]
    ah_back_segs = np.array([np.linalg.norm(ah_back_pts[k+1] - ah_back_pts[k]) for k in range(N_AH)])

    vr, fr, er = make_sleeve(1, ah_front_segs, ah_back_segs)
    vl, fl, el = make_sleeve(-1, ah_front_segs, ah_back_segs)

    parts = [vf, vb, vr, vl]
    fparts = [ff, fb, fr, fl]
    eparts = [ef, eb, er, el]

    offs = [0]
    for p in parts[:-1]:
        offs.append(offs[-1] + len(p))

    positions = np.vstack(parts)
    faces = np.vstack([f + off for f, off in zip(fparts, offs)])
    normal_edges = set()
    for e, off in zip(eparts, offs):
        normal_edges.update((a + off, b + off) for a, b in e)

    off_f, off_b, off_r, off_l = offs

    sew_pairs = []
    group_map = {}

    def add_pair(a: int, b: int, grp_name: str):
        p = (min(a, b), max(a, b))
        if p[0] != p[1] and p not in sew_pairs:
            sew_pairs.append(p)
            group_map.setdefault(grp_name, []).append(len(sew_pairs) - 1)

    # 1. 脇線 (Side Seams)
    for j in range(N_BODY + 1):
        add_pair(off_f + j * (N_X + 1) + 0, off_b + j * (N_X + 1) + 0, "side_seams_left")
        add_pair(off_f + j * (N_X + 1) + N_X, off_b + j * (N_X + 1) + N_X, "side_seams_right")

    # 2. 肩線 (Shoulder Seams)
    for i in range(N_SH + 1):
        add_pair(off_f + N_Y * (N_X + 1) + i, off_b + N_Y * (N_X + 1) + i, "shoulder_left")
    for k in range(N_SH + 1):
        i = (N_X - N_SH) + k
        add_pair(off_f + N_Y * (N_X + 1) + i, off_b + N_Y * (N_X + 1) + i, "shoulder_right")

    # 3. 袖下線 (Underarm Seams)
    for j in range(N_SLEEVE_V + 1):
        add_pair(off_r + j * (N_SLEEVE_U + 1) + 0, off_r + j * (N_SLEEVE_U + 1) + N_SLEEVE_U, "sleeve_inseam_right")
        add_pair(off_l + j * (N_SLEEVE_U + 1) + 0, off_l + j * (N_SLEEVE_U + 1) + N_SLEEVE_U, "sleeve_inseam_left")

    # 4. アームホールと袖山
    for k in range(N_AH + 1):
        add_pair(off_f + (N_BODY + k) * (N_X + 1) + N_X,
                 off_r + N_SLEEVE_V * (N_SLEEVE_U + 1) + k, "armhole_right")
        add_pair(off_b + (N_BODY + k) * (N_X + 1) + N_X,
                 off_r + N_SLEEVE_V * (N_SLEEVE_U + 1) + (2 * N_AH - k), "armhole_right")

    for k in range(N_AH + 1):
        add_pair(off_f + (N_BODY + k) * (N_X + 1) + 0,
                 off_l + N_SLEEVE_V * (N_SLEEVE_U + 1) + k, "armhole_left")
        add_pair(off_b + (N_BODY + k) * (N_X + 1) + 0,
                 off_l + N_SLEEVE_V * (N_SLEEVE_U + 1) + (2 * N_AH - k), "armhole_left")

    seam_groups = []
    for gname, p_indices in group_map.items():
        seam_groups.append({
            "name": gname,
            "pairs": p_indices,
            "tolerance_mm": 5.0,
            "ease_mm": 0.0,
        })

    return positions, faces, normal_edges, sew_pairs, seam_groups


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

    # 2. 1cm方眼ベースTシャツ型紙
    positions, faces, edge_set, sew_list, seam_groups = make_tshirt()
    sew = np.array(sew_list, dtype=np.uint32)
    normal_edges = sorted(edge_set)

    panels_obj = TSHIRT_DIR / "tshirt_panels.obj"
    export_obj(panels_obj, positions, faces)
    (TSHIRT_DIR / "tshirt_sewing.json").write_text(
        json.dumps({"sewing_springs": sew.tolist(),
                    "normal_edges": [list(e) for e in normal_edges],
                    "seam_groups": seam_groups,
                    "pieces": {"bodice": "front+back 1cm square grid (neckline/shoulder/armhole shaped by Coons Patch)",
                               "sleeves": "L/R 1cm square grid 1-piece set-in sleeves wrapped around arms",
                               "open": ["neckline", "hem", "sleeve cuffs"]}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    scene = {"object_name": "TShirtOnSiroino",
             "mesh_file": "tshirt_panels.obj", "sewing_file": "tshirt_sewing.json",
             "collider_mesh_file": "../../bodies/siroino/SiroinoSotai_Mobile_collider.obj",
             "collider_type": "MESH", "collider_thickness": 0.008,
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
