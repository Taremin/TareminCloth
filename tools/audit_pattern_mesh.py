"""
tools/audit_pattern_mesh.py

型紙（Pattern Mesh）の幾何学的健全性と縫合トポロジーを包括的に監査するCLIツール。
以下の欠陥・異常を自動検出し、早期発見します：
1. 面の裏返り・反転（Inverted / Flipped Faces: 法線反転・負の符号付き面積）
2. 同一平面内の折り返重なり（Folded Adjacent Edges: 二面角 <= 0 による自己めり込み）
3. 型紙境界の未縫合大穴（Unsewn Boundary Holes: 開口部以外の未縫合境界頂点）
4. 縫合トポロジー分岐・異常（Branching / Multi-Endpoints: 次数 >= 2 の不正合流）
5. ゼロ面積縮退面（Degenerate Faces）
6. 非隣接面の物理的貫通（BVH Overlaps）

使用法:
  python tools/audit_pattern_mesh.py [obj_name]
"""

import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent
_python_dir = _repo_root / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from taremin_cloth.utils.validation import validate_cloth_mesh

def audit_cloth_object(obj):
    if not obj or obj.type != 'MESH':
        print(f"Error: {obj} is not a valid MESH object.")
        return False

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.normal_update()
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    report = {
        "obj_name": obj.name,
        "total_verts": len(bm.verts),
        "total_edges": len(bm.edges),
        "total_faces": len(bm.faces),
        "flipped_faces": [],
        "folded_edges": [],
        "unsewn_holes": [],
        "bvh_overlaps": [],
        "addon_issues": [],
    }

    # 1. 面法線の反転（裏返り・自己交差）チェック
    for f in bm.faces:
        c = f.calc_center_median()
        y_span = max(v.co.y for v in f.verts) - min(v.co.y for v in f.verts)
        x_span = max(v.co.x for v in f.verts) - min(v.co.x for v in f.verts)

        if "Pants" in obj.name:
            # パンツ (前パネル Y ≈ -0.14m, 後パネル Y ≈ 0.09m)
            if y_span < 1e-3 and c.y < 0.0:
                if f.normal.y > -0.5:
                    report["flipped_faces"].append((f.index, c, f.normal.copy()))
            elif y_span < 1e-3 and c.y > 0.0:
                if f.normal.y < 0.5:
                    report["flipped_faces"].append((f.index, c, f.normal.copy()))
        else:
            # トップス共通 (Tシャツ, Yシャツ, フード付きパーカー)
            # 1. 前身頃 (XZ平面: y_span ≈ 0, Y < 0, Z < 1.05m)
            if y_span < 1e-3 and c.y < -0.05 and c.z < 1.05:
                if f.normal.y > -0.5:
                    report["flipped_faces"].append((f.index, c, f.normal.copy()))
            # 2. 後身頃 (XZ平面: y_span ≈ 0, Y > 0.05, Z < 1.05m)
            elif y_span < 1e-3 and c.y > 0.05 and c.z < 1.05:
                if f.normal.y < 0.5:
                    report["flipped_faces"].append((f.index, c, f.normal.copy()))
            # 3. フード頭部パネル (YZ平面またはハの字傾斜YZ平面: Z >= 1.035m, 袖 |X| >= 0.085 を除く)
            elif "Hood" in obj.name and c.z >= 1.035 and (c.z > 1.06 or abs(c.x) < 0.085):
                if c.x > 0.0 and f.normal.x < 0.5:
                    report["flipped_faces"].append((f.index, c, f.normal.copy()))
                elif c.x < 0.0 and f.normal.x > -0.5:
                    report["flipped_faces"].append((f.index, c, f.normal.copy()))
            # 4. 台襟 (首周り帯状パネル: Z >= 1.035m, 身頃中央)
            elif "Shirt" in obj.name and c.z >= 1.035:
                pass
            # 5. 袖 (腕の高さ, 身頃外側: normal.z は +Z方向)
            elif abs(c.x) >= 0.085 and 0.88 <= c.z <= 1.06:
                if f.normal.z < 0.5:
                    report["flipped_faces"].append((f.index, c, f.normal.copy()))

    # 2. 同一平面内の折り返重なりエッジ（二面角内積 <= 0）
    for e in bm.edges:
        if len(e.link_faces) == 2:
            f1, f2 = e.link_faces[0], e.link_faces[1]
            dot = f1.normal.dot(f2.normal)
            if dot < 0.0:
                c = (e.verts[0].co + e.verts[1].co) * 0.5
                report["folded_edges"].append((e.index, f1.index, f2.index, dot, c))

    # 3. 未縫合境界頂点（大穴）の検出
    # 首ぐり開口部、裾、袖口、ズボン裾・ウエスト、フード顔穴以外の境界頂点で縫合線のない頂点
    for v in bm.verts:
        b_edges = [e for e in v.link_edges if len(e.link_faces) == 1]
        if b_edges:
            s_edges = [e for e in v.link_edges if len(e.link_faces) == 0]
            c = v.co
            if not s_edges:
                # 衣服ごとの開口部判定
                is_legit_opening = False
                if "TShirt" in obj.name or "Shirt" in obj.name or "Hood" in obj.name:
                    # 裾 (Z < 0.70)
                    if c.z < 0.70:
                        is_legit_opening = True
                    # 首ぐり開口部 (Tシャツのみ: Z > 0.97, |X| < 0.045)
                    elif "TShirt" in obj.name and c.z > 0.97 and abs(c.x) < 0.045:
                        is_legit_opening = True
                    # 台襟開口部 (上端 Z > 1.06 または 左右端短辺 |X| > 0.120)
                    elif "Shirt" in obj.name and c.z >= 1.048 and (c.z > 1.06 or abs(c.x) > 0.120):
                        is_legit_opening = True
                    # シャツ前立て開口部 (ボタン間: |X| <= 0.015, Y in [-0.125, -0.105])
                    elif "Shirt" in obj.name and abs(c.x) <= 0.015 and -0.125 <= c.y <= -0.105:
                        is_legit_opening = True
                    # 袖口 (袖外側端: |X| > 0.22)
                    elif abs(c.x) > 0.22:
                        is_legit_opening = True
                    # フード顔開口部 (Z >= 1.035, Y <= -0.04)
                    elif "Hood" in obj.name and c.z >= 1.035 and c.y <= -0.04:
                        is_legit_opening = True
                elif "Pants" in obj.name:
                    # 裾 (Z < 0.20) または ウエスト (Z > 0.78)
                    if c.z < 0.20 or c.z > 0.78:
                        is_legit_opening = True

                if not is_legit_opening:
                    report["unsewn_holes"].append((v.index, c))

    # 4. BVH自己交差判定（非隣接面交差）
    bvh = BVHTree.FromBMesh(bm, epsilon=0.0001)
    overlaps = bvh.overlap(bvh)
    for f1_idx, f2_idx in overlaps:
        if f1_idx >= f2_idx:
            continue
        f1, f2 = bm.faces[f1_idx], bm.faces[f2_idx]
        if len(set(f1.verts).intersection(set(f2.verts))) == 0:
            report["bvh_overlaps"].append((f1_idx, f2_idx))

    # 5. アドオン公式バリデータ
    rep = validate_cloth_mesh(obj)
    if rep.degenerate_faces.has_issues:
        deg = rep.degenerate_faces
        count = deg.zero_area_poly_count or deg.degenerate_tri_count
        report["addon_issues"].append(f"縮退面: {count}個")
    if rep.sewing_topology.has_issues:
        for b in rep.sewing_topology.branching_issues:
            report["addon_issues"].append(f"縫合トポロジー異常: {b.issue_type} ({b.message})")

    # 6. 紐パーツ (string_parts) の異常長ジャンプエッジ検査 (シームブリッジ縫合区間を除く実構造エッジ)
    settings = getattr(obj, "taremin_cloth", None)
    if settings is not None and hasattr(settings, "string_parts"):
        import numpy as np
        for sp in settings.string_parts:
            c_obj = getattr(sp, "curve_object", None)
            if c_obj and getattr(c_obj, "type", None) == 'CURVE' and c_obj.data:
                guide_str = getattr(sp, "guide_vert_indices_str", "")
                guide_cloth_verts = [int(v.strip()) for v in guide_str.split(",") if v.strip().isdigit()]
                seam_bridge_str = getattr(sp, "seam_bridge_pairs_str", "")
                seam_bridges = set()
                if seam_bridge_str:
                    for pair_item in seam_bridge_str.split(";"):
                        parts = [int(x.strip()) for x in pair_item.split(",") if x.strip().isdigit()]
                        if len(parts) == 2:
                            seam_bridges.add((min(parts[0], parts[1]), max(parts[0], parts[1])))
                tail_left = int(getattr(sp, "tail_vert_count_left", 8) or 0)
                tail_right = int(getattr(sp, "tail_vert_count_right", 8) or 0)

                for spline in c_obj.data.splines:
                    pts = [np.array(p.co[:3], dtype=np.float64) for p in spline.points]
                    n_c = len(pts)
                    if n_c >= 3:
                        ch_start = min(tail_left, n_c)
                        ch_end = max(ch_start, n_c - tail_right)
                        n_ch = ch_end - ch_start
                        bridge_set = set()
                        if seam_bridges and len(guide_cloth_verts) >= 2 and n_ch >= 2:
                            n_g = len(guide_cloth_verts)
                            for g_idx in range(n_g - 1):
                                u_c, v_c = guide_cloth_verts[g_idx], guide_cloth_verts[g_idx + 1]
                                if (min(u_c, v_c), max(u_c, v_c)) in seam_bridges:
                                    v_r0 = int(round(ch_start + (float(g_idx) / max(1, n_g - 1)) * (n_ch - 1)))
                                    v_r1 = int(round(ch_start + (float(g_idx + 1) / max(1, n_g - 1)) * (n_ch - 1)))
                                    for e_l in range(min(v_r0, v_r1), min(max(v_r0, v_r1), n_c - 1)):
                                        bridge_set.add(e_l)

                        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                        valid_indices = [i for i in range(len(seg_lens)) if i not in bridge_set]
                        if valid_indices:
                            valid_lens = seg_lens[valid_indices]
                            med_len = float(np.median(valid_lens))
                            max_pos = int(np.argmax(valid_lens))
                            max_idx = valid_indices[max_pos]
                            max_len = float(valid_lens[max_pos])
                            if med_len > 1e-5 and max_len > max(0.03, med_len * 5.0):
                                report["addon_issues"].append(
                                    f"紐パーツ異常長エッジ ({c_obj.name}): pt[{max_idx}]-pt[{max_idx+1}] 長さ {max_len*1000:.1f}mm (中央値 {med_len*1000:.1f}mm)"
                                )

    bm.free()

    # レポート表示
    print(f"======================================================================")
    print(f"【型紙メッシュ品質監査】: {obj.name}")
    print(f"  頂点数: {report['total_verts']:,}, エッジ数: {report['total_edges']:,}, 面数: {report['total_faces']:,}")
    print(f"----------------------------------------------------------------------")
    print(f"  1. 反転・裏返り面 (Inverted Faces)    : {len(report['flipped_faces']):4d} 件 (合格: 0)")
    print(f"  2. 同一平面折り畳みエッジ (Folded)   : {len(report['folded_edges']):4d} 件 (合格: 0)")
    print(f"  3. 肩・AH周辺の未縫合穴 (Holes)      : {len(report['unsewn_holes']):4d} 個 (合格: 0)")
    print(f"  4. 非隣接面交差・貫通 (BVH Overlaps) : {len(report['bvh_overlaps']):4d} 件 (合格: 0)")
    print(f"  5. アドオン検証異常 (Addon Issues)   : {len(report['addon_issues']):4d} 件 (合格: 0)")

    is_clean = (
        len(report['flipped_faces']) == 0
        and len(report['folded_edges']) == 0
        and len(report['unsewn_holes']) == 0
        and len(report['bvh_overlaps']) == 0
        and len(report['addon_issues']) == 0
    )

    if not is_clean:
        print(f"\n[!] 検出された問題の詳細:")
        for f in report['flipped_faces'][:5]:
            print(f"    - 反転面 {f[0]} at ({f[1].x:.3f}, {f[1].y:.3f}, {f[1].z:.3f}), normal=({f[2].x:.2f}, {f[2].y:.2f}, {f[2].z:.2f})")
        for e in report['folded_edges'][:5]:
            print(f"    - 折り畳みエッジ {e[0]} (Face {e[1]} ↔ {e[2]}) at ({e[4].x:.3f}, {e[4].y:.3f}, {e[4].z:.3f})")
        for v in report['unsewn_holes'][:5]:
            print(f"    - 未縫合穴あき頂点 v{v[0]} at ({v[1].x:.3f}, {v[1].y:.3f}, {v[1].z:.3f})")
        for ov in report['bvh_overlaps'][:5]:
            print(f"    - BVH交差面ペア: Face {ov[0]} ↔ Face {ov[1]}")
        for iss in report['addon_issues']:
            print(f"    - {iss}")

    print(f"  総合判定: {'PASS (完全健全)' if is_clean else 'FAIL (不具合あり)'}")
    print(f"======================================================================\n")
    return is_clean

def audit_all_patterns():
    targets = ["Pattern_TShirt", "Pattern_Shirt", "Pattern_Pants", "Pattern_Hood"]
    all_pass = True
    for name in targets:
        obj = bpy.data.objects.get(name)
        if obj:
            ok = audit_cloth_object(obj)
            if not ok:
                all_pass = False
        else:
            print(f"Object {name} is not present in the current scene.")
    return all_pass

if __name__ == "__main__":
    audit_all_patterns()
