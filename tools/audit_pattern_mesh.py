"""
tools/audit_pattern_mesh.py

型紙（Pattern Mesh）の幾何学的健全性と縫合トポロジーを汎用的に監査するCLIツール。
任意のクロスオブジェクトに対して以下の欠陥・異常を自動検出します：
1. 共有エッジのループ巻き方向不一致（局所的な面の裏返り・法線反転）
2. 同一平面内の折り返し重なり（隣接面法線内積 < 0 による自己めり込み）
3. 自由境界（未縫合境界）の連結ループおよび境界頂点統計
4. 非隣接面の物理的貫通（BVH Overlaps）
5. 公式バリデータによるゼロ面積縮退面・縫合トポロジー分岐異常（Validate Issues）
6. 紐パーツ（string_parts）のシームブリッジ区間を除く異常長エッジ

使用法:
  blender -b scene.blend --python tools/audit_pattern_mesh.py -- [obj_name ...]
  python tools/audit_pattern_mesh.py [obj_name ...]
"""

import argparse
from collections import deque
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_repo_root = Path(__file__).resolve().parent.parent
_python_dir = _repo_root / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import bpy
import bmesh
from mathutils.bvhtree import BVHTree
from taremin_cloth.utils.validation import validate_cloth_mesh


def _count_boundary_loops(bm: Any) -> tuple[int, int]:
    """未縫合の境界頂点数と連結境界ループ数を算出する"""
    boundary_verts = {
        v for v in bm.verts
        if any(len(e.link_faces) == 1 for e in v.link_edges)
        and not any(len(e.link_faces) == 0 for e in v.link_edges)
    }
    boundary_edges = {
        e for e in bm.edges
        if len(e.link_faces) == 1
        and all(v in boundary_verts for v in e.verts)
    }
    visited: set = set()
    loop_count = 0
    for start_v in boundary_verts:
        if start_v in visited:
            continue
        loop_count += 1
        queue = deque([start_v])
        visited.add(start_v)
        while queue:
            v = queue.popleft()
            for e in v.link_edges:
                if e in boundary_edges:
                    other = e.other_vert(v)
                    if other in boundary_verts and other not in visited:
                        visited.add(other)
                        queue.append(other)
    return len(boundary_verts), loop_count


def _collect_string_part_issues(obj: Any) -> List[str]:
    """紐パーツ (string_parts) のシームブリッジを除く異常長エッジを検査する"""
    issues: List[str] = []
    settings = getattr(obj, "taremin_cloth", None)
    if settings is None or not hasattr(settings, "string_parts"):
        return issues

    import numpy as np

    for sp in settings.string_parts:
        c_obj = getattr(sp, "curve_object", None)
        if not (c_obj and getattr(c_obj, "type", None) == 'CURVE' and c_obj.data):
            continue
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
            if n_c < 3:
                continue
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
                    issues.append(
                        f"紐パーツ異常長エッジ ({c_obj.name}): pt[{max_idx}]-pt[{max_idx+1}] 長さ {max_len*1000:.1f}mm (中央値 {med_len*1000:.1f}mm)"
                    )
    return issues


def audit_cloth_object(obj: Any) -> bool:
    """単一の布メッシュオブジェクトを汎用幾何・トポロジー基準で監査する"""
    if not obj or obj.type != 'MESH':
        print(f"Error: {obj} is not a valid MESH object.")
        return False

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    winding_mismatches = []
    folded_edges = []
    non_manifold_edges = []
    bvh_overlaps = []
    addon_issues: List[str] = []

    # 1. 共有エッジのループ巻き方向不一致 & 非多様体 & 同一平面折り畳み検出
    for e in bm.edges:
        n_faces = len(e.link_faces)
        if n_faces > 2:
            c = (e.verts[0].co + e.verts[1].co) * 0.5
            non_manifold_edges.append((e.index, n_faces, c))
            continue
        if n_faces == 2:
            f1, f2 = e.link_faces[0], e.link_faces[1]
            # 2面の共有ループ向きが一致していれば巻き順(法線表裏)が反転している
            if len(e.link_loops) == 2:
                l1, l2 = e.link_loops[0], e.link_loops[1]
                dir1 = (l1.vert.index, l1.link_loop_next.vert.index)
                dir2 = (l2.vert.index, l2.link_loop_next.vert.index)
                if dir1 == dir2:
                    c = (e.verts[0].co + e.verts[1].co) * 0.5
                    winding_mismatches.append((e.index, f1.index, f2.index, c))
            if f1.normal.length > 0.5 and f2.normal.length > 0.5:
                dot = f1.normal.dot(f2.normal)
                if dot < 0.0:
                    c = (e.verts[0].co + e.verts[1].co) * 0.5
                    folded_edges.append((e.index, f1.index, f2.index, dot, c))

    # 2. 自由境界ループ統計（参考メトリクス）
    boundary_vert_count, boundary_loop_count = _count_boundary_loops(bm)

    # 3. BVH自己交差判定（非隣接面交差）
    bvh = BVHTree.FromBMesh(bm, epsilon=0.0001)
    for f1_idx, f2_idx in bvh.overlap(bvh):
        if f1_idx >= f2_idx:
            continue
        f1, f2 = bm.faces[f1_idx], bm.faces[f2_idx]
        if len(set(f1.verts).intersection(set(f2.verts))) == 0:
            bvh_overlaps.append((f1_idx, f2_idx))

    # 4. 公式バリデータ（縮退面・縫合分岐異常）
    rep = validate_cloth_mesh(obj)
    if rep.degenerate_faces.has_issues:
        deg = rep.degenerate_faces
        count = deg.zero_area_poly_count or deg.degenerate_tri_count
        addon_issues.append(f"縮退面: {count}個")
    if rep.sewing_topology.has_issues:
        for b in rep.sewing_topology.branching_issues:
            addon_issues.append(f"縫合トポロジー異常: {b.issue_type} ({b.message})")

    # 5. 紐パーツ異常長ジャンプエッジ検査
    addon_issues.extend(_collect_string_part_issues(obj))

    total_verts = len(bm.verts)
    total_edges = len(bm.edges)
    total_faces = len(bm.faces)
    bm.free()

    print("======================================================================")
    print(f"【型紙メッシュ品質監査】: {obj.name}")
    print(f"  頂点数: {total_verts:,}, エッジ数: {total_edges:,}, 面数: {total_faces:,}")
    print(f"  自由境界統計: 境界頂点 {boundary_vert_count:,} 個 / 連結境界ループ {boundary_loop_count} 本")
    print("----------------------------------------------------------------------")
    print(f"  1. 巻き方向不一致エッジ (Winding Flip) : {len(winding_mismatches):4d} 件 (合格: 0)")
    print(f"  2. 同一平面折り畳みエッジ (Folded)     : {len(folded_edges):4d} 件 (合格: 0)")
    print(f"  3. 非多様体エッジ (Non-manifold)       : {len(non_manifold_edges):4d} 件 (合格: 0)")
    print(f"  4. 非隣接面交差・貫通 (BVH Overlaps)   : {len(bvh_overlaps):4d} 件 (合格: 0)")
    print(f"  5. アドオン検証異常 (Addon Issues)     : {len(addon_issues):4d} 件 (合格: 0)")

    is_clean = (
        len(winding_mismatches) == 0
        and len(folded_edges) == 0
        and len(non_manifold_edges) == 0
        and len(bvh_overlaps) == 0
        and len(addon_issues) == 0
    )

    if not is_clean:
        print("\n[!] 検出された問題の詳細:")
        for e in winding_mismatches[:5]:
            print(f"    - 巻き反転エッジ {e[0]} (Face {e[1]} <-> {e[2]}) at ({e[3].x:.3f}, {e[3].y:.3f}, {e[3].z:.3f})")
        for e in folded_edges[:5]:
            print(f"    - 折り畳みエッジ {e[0]} (Face {e[1]} <-> {e[2]}, dot={e[3]:.2f}) at ({e[4].x:.3f}, {e[4].y:.3f}, {e[4].z:.3f})")
        for e in non_manifold_edges[:5]:
            print(f"    - 非多様体エッジ {e[0]} (link_faces={e[1]}) at ({e[2].x:.3f}, {e[2].y:.3f}, {e[2].z:.3f})")
        for ov in bvh_overlaps[:5]:
            print(f"    - BVH交差面ペア: Face {ov[0]} <-> Face {ov[1]}")
        for iss in addon_issues:
            print(f"    - {iss}")

    print(f"  総合判定: {'PASS (完全健全)' if is_clean else 'FAIL (不具合あり)'}")
    print("======================================================================\n")
    return is_clean


def _resolve_target_objects(target_names: Optional[List[str]] = None) -> List[Any]:
    """監査対象の Blender メッシュオブジェクト一覧を解決する"""
    if target_names:
        resolved = []
        for name in target_names:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                resolved.append(obj)
            else:
                print(f"Warning: Object '{name}' is not present in the current scene.")
        return resolved

    cloth_objs = [
        obj for obj in bpy.data.objects
        if obj.type == 'MESH' and getattr(getattr(obj, "taremin_cloth", None), "is_cloth", False)
    ]
    if cloth_objs:
        return cloth_objs

    active = getattr(getattr(bpy, "context", None), "active_object", None)
    if active is not None and active.type == 'MESH':
        return [active]
    return []


def audit_all_patterns(target_names: Optional[List[str]] = None) -> bool:
    """指定またはシーン内のクロスオブジェクトをすべて監査する"""
    targets = _resolve_target_objects(target_names)
    if not targets:
        print("No target cloth/mesh objects found in the current scene.")
        return False

    all_pass = True
    for obj in targets:
        if not audit_cloth_object(obj):
            all_pass = False
    return all_pass


def _parse_cli_args(raw_argv: List[str]) -> argparse.Namespace:
    if "--" in raw_argv:
        raw_argv = raw_argv[raw_argv.index("--") + 1 :]
    else:
        raw_argv = raw_argv[1:]
    parser = argparse.ArgumentParser(description="型紙・クロス メッシュ幾何品質監査ツール")
    parser.add_argument("obj_names", nargs="*", help="監査対象のオブジェクト名（省略時は is_cloth メッシュまたはアクティブメッシュ）")
    return parser.parse_args(raw_argv)


if __name__ == "__main__":
    cli_args = _parse_cli_args(sys.argv)
    ok = audit_all_patterns(cli_args.obj_names or None)
    sys.exit(0 if ok else 1)
