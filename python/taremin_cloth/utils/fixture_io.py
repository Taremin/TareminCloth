"""E2E fixture manifest の読込口 (Blender非依存).

scene.json は頂点を直埋めせず、OBJ参照で幾何を共有する。
manifest -> SceneInitData相当dict -> ClothSimulator / GUI InitScene の双方へ渡せる。
"""
import json
import os
from typing import Any, Dict, List

import numpy as np


def load_obj(path: str):
    verts, faces = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:4]]
                if len(idx) == 3:
                    faces.append(idx)
    return (np.array(verts, dtype=np.float32),
            np.array(faces, dtype=np.uint32))


def load_scene_manifest(path: str) -> Dict[str, Any]:
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as f:
        scene = json.load(f)
    gv, gf = load_obj(os.path.join(base, scene["mesh_file"]))
    with open(os.path.join(base, scene["sewing_file"]), encoding="utf-8") as f:
        sj = json.load(f)
    out: Dict[str, Any] = {
        "object_name": scene["object_name"],
        "positions": gv,
        "faces": gf,
        "edges": np.array(sj["normal_edges"], dtype=np.uint32),
        "sewing_springs": np.array(sj["sewing_springs"], dtype=np.uint32),
        "seam_groups": sj.get("seam_groups", []),
        "config": scene.get("config", {}),
    }
    col_file = scene.get("collider_mesh_file")
    if col_file:
        cv, cf = load_obj(os.path.normpath(os.path.join(base, col_file)))
        out["collider_positions"] = cv
        out["collider_faces"] = cf
        out["collider_thickness"] = float(scene.get("collider_thickness", 0.01))
        out["collider_friction"] = float(scene.get("collider_friction", 0.5))
        out["collider_single_sided"] = bool(scene.get("collider_single_sided", True))
    return out


def audit_seams(positions: np.ndarray, faces: np.ndarray,
                sewing_springs: np.ndarray,
                seam_groups: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """縫合ペアの辺長整合を検査する.

    境界辺で隣り合うペア同士を鎖 (chain) につなぎ、鎖ごとの両側ポリライン
    長を比べる。対応辺の長さが合わない型紙 (いせ込み過大・取付不可) を検出する。
    seam_groups を渡すとグループ単位の辺長検査も行う。グループの ignore が
    真の場合は検査を免除する (lint の ignore 宣言に相当。reason 必須)。
    """
    pos = np.asarray(positions, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    pairs = [(int(a), int(b)) for a, b in np.asarray(sewing_springs, dtype=np.int64)]

    use: Dict[tuple, int] = {}
    for f in tri:
        for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (min(u, v), max(u, v))
            use[key] = use.get(key, 0) + 1
    boundary = {k for k, c in use.items() if c == 1}
    adj: Dict[int, set] = {}
    for u, v in boundary:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    partner: Dict[int, int] = {}
    orphans: List[List[int]] = []
    for a, b in pairs:
        if a in partner or b in partner:
            orphans.append([a, b])
            continue
        partner[a] = b
        partner[b] = a

    gaps = [float(np.linalg.norm(pos[a] - pos[b])) for a, b in pairs]
    pair_index = {tuple(sorted(p)): i for i, p in enumerate(pairs)}
    visited: set = set()
    chains: List[Dict[str, Any]] = []
    for a0, b0 in pairs:
        if (a0, b0) in visited:
            continue
        run = [(a0, b0)]
        visited.add((a0, b0))

        def step_end(a: int, b: int):
            for a2 in adj.get(a, ()):
                if a2 in partner:
                    b2 = partner[a2]
                    if b2 in adj.get(b, ()) and (a2, b2) not in visited and (b2, a2) not in visited:
                        return (a2, b2)
            return None

        for _ in range(len(pairs)):
            nxt = step_end(*run[-1])
            if nxt is None:
                break
            run.append(nxt)
            visited.add(nxt)
        for _ in range(len(pairs)):
            nxt = step_end(*run[0])
            if nxt is None:
                break
            run.insert(0, nxt)
            visited.add(nxt)
        la = sum(float(np.linalg.norm(pos[run[i][0]] - pos[run[i + 1][0]])) for i in range(len(run) - 1))
        lb = sum(float(np.linalg.norm(pos[run[i][1]] - pos[run[i + 1][1]])) for i in range(len(run) - 1))
        chains.append({"length": len(run), "side_a_m": la, "side_b_m": lb,
                       "diff_m": abs(la - lb),
                       "pairs": [pair_index[tuple(sorted(p))] for p in run]})
    group_results: List[Dict[str, Any]] = []
    if seam_groups:
        pair_set = {tuple(sorted(p)) for p in pairs}
        for g in seam_groups:
            name = str(g.get("name", "?"))
            tol = float(g.get("tolerance_mm", 5.0)) / 1000.0
            ease = float(g.get("ease_mm", 0.0)) / 1000.0
            if g.get("ignore"):
                if not g.get("reason"):
                    raise ValueError(f"seam group '{name}' ignores without reason")
                group_results.append({"name": name, "status": "ignored",
                                      "reason": str(g.get("reason", ""))})
                continue
            idx = [int(i) for i in g.get("pairs", [])]
            missing = [i for i in idx if tuple(sorted(pairs[i])) not in pair_set] if idx else []
            va, vb = set(), set()
            for i in idx:
                if 0 <= i < len(pairs):
                    va.add(pairs[i][0])
                    vb.add(pairs[i][1])
            la = sum(float(np.linalg.norm(pos[u] - pos[v]))
                     for u, v in boundary if u in va and v in va)
            lb = sum(float(np.linalg.norm(pos[u] - pos[v]))
                     for u, v in boundary if u in vb and v in vb)
            diff = abs(la - lb)
            group_results.append({"name": name, "status": "ok" if diff <= tol + ease else "ng",
                                  "side_a_m": la, "side_b_m": lb, "diff_m": diff,
                                  "tolerance_m": tol, "ease_m": ease,
                                  "missing_pairs": missing})
    return {"num_pairs": len(pairs), "num_chains": len(chains),
            "gap_max_m": max(gaps) if gaps else 0.0,
            "gap_mean_m": sum(gaps) / len(gaps) if gaps else 0.0,
            "orphan_pairs": orphans, "chains": chains,
            "groups": group_results}


def to_gui_init_scene(m: Dict[str, Any]) -> Dict[str, Any]:
    """ClothGuiClient.send_init_scene と同形のdictを組み立てる (送信は呼び出し側)."""
    cfg = m.get("config", {})
    n = len(m["positions"])
    tris = None
    if "collider_positions" in m:
        cp, cf = m["collider_positions"], m["collider_faces"]
        tris = [[cp[cf[i, 0]].tolist(), cp[cf[i, 1]].tolist(), cp[cf[i, 2]].tolist()]
                for i in range(len(cf))]
    return {
        "object_name": m["object_name"],
        "positions": m["positions"].tolist(),
        "faces": m["faces"].tolist(),
        "edges": m["edges"].tolist(),
        "sewing_springs": m["sewing_springs"].tolist(),
        "inv_masses": [1.0] * n,
        "thickness": float(cfg.get("thickness", 0.005)),
        "stiffness": float(cfg.get("tension_stiffness", 10000.0)),
        "bending_stiffness": float(cfg.get("bending_stiffness", 20.0)),
        "sewing_shrink_speed": float(cfg.get("sewing_shrink_speed", 0.5)),
        "substeps": int(cfg.get("substeps", 20)),
        "solver_iterations": int(cfg.get("solver_iterations", 10)),
        "mesh_triangles": [{"p0": t[0], "p1": t[1], "p2": t[2],
                            "friction": m.get("collider_friction", 0.5),
                            "thickness": m.get("collider_thickness", 0.01),
                            "restitution": 0.0, "flags": 0} for t in (tris or [])],
    }
