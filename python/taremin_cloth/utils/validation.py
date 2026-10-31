"""
メッシュトポロジー検証ユーティリティ (taremin_cloth.utils.validation)

シミュレーション破綻やメッシュ異常伸長を未然に防ぐため、以下の検証を提供する:
1. 面積0の縮退面 (Degenerate Faces / Degenerate Triangles) の検出
2. 縫合エッジ（ルーズエッジ＋中継点）の枝分かれ（Branching）検出および単一経路保証
"""

from typing import NamedTuple, Optional, List, Set, Tuple, Dict
from collections import defaultdict
import numpy as np

from .logger import logger


class DegenerateFaceReport(NamedTuple):
    """縮退面（初期面積0の面）の検出結果"""
    zero_area_poly_count: int
    zero_area_poly_indices: List[int]
    degenerate_tri_count: int
    degenerate_tri_indices: List[int]

    @property
    def has_issues(self) -> bool:
        return self.zero_area_poly_count > 0 or self.degenerate_tri_count > 0


class BranchingIssue(NamedTuple):
    """縫合線のトポロジー異常情報"""
    issue_type: str  # "branching", "cycle", "dangling", "multi_endpoints"
    vertices: List[int]
    message: str


class SewingTopologyReport(NamedTuple):
    """縫合線トポロジーの検証結果"""
    valid_simple_count: int
    valid_waypoint_count: int
    branching_issues: List[BranchingIssue]

    @property
    def has_issues(self) -> bool:
        return len(self.branching_issues) > 0


class MeshValidationReport(NamedTuple):
    """オブジェクト全体のメッシュ検証結果"""
    degenerate_faces: DegenerateFaceReport
    sewing_topology: SewingTopologyReport

    @property
    def has_errors(self) -> bool:
        return self.sewing_topology.has_issues

    @property
    def has_warnings(self) -> bool:
        return self.degenerate_faces.has_issues


def detect_degenerate_faces(
    positions: np.ndarray,
    faces: Optional[np.ndarray],
    *,
    mesh=None,
    threshold: float = 1e-7,
) -> DegenerateFaceReport:
    """初期面積が極小・0の縮退面（三角形およびポリゴン）を検出する。

    Args:
        positions: 頂点座標配列 [N, 3]
        faces: 三角形インデックス配列 [F, 3] (None可)
        mesh: bpy.types.Mesh オブジェクト (任意)
        threshold: 面積ゼロ判定閾値 (m^2)

    Returns:
        DegenerateFaceReport
    """
    zero_poly_indices = []
    if mesh is not None and hasattr(mesh, "polygons"):
        for p in mesh.polygons:
            area = getattr(p, "area", None)
            if isinstance(area, (int, float)) and area < threshold:
                zero_poly_indices.append(getattr(p, "index", 0))

    zero_tri_indices = []
    if faces is not None and len(faces) > 0 and len(positions) > 0:
        tris = np.asarray(faces, dtype=np.int64)
        v0 = positions[tris[:, 0]]
        v1 = positions[tris[:, 1]]
        v2 = positions[tris[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        # 面積 = 0.5 * ||(v1 - v0) x (v2 - v0)||
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        zero_tri_indices = np.where(areas < threshold)[0].tolist()

    return DegenerateFaceReport(
        zero_area_poly_count=len(zero_poly_indices),
        zero_area_poly_indices=zero_poly_indices,
        degenerate_tri_count=len(zero_tri_indices),
        degenerate_tri_indices=zero_tri_indices,
    )


def validate_sewing_topology(
    loose_edges: List[Tuple[int, int]],
    face_vertex_set: Set[int],
) -> SewingTopologyReport:
    """面を持たないルーズエッジ群を解析し、縫合線が単一経路（枝分かれなし）であるかを検証する。

    正当な縫合パスの要件:
      - 直接縫合: 端点2つ（面頂点）、内部頂点0、エッジ1本
      - 中継点パス: 端点2つ（面頂点）、内部頂点 k >= 1、端点次数=1、全内部頂点次数=2、閉路なし (A - W1 - ... - Wk - B)

    不正なトポロジー:
      - 枝分かれ (branching): 次数 >= 3 の頂点が存在（T字・Y字分岐など）
      - 端点数異常: 端点が 2 以外（0=完全ループ、1=行き止まり/宙ぶらりん、3以上=複数布への分岐）
      - サイクル (cycle): 中継点同士が閉路を形成

    Args:
        loose_edges: ルーズエッジ [(u, v), ...] のリスト
        face_vertex_set: 面を構成する頂点インデックスの集合

    Returns:
        SewingTopologyReport
    """
    if not loose_edges:
        return SewingTopologyReport(0, 0, [])

    adj: Dict[int, Set[int]] = defaultdict(set)
    for e in loose_edges:
        u, v = int(e[0]), int(e[1])
        if u != v:
            adj[u].add(v)
            adj[v].add(u)

    visited: Set[int] = set()
    valid_simple = 0
    valid_waypoint = 0
    issues: List[BranchingIssue] = []

    for start_node in list(adj.keys()):
        if start_node in visited:
            continue

        # 連結成分（Connected Component）をBFSで収集
        comp: List[int] = []
        queue = [start_node]
        visited.add(start_node)
        while queue:
            curr = queue.pop(0)
            comp.append(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # 連結成分内の面頂点（端点）と内部頂点（中継点）を分類
        endpoints = [x for x in comp if x in face_vertex_set]
        internal = [x for x in comp if x not in face_vertex_set]

        # 内部頂点（中継点）の有無で処理を分岐
        # ケースA: 内部頂点なし（純粋な直接縫合ネットワーク）
        # 布頂点同士を直接結ぶエッジは、洋裁の脇下十字点などで同一頂点に複数接続することが
        # 物理的・構造的に正当であるため、単一パス制約（次数1）は中継点を含むパスにのみ適用する。
        if len(internal) == 0:
            for cu in comp:
                for cv in adj[cu]:
                    if cu < cv:
                        valid_simple += 1
            continue

        # ケースB: 内部頂点あり（中継点パス）
        # 中継点を含むパスは、始点・終点のみが布頂点で中継点が1列に並ぶ単一経路でなければならない。
        branch_nodes = [x for x in comp if len(adj[x]) >= 3]
        if branch_nodes:
            issues.append(BranchingIssue(
                issue_type="branching",
                vertices=branch_nodes,
                message=f"Branching detected at vertex {branch_nodes} (degree >= 3). Sewing paths must not branch.",
            ))
            continue

        if len(endpoints) == 0:
            issues.append(BranchingIssue(
                issue_type="cycle",
                vertices=comp,
                message=f"Closed cycle with no cloth endpoints detected in vertices {comp}.",
            ))
            continue

        if len(endpoints) == 1:
            issues.append(BranchingIssue(
                issue_type="dangling",
                vertices=comp,
                message=f"Dangling seam with only one cloth endpoint ({endpoints[0]}) detected.",
            ))
            continue

        if len(endpoints) >= 3:
            issues.append(BranchingIssue(
                issue_type="multi_endpoints",
                vertices=endpoints,
                message=f"Seam connects {len(endpoints)} cloth endpoints ({endpoints}). Expected exactly 2.",
            ))
            continue

        # ここで len(endpoints) == 2 かつ len(internal) >= 1 が確定
        ep1, ep2 = endpoints[0], endpoints[1]

        # 次数チェック: 端点は次数1、内部頂点は全て次数2
        if len(adj[ep1]) != 1 or len(adj[ep2]) != 1 or any(len(adj[m]) != 2 for m in internal):
            issues.append(BranchingIssue(
                issue_type="branching",
                vertices=comp,
                message=f"Invalid degree in waypoint seam path between {ep1} and {ep2}.",
            ))
            continue

        # パス追跡（ep1 -> ep2）
        path = [ep1]
        prev = None
        curr = ep1
        while curr != ep2:
            candidates = [n for n in adj[curr] if n != prev]
            if not candidates:
                break
            nxt = candidates[0]
            path.append(nxt)
            prev = curr
            curr = nxt

        # 追跡結果の検証: パスが ep2 に到達し、かつ全ノードを過不足なく辿っているか
        if path[-1] == ep2 and len(path) == len(comp):
            valid_waypoint += 1
        else:
            issues.append(BranchingIssue(
                issue_type="cycle",
                vertices=comp,
                message=f"Cycle or disconnected sub-path detected in seam between {ep1} and {ep2}.",
            ))

    return SewingTopologyReport(
        valid_simple_count=valid_simple,
        valid_waypoint_count=valid_waypoint,
        branching_issues=issues,
    )


# UIパネル等での毎フレーム再計算を抑止するためのキャッシュ
_validation_cache: Dict[str, Tuple[int, int, int, MeshValidationReport]] = {}
_collider_validation_cache: Dict[str, Tuple[int, int, int, "OpenBoundaryReport"]] = {}


class OpenBoundaryReport(NamedTuple):
    """コライダーまたはメッシュの開放境界（穴・非多様体エッジ）検証結果"""
    has_open_boundaries: bool
    open_edge_count: int
    open_loop_count: int
    non_manifold_edge_count: int


def detect_open_boundaries(
    faces: Optional[np.ndarray],
    *,
    num_verts: Optional[int] = None,
) -> OpenBoundaryReport:
    """三角形面インデックス配列から開放境界エッジおよび境界ループを検出する（Blender非依存）。"""
    if faces is None or len(faces) == 0:
        return OpenBoundaryReport(
            has_open_boundaries=False,
            open_edge_count=0,
            open_loop_count=0,
            non_manifold_edge_count=0,
        )

    tris = np.asarray(faces, dtype=np.int32)
    if tris.ndim != 2 or tris.shape[1] < 3:
        return OpenBoundaryReport(
            has_open_boundaries=False,
            open_edge_count=0,
            open_loop_count=0,
            non_manifold_edge_count=0,
        )

    n_verts = int(num_verts) if num_verts is not None else (int(tris.max()) + 1 if len(tris) > 0 else 0)
    edge_counts: Dict[Tuple[int, int], int] = {}
    directed_edges: List[Tuple[int, int]] = []

    for row in tris:
        i0, i1, i2 = int(row[0]), int(row[1]), int(row[2])
        if i0 < 0 or i1 < 0 or i2 < 0 or i0 == i1 or i1 == i2 or i2 == i0:
            continue
        if n_verts > 0 and (i0 >= n_verts or i1 >= n_verts or i2 >= n_verts):
            continue
        for u, v in ((i0, i1), (i1, i2), (i2, i0)):
            key = (u, v) if u < v else (v, u)
            edge_counts[key] = edge_counts.get(key, 0) + 1
            directed_edges.append((u, v))

    open_edge_count = 0
    non_manifold_edge_count = 0
    for cnt in edge_counts.values():
        if cnt == 1:
            open_edge_count += 1
        elif cnt > 2:
            non_manifold_edge_count += 1

    if open_edge_count == 0:
        return OpenBoundaryReport(
            has_open_boundaries=False,
            open_edge_count=0,
            open_loop_count=0,
            non_manifold_edge_count=non_manifold_edge_count,
        )

    boundary_edges: List[Tuple[int, int]] = []
    next_map: Dict[int, List[int]] = {}
    for u, v in directed_edges:
        key = (u, v) if u < v else (v, u)
        if edge_counts.get(key) == 1:
            idx = len(boundary_edges)
            boundary_edges.append((u, v))
            next_map.setdefault(u, []).append(idx)

    used = [False] * len(boundary_edges)
    open_loop_count = 0
    max_steps = max(n_verts, len(boundary_edges)) + 1

    for start_idx in range(len(boundary_edges)):
        if used[start_idx]:
            continue
        used[start_idx] = True
        start_u, start_v = boundary_edges[start_idx]
        curr = start_v
        steps = 1
        is_closed = False
        while steps <= max_steps:
            if curr == start_u:
                is_closed = True
                break
            found_next = None
            for e_idx in next_map.get(curr, ()):
                if not used[e_idx]:
                    used[e_idx] = True
                    found_next = boundary_edges[e_idx][1]
                    break
            if found_next is not None:
                curr = found_next
                steps += 1
            else:
                break
        if is_closed and steps >= 3:
            open_loop_count += 1

    return OpenBoundaryReport(
        has_open_boundaries=True,
        open_edge_count=open_edge_count,
        open_loop_count=open_loop_count,
        non_manifold_edge_count=non_manifold_edge_count,
    )


def close_open_boundary_loops_numpy(
    positions: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """立体メッシュの開放境界ループを検出して扇状キャップ三角形で閉じる（Blender非依存）。

    Returns:
        (capped_positions, capped_faces, closed_loop_count)
    """
    pos = np.asarray(positions, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int32)
    n_verts = len(pos)
    if len(tris) < 4 or n_verts < 4:
        return pos, tris, 0

    edge_counts: Dict[Tuple[int, int], int] = {}
    existing_tri_sets = set()
    for row in tris:
        i0, i1, i2 = int(row[0]), int(row[1]), int(row[2])
        if (
            i0 < 0 or i1 < 0 or i2 < 0
            or i0 >= n_verts or i1 >= n_verts or i2 >= n_verts
            or i0 == i1 or i1 == i2 or i2 == i0
        ):
            continue
        existing_tri_sets.add(tuple(sorted((i0, i1, i2))))
        for u, v in ((i0, i1), (i1, i2), (i2, i0)):
            key = (u, v) if u < v else (v, u)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    boundary_edges: List[Tuple[int, int]] = []
    boundary_adj_normals: List[np.ndarray] = []
    next_map: Dict[int, List[int]] = {}

    for row in tris:
        i0, i1, i2 = int(row[0]), int(row[1]), int(row[2])
        if (
            i0 < 0 or i1 < 0 or i2 < 0
            or i0 >= n_verts or i1 >= n_verts or i2 >= n_verts
            or i0 == i1 or i1 == i2 or i2 == i0
        ):
            continue
        has_open = False
        for u, v in ((i0, i1), (i1, i2), (i2, i0)):
            key = (u, v) if u < v else (v, u)
            if edge_counts.get(key) == 1:
                has_open = True
                break
        if not has_open:
            continue
        p0, p1, p2 = pos[i0], pos[i1], pos[i2]
        tn = np.cross(p1 - p0, p2 - p0)
        norm_len = float(np.linalg.norm(tn))
        tn = tn / norm_len if norm_len > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

        for u, v in ((i0, i1), (i1, i2), (i2, i0)):
            key = (u, v) if u < v else (v, u)
            if edge_counts.get(key) == 1:
                idx = len(boundary_edges)
                boundary_edges.append((u, v))
                boundary_adj_normals.append(tn)
                next_map.setdefault(u, []).append(idx)

    if not boundary_edges:
        return pos, tris, 0

    used = [False] * len(boundary_edges)
    new_verts: List[np.ndarray] = []
    new_tris: List[Tuple[int, int, int]] = []
    closed_loops = 0

    for start_idx in range(len(boundary_edges)):
        if used[start_idx]:
            continue
        used[start_idx] = True
        start_u, start_v = boundary_edges[start_idx]
        loop_verts = [start_u, start_v]
        loop_edge_indices = [start_idx]
        curr = start_v
        is_closed = False

        while len(loop_verts) <= n_verts:
            if curr == start_u:
                loop_verts.pop()
                is_closed = True
                break
            found_next = None
            for e_idx in next_map.get(curr, ()):
                if not used[e_idx]:
                    used[e_idx] = True
                    loop_edge_indices.append(e_idx)
                    found_next = boundary_edges[e_idx][1]
                    break
            if found_next is not None:
                loop_verts.append(found_next)
                curr = found_next
            else:
                break

        k = len(loop_verts)
        if not is_closed or k < 3 or k >= n_verts:
            continue

        loop_pts = pos[loop_verts]
        center = np.mean(loop_pts, axis=0, dtype=np.float32)

        cap_n_sum = np.zeros(3, dtype=np.float32)
        for idx in range(k):
            pu = pos[loop_verts[idx]]
            pv = pos[loop_verts[(idx + 1) % k]]
            cap_n_sum += np.cross(pu - pv, center - pv)
        cn_len = float(np.linalg.norm(cap_n_sum))
        if cn_len <= 1e-12:
            continue
        cap_normal = cap_n_sum / cn_len

        adj_dots = [float(np.dot(boundary_adj_normals[e_idx], cap_normal)) for e_idx in loop_edge_indices]
        mean_adj_dot = sum(adj_dots) / max(len(adj_dots), 1)
        if mean_adj_dot <= -0.85:
            continue

        if k == 3:
            st = tuple(sorted((loop_verts[0], loop_verts[1], loop_verts[2])))
            if st in existing_tri_sets:
                continue
            new_tris.append((loop_verts[0], loop_verts[2], loop_verts[1]))
            existing_tri_sets.add(st)
            closed_loops += 1
            continue

        center_idx = n_verts + len(new_verts)
        new_verts.append(center)
        for idx in range(k):
            u = loop_verts[idx]
            v = loop_verts[(idx + 1) % k]
            new_tris.append((v, u, center_idx))
        closed_loops += 1

    if closed_loops == 0:
        return pos, tris, 0

    out_pos = np.vstack([pos, np.asarray(new_verts, dtype=np.float32)]) if new_verts else pos
    out_tris = np.vstack([tris, np.asarray(new_tris, dtype=np.int32)]) if new_tris else tris
    return out_pos, out_tris, closed_loops


def validate_collider_mesh(obj) -> Optional[OpenBoundaryReport]:
    """コライダーメッシュオブジェクトの開放境界（穴・切り口）を検証する（キャッシュ付き）。"""
    if not obj or obj.type != 'MESH' or not obj.data:
        return None

    mesh = obj.data
    vert_count = len(mesh.vertices)
    face_count = len(mesh.polygons)
    edge_count = len(mesh.edges)
    cache_key = obj.name

    cached = _collider_validation_cache.get(cache_key)
    if cached is not None:
        c_verts, c_faces, c_edges, report = cached
        if c_verts == vert_count and c_faces == face_count and c_edges == edge_count:
            return report

    if vert_count == 0:
        return None

    mesh.calc_loop_triangles()
    n_tris = len(mesh.loop_triangles)
    faces = None
    if n_tris > 0:
        tri_indices = np.empty(n_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tri_indices)
        faces = tri_indices.reshape((n_tris, 3))

    report = detect_open_boundaries(faces, num_verts=vert_count)
    _collider_validation_cache[cache_key] = (vert_count, face_count, edge_count, report)
    return report


def validate_cloth_mesh(obj) -> Optional[MeshValidationReport]:
    """布メッシュオブジェクトのトポロジー検証をワンストップで実行する。

    頂点数・面数・エッジ数のシグネチャによるキャッシュ機構を備え、
    UIパネル描画等の高頻度呼び出しでもパフォーマンス負荷を生じない。

    Args:
        obj: Blenderメッシュオブジェクト

    Returns:
        MeshValidationReport または None
    """
    if not obj or obj.type != 'MESH' or not obj.data:
        return None

    mesh = obj.data
    vert_count = len(mesh.vertices)
    face_count = len(mesh.polygons)
    edge_count = len(mesh.edges)
    cache_key = obj.name

    cached = _validation_cache.get(cache_key)
    if cached is not None:
        c_verts, c_faces, c_edges, report = cached
        if c_verts == vert_count and c_faces == face_count and c_edges == edge_count:
            return report

    # 1. 座標と三角形面の抽出
    n_verts = vert_count
    if n_verts == 0:
        return None

    coords = np.empty(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    positions = coords.reshape((n_verts, 3))

    mesh.calc_loop_triangles()
    n_tris = len(mesh.loop_triangles)
    faces = None
    if n_tris > 0:
        tri_indices = np.empty(n_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tri_indices)
        faces = tri_indices.reshape((n_tris, 3))

    # 2. 縮退面検出
    deg_report = detect_degenerate_faces(positions, faces, mesh=mesh)

    # 3. ルーズエッジ抽出と縫合線検証
    face_edge_set = set()
    face_vert_set = set()
    if faces is not None:
        for tri in faces:
            face_vert_set.add(int(tri[0]))
            face_vert_set.add(int(tri[1]))
            face_vert_set.add(int(tri[2]))
            face_edge_set.add((min(int(tri[0]), int(tri[1])), max(int(tri[0]), int(tri[1]))))
            face_edge_set.add((min(int(tri[1]), int(tri[2])), max(int(tri[1]), int(tri[2]))))
            face_edge_set.add((min(int(tri[2]), int(tri[0])), max(int(tri[2]), int(tri[0]))))

    loose_edges = []
    for e in mesh.edges:
        pair = (min(e.vertices[0], e.vertices[1]), max(e.vertices[0], e.vertices[1]))
        if pair not in face_edge_set:
            loose_edges.append((e.vertices[0], e.vertices[1]))

    sew_report = validate_sewing_topology(loose_edges, face_vert_set)

    report = MeshValidationReport(
        degenerate_faces=deg_report,
        sewing_topology=sew_report,
    )

    _validation_cache[cache_key] = (vert_count, face_count, edge_count, report)
    return report


def clear_validation_cache(obj_name: Optional[str] = None):
    """検証結果キャッシュをクリアする"""
    global _validation_cache, _collider_validation_cache
    if obj_name is not None:
        _validation_cache.pop(obj_name, None)
        _collider_validation_cache.pop(obj_name, None)
    else:
        _validation_cache.clear()
        _collider_validation_cache.clear()
