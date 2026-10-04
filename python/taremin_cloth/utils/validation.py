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

        # 1. 分岐チェック: 次数 >= 3 の頂点が存在するか
        branch_nodes = [x for x in comp if len(adj[x]) >= 3]
        if branch_nodes:
            issues.append(BranchingIssue(
                issue_type="branching",
                vertices=branch_nodes,
                message=f"Branching detected at vertex {branch_nodes} (degree >= 3). Sewing paths must not branch.",
            ))
            continue

        # 2. 端点数の検証
        if len(endpoints) == 0:
            # 端点が存在しない（布に繋がっていない浮遊ループ）
            issues.append(BranchingIssue(
                issue_type="cycle",
                vertices=comp,
                message=f"Closed cycle with no cloth endpoints detected in vertices {comp}.",
            ))
            continue

        if len(endpoints) == 1:
            # 端点が1つだけ（行き止まり / 宙ぶらりんのエッジ）
            issues.append(BranchingIssue(
                issue_type="dangling",
                vertices=comp,
                message=f"Dangling seam with only one cloth endpoint ({endpoints[0]}) detected.",
            ))
            continue

        if len(endpoints) >= 3:
            # 3つ以上の布端点に繋がっている（複数端点合流）
            issues.append(BranchingIssue(
                issue_type="multi_endpoints",
                vertices=endpoints,
                message=f"Seam connects {len(endpoints)} cloth endpoints ({endpoints}). Expected exactly 2.",
            ))
            continue

        # ここで len(endpoints) == 2 が確定
        ep1, ep2 = endpoints[0], endpoints[1]

        # 3. ケースA: 直接縫合 (内部頂点なし)
        if len(internal) == 0:
            if len(comp) == 2 and ep2 in adj[ep1]:
                valid_simple += 1
            else:
                issues.append(BranchingIssue(
                    issue_type="invalid_simple",
                    vertices=comp,
                    message=f"Invalid direct seam topology between {ep1} and {ep2}.",
                ))
            continue

        # 4. ケースB: 中継点パス (内部頂点あり)
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
    global _validation_cache
    if obj_name is not None:
        _validation_cache.pop(obj_name, None)
    else:
        _validation_cache.clear()
