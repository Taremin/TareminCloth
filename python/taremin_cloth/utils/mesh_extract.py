"""
メッシュデータ高速抽出ユーティリティ
Blenderの bpy.types.Mesh から NumPy 配列へのゼロコピー/一括データ転送を行う。
"""

import numpy as np


def extract_mesh_data(obj):
    """
    Blenderメッシュオブジェクトからシミュレーションに必要な生NumPy配列を一括抽出する
    戻り値:
        (positions, edges, faces, sewing_edges, pin_weights)
    """
    mesh = obj.data
    n_verts = len(mesh.vertices)
    coords = np.empty(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    pos_2d = coords.reshape((n_verts, 3))

    mesh.calc_loop_triangles()
    tri_list = [tri.vertices for tri in mesh.loop_triangles]
    faces_2d = np.array(tri_list, dtype=np.uint32) if tri_list else None

    n_edges = len(mesh.edges)
    edge_indices = np.empty(n_edges * 2, dtype=np.uint32)
    mesh.edges.foreach_get("vertices", edge_indices)
    all_edges_2d = edge_indices.reshape((n_edges, 2))

    face_edge_set = set()
    if faces_2d is not None:
        for f in faces_2d:
            face_edge_set.add((min(f[0], f[1]), max(f[0], f[1])))
            face_edge_set.add((min(f[1], f[2]), max(f[1], f[2])))
            face_edge_set.add((min(f[2], f[0]), max(f[2], f[0])))

    normal_edges = []
    sewing_edges = []

    for e in all_edges_2d:
        pair = (min(e[0], e[1]), max(e[0], e[1]))
        if pair in face_edge_set:
            normal_edges.append(e)
        else:
            sewing_edges.append(e)

    # 四角面（Quad）が存在する場合、両対角線のエッジ (v0, v2) と (v1, v3) を
    # せん断拘束（Shear）の対称化のために抽出
    diag_edges = []
    for poly in mesh.polygons:
        if len(poly.vertices) == 4:
            v = poly.vertices
            diag_edges.append([min(v[0], v[2]), max(v[0], v[2])])
            diag_edges.append([min(v[1], v[3]), max(v[1], v[3])])

    if diag_edges:
        # 重複を排除しつつ normal_edges に追加、あるいは Rust core でせん断拘束として扱えるように
        # 既存 edges と重複しない対角線エッジを特定
        diag_set = set((e[0], e[1]) for e in diag_edges)
        curr_edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in normal_edges)
        new_diags = [list(e) for e in diag_set if e not in curr_edge_set]
        # 注意: normal_edgesに直接加えると伸び拘束（Tension）にもなってしまうため、
        # 後述のShear拘束等で利用できるようにするか、Rust coreのShear生成ロジックに適合させる。
        # 現在のRust coreはfaces (triangles) から隣接三角の対向頂点をShear拘束として自動生成している。

    edges_2d = np.array(normal_edges, dtype=np.uint32) if normal_edges else np.empty((0, 2), dtype=np.uint32)
    sew_2d = np.array(sewing_edges, dtype=np.uint32) if sewing_edges else None

    return pos_2d, edges_2d, faces_2d, sew_2d
