# -*- coding: utf-8 -*-
"""型紙ドキュメントからBlenderメッシュへの非破壊3D統合ビルド (build.py)."""

from __future__ import annotations

from typing import Any, Optional
import bpy

import numpy as np

from ..engine import clear_simulators
from .build_types import build_garment_mesh_geometry, transfer_vertex_weights_by_uv
from .document import Document


def _normalize_piece_uvs_local(uvs: np.ndarray, *, min_uv: np.ndarray, max_span: float) -> np.ndarray:
    """ピース内 UV を最小原点・等方スパン基準で [0, 1] に正規化する."""
    scale = max(float(max_span), 1e-6)
    return (uvs - min_uv) / scale


def _capture_user_vertex_groups_for_transfer(
    obj: bpy.types.Object,
    mesh: bpy.types.Mesh,
) -> Optional[tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], dict[str, Any]]]:
    """トポロジー更新前に非 Piece_* 頂点グループウェイト、旧 UV 三角形データ、およびピース対応を退避する."""
    preserve_vgs = [vg for vg in obj.vertex_groups if not vg.name.startswith("Piece_")]
    if not preserve_vgs or len(mesh.vertices) == 0 or len(mesh.polygons) == 0:
        return None
    uv_layer = mesh.uv_layers.active
    if uv_layer is None or len(uv_layer.data) == 0:
        return None

    mesh.calc_loop_triangles()
    lt_list = mesh.loop_triangles
    if not lt_list:
        return None

    n_old_verts = len(mesh.vertices)
    vg_idx_to_name = {vg.index: vg.name for vg in preserve_vgs}
    piece_vg_idx_to_name = {
        vg.index: vg.name[len("Piece_"):]
        for vg in obj.vertex_groups
        if vg.name.startswith("Piece_")
    }
    old_vert_pieces: list[Optional[str]] = [None] * n_old_verts
    old_vert_piece_w = np.zeros(n_old_verts, dtype=np.float32)

    vert_weights: dict[str, np.ndarray] = {
        vg.name: np.zeros(n_old_verts, dtype=np.float32) for vg in preserve_vgs
    }
    for v_idx, vert in enumerate(mesh.vertices):
        for g in vert.groups:
            name = vg_idx_to_name.get(g.group)
            if name is not None:
                vert_weights[name][v_idx] = float(g.weight)
            p_name = piece_vg_idx_to_name.get(g.group)
            if p_name is not None and float(g.weight) > old_vert_piece_w[v_idx] and float(g.weight) > 0.5:
                old_vert_piece_w[v_idx] = float(g.weight)
                old_vert_pieces[v_idx] = p_name

    active_names = [name for name, arr in vert_weights.items() if float(np.max(arr)) > 1e-6]
    if not active_names:
        return None

    n_tris = len(lt_list)
    old_tri_uvs = np.zeros((n_tris, 3, 2), dtype=np.float32)
    tri_vert_indices = np.zeros((n_tris, 3), dtype=np.int32)
    vert_uv_fallback = np.zeros((n_old_verts, 2), dtype=np.float32)
    old_tri_pieces: list[Optional[str]] = [None] * n_tris

    uv_data = uv_layer.data
    for t_i, lt in enumerate(lt_list):
        p_cands = []
        for k in range(3):
            vi = int(lt.vertices[k])
            li = int(lt.loops[k])
            u, v = uv_data[li].uv
            old_tri_uvs[t_i, k, 0] = u
            old_tri_uvs[t_i, k, 1] = v
            tri_vert_indices[t_i, k] = vi
            vert_uv_fallback[vi, 0] = u
            vert_uv_fallback[vi, 1] = v
            if old_vert_pieces[vi] is not None:
                p_cands.append(old_vert_pieces[vi])
        if p_cands and len(set(p_cands)) == 1:
            old_tri_pieces[t_i] = p_cands[0]

    old_tri_weights = {name: vert_weights[name][tri_vert_indices] for name in active_names}
    old_point_weights = {name: vert_weights[name] for name in active_names}
    piece_meta = {
        "vert_pieces": old_vert_pieces,
        "tri_pieces": old_tri_pieces,
    }
    return (old_tri_uvs, old_tri_weights, vert_uv_fallback, old_point_weights, piece_meta)


def build_garment_mesh_in_place(
    doc: Document,
    *,
    target_object_name: str = "Garment_Cloth",
    scene: Optional[bpy.types.Scene] = None,
) -> Optional[bpy.types.Object]:
    """Document から3D衣服メッシュを構築し、既存オブジェクトが存在する場合は非破壊にIn-Place更新する.
    
    オブジェクトを削除・再生成せず、clear_simulators() 後に mesh.clear_geometry() と
    from_pydata() を用いることで、マテリアルやモディファイア、物理設定を保持する.
    """
    sc = scene or bpy.context.scene

    bones = None
    if sc is not None:
        try:
            from ..utils.view3d import extract_armature_bones_for_canvas
            bones = extract_armature_bones_for_canvas(sc)
        except Exception:
            pass

    # 1. 幾何データの構築 (Blender非依存コア)
    geom = build_garment_mesh_geometry(doc, bones=bones)
    if not geom.vertices_3d or not geom.faces:
        return None

    # シミュレーターキャッシュおよび縫合描画キャッシュのクリア（トポロジー変更対応）
    clear_simulators()
    try:
        from ..utils.drawing import clear_sewing_cache
        clear_sewing_cache()
    except Exception:
        pass

    # 2. オブジェクトおよびメッシュの取得または作成
    obj = sc.objects.get(target_object_name)
    captured_weights = None
    if obj is None or obj.type != 'MESH':
        mesh = bpy.data.meshes.new(name=f"{target_object_name}_Mesh")
        obj = bpy.data.objects.new(name=target_object_name, object_data=mesh)
        sc.collection.objects.link(obj)
    else:
        mesh = obj.data
        captured_weights = _capture_user_vertex_groups_for_transfer(obj, mesh)

    # 3. ジオメトリの In-Place 非破壊更新
    mesh.clear_geometry()
    # 面と縫合Loose Edgeを同時に登録
    mesh.from_pydata(geom.vertices_3d, geom.sewing_edges, geom.faces)
    mesh.update()

    # 4. UVレイヤーの設定
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        uv_layer = mesh.uv_layers.new(name="UVMap")

    if uv_layer and geom.face_uvs:
        # loop_triangles または loops への書き込み
        for i, loop in enumerate(mesh.loops):
            if i < len(geom.face_uvs):
                uv_layer.data[i].uv = geom.face_uvs[i]

    # 5. 面属性 (Attributes) の登録
    # pattern_piece_index
    attr_piece = mesh.attributes.get("pattern_piece_index")
    if attr_piece is None:
        attr_piece = mesh.attributes.new(name="pattern_piece_index", type='INT', domain='FACE')
    for i, p_idx in enumerate(geom.face_piece_indices):
        if i < len(attr_piece.data):
            attr_piece.data[i].value = p_idx

    # cloth_layer
    attr_layer = mesh.attributes.get("cloth_layer")
    if attr_layer is None:
        attr_layer = mesh.attributes.new(name="cloth_layer", type='INT', domain='FACE')
    for i, l_id in enumerate(geom.face_layer_ids):
        if i < len(attr_layer.data):
            attr_layer.data[i].value = l_id

    # 6. 頂点グループの作成・更新
    # 既存の Piece_* 頂点グループをクリアして再作成
    for vg_name in list(obj.vertex_groups.keys()):
        if vg_name.startswith("Piece_") or vg_name == "Cloth_Pin":
            obj.vertex_groups.remove(obj.vertex_groups[vg_name])

    for p_name, v_indices in geom.piece_vertex_map.items():
        vg = obj.vertex_groups.new(name=f"Piece_{p_name}")
        if v_indices:
            vg.add(v_indices, 1.0, 'REPLACE')

    # 再ビルド前に退避したユーザー定義頂点グループ（および手描き Cloth_Pin）を UV 転写で復元
    # Loose 縫合中継点（面を持たない頂点）は誤転写を防ぐため対象外とし、
    # ピース対応が取れる箇所はピースローカル正規化 UV で別ピース混入やリパックずれを防ぐ。
    if captured_weights is not None and geom.face_uvs:
        old_tri_uvs, old_tri_weights, old_pt_uvs, old_pt_weights, piece_meta = captured_weights
        n_new_verts = len(geom.vertices_3d)
        new_vert_uvs = np.zeros((n_new_verts, 2), dtype=np.float32)
        has_face_uv = np.zeros(n_new_verts, dtype=bool)
        for f_i, tri in enumerate(geom.faces):
            for k in range(3):
                l_idx = f_i * 3 + k
                if l_idx < len(geom.face_uvs):
                    v_i = int(tri[k])
                    new_vert_uvs[v_i] = geom.face_uvs[l_idx]
                    has_face_uv[v_i] = True

        transferred: dict[str, np.ndarray] = {
            name: np.zeros(n_new_verts, dtype=np.float32) for name in old_tri_weights
        }
        handled_mask = np.zeros(n_new_verts, dtype=bool)

        old_vert_pieces = piece_meta.get("vert_pieces", [])
        old_tri_pieces = piece_meta.get("tri_pieces", [])
        for p_name, v_indices in geom.piece_vertex_map.items():
            target_verts = [int(vi) for vi in v_indices if 0 <= int(vi) < n_new_verts and has_face_uv[int(vi)]]
            if not target_verts:
                continue
            old_t_idx = [i for i, tp in enumerate(old_tri_pieces) if tp == p_name]
            old_p_idx = [i for i, vp in enumerate(old_vert_pieces) if vp == p_name]
            if not old_t_idx and not old_p_idx:
                continue

            sub_new_uvs = new_vert_uvs[target_verts]
            new_min = np.min(sub_new_uvs, axis=0)
            new_span = float(np.max(np.max(sub_new_uvs, axis=0) - new_min))
            norm_new_uvs = _normalize_piece_uvs_local(sub_new_uvs, min_uv=new_min, max_span=new_span)

            if old_t_idx:
                sub_old_tris = old_tri_uvs[old_t_idx]
                old_tri_flat = sub_old_tris.reshape(-1, 2)
                old_min = np.min(old_tri_flat, axis=0)
                old_span = float(np.max(np.max(old_tri_flat, axis=0) - old_min))
                norm_old_tris = _normalize_piece_uvs_local(sub_old_tris, min_uv=old_min, max_span=old_span)
                sub_tri_weights = {name: arr[old_t_idx] for name, arr in old_tri_weights.items()}
            else:
                norm_old_tris = np.empty((0, 3, 2), dtype=np.float32)
                sub_tri_weights = {name: np.empty((0, 3), dtype=np.float32) for name in old_tri_weights}
                old_pt_flat = old_pt_uvs[old_p_idx]
                old_min = np.min(old_pt_flat, axis=0)
                old_span = float(np.max(np.max(old_pt_flat, axis=0) - old_min))

            if old_p_idx:
                sub_old_pts = _normalize_piece_uvs_local(
                    old_pt_uvs[old_p_idx], min_uv=old_min, max_span=old_span
                )
                sub_pt_weights = {name: arr[old_p_idx] for name, arr in old_pt_weights.items()}
            else:
                sub_old_pts = None
                sub_pt_weights = None

            sub_transferred = transfer_vertex_weights_by_uv(
                old_tri_uvs=norm_old_tris,
                old_tri_weights=sub_tri_weights,
                new_uvs=norm_new_uvs,
                old_point_uvs=sub_old_pts,
                old_point_weights=sub_pt_weights,
            )
            for vg_name, w_sub in sub_transferred.items():
                transferred[vg_name][target_verts] = w_sub
            handled_mask[target_verts] = True

        # 旧 Piece_* が無いメッシュ等、未対応の面頂点のみ全体 UV でフォールバック転写
        fallback_verts = np.flatnonzero(has_face_uv & (~handled_mask))
        if len(fallback_verts) > 0:
            fb_transferred = transfer_vertex_weights_by_uv(
                old_tri_uvs=old_tri_uvs,
                old_tri_weights=old_tri_weights,
                new_uvs=new_vert_uvs[fallback_verts],
                old_point_uvs=old_pt_uvs,
                old_point_weights=old_pt_weights,
            )
            for vg_name, w_fb in fb_transferred.items():
                transferred[vg_name][fallback_verts] = w_fb

        for vg_name, w_arr in transferred.items():
            vg = obj.vertex_groups.get(vg_name)
            if vg is None:
                vg = obj.vertex_groups.new(name=vg_name)
            else:
                # 既存の古い頂点インデックス残留をクリアして新規再構築
                all_indices = list(range(len(geom.vertices_3d)))
                if all_indices:
                    vg.remove(all_indices)
            for v_idx in np.flatnonzero(w_arr > 1e-4):
                vg.add([int(v_idx)], float(w_arr[v_idx]), 'REPLACE')

    if geom.pin_vertex_indices:
        vg_pin = obj.vertex_groups.get("Cloth_Pin") or obj.vertex_groups.new(name="Cloth_Pin")
        vg_pin.add(geom.pin_vertex_indices, 1.0, 'REPLACE')
        # アドオン設定の pin_vertex_group に自動指定
        if hasattr(obj, "taremin_cloth"):
            obj.taremin_cloth.pin_vertex_group = "Cloth_Pin"

    # 7. アドオンの布シミュレーション設定を有効化
    if hasattr(obj, "taremin_cloth"):
        obj.taremin_cloth.is_cloth = True
        if geom.sewing_edges:
            obj.taremin_cloth.enable_sewing = True

    return obj
