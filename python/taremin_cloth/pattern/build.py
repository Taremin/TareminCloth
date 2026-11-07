# -*- coding: utf-8 -*-
"""型紙ドキュメントからBlenderメッシュへの非破壊3D統合ビルド (build.py)."""

from __future__ import annotations

from typing import Optional
import bpy

from ..engine import clear_simulators
from .build_types import build_garment_mesh_geometry
from .document import Document


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
    if obj is None or obj.type != 'MESH':
        mesh = bpy.data.meshes.new(name=f"{target_object_name}_Mesh")
        obj = bpy.data.objects.new(name=target_object_name, object_data=mesh)
        sc.collection.objects.link(obj)
    else:
        mesh = obj.data

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

    if geom.pin_vertex_indices:
        vg_pin = obj.vertex_groups.new(name="Cloth_Pin")
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
