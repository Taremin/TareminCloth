"""
taremin_cloth コライダー形状自動判別ユーティリティ
オブジェクトの構造（モディファイア、ボーンウェイト、トポロジー、寸法）を解析し、
最適なコライダータイプを自動推定する。
"""

import bpy


def _get_dimensions(obj):
    """オブジェクトの3軸寸法 (dx, dy, dz) を取得する"""
    dims = getattr(obj, "dimensions", None)
    if dims is not None:
        return (dims[0], dims[1], dims[2])
    bbox = getattr(obj, "bound_box", None)
    if bbox and len(bbox) >= 8:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        zs = [p[2] for p in bbox]
        return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    return None


def detect_collider_type(obj) -> str:
    """
    オブジェクトの構造を解析し、最適なコライダータイプを返す。
    返り値: 'BONE_SDF' | 'PLANE' | 'SPHERE' | 'MESH_SDF' | 'MESH'
    """
    if not obj or getattr(obj, "type", None) != 'MESH':
        return 'MESH'

    # 1. Armatureモディファイアを持ち、かつ頂点グループ（ボーンウェイト）が存在する -> BONE_SDF (素体・アバター)
    has_armature = False
    modifiers = getattr(obj, "modifiers", [])
    for mod in modifiers:
        if getattr(mod, "type", None) == 'ARMATURE' and getattr(mod, "object", None):
            has_armature = True
            break

    vertex_groups = getattr(obj, "vertex_groups", [])
    if has_armature and len(vertex_groups) > 0:
        return 'BONE_SDF'

    # 2. メッシュデータの基本情報を取得
    data = getattr(obj, "data", None)
    if not data:
        return 'MESH'

    polygons = getattr(data, "polygons", [])
    n_polys = len(polygons)

    dims = _get_dimensions(obj)

    # 3. 平面（Floor / Ground）判定
    # 厚み比が極めて薄い (2%未満) かつポリゴン数が控えめな場合
    if dims:
        dx, dy, dz = dims
        min_dim = min(dx, dy, dz)
        max_dim = max(dx, dy, dz)
        if max_dim > 1e-4 and (min_dim / max_dim) < 0.02 and n_polys <= 100:
            return 'PLANE'

    # 4. 球体（Sphere）判定
    # 3軸寸法比がほぼ均等 (誤差5%以内) かつオブジェクト名に球関連キーワードを含む場合
    name_lower = getattr(obj, "name", "").lower()
    if dims:
        dx, dy, dz = dims
        max_dim = max(dx, dy, dz)
        if max_dim > 1e-4:
            diff_xy = abs(dx - dy) / max_dim
            diff_yz = abs(dy - dz) / max_dim
            diff_zx = abs(dz - dx) / max_dim
            if diff_xy < 0.05 and diff_yz < 0.05 and diff_zx < 0.05:
                if any(kw in name_lower for kw in ("sphere", "ball", "orb", "globe")):
                    return 'SPHERE'

    # 5. 中〜高密度メッシュ（ポリゴン数 >= 300） -> MESH_SDF (マネキン・家具・剛体)
    if n_polys >= 300:
        return 'MESH_SDF'

    # 6. 低ポリゴンメッシュ -> MESH (直接ポリゴン接触)
    return 'MESH'


PURPOSE_TO_COLLIDER_TYPE = {
    'AUTO': None,
    'CHARACTER': 'BONE_SDF',
    'MANNEQUIN': 'MESH_SDF',
    'FLOOR': 'PLANE',
    'SPHERE': 'SPHERE',
    'CUSTOM': None,
}

COLLIDER_TYPE_TO_PURPOSE = {
    'BONE_SDF': 'CHARACTER',
    'MESH_SDF': 'MANNEQUIN',
    'PLANE': 'FLOOR',
    'SPHERE': 'SPHERE',
    'CAPSULE': 'CUSTOM',
    'MESH': 'CUSTOM',
}
