# -*- coding: utf-8 -*-
"""3D配置プロキシ & クイック配置プリセット管理 (pattern_proxy.py)."""

import math
import bpy

try:
    from mathutils import Vector, Euler
    if hasattr(Vector, "_mock_name"):
        raise ImportError
except Exception:
    class Vector(tuple):
        def __new__(cls, val):
            return super().__new__(cls, tuple(float(x) for x in val))
        @property
        def x(self): return self[0]
        @property
        def y(self): return self[1]
        @property
        def z(self): return self[2]

    class Euler(tuple):
        def __new__(cls, val, order='XYZ'):
            return super().__new__(cls, tuple(float(x) for x in val))
        @property
        def x(self): return self[0]
        @property
        def y(self): return self[1]
        @property
        def z(self): return self[2]

from .pattern_ir import sample_pattern_boundary_loop


QUICK_PLACEMENT_PRESETS = [
    ('CHEST_FRONT', "Front (Chest)", "Front body / chest placement", 'TRIA_UP', 0),
    ('BACK', "Back", "Back body placement", 'TRIA_DOWN', 1),
    ('ARM_LEFT', "Left Arm / Sleeve", "Left arm sleeve placement", 'TRIA_LEFT', 2),
    ('ARM_RIGHT', "Right Arm / Sleeve", "Right arm sleeve placement", 'TRIA_RIGHT', 3),
    ('COLLAR', "Collar / Neck", "Neck collar placement", 'SNAP_FACE', 4),
    ('WAIST_SKIRT', "Waist / Skirt", "Waistband or skirt placement", 'SPHERE', 5),
]


def estimate_body_reference(scene: bpy.types.Scene) -> dict[str, float]:
    """シーン内のコライダーやアーマチュアから素体の寸法目安を推定する."""
    chest_z = 1.15
    arm_x = 0.35
    arm_z = 1.20
    waist_z = 0.85
    thickness_y = 0.15

    if not scene:
        return {
            "chest_z": chest_z,
            "arm_x": arm_x,
            "arm_z": arm_z,
            "waist_z": waist_z,
            "thickness_y": thickness_y,
        }

    # コライダーオブジェクトを走査してバウンディングボックスの目安を算出
    collider_objs = [
        obj for obj in scene.objects
        if hasattr(obj, "taremin_collider") and obj.taremin_collider.is_collider
    ]
    if collider_objs:
        all_z = []
        all_x = []
        all_y = []
        for obj in collider_objs:
            if obj.type == 'MESH':
                for corner in obj.bound_box:
                    world_co = obj.matrix_world @ Vector(corner)
                    all_x.append(abs(world_co.x))
                    all_y.append(abs(world_co.y))
                    all_z.append(world_co.z)
        if all_z:
            min_z = min(all_z)
            max_z = max(all_z)
            h = max_z - min_z
            if h > 0.5:
                chest_z = min_z + h * 0.70
                arm_z = min_z + h * 0.72
                waist_z = min_z + h * 0.50
            if all_x:
                max_x = max(all_x)
                if max_x > 0.1:
                    arm_x = max(0.25, max_x * 0.85)
            if all_y:
                max_y = max(all_y)
                if max_y > 0.05:
                    thickness_y = max(0.10, max_y * 0.75)

    return {
        "chest_z": chest_z,
        "arm_x": arm_x,
        "arm_z": arm_z,
        "waist_z": waist_z,
        "thickness_y": thickness_y,
    }


def get_quick_placement_transform(preset_name: str, scene: bpy.types.Scene = None) -> tuple[Vector, Euler]:
    """プリセット名に応じたオフセットベクトルとオイラー角を返す."""
    ref = estimate_body_reference(scene)
    cz = ref["chest_z"]
    ax = ref["arm_x"]
    az = ref["arm_z"]
    wz = ref["waist_z"]
    ty = ref["thickness_y"]

    if preset_name == 'CHEST_FRONT':
        offset = Vector((0.0, -ty, cz))
        euler = Euler((math.radians(90.0), 0.0, 0.0), 'XYZ')
    elif preset_name == 'BACK':
        offset = Vector((0.0, ty, cz))
        euler = Euler((math.radians(90.0), 0.0, math.radians(180.0)), 'XYZ')
    elif preset_name == 'ARM_LEFT':
        offset = Vector((-ax, 0.0, az))
        euler = Euler((math.radians(90.0), 0.0, math.radians(-90.0)), 'XYZ')
    elif preset_name == 'ARM_RIGHT':
        offset = Vector((ax, 0.0, az))
        euler = Euler((math.radians(90.0), 0.0, math.radians(90.0)), 'XYZ')
    elif preset_name == 'COLLAR':
        offset = Vector((0.0, 0.0, cz + 0.25))
        euler = Euler((0.0, 0.0, 0.0), 'XYZ')
    elif preset_name == 'WAIST_SKIRT':
        offset = Vector((0.0, 0.0, wz))
        euler = Euler((0.0, 0.0, 0.0), 'XYZ')
    else:
        offset = Vector((0.0, 0.0, 0.0))
        euler = Euler((0.0, 0.0, 0.0), 'XYZ')

    return offset, euler


def get_or_create_arrangement_proxy(context: bpy.types.Context, pattern_obj: bpy.types.Object) -> bpy.types.Object:
    """型紙オブジェクトに対応する3D配置プロキシオブジェクトを取得または生成する."""
    proxy_name = f"{pattern_obj.name}_Proxy3D"
    proxy_obj = bpy.data.objects.get(proxy_name)

    # 境界ループの抽出
    bnd_pts = None
    try:
        from .pattern_builder import extract_pattern_contour
        contour = extract_pattern_contour(pattern_obj)
        loop_points, _ = sample_pattern_boundary_loop(contour)
        if len(loop_points) >= 3:
            bnd_pts = [(float(p[0]), float(p[1])) for p in loop_points]
    except Exception:
        pass

    if not bnd_pts or len(bnd_pts) < 3:
        bnd_pts = [(0.0, 0.0), (0.2, 0.0), (0.2, 0.3), (0.0, 0.3)]

    # 重心を計算してローカル座標系を中心化
    cx = sum(p[0] for p in bnd_pts) / len(bnd_pts)
    cy = sum(p[1] for p in bnd_pts) / len(bnd_pts)
    verts = [(p[0] - cx, p[1] - cy, 0.0) for p in bnd_pts]
    edges = [[i, (i + 1) % len(verts)] for i in range(len(verts))]
    faces = [list(range(len(verts)))]

    mesh_name = f"{proxy_name}_Mesh"
    proxy_mesh = bpy.data.meshes.get(mesh_name)
    if not proxy_mesh:
        proxy_mesh = bpy.data.meshes.new(mesh_name)
    proxy_mesh.clear_geometry()
    proxy_mesh.from_pydata(verts, edges, faces)
    proxy_mesh.update()

    if not proxy_obj:
        proxy_obj = bpy.data.objects.new(proxy_name, proxy_mesh)
        context.collection.objects.link(proxy_obj)
    else:
        proxy_obj.data = proxy_mesh

    # 表示設定（ワイヤーフレーム表示・前面表示で視認性を確保）
    proxy_obj.display_type = 'WIRE'
    proxy_obj.show_in_front = True

    # トランスフォーム同期
    pat = pattern_obj.taremin_pattern
    proxy_obj.location = Vector(pat.arrangement_offset)
    proxy_obj.rotation_euler = Euler(pat.arrangement_euler, 'XYZ')

    proxy_obj["taremin_pattern_source"] = pattern_obj.name
    return proxy_obj


def sync_proxy_to_pattern(proxy_obj: bpy.types.Object, scene: bpy.types.Scene = None) -> None:
    """3Dプロキシの移動・回転を型紙のarrangement_offset / arrangement_euler に反映する."""
    if not proxy_obj or "taremin_pattern_source" not in proxy_obj:
        return
    source_name = proxy_obj["taremin_pattern_source"]
    source_obj = bpy.data.objects.get(source_name)
    if not source_obj or not hasattr(source_obj, "taremin_pattern"):
        return

    pat = source_obj.taremin_pattern
    pat.arrangement_offset = proxy_obj.location
    pat.arrangement_euler = proxy_obj.rotation_euler


def sync_pattern_to_proxy(pattern_obj: bpy.types.Object) -> None:
    """型紙のトランスフォームを3Dプロキシに反映する."""
    proxy_name = f"{pattern_obj.name}_Proxy3D"
    proxy_obj = bpy.data.objects.get(proxy_name)
    if not proxy_obj:
        return
    pat = pattern_obj.taremin_pattern
    proxy_obj.location = Vector(pat.arrangement_offset)
    proxy_obj.rotation_euler = Euler(pat.arrangement_euler, 'XYZ')
