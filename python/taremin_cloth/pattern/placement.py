# -*- coding: utf-8 -*-
"""型紙パーツの3D配置（Placement）プロキシ統合モジュール.

Document.placements と 3D ビューポート内のプロキシオブジェクトを双方向同期し、
素体周りへのクイック配置プリセットを提供する。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .document import Document, Placement
from .geometry import triangulate_polygon_2d

if TYPE_CHECKING:
    import bpy


QUICK_PLACEMENT_PRESETS: list[tuple[str, str, str, str, int]] = [
    ("CHEST_FRONT", "前身頃 (Chest/Front)", "胴体前面への配置", "TRIA_UP", 0),
    ("BACK", "後身頃 (Back)", "胴体背面への配置", "TRIA_DOWN", 1),
    ("ARM_LEFT", "左袖 (Left Arm)", "左腕周囲への配置", "TRIA_LEFT", 2),
    ("ARM_RIGHT", "右袖 (Right Arm)", "右腕周囲への配置", "TRIA_RIGHT", 3),
    ("COLLAR", "襟 (Collar)", "首周りへの配置", "SNAP_FACE", 4),
    ("WAIST_SKIRT", "ウエスト・スカート (Waist)", "腰・下半身周囲への配置", "SPHERE", 5),
]


def estimate_body_reference_pure(
    *,
    body_height: float = 1.7,
    chest_z_ratio: float = 0.70,
    arm_z_ratio: float = 0.72,
    waist_z_ratio: float = 0.50,
) -> dict[str, float]:
    """素体の推定寸法を純粋な計算で返す (Blender非依存)."""
    h = max(0.5, float(body_height))
    return {
        "chest_z": h * chest_z_ratio,
        "arm_x": max(0.25, h * 0.22),
        "arm_z": h * arm_z_ratio,
        "waist_z": h * waist_z_ratio,
        "thickness_y": max(0.10, h * 0.09),
    }


def get_quick_placement_transform(
    preset_name: str,
    *,
    ref: Optional[dict[str, float]] = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """プリセット名に応じた (offset_3d, euler_3d) タプルを返す (Blender非依存)."""
    if ref is None:
        ref = estimate_body_reference_pure()

    cz = ref["chest_z"]
    ax = ref["arm_x"]
    az = ref["arm_z"]
    wz = ref["waist_z"]
    ty = ref["thickness_y"]

    pi_half = math.pi * 0.5

    if preset_name == "CHEST_FRONT":
        offset = (0.0, -ty, cz)
        euler = (pi_half, 0.0, 0.0)
    elif preset_name == "BACK":
        offset = (0.0, ty, cz)
        euler = (pi_half, 0.0, math.pi)
    elif preset_name == "ARM_LEFT":
        offset = (-ax, 0.0, az)
        euler = (pi_half, 0.0, -pi_half)
    elif preset_name == "ARM_RIGHT":
        offset = (ax, 0.0, az)
        euler = (pi_half, 0.0, pi_half)
    elif preset_name == "COLLAR":
        offset = (0.0, 0.0, cz + 0.20)
        euler = (0.0, 0.0, 0.0)
    elif preset_name == "WAIST_SKIRT":
        offset = (0.0, 0.0, wz)
        euler = (0.0, 0.0, 0.0)
    else:
        offset = (0.0, 0.0, 0.0)
        euler = (0.0, 0.0, 0.0)

    return (offset, euler)


def compute_bone_aligned_frame(
    head_3d: tuple[float, float, float],
    tail_3d: tuple[float, float, float],
    *,
    bone_t: float = 0.5,
    bone_angle_deg: float = 0.0,
    surface_offset: float = 0.08,
) -> tuple[Any, Any, Any, Any, Any]:
    """ボーンの始点・終点からボーン軸平行の平面配置フレーム基底を計算して返す (Blender非依存).
    
    戻り値:
        (plane_center, ex, normal, ez, axis_origin)
    """
    h = np.array(head_3d, dtype=np.float64)
    t = np.array(tail_3d, dtype=np.float64)
    v_axis = t - h
    axis_len = float(np.linalg.norm(v_axis))
    if axis_len < 1e-6:
        u_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        u_axis = v_axis / axis_len

    # ワールド +Z 方向との内積を見て型紙の縦軸 ez が上向きになるよう整列
    if u_axis[2] < -1e-6:
        ez = -u_axis
    else:
        ez = u_axis.copy()

    # 正面基準法線 n0 (Blender の正面視線は -Y)
    ref_front = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    proj_front = ref_front - float(np.dot(ref_front, u_axis)) * u_axis
    norm_proj = float(np.linalg.norm(proj_front))
    if norm_proj < 1e-4:
        ref_alt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        proj_front = ref_alt - float(np.dot(ref_alt, u_axis)) * u_axis
        norm_proj = float(np.linalg.norm(proj_front))
        if norm_proj < 1e-4:
            proj_front = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            norm_proj = 1.0

    n0 = proj_front / norm_proj
    # 右方向接線 ex0 (ez x n0)
    ex0 = np.cross(ez, n0)
    norm_ex0 = float(np.linalg.norm(ex0))
    if norm_ex0 < 1e-4:
        ex0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ex0 = ex0 / norm_ex0

    # ロール角度 theta による法線・接線回転
    theta_rad = math.radians(float(bone_angle_deg))
    normal = math.cos(theta_rad) * n0 + math.sin(theta_rad) * ex0
    norm_n = float(np.linalg.norm(normal))
    normal = normal / max(1e-6, norm_n)

    # 展開平面の横軸 ex (ez x normal)
    ex = np.cross(ez, normal)
    norm_ex = float(np.linalg.norm(ex))
    ex = ex / max(1e-6, norm_ex)

    clamped_t = max(0.0, min(1.0, float(bone_t)))
    axis_origin = (1.0 - clamped_t) * h + clamped_t * t
    d = max(0.005, float(surface_offset))
    plane_center = axis_origin + d * normal

    return (plane_center, ex, normal, ez, axis_origin)


def project_2d_points_to_bone_plane(
    points_2d: Any,
    *,
    head_3d: tuple[float, float, float],
    tail_3d: tuple[float, float, float],
    bone_t: float = 0.5,
    bone_angle_deg: float = 0.0,
    surface_offset: float = 0.08,
) -> np.ndarray:
    """2D頂点列の重心を差し引いてボーン追従3D平面上のワールド座標列 (N, 3) へ投影する (Blender非依存)."""
    pts_arr = np.asarray(points_2d, dtype=np.float64)
    if len(pts_arr) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    plane_center, ex, _, ez, _ = compute_bone_aligned_frame(
        head_3d,
        tail_3d,
        bone_t=bone_t,
        bone_angle_deg=bone_angle_deg,
        surface_offset=surface_offset,
    )
    c2d = np.mean(pts_arr, axis=0)
    du = pts_arr[:, 0] - c2d[0]
    dv = pts_arr[:, 1] - c2d[1]
    world_3d = (
        plane_center[np.newaxis, :]
        + du[:, np.newaxis] * ex[np.newaxis, :]
        + dv[:, np.newaxis] * ez[np.newaxis, :]
    ).astype(np.float32)
    return world_3d


def compute_tubular_waypoints_for_pair(
    pos_a: Any,
    pos_b: Any,
    *,
    head_3d: tuple[float, float, float],
    tail_3d: tuple[float, float, float],
    bone_t: float = 0.5,
    bone_angle_deg: float = 0.0,
    surface_offset: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """筒モード縫合ペア (pos_a, pos_b) に対して、ボーン裏側 (+120°, -120°) を迂回する2つの中継点 (wp1, wp2) を算出する."""
    h_bone = np.asarray(head_3d, dtype=np.float32)
    t_bone = np.asarray(tail_3d, dtype=np.float32)
    v_axis = t_bone - h_bone
    axis_len = float(np.linalg.norm(v_axis))
    u_axis = v_axis / max(1e-6, axis_len)

    _, ex, normal, _, _ = compute_bone_aligned_frame(
        head_3d,
        tail_3d,
        bone_t=bone_t,
        bone_angle_deg=bone_angle_deg,
        surface_offset=surface_offset,
    )
    r_offset = max(0.04, float(surface_offset) * 1.15)
    # ボーン裏側 (+120°, -120°) への振り分け単位ベクトル
    dir_plus = (-0.5 * normal + 0.8660254 * ex).astype(np.float32)
    dir_minus = (-0.5 * normal - 0.8660254 * ex).astype(np.float32)

    pa = np.asarray(pos_a, dtype=np.float32)
    pb = np.asarray(pos_b, dtype=np.float32)
    mid_pos = 0.5 * (pa + pb)

    # ボーン軸への正射影点
    proj_t = float(np.dot(mid_pos - h_bone, u_axis))
    axis_pt = h_bone + proj_t * u_axis

    # pa 側の横方向符号に合わせて wp1 -> wp2 の順序を決定（裏側での交差防止）
    side_a = float(np.dot(pa - pb, ex))
    if side_a >= 0.0:
        wp1 = axis_pt + r_offset * dir_plus
        wp2 = axis_pt + r_offset * dir_minus
    else:
        wp1 = axis_pt + r_offset * dir_minus
        wp2 = axis_pt + r_offset * dir_plus

    return (wp1, wp2)


def compute_bone_placement_transform(
    head_3d: tuple[float, float, float],
    tail_3d: tuple[float, float, float],
    *,
    bone_t: float = 0.5,
    bone_angle_deg: float = 0.0,
    surface_offset: float = 0.08,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """ボーン情報から Placement 用の (offset_3d, euler_3d) を算出する (Blender非依存)."""
    plane_center, ex, normal, ez, _ = compute_bone_aligned_frame(
        head_3d,
        tail_3d,
        bone_t=bone_t,
        bone_angle_deg=bone_angle_deg,
        surface_offset=surface_offset,
    )

    # 回転行列 R = [ex, -normal, ez]
    R = np.column_stack([ex, -normal, ez])

    # オイラー角 (XYZ順) の逆算
    # R = R_x * R_y * R_z
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0

    offset_3d = (float(plane_center[0]), float(plane_center[1]), float(plane_center[2]))
    euler_3d = (float(rx), float(ry), float(rz))
    return (offset_3d, euler_3d)


def flip_bone_name_left_right(bone_name: str) -> str:
    """ボーン名の左右表記 (.L / .R, _L / _R, Left / Right) を反転して返す."""
    if not bone_name:
        return ""
    if bone_name.endswith(".L"):
        return bone_name[:-2] + ".R"
    elif bone_name.endswith(".R"):
        return bone_name[:-2] + ".L"
    elif bone_name.endswith("_L"):
        return bone_name[:-2] + "_R"
    elif bone_name.endswith("_R"):
        return bone_name[:-2] + "_L"
    elif "Left" in bone_name:
        return bone_name.replace("Left", "Right")
    elif "Right" in bone_name:
        return bone_name.replace("Right", "Left")
    elif "left" in bone_name:
        return bone_name.replace("left", "right")
    elif "right" in bone_name:
        return bone_name.replace("right", "left")
    return bone_name


def sync_document_to_proxy_objects(
    context: bpy.types.Context,
    doc: Document,
) -> list[bpy.types.Object]:
    """Documentの全ピースに対応する3D配置プロキシメッシュを同期・生成する."""
    import bpy
    from mathutils import Euler, Vector

    created_proxies: list[bpy.types.Object] = []

    for piece_id, piece in doc.pieces.items():
        proxy_name = f"Proxy_{piece.name}"
        proxy_obj = bpy.data.objects.get(proxy_name)

        # ピースの配置トランスフォーム
        placement = doc.placements.get(piece_id)
        if not placement:
            placement = Placement(piece_id=piece_id)
            doc.placements[piece_id] = placement

        # 外周ポリゴンの取得 (ベジェサンプリング & 三角形化)
        outer = piece.paths.get(piece.outer_path_id)
        if outer and len(outer.nodes) >= 3:
            pts_2d = outer.sample_outline_points(steps_per_segment=16)
        else:
            pts_2d = [(0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)]

        # 重心を計算してローカル中心化
        cx = sum(p[0] for p in pts_2d) / len(pts_2d)
        cy = sum(p[1] for p in pts_2d) / len(pts_2d)
        verts = [(p[0] - cx, 0.0, p[1] - cy) for p in pts_2d]  # (X, Z) 平面
        edges = [[i, (i + 1) % len(verts)] for i in range(len(verts))]
        tri_faces = triangulate_polygon_2d(pts_2d)
        faces = [list(tri) for tri in tri_faces] if tri_faces else [list(range(len(verts)))]

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

        # 表示設定
        proxy_obj.display_type = "WIRE"
        proxy_obj.show_in_front = True

        # トランスフォームの適用
        proxy_obj.location = Vector(placement.offset_3d)
        proxy_obj.rotation_euler = Euler(placement.euler_3d, "XYZ")

        proxy_obj["taremin_piece_id"] = piece_id
        created_proxies.append(proxy_obj)

    return created_proxies


def sync_proxy_objects_to_document(
    context: bpy.types.Context,
    doc: Document,
) -> bool:
    """3D配置プロキシオブジェクトの移動・回転を Document.placements に反映する."""
    import bpy

    modified = False
    for obj in context.scene.objects:
        piece_id = obj.get("taremin_piece_id")
        if not piece_id or piece_id not in doc.pieces:
            continue

        placement = doc.placements.get(piece_id)
        if not placement:
            placement = Placement(piece_id=piece_id)
            doc.placements[piece_id] = placement

        loc = (float(obj.location.x), float(obj.location.y), float(obj.location.z))
        rot = (float(obj.rotation_euler.x), float(obj.rotation_euler.y), float(obj.rotation_euler.z))

        if placement.offset_3d != loc or placement.euler_3d != rot:
            placement.offset_3d = loc
            placement.euler_3d = rot
            modified = True

    return modified

