#!/usr/bin/env python3
"""
洋裁（パターンメイキング）構造に準拠した Taremin Cloth テスト衣服パターン生成ツール。

Blender の新規シーン（空のシーンまたは初期シーン）から、素体コライダーの読み込みと
4種類のテスト衣服（Tシャツ・ズボン・Yシャツ・フード付きパーカー＋引き紐）を一括構築します。

主な特徴:
- 数学的単調性: 各パネル行の座標が単調増加する連続関数で構成され、初期自己交差・反転面・折り畳みエッジが 0 件。
- 完全 1 対 1 縫合と中継点（Waypoint）縫合: 肩・アームホール・袖下・台襟を分岐なく接続し、腕や首の挟み込みを防止。
- フード等長剛体配置: 2D 型紙の辺長・面積・角度を 100% 維持したまま純粋 3D 剛体回転（SO(3)）で立体配置（初期歪み 0%）。
- 2 通りの実行方法に対応:
  1. Blender GUI 上で新規シーンを開き、Scripting タブから本スクリプトを開いて実行
  2. CLI からヘッドレス実行して .blend ファイルへ書き出し:
     python tools/setup_tailored_patterns.py --output tmp/test_pattern.blend
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

GARMENT_OBJECT_NAMES: tuple[str, ...] = (
    "Pattern_TShirt",
    "Pattern_Pants",
    "Pattern_Shirt",
    "Pattern_Hood",
)
DRAWSTRING_OBJECT_NAME = "Pattern_Hood_Drawstring"


def _resolve_repo_root() -> Path:
    """スクリプト位置またはカレントディレクトリからリポジトリルートを特定する。"""
    candidates: list[Path] = []
    if "__file__" in globals():
        script_path = Path(__file__).resolve()
        candidates.extend([script_path.parent.parent, script_path.parent])
    candidates.append(Path.cwd().resolve())

    try:
        import bpy  # type: ignore

        bdata = getattr(bpy, "data", None)
        if bdata is not None:
            for text_block in bdata.texts:
                if text_block.filepath:
                    tp = Path(bpy.path.abspath(text_block.filepath)).resolve()
                    candidates.extend([tp.parent.parent, tp.parent])
            if bdata.filepath:
                bp = Path(bdata.filepath).resolve()
                candidates.extend([bp.parent.parent, bp.parent])
    except (ImportError, AttributeError):
        pass

    for cand in candidates:
        if (cand / "python" / "taremin_cloth").is_dir():
            return cand
    return Path.cwd().resolve()


# ==============================================================================
# Blender 内部構築ロジック (bpy 依存部)
# ==============================================================================
def _ensure_addon_registered(repo_root: Path) -> None:
    """Blender 内で taremin_cloth アドオンが未登録の場合にパス解決と登録を行う。"""
    import bpy

    python_dir = repo_root / "python"
    for p in (python_dir, repo_root):
        p_str = str(p)
        if p.is_dir() and p_str not in sys.path:
            sys.path.insert(0, p_str)

    import taremin_cloth

    if not hasattr(bpy.types.Object, "taremin_cloth"):
        taremin_cloth.register()


def _remove_object_and_data(obj_name: str) -> None:
    """指定名のオブジェクトとそのメッシュまたはカーブデータを安全に削除する。"""
    import bpy

    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        return
    data = obj.data
    obj_type = obj.type
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        if obj_type == "MESH":
            bpy.data.meshes.remove(data, do_unlink=True)
        elif obj_type == "CURVE":
            bpy.data.curves.remove(data, do_unlink=True)


def _clean_scene_for_setup(*, clear_default_cube: bool = True) -> None:
    """既存のシミュレータキャッシュ・旧テスト衣服・初期 Cube をクリーンアップする。"""
    import bpy
    from taremin_cloth.engine import clear_collider_cache, clear_simulators

    clear_simulators()
    clear_collider_cache()

    target_names = [
        *GARMENT_OBJECT_NAMES,
        "Pattern_Hood.001",
        DRAWSTRING_OBJECT_NAME,
        "Drawstring_Hood",
    ]
    for name in target_names:
        _remove_object_and_data(name)

    if clear_default_cube:
        cube = bpy.data.objects.get("Cube")
        if cube is not None and cube.type == "MESH":
            cloth = getattr(cube, "taremin_cloth", None)
            col = getattr(cube, "taremin_cloth_collider", None)
            is_cloth = bool(cloth and getattr(cloth, "is_cloth", False))
            is_col = bool(col and getattr(col, "is_collider", False))
            if not is_cloth and not is_col and len(cube.data.vertices) == 8:
                _remove_object_and_data("Cube")


def _setup_collections() -> dict[str, "bpy.types.Collection"]:
    """テスト用のコレクション階層 TareminCloth_Test を構築して返す。"""
    import bpy

    scene = bpy.context.scene
    root_name = "TareminCloth_Test"
    main_col = bpy.data.collections.get(root_name)
    if main_col is None:
        main_col = bpy.data.collections.new(root_name)
    if main_col.name not in scene.collection.children:
        scene.collection.children.link(main_col)

    def ensure_subcol(col_name: str):
        sub = bpy.data.collections.get(col_name)
        if sub is None:
            sub = bpy.data.collections.new(col_name)
        if sub.name not in main_col.children:
            main_col.children.link(sub)
        return sub

    return {
        "sotai": ensure_subcol("01_Sotai_Collider"),
        "tshirt": ensure_subcol("02_Cloth_TShirt"),
        "shirt": ensure_subcol("03_Cloth_Shirt"),
        "pants": ensure_subcol("04_Cloth_Pants"),
        "hood": ensure_subcol("05_Cloth_Hood"),
    }


def _move_object_to_collection(obj, target_col) -> None:
    """オブジェクトを既存の全コレクションから外し、指定コレクションのみにリンクする。"""
    for col in list(obj.users_collection):
        if col != target_col:
            col.objects.unlink(obj)
    if obj.name not in target_col.objects:
        target_col.objects.link(obj)


def _setup_sotai_collider(
    sotai_col,
    *,
    body_blend_path: Path | None = None,
) -> tuple[object | None, object | None]:
    """素体メッシュ SiroinoSotai_PC と Armature をシーンに用意し、BONE_SDF コライダーを設定する。"""
    import bpy

    body_obj = bpy.data.objects.get("SiroinoSotai_PC")
    arm_obj = bpy.data.objects.get("Armature")

    if (body_obj is None or arm_obj is None) and body_blend_path is not None:
        if not body_blend_path.exists():
            raise FileNotFoundError(
                f"素体ファイルが見つかりません: {body_blend_path}"
            )
        wanted = []
        if arm_obj is None:
            wanted.append("Armature")
        if body_obj is None:
            wanted.append("SiroinoSotai_PC")

        with bpy.data.libraries.load(str(body_blend_path), link=False) as (data_from, data_to):
            available = set(data_from.objects)
            data_to.objects = [name for name in wanted if name in available]

        for loaded_obj in data_to.objects:
            if loaded_obj is None:
                continue
            if loaded_obj.name == "Armature":
                arm_obj = loaded_obj
            elif loaded_obj.name == "SiroinoSotai_PC":
                body_obj = loaded_obj

    if arm_obj is not None:
        _move_object_to_collection(arm_obj, sotai_col)
        arm_obj.hide_viewport = False
        arm_obj.hide_render = True

    if body_obj is not None:
        _move_object_to_collection(body_obj, sotai_col)
        body_obj.hide_viewport = False
        body_obj.hide_render = False

        # Armature モディファイアの参照整合性を保証
        if arm_obj is not None:
            has_arm_mod = False
            for mod in body_obj.modifiers:
                if mod.type == "ARMATURE":
                    mod.object = arm_obj
                    has_arm_mod = True
            if not has_arm_mod:
                arm_mod = body_obj.modifiers.new(name="Armature", type="ARMATURE")
                arm_mod.object = arm_obj

        # 素体確認用マテリアルの設定
        mat_name = "Mat_SiroinoSotai"
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(name=mat_name)
        mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
        if mat.node_tree is not None:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None and "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = (0.85, 0.85, 0.88, 1.0)
        if len(body_obj.data.materials) == 0:
            body_obj.data.materials.append(mat)
        else:
            body_obj.data.materials[0] = mat

        # コライダー設定 (開放境界キャップ閉鎖を含む BONE_SDF 標準構成)
        col_settings = body_obj.taremin_cloth_collider
        col_settings.is_collider = True
        col_settings.enabled = True
        col_settings.collider_type = "BONE_SDF"
        col_settings.sdf_resolution = "64"
        col_settings.sdf_margin = 0.2
        col_settings.weight_threshold = 0.02
        col_settings.blend_k = 0.05
        col_settings.sdf_update_mode = "STATIC"
        col_settings.enable_joint_mesh = True
        col_settings.thickness = 0.005
        col_settings.single_sided = True
        col_settings.enable_single_sided_recovery = True
        col_settings.close_open_boundaries = True
        col_settings.friction = 0.5
        col_settings.restitution = 0.0

    return body_obj, arm_obj


def _setup_lighting_and_camera() -> None:
    """新規空シーンでもそのまま確認できるよう、不足しているライトとカメラを配置する。"""
    import bpy
    from mathutils import Euler, Vector

    scene = bpy.context.scene
    root_col = scene.collection

    if bpy.data.objects.get("Light") is None:
        light_data = bpy.data.lights.new(name="Light", type="SUN")
        light_data.energy = 3.0
        light_obj = bpy.data.objects.new(name="Light", object_data=light_data)
        light_obj.location = Vector((0.2, -1.0, 1.3))
        light_obj.rotation_euler = Euler((0.650328, 0.055217, 1.866391), "XYZ")
        light_obj.hide_viewport = True
        root_col.objects.link(light_obj)

    if bpy.data.objects.get("Back_Light") is None:
        back_data = bpy.data.lights.new(name="Back_Light", type="POINT")
        back_data.energy = 120.0
        back_obj = bpy.data.objects.new(name="Back_Light", object_data=back_data)
        back_obj.location = Vector((0.0, 1.5, 1.2))
        root_col.objects.link(back_obj)

    cam_obj = bpy.data.objects.get("Camera")
    if cam_obj is None:
        cam_data = bpy.data.cameras.new(name="Camera")
        cam_obj = bpy.data.objects.new(name="Camera", object_data=cam_data)
        root_col.objects.link(cam_obj)
    cam_obj.location = Vector((0.0, -1.45, 0.95))
    cam_obj.rotation_euler = Euler((math.radians(88.0), 0.0, 0.0), "XYZ")
    cam_obj.hide_viewport = True
    if scene.camera is None:
        scene.camera = cam_obj


# ==============================================================================
# 共通 BMesh / 型紙構築ユーティリティ
# ==============================================================================
def _add_seam(bm, v1, v2):  # audit: allow-positional
    """1 対 1 縫合エッジ（面を持たない Loose Edge）を作成する。"""
    if not v1 or not v2 or not v1.is_valid or not v2.is_valid or v1 == v2:
        return None
    bm.edges.ensure_lookup_table()
    edge = bm.edges.get((v1, v2))
    if edge is None:
        edge = bm.edges.new((v1, v2))
    return edge


def _add_waypoint_seam(bm, vert_a, vert_b, *, waypoint_coord):  # audit: allow-positional
    """中継点（Waypoint）を持つ縫合パス A - W - B を作成する。"""
    if not vert_a.is_valid or not vert_b.is_valid:
        return None
    w = bm.verts.new(waypoint_coord)
    bm.verts.ensure_lookup_table()
    e1 = _add_seam(bm, vert_a, w)
    e2 = _add_seam(bm, w, vert_b)
    return w, e1, e2


def _create_quad_grid_faces(
    bm,
    grid: Sequence[Sequence[object]],
    *,
    reverse_winding: bool = False,
    face_layer_attr=None,
    layer_value: int = 0,
) -> None:
    """2 次元頂点グリッドから一貫した巻き順の四角形ポリゴン群を生成する。"""
    num_v = len(grid)
    num_u = len(grid[0])
    for vi in range(num_v - 1):
        for ui in range(num_u - 1):
            v00 = grid[vi][ui]
            v10 = grid[vi][ui + 1]
            v11 = grid[vi + 1][ui + 1]
            v01 = grid[vi + 1][ui]
            verts = (v00, v01, v11, v10) if reverse_winding else (v00, v10, v11, v01)
            face = bm.faces.new(verts)
            if face_layer_attr is not None:
                face[face_layer_attr] = layer_value


def _bodice_half_width(t_v: float, *, hem_hw: float, waist_hw: float) -> float:
    """身頃の縦位置比率 t_v (0=裾 .. 1=肩) に対する単調な半幅を返す。"""
    if t_v < 0.6:
        return hem_hw - (t_v / 0.6) * (hem_hw - waist_hw)
    t_ah = (t_v - 0.6) / 0.4
    s = 0.5 * (1.0 - math.cos(t_ah * math.pi))
    return waist_hw - s * 0.045


def _bodice_top_z(x: float, *, is_front: bool, front_neck_z: float) -> float:
    """肩線および襟ぐり開口部の高さ Z 座標を返す。"""
    ax = abs(x)
    if ax >= 0.045:
        t = (ax - 0.045) / (0.085 - 0.045)
        return 1.035 - t * 0.015
    t = ax / 0.045
    base_z = front_neck_z if is_front else 1.020
    depth = (1.035 - front_neck_z) if is_front else 0.015
    return base_z + depth * (1.0 - math.cos(t * math.pi * 0.5))


def _build_bodice_grid(
    bm,
    y_val: float,
    *,
    is_front: bool,
    z_bot: float,
    hem_hw: float,
    waist_hw: float,
    front_neck_z: float,
    face_layer_attr=None,
    layer_value: int = 0,
) -> list[list[object]]:
    """前後一体幅の身頃パネル (49x71) を構築する。"""
    import numpy as np
    from mathutils import Vector

    num_u = 49
    num_v = 71
    u_ratios = np.linspace(-1.0, 1.0, num_u)
    grid: list[list[object]] = []

    for vi in range(num_v):
        t_v = vi / (num_v - 1)
        hw = _bodice_half_width(t_v, hem_hw=hem_hw, waist_hw=waist_hw)
        row = []
        for u_r in u_ratios:
            x = float(u_r) * hw
            z_top = _bodice_top_z(float(u_r) * 0.085, is_front=is_front, front_neck_z=front_neck_z)
            z = z_bot + t_v * (z_top - z_bot)
            row.append(bm.verts.new(Vector((x, y_val, z))))
        grid.append(row)

    _create_quad_grid_faces(
        bm,
        grid,
        reverse_winding=not is_front,
        face_layer_attr=face_layer_attr,
        layer_value=layer_value,
    )
    return grid


def _build_flat_sleeve_grid(
    bm,
    x_sign: float,
    *,
    sleeve_u: int,
    sleeve_v: int,
    sleeve_len: float,
    cuff_half_w: float,
    face_layer_attr=None,
    layer_value: int = 0,
) -> list[list[object]]:
    """イセ込み付き平面袖型紙（表側 +Z 法線）を構築する。"""
    import numpy as np
    from mathutils import Vector

    armhole_half_w = 0.145
    cap_height = 0.040
    grid: list[list[object]] = []

    for vi in range(sleeve_v):
        t_v = vi / (sleeve_v - 1)
        dist = t_v * sleeve_len
        x_pos = (0.130 + dist) * x_sign
        half_w = armhole_half_w - t_v * (armhole_half_w - cuff_half_w)
        row_ys = np.linspace(-half_w, half_w, sleeve_u)
        cap_weight = 1.0 - t_v
        row = []
        for y in row_ys:
            yf = float(y)
            ratio_sq = (yf / half_w) ** 2
            cap_x = (1.0 - ratio_sq) * cap_height * (-x_sign) * cap_weight
            z_pos = 1.025 + (1.0 - ratio_sq) * 0.015
            row.append(bm.verts.new(Vector((x_pos + cap_x, yf, z_pos))))
        grid.append(row)

    _create_quad_grid_faces(
        bm,
        grid,
        reverse_winding=(x_sign > 0.0),
        face_layer_attr=face_layer_attr,
        layer_value=layer_value,
    )
    return grid


def _sew_underarm_waypoints(
    bm,
    *,
    sleeve_l: Sequence[Sequence[object]],
    sleeve_r: Sequence[Sequence[object]],
    z_armpit: float = 0.885,
    z_cuff: float = 0.960,
) -> None:
    """腕コライダーの下側を迂回する袖下線の中継点（Waypoint）縫合を作成する。"""
    from mathutils import Vector

    sleeve_v = len(sleeve_l)
    for vi in range(1, sleeve_v):
        t_s = vi / (sleeve_v - 1)
        wp_z = z_armpit + t_s * (z_cuff - z_armpit)
        wp_x_l = 0.080 + t_s * (sleeve_l[vi][0].co.x - 0.080)
        wp_x_r = -0.080 + t_s * (sleeve_r[vi][0].co.x - (-0.080))
        _add_waypoint_seam(
            bm,
            sleeve_l[vi][0],
            sleeve_l[vi][-1],
            waypoint_coord=Vector((wp_x_l, 0.000, wp_z)),
        )
        _add_waypoint_seam(
            bm,
            sleeve_r[vi][0],
            sleeve_r[vi][-1],
            waypoint_coord=Vector((wp_x_r, 0.000, wp_z)),
        )


def _apply_common_cloth_settings(
    obj,
    *,
    fabric_preset: str,
    thickness: float,
    sewing_shrink_speed: float = 5.0,
    enable_sewing_lock: bool = True,
    pin_vertex_group: str = "Pin",
) -> None:
    """検証済みの安定布シミュレーション設定をオブジェクトへ一括適用する。"""
    cloth = obj.taremin_cloth
    cloth.is_cloth = True
    cloth.enabled = True
    cloth.last_fabric_preset = fabric_preset

    cloth.tension_stiffness = 1500.0
    cloth.compression_stiffness = 150.0
    cloth.shear_stiffness = 100.0
    cloth.bending_stiffness = 15.0
    cloth.stiffness = 1500.0
    cloth.tension_damping = 5.0
    cloth.compression_damping = 5.0
    cloth.shear_damping = 5.0
    cloth.bending_damping = 0.5
    cloth.air_damping = 1.0
    cloth.areal_density = 0.15

    cloth.substeps = 20
    cloth.solver_iterations = 2
    cloth.thickness = thickness

    cloth.enable_self_collision = True
    cloth.self_collision_algorithm = "DIRECT"
    cloth.coupled_self_collision_mode = "OFF"
    cloth.post_collision_relaxation_iters = 0
    cloth.self_collision_max_iterations = "512"
    cloth.self_collision_relief_factor = 1.0
    cloth.self_collision_max_displacement_ratio = 0.5
    cloth.enable_normal_untangling = True

    cloth.enable_sewing = True
    cloth.sewing_shrink_speed = sewing_shrink_speed
    cloth.sewing_stiffness = 30000.0
    cloth.enable_sewing_lock = enable_sewing_lock
    cloth.sewing_lock_distance = 0.02
    cloth.enable_sewing_priority = True
    cloth.pin_vertex_group = pin_vertex_group


# ==============================================================================
# 1. Tシャツ (Pattern_TShirt: 10,204 頂点, 9,744 面)
# ==============================================================================
def build_tshirt(col):
    """半袖 T シャツ型紙を構築する。"""
    import bpy
    import bmesh

    mesh = bpy.data.meshes.new("Pattern_TShirt")
    obj = bpy.data.objects.new("Pattern_TShirt", mesh)
    col.objects.link(obj)
    bm = bmesh.new()

    f_bodice = _build_bodice_grid(
        bm, -0.115, is_front=True, z_bot=0.68, hem_hw=0.170, waist_hw=0.130, front_neck_z=0.980
    )
    b_bodice = _build_bodice_grid(
        bm, 0.075, is_front=False, z_bot=0.68, hem_hw=0.170, waist_hw=0.130, front_neck_z=0.980
    )
    sleeve_l = _build_flat_sleeve_grid(
        bm, 1.0, sleeve_u=57, sleeve_v=28, sleeve_len=0.12, cuff_half_w=0.145
    )
    sleeve_r = _build_flat_sleeve_grid(
        bm, -1.0, sleeve_u=57, sleeve_v=28, sleeve_len=0.12, cuff_half_w=0.145
    )
    bm.verts.ensure_lookup_table()

    # 脇線 (vi=0..42)
    for vi in range(43):
        _add_seam(bm, f_bodice[vi][-1], b_bodice[vi][-1])
        _add_seam(bm, f_bodice[vi][0], b_bodice[vi][0])

    # 肩線 (ui=0..11, 37..48)
    for ui in range(0, 12):
        _add_seam(bm, f_bodice[70][ui], b_bodice[70][ui])
    for ui in range(37, 49):
        _add_seam(bm, f_bodice[70][ui], b_bodice[70][ui])

    # アームホール (vi=42..70 <-> 袖山 0..28, 56..28)
    for i, vi in enumerate(range(42, 71)):
        _add_seam(bm, f_bodice[vi][-1], sleeve_l[0][i])
        _add_seam(bm, f_bodice[vi][0], sleeve_r[0][i])
        _add_seam(bm, b_bodice[vi][-1], sleeve_l[0][56 - i])
        _add_seam(bm, b_bodice[vi][0], sleeve_r[0][56 - i])

    # 袖下線 Waypoint 縫合
    _sew_underarm_waypoints(bm, sleeve_l=sleeve_l, sleeve_r=sleeve_r, z_armpit=0.885, z_cuff=0.965)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    _apply_common_cloth_settings(obj, fabric_preset="COTTON", thickness=0.00157)
    return obj


# ==============================================================================
# 2. ズボン (Pattern_Pants: 9,240 頂点, 8,832 面 + ウエストゴム)
# ==============================================================================
def build_pants(col):
    """テーラードスラックス型紙（ウエストゴム付き・ピン不要自立設計）を構築する。"""
    import bpy
    import bmesh
    import numpy as np
    from mathutils import Vector

    mesh = bpy.data.meshes.new("Pattern_Pants")
    obj = bpy.data.objects.new("Pattern_Pants", mesh)
    col.objects.link(obj)
    bm = bmesh.new()

    leg_zs = np.linspace(0.18, 0.62, 46)
    rise_zs_f = np.linspace(0.625, 0.850, 24)
    rise_zs_b = np.linspace(0.625, 0.865, 24)
    num_u = 33
    shift_x_val = 0.080

    def make_pant_panel(*, is_front: bool, is_left: bool) -> list[list[object]]:
        y_val = -0.130 if is_front else 0.095
        x_sign = 1.0 if is_left else -1.0
        grid: list[list[object]] = []

        outer_x_base = 0.120
        width = 0.190
        for zi, z in enumerate(leg_zs):
            t_leg = zi / (len(leg_zs) - 1)
            shift_x = (t_leg * shift_x_val) * x_sign
            outer_x_base = 0.090 + t_leg * 0.030
            width = (0.110 + t_leg * 0.080) if is_front else (0.110 + t_leg * 0.095)
            inner_x_base = outer_x_base - width

            pt_outer = (outer_x_base * x_sign) + shift_x
            pt_inner = (inner_x_base * x_sign) + shift_x
            xs = np.linspace(min(pt_outer, pt_inner), max(pt_outer, pt_inner), num_u)
            grid.append([bm.verts.new(Vector((float(x), y_val, float(z)))) for x in xs])

        rise_zs = rise_zs_f if is_front else rise_zs_b
        shift_x = shift_x_val * x_sign
        crotch_inner_x = outer_x_base - width

        for zi, z in enumerate(rise_zs):
            t_rise = zi / (len(rise_zs) - 1)
            w_waist = 0.080 if is_front else 0.085
            if t_rise < 0.35:
                ox = 0.120
            else:
                s = (t_rise - 0.35) / 0.65
                ox = 0.120 + (w_waist - 0.120) * 0.5 * (1.0 - math.cos(s * math.pi))

            j_curve = (1.0 - t_rise) ** 2.2
            ix = crotch_inner_x * j_curve
            pt_outer = (ox * x_sign) + shift_x
            pt_inner = (ix * x_sign) + shift_x
            xs = np.linspace(min(pt_outer, pt_inner), max(pt_outer, pt_inner), num_u)
            grid.append([bm.verts.new(Vector((float(x), y_val, float(z)))) for x in xs])

        _create_quad_grid_faces(bm, grid, reverse_winding=not is_front)
        return grid

    fl = make_pant_panel(is_front=True, is_left=True)
    fr = make_pant_panel(is_front=True, is_left=False)
    bl = make_pant_panel(is_front=False, is_left=True)
    br = make_pant_panel(is_front=False, is_left=False)
    bm.verts.ensure_lookup_table()

    # 外側脇線 (足首裾 vi=0 .. ウエスト)
    for vi in range(len(fl)):
        _add_seam(bm, fl[vi][-1], bl[vi][-1])
        _add_seam(bm, fr[vi][0], br[vi][0])

    # 股上線 (股十字点 vi=45 .. ウエスト)
    for vi in range(45, len(fl)):
        _add_seam(bm, fl[vi][0], fr[vi][-1])
        _add_seam(bm, bl[vi][0], br[vi][-1])

    # 内股線 (足首裾 vi=0 .. 股十字点 vi=45)
    for vi in range(46):
        _add_seam(bm, fl[vi][0], bl[vi][0])
        _add_seam(bm, fr[vi][-1], br[vi][-1])

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    _apply_common_cloth_settings(
        obj,
        fabric_preset="DENIM",
        thickness=0.0015,
        sewing_shrink_speed=6.0,
        pin_vertex_group="",
    )

    # ウエストゴム (Z > 0.835m の辺を 20% 収縮させて骨盤上で保持)
    waist_edges = [
        edge.index
        for edge in mesh.edges
        if mesh.vertices[edge.vertices[0]].co.z > 0.835
        and mesh.vertices[edge.vertices[1]].co.z > 0.835
    ]
    if waist_edges:
        eg = obj.taremin_cloth.elastic_groups.add()
        eg.name = "Waist_Elastic"
        eg.scale = 0.80
        eg.stiffness_multiplier = 4.0
        eg.edge_indices_str = ",".join(map(str, waist_edges))

    return obj


# ==============================================================================
# 3. Yシャツ (Pattern_Shirt: 13,294 頂点, 12,596 面, 前開き持ち出し + 台襟)
# ==============================================================================
def build_shirt(col):
    """前立て重なりレイヤー（cloth_layer）と台襟を備えた長袖 Y シャツ型紙を構築する。"""
    import bpy
    import bmesh
    import numpy as np
    from mathutils import Vector

    mesh = bpy.data.meshes.new("Pattern_Shirt")
    obj = bpy.data.objects.new("Pattern_Shirt", mesh)
    col.objects.link(obj)
    bm = bmesh.new()

    layer_attr = bm.faces.layers.int.new("cloth_layer")
    num_v = 71
    num_u_half = 25

    def make_front_half(*, is_left: bool) -> list[list[object]]:
        y_val = -0.112 if is_left else -0.118
        overlap = 0.015
        grid: list[list[object]] = []
        for vi in range(num_v):
            t_v = vi / (num_v - 1)
            hw = _bodice_half_width(t_v, hem_hw=0.170, waist_hw=0.130)
            if is_left:
                xs: list[float] = [-overlap, -overlap * 0.5]
                xs.extend(float(x) for x in np.linspace(0.0, hw, 23))
            else:
                xs = [float(x) for x in np.linspace(-hw, 0.0, 23)[:-1]]
                xs.extend([0.0, overlap * 0.5, overlap])

            row = []
            for x in xs:
                z_bot = 0.66
                z_top = _bodice_top_z(abs(x), is_front=True, front_neck_z=0.990)
                z = z_bot + t_v * (z_top - z_bot)
                row.append(bm.verts.new(Vector((x, y_val, z))))
            grid.append(row)

        assert len(grid[0]) == num_u_half
        _create_quad_grid_faces(
            bm,
            grid,
            reverse_winding=False,
            face_layer_attr=layer_attr,
            layer_value=(0 if is_left else 1),
        )
        return grid

    fl = make_front_half(is_left=True)
    fr = make_front_half(is_left=False)
    b_bodice = _build_bodice_grid(
        bm,
        0.075,
        is_front=False,
        z_bot=0.66,
        hem_hw=0.170,
        waist_hw=0.130,
        front_neck_z=0.990,
        face_layer_attr=layer_attr,
        layer_value=0,
    )

    sleeve_l = _build_flat_sleeve_grid(
        bm,
        1.0,
        sleeve_u=57,
        sleeve_v=51,
        sleeve_len=0.31,
        cuff_half_w=0.090,
        face_layer_attr=layer_attr,
        layer_value=0,
    )
    sleeve_r = _build_flat_sleeve_grid(
        bm,
        -1.0,
        sleeve_u=57,
        sleeve_v=51,
        sleeve_len=0.31,
        cuff_half_w=0.090,
        face_layer_attr=layer_attr,
        layer_value=0,
    )

    # 台襟 (47x7)
    collar_u = 47
    collar_v = 7
    c_xs = np.linspace(-0.1225, 0.1225, collar_u)
    collar_grid: list[list[object]] = []
    for vi in range(collar_v):
        t_v = vi / (collar_v - 1)
        z = 1.050 + t_v * 0.028
        collar_grid.append([bm.verts.new(Vector((float(x), 0.095, z))) for x in c_xs])
    _create_quad_grid_faces(
        bm,
        collar_grid,
        reverse_winding=True,
        face_layer_attr=layer_attr,
        layer_value=1,
    )
    bm.verts.ensure_lookup_table()

    # 1. 前立てボタン留め
    for b_vi in (4, 16, 28, 40, 52, 64):
        _add_seam(bm, fr[b_vi][22], fl[b_vi][2])

    # 2. 脇線
    for vi in range(43):
        _add_seam(bm, fl[vi][-1], b_bodice[vi][-1])
        _add_seam(bm, fr[vi][0], b_bodice[vi][0])

    # 3. 肩線
    for i in range(12):
        _add_seam(bm, fl[70][13 + i], b_bodice[70][37 + i])
        _add_seam(bm, fr[70][11 - i], b_bodice[70][11 - i])

    # 4. アームホール
    for i, vi in enumerate(range(42, 71)):
        _add_seam(bm, fl[vi][-1], sleeve_l[0][i])
        _add_seam(bm, fr[vi][0], sleeve_r[0][i])
        _add_seam(bm, b_bodice[vi][-1], sleeve_l[0][56 - i])
        _add_seam(bm, b_bodice[vi][0], sleeve_r[0][56 - i])

    # 5. 袖下線 Waypoint 縫合
    _sew_underarm_waypoints(bm, sleeve_l=sleeve_l, sleeve_r=sleeve_r, z_armpit=0.885, z_cuff=0.960)

    # 6. 台襟 360 度縫合 (前襟ぐりは首側面を迂回する Waypoint を配置)
    bodice_neck: list[object] = []
    for ui in range(22, 11, -1):
        bodice_neck.append(fr[70][ui])
    for ui in range(12, 37):
        bodice_neck.append(b_bodice[70][ui])
    for ui in range(12, 1, -1):
        bodice_neck.append(fl[70][ui])

    for i in range(collar_u):
        if i < 11:
            t = (10 - i) / 10.0
            wp = Vector((-0.065 - t * 0.045, -0.015 - t * 0.035, 1.042 + t * 0.008))
            _add_waypoint_seam(bm, collar_grid[0][i], bodice_neck[i], waypoint_coord=wp)
        elif i >= 36:
            t = (i - 36) / 10.0
            wp = Vector((0.065 + t * 0.045, -0.015 - t * 0.035, 1.042 + t * 0.008))
            _add_waypoint_seam(bm, collar_grid[0][i], bodice_neck[i], waypoint_coord=wp)
        else:
            _add_seam(bm, collar_grid[0][i], bodice_neck[i])

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    _apply_common_cloth_settings(obj, fabric_preset="COTTON", thickness=0.0015)
    return obj


# ==============================================================================
# 4. フード付きパーカー + 引き紐 (Pattern_Hood: 13,462 頂点 + Drawstring 94 点)
# ==============================================================================
def _build_rigid_angled_hood_panel(bm, x_sign: float) -> list[list[object]]:
    """
    2D 平面型紙 (YZ 平面: 25x35) に対して純粋な 3D 剛体回転 (SO(3)) と平行移動のみを適用し、
    初期歪み 0% を保ったまま頭・首コライダーを避ける「ハの字」立体配置のフードパネルを生成する。
    """
    from mathutils import Vector

    num_u_hood = 25
    num_v_hood = 35

    pivot_y = 0.0025
    pivot_z = 1.1125
    cy = math.cos(math.radians(20.0))
    sy = math.sin(math.radians(20.0))
    cz = math.cos(math.radians(15.0))
    sz = math.sin(math.radians(15.0))

    grid: list[list[object]] = []
    for vi in range(num_v_hood):
        t_v = vi / (num_v_hood - 1)
        z_base = 1.040 + t_v * (1.185 - 1.040)
        y_face = -0.070 + 0.025 * (t_v ** 2)
        y_back = 0.075 - 0.015 * t_v

        row = []
        for ui in range(num_u_hood):
            t_u = ui / (num_u_hood - 1)
            y_base = y_face + t_u * (y_back - y_face)

            dy = y_base - pivot_y
            dz = z_base - pivot_z

            # 1) Z 軸周り回転: 後中心シーム側 (dy > 0) を内側 (X=0) に寄せる
            x1 = -x_sign * dy * sz
            y1 = dy * cz
            z1 = dz

            # 2) Y 軸周り回転: クラウン上端 (z1 > 0) を内側 (X=0) に傾け、襟ぐり下端を外へ開く
            x2 = x1 * cy - x_sign * z1 * sy
            y2 = y1
            z2 = x_sign * x1 * sy + z1 * cy

            x = x_sign * 0.072 + x2
            y = pivot_y + y2
            z = (pivot_z + 0.018) + z2
            row.append(bm.verts.new(Vector((x, y, z))))
        grid.append(row)

    # 左フード (x_sign > 0) は +X 外向き、右フード (x_sign < 0) は -X 外向き
    _create_quad_grid_faces(bm, grid, reverse_winding=(x_sign < 0.0))
    return grid


def _setup_hood_drawstring(hood_obj, col, *, start_hood_vert: int) -> object:
    """フード顔開口部のガイド頂点列に沿った 3D POLY カーブ引き紐を構築して登録する。"""
    import bpy
    import numpy as np

    num_u_hood = 25
    num_v_hood = 35
    l_start = start_hood_vert
    r_start = start_hood_vert + num_u_hood * num_v_hood

    # 左顔周り下端 -> 左クラウン -> 右クラウン -> 右顔周り下端 (計 70 頂点)
    guide_verts = [l_start + vi * num_u_hood for vi in range(num_v_hood)] + [
        r_start + vi * num_u_hood for vi in range(num_v_hood - 1, -1, -1)
    ]
    v_l_crown = guide_verts[num_v_hood - 1]
    v_r_crown = guide_verts[num_v_hood]
    seam_bridges = {(min(v_l_crown, v_r_crown), max(v_l_crown, v_r_crown))}

    verts = hood_obj.data.vertices
    channel_pts = np.array([verts[v].co[:] for v in guide_verts], dtype=np.float32)

    tail_count = 12
    p_start = channel_pts[0]
    left_tail = [
        p_start + np.array([-0.010 * (k / tail_count), -0.050 * (k / tail_count), -0.012 * k], dtype=np.float32)
        for k in range(tail_count, 0, -1)
    ]
    p_end = channel_pts[-1]
    right_tail = [
        p_end + np.array([+0.010 * (k / tail_count), -0.050 * (k / tail_count), -0.012 * k], dtype=np.float32)
        for k in range(1, tail_count + 1)
    ]

    ds_pts = np.vstack([left_tail, channel_pts, right_tail])
    total_pts = len(ds_pts)

    curve_data = bpy.data.curves.new(name=DRAWSTRING_OBJECT_NAME, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = 0.0022
    curve_data.bevel_resolution = 3

    spline = curve_data.splines.new("POLY")
    spline.points.add(total_pts - 1)
    raw_points_4d = np.ones((total_pts, 4), dtype=np.float32)
    raw_points_4d[:, :3] = ds_pts
    spline.points.foreach_set("co", raw_points_4d.ravel())

    curve_obj = bpy.data.objects.new(name=DRAWSTRING_OBJECT_NAME, object_data=curve_data)
    curve_obj.matrix_world = hood_obj.matrix_world.copy()
    col.objects.link(curve_obj)

    settings = hood_obj.taremin_cloth
    settings.string_parts.clear()
    part = settings.string_parts.add()
    part.name = "Hood_Drawstring"
    part.enabled = True
    part.curve_object = curve_obj
    part.set_guide_vert_indices(guide_verts)
    part.set_seam_bridge_pairs(seam_bridges)
    part.tail_vert_count_left = tail_count
    part.tail_vert_count_right = tail_count
    part.eyelet_stride = 1
    part.radius = 0.0022
    part.linear_density = 0.002
    settings.active_string_part_index = 0

    return curve_obj


def build_hood(col):
    """フード一体型パーカーと引き紐カーブを構築する。"""
    import bpy
    import bmesh

    mesh = bpy.data.meshes.new("Pattern_Hood")
    obj = bpy.data.objects.new("Pattern_Hood", mesh)
    col.objects.link(obj)
    bm = bmesh.new()

    f_bodice = _build_bodice_grid(
        bm, -0.115, is_front=True, z_bot=0.64, hem_hw=0.175, waist_hw=0.135, front_neck_z=0.995
    )
    b_bodice = _build_bodice_grid(
        bm, 0.075, is_front=False, z_bot=0.64, hem_hw=0.175, waist_hw=0.135, front_neck_z=0.995
    )
    sleeve_l = _build_flat_sleeve_grid(
        bm, 1.0, sleeve_u=57, sleeve_v=41, sleeve_len=0.30, cuff_half_w=0.095
    )
    sleeve_r = _build_flat_sleeve_grid(
        bm, -1.0, sleeve_u=57, sleeve_v=41, sleeve_len=0.30, cuff_half_w=0.095
    )

    start_hood_vert = len(bm.verts)  # 11,632
    l_hood = _build_rigid_angled_hood_panel(bm, 1.0)
    r_hood = _build_rigid_angled_hood_panel(bm, -1.0)
    end_hood_vert = len(bm.verts)    # 13,382
    bm.verts.ensure_lookup_table()

    # 1. 身頃脇線
    for vi in range(43):
        _add_seam(bm, f_bodice[vi][-1], b_bodice[vi][-1])
        _add_seam(bm, f_bodice[vi][0], b_bodice[vi][0])

    # 2. 身頃肩線 (ui=0..12, 36..48 で襟ぐり接続点まで隙間なく縫合)
    for ui in range(0, 13):
        _add_seam(bm, f_bodice[70][ui], b_bodice[70][ui])
    for ui in range(36, 49):
        _add_seam(bm, f_bodice[70][ui], b_bodice[70][ui])

    # 3. アームホール
    for i, vi in enumerate(range(42, 71)):
        _add_seam(bm, f_bodice[vi][-1], sleeve_l[0][i])
        _add_seam(bm, f_bodice[vi][0], sleeve_r[0][i])
        _add_seam(bm, b_bodice[vi][-1], sleeve_l[0][56 - i])
        _add_seam(bm, b_bodice[vi][0], sleeve_r[0][56 - i])

    # 4. フード後頭部 (ui=24) と頭頂部クラウン (vi=34, ui=0..23)
    num_u_hood = 25
    num_v_hood = 35
    for vi in range(num_v_hood):
        _add_seam(bm, l_hood[vi][-1], r_hood[vi][-1])
    for ui in range(num_u_hood - 1):
        _add_seam(bm, l_hood[-1][ui], r_hood[-1][ui])

    # 5. フード後襟ぐり (ui=12..24) と前襟ぐり (ui=0..12)
    for i in range(13):
        _add_seam(bm, l_hood[0][12 + i], b_bodice[70][36 - i])
        _add_seam(bm, r_hood[0][12 + i], b_bodice[70][12 + i])
    for i in range(13):
        _add_seam(bm, l_hood[0][i], f_bodice[70][24 + i])
        _add_seam(bm, r_hood[0][i], f_bodice[70][24 - i])

    # 6. 袖下線 Waypoint 縫合 (フード頂点範囲 11632..13381 の後ろに 80 頂点追加 -> 計 13,462 頂点)
    _sew_underarm_waypoints(bm, sleeve_l=sleeve_l, sleeve_r=sleeve_r, z_armpit=0.885, z_cuff=0.960)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    # レイヤー階層属性 (身頃・袖・Waypoint = Layer 0, フード = Layer 1)
    layer_attr = mesh.attributes.new(name="cloth_layer", type="INT", domain="POINT")
    layer_vals = [0] * len(mesh.vertices)
    for idx in range(start_hood_vert, end_hood_vert):
        layer_vals[idx] = 1
    layer_attr.data.foreach_set("value", layer_vals)
    mesh.update()

    _apply_common_cloth_settings(
        obj,
        fabric_preset="FLEECE",
        thickness=0.001,
        sewing_shrink_speed=0.5,
        enable_sewing_lock=False,
    )
    _setup_hood_drawstring(obj, col, start_hood_vert=start_hood_vert)
    return obj


# ==============================================================================
# シーン一括セットアップ・CLI エントリポイント
# ==============================================================================
def setup_tailored_scene(
    *,
    body_blend_path: Path | None = None,
    active_garment: str = "Pattern_Hood",
    include_body: bool = True,
) -> dict[str, object]:
    """
    現在の Blender シーン上に素体コライダーと 4 着の洋裁テストパターンを一括構築する。
    """
    import bpy

    repo_root = _resolve_repo_root()
    _ensure_addon_registered(repo_root)
    from taremin_cloth.engine.runner import cache_rest_positions

    if body_blend_path is None:
        env_body = os.environ.get("TAREMIN_CLOTH_BODY_BLEND", "").strip()
        if env_body:
            body_blend_path = Path(env_body)
        else:
            for candidate in (
                repo_root / "tests" / "fixtures" / "bodies" / "siroino" / "SiroinoSotai.blend",
                repo_root / "tmp" / "SiroinoSotai.blend",
            ):
                if candidate.exists():
                    body_blend_path = candidate
                    break

    _clean_scene_for_setup(clear_default_cube=True)
    cols = _setup_collections()

    body_obj = None
    arm_obj = None
    if include_body:
        body_obj, arm_obj = _setup_sotai_collider(cols["sotai"], body_blend_path=body_blend_path)
    _setup_lighting_and_camera()

    garments = {
        "Pattern_TShirt": build_tshirt(cols["tshirt"]),
        "Pattern_Pants": build_pants(cols["pants"]),
        "Pattern_Shirt": build_shirt(cols["shirt"]),
        "Pattern_Hood": build_hood(cols["hood"]),
    }

    # レスト座標を全布オブジェクトに事前キャッシュし、アクティブ対象のみを表示する
    show_all = active_garment.upper() == "ALL"
    for name, g_obj in garments.items():
        cache_rest_positions(g_obj, force=True)
        is_visible = show_all or (name == active_garment)
        g_obj.hide_viewport = not is_visible
        g_obj.hide_render = not is_visible
        if name == "Pattern_Hood":
            ds_obj = bpy.data.objects.get(DRAWSTRING_OBJECT_NAME)
            if ds_obj is not None:
                ds_obj.hide_viewport = not is_visible
                ds_obj.hide_render = not is_visible

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 250
    scene.frame_set(1)
    bpy.context.view_layer.update()

    active_obj = garments.get(active_garment) or garments["Pattern_Hood"]
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    if not active_obj.hide_viewport:
        active_obj.select_set(True)
        bpy.context.view_layer.objects.active = active_obj

    print(
        f"[setup_tailored_patterns] 生成完了: "
        f"body={'OK' if body_obj else 'SKIPPED'}, "
        f"garments={ {k: len(v.data.vertices) for k, v in garments.items()} }"
    )
    return {
        "body": body_obj,
        "armature": arm_obj,
        "garments": garments,
    }


def _parse_cli_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="新規 Blender シーンに Taremin Cloth 洋裁テストパターン一式を構築して .blend に保存するツール"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="書き出し先の .blend ファイルパス (例: tmp/test_pattern.blend)",
    )
    parser.add_argument(
        "--body-blend",
        type=Path,
        default=None,
        help="素体 (SiroinoSotai_PC / Armature) を読み込む元 .blend パス (省略時は tests/fixtures/bodies/siroino/SiroinoSotai.blend)",
    )
    parser.add_argument(
        "--no-body",
        action="store_true",
        help="素体コライダーのアペンドをスキップして衣服メッシュのみを生成する",
    )
    parser.add_argument(
        "--active",
        type=str,
        default="Pattern_Hood",
        choices=[*GARMENT_OBJECT_NAMES, "ALL"],
        help="初期表示を有効にする衣服オブジェクト名 (デフォルト: Pattern_Hood)",
    )
    parser.add_argument(
        "--blender",
        type=str,
        default=None,
        help="CLI 実行時に使用する Blender バージョン (例: 5.2, latest-lts)",
    )
    return parser.parse_args(argv)


def _run_inside_blender(cli_argv: Sequence[str]) -> None:
    """Blender プロセス内部からの実行（Scripting タブ実行または --python 経由）。"""
    import bpy

    args = _parse_cli_args(cli_argv)
    if args.output is not None:
        # CLI から --output 指定で呼ばれた場合は完全な空の新規シーンに初期化する
        bpy.ops.wm.read_homefile(use_empty=True)

    setup_tailored_scene(
        body_blend_path=args.body_blend,
        active_garment=args.active,
        include_body=not args.no_body,
    )

    if args.output is not None:
        out_path = args.output.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
        print(f"[setup_tailored_patterns] .blend ファイルを書き出しました: {out_path}")


def _run_from_system_python(cli_argv: Sequence[str]) -> int:
    """通常の Python 環境から resolve_blender() でヘッドレス Blender を起動して実行する。"""
    repo_root = _resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.blender_manager import resolve_blender

    args = _parse_cli_args(cli_argv)
    output_path = args.output or (repo_root / "tmp" / "test_pattern.blend")
    blender_exe = resolve_blender(args.blender or "latest-lts")

    forwarded_args = ["--output", str(output_path.resolve()), "--active", args.active]
    if args.body_blend is not None:
        forwarded_args.extend(["--body-blend", str(args.body_blend.resolve())])
    if args.no_body:
        forwarded_args.append("--no-body")

    cmd = [
        str(blender_exe),
        "-b",
        "--factory-startup",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        *forwarded_args,
    ]
    res = subprocess.run(cmd)
    return res.returncode


def _is_running_inside_blender_app() -> bool:
    """pip 版 bpy 等ではなく実際の Blender プロセス内部で実行されているかを判定する。"""
    try:
        import bpy  # type: ignore

        app = getattr(bpy, "app", None)
        binary_path = getattr(app, "binary_path", "") if app is not None else ""
        return bool(binary_path) and getattr(bpy, "data", None) is not None
    except ImportError:
        return False


def main() -> None:
    if _is_running_inside_blender_app():
        custom_argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
        _run_inside_blender(custom_argv)
    else:
        raise SystemExit(_run_from_system_python(sys.argv[1:]))


if __name__ == "__main__":
    main()
