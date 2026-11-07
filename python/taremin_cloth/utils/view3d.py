"""
3Dビューポート・アニメーション制御ユーティリティ (taremin_cloth.utils.view3d)
Blenderの画面再描画要求およびアニメーション再生停止を一元的に提供する。
"""

from __future__ import annotations

from typing import Any

try:
    import bpy
except ImportError:
    bpy = None


def tag_redraw_view3d(context=None, *, extra_area_types=None):
    """
    現在のスクリーン内のすべての 3D Viewport エリア（および必要に応じて追加エリア）に対して再描画（tag_redraw）を要求する。
    context が指定されていない場合は bpy.context を参照する。
    """
    ctx = context or getattr(bpy, "context", None)
    if not ctx:
        return

    target_types = {'VIEW_3D'}
    if extra_area_types:
        target_types.update(extra_area_types)

    screen = getattr(ctx, "screen", None)
    if not screen:
        # window_manager経由でのフォールバック
        wm = getattr(ctx, "window_manager", None)
        if wm and hasattr(wm, "windows"):
            for win in wm.windows:
                if win.screen:
                    for area in win.screen.areas:
                        if area.type in target_types:
                            area.tag_redraw()
        return

    for area in getattr(screen, "areas", []):
        if area.type in target_types:
            area.tag_redraw()


def stop_animation(context=None):
    """
    アニメーション再生中であれば安全に停止する。
    """
    ctx = context or getattr(bpy, "context", None)
    if not ctx:
        return

    screen = getattr(ctx, "screen", None)
    if screen and getattr(screen, "is_animation_playing", False):
        try:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        except Exception:
            pass


def force_redraw_view3d(context=None):
    """
    現在の3Dビューポートの即時再描画を強制する。
    同期的に実行されるオペレーター（一瞬ドレープ等）の実行途中でHUDプログレスを即座に画面へ反映させる。
    """
    tag_redraw_view3d(context)
    try:
        if not getattr(bpy.app, "background", False):
            wm = getattr(bpy.ops, "wm", None)
            if wm and hasattr(wm, "redraw_timer"):
                wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    except Exception:
        pass


def get_viewport_overlay_offsets(*, area=None) -> tuple[float, float, float, float]:
    """3D Viewport エリア内の各リージョン（HEADER, TOOL_HEADER, TOOLS, UI）のオーバーレイ幅・高さを算出する.
    
    (left_offset, right_offset, top_offset, bottom_offset) のピクセルタプルを返す.
    """
    if not area:
        ctx = getattr(bpy, "context", None)
        area = getattr(ctx, "area", None) if ctx else None
    if not area or getattr(area, "type", "") != 'VIEW_3D':
        return 0.0, 0.0, 0.0, 0.0

    win = next((r for r in getattr(area, "regions", []) if getattr(r, "type", "") == 'WINDOW'), None)
    if not win:
        return 0.0, 0.0, 0.0, 0.0

    win_top = win.y + win.height
    win_bottom = win.y

    top_off = 0.0
    bottom_off = 0.0
    left_off = 0.0
    right_off = 0.0

    for r in getattr(area, "regions", []):
        if r.width <= 1 or r.height <= 1:
            continue
        r_type = getattr(r, "type", "")
        if r_type in ('HEADER', 'TOOL_HEADER'):
            if r.y >= win_bottom + win.height * 0.5:
                overlap = win_top - r.y
                if overlap > top_off:
                    top_off = float(overlap)
            else:
                overlap = (r.y + r.height) - win_bottom
                if overlap > bottom_off:
                    bottom_off = float(overlap)
        elif r_type == 'TOOLS':
            left_off = max(left_off, float(r.width))
        elif r_type == 'UI':
            right_off = max(right_off, float(r.width))

    return left_off, right_off, top_off, bottom_off


def sync_view2d_from_region_3d(
    region: bpy.types.Region,
    rv3d: bpy.types.RegionView3D,
    *,
    view_2d: Any,
) -> bool:
    """3Dビューポートの投影位置とスケールをパターンエディタの ViewTransform2D へ 1:1 実寸同期する."""
    try:
        if hasattr(rv3d, "update"):
            rv3d.update()
        from bpy_extras import view3d_utils
        from mathutils import Vector
        p0 = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector((0.0, 0.0, 0.0)))
        p_right = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector((1.0, 0.0, 0.0)))
        if p0 is None or p_right is None:
            return False

        dx = p_right[0] - p0[0]
        dy = p_right[1] - p0[1]
        zoom = (dx * dx + dy * dy) ** 0.5
        if zoom < 1.0:
            # 側面視点 (NUMPAD_3) 等で +X 軸が視線方向と重なる場合は +Z 軸（1m）から実寸スケールを算出
            p_up = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector((0.0, 0.0, 1.0)))
            if p_up is not None:
                ux = p_up[0] - p0[0]
                uy = p_up[1] - p0[1]
                zoom = (ux * ux + uy * uy) ** 0.5
            if zoom < 1.0:
                return False

        view_2d.pan_x = float(p0[0])
        view_2d.pan_y = float(p0[1])
        view_2d.zoom = float(zoom)
        return True
    except Exception:
        return False


def pan_region_3d_by_pixels(
    rv3d: bpy.types.RegionView3D,
    *,
    dx_px: float,
    dy_px: float,
    zoom_px_per_m: float,
) -> None:
    """2Dピクセル移動量に基づき3Dビューポートの注視点 (view_location) を平行移動する."""
    try:
        from mathutils import Vector
        zoom = max(10.0, zoom_px_per_m)
        R = rv3d.view_matrix.to_3x3()
        cam_right = Vector((R[0][0], R[0][1], R[0][2]))
        cam_up = Vector((R[1][0], R[1][1], R[1][2]))
        delta = (-dx_px / zoom) * cam_right + (-dy_px / zoom) * cam_up
        rv3d.view_location += delta
        if hasattr(rv3d, "update"):
            rv3d.update()
    except Exception:
        pass


def zoom_region_3d(
    rv3d: bpy.types.RegionView3D,
    *,
    factor: float,
) -> None:
    """3Dビューポートのズーム倍率を変更する."""
    try:
        if factor > 1e-4:
            rv3d.view_distance /= factor
            if hasattr(rv3d, "update"):
                rv3d.update()
    except Exception:
        pass


def extract_armature_bones_for_canvas(
    scene: bpy.types.Scene,
    *,
    armature_name: str = "",
) -> dict[str, Any]:
    """シーン内のアーマチュアからボーンのワールド3D座標情報 (BoneInfo3D) を抽出する."""
    bones_map: dict[str, Any] = {}
    try:
        import bpy
        from ..pattern.session import BoneInfo3D

        arm_obj = None
        if armature_name and armature_name in scene.objects:
            arm_obj = scene.objects[armature_name]
        elif scene.objects:
            # アクティブオブジェクトがアーマチュアなら最優先
            act = bpy.context.view_layer.objects.active if hasattr(bpy.context, "view_layer") else None
            if act and act.type == 'ARMATURE':
                arm_obj = act
            else:
                for obj in scene.objects:
                    if obj.type == 'ARMATURE':
                        arm_obj = obj
                        break

        if not arm_obj or not arm_obj.data:
            return bones_map

        mat = arm_obj.matrix_world
        # ポーズボーンがあれば優先、なければデータボーン
        bones_source = arm_obj.pose.bones if arm_obj.pose else arm_obj.data.bones

        for b in bones_source:
            h_w = mat @ b.head
            t_w = mat @ b.tail
            bone_data = getattr(b, "bone", b)
            r_h = float(getattr(bone_data, "head_radius", 0.05))
            r_t = float(getattr(bone_data, "tail_radius", 0.05))
            r = max(r_h, r_t)
            if r <= 1e-4:
                b_len = float(getattr(b, "length", (t_w - h_w).length))
                r = max(0.015, b_len * 0.08)

            b_info = BoneInfo3D(
                name=b.name,
                head_3d=(float(h_w.x), float(h_w.y), float(h_w.z)),
                tail_3d=(float(t_w.x), float(t_w.y), float(t_w.z)),
                radius=r,
            )
            bones_map[b.name] = b_info

    except Exception:
        pass

    return bones_map
