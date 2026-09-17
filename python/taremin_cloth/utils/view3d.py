"""
3Dビューポート・アニメーション制御ユーティリティ (taremin_cloth.utils.view3d)
Blenderの画面再描画要求およびアニメーション再生停止を一元的に提供する。
"""

import bpy


def tag_redraw_view3d(context=None, extra_area_types=None):
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
