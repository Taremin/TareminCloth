# -*- coding: utf-8 -*-
"""Milestone 4: 2D特化ビューポート制御 (pattern_view.py)."""

import bpy
from mathutils import Quaternion


def _persistent(func):
    """Blender環境では@persistentを適用し、スタンドアロンテスト環境では透過するデコレータ"""
    app = getattr(bpy, "app", None)
    handlers = getattr(app, "handlers", None)
    persistent_fn = getattr(handlers, "persistent", None)
    if (
        callable(persistent_fn)
        and "Mock" not in type(persistent_fn).__name__
        and not hasattr(persistent_fn, "_mock_name")
    ):
        return persistent_fn(func)
    return func




def get_pattern_objects(scene: bpy.types.Scene) -> list[bpy.types.Object]:
    """シーン内の全型紙パーツオブジェクトを取得する."""
    if not scene:
        return []
    return [
        obj for obj in scene.objects
        if hasattr(obj, "taremin_pattern") and obj.taremin_pattern.is_pattern_piece
    ]


def enter_pattern_local_view(
    context: bpy.types.Context,
    *,
    target_objects: list[bpy.types.Object] | None = None,
) -> None:
    """型紙オブジェクトを対象にローカルビュー（Local View）に入り、素体等を非表示化する.

    target_objects が指定された場合はそのオブジェクトのみ、
    None の場合はアクティブな型紙パーツまたは全型紙パーツを対象とする。
    すでにローカルビュー中の場合は一度解除してから新しい対象のみでローカルビューに入り直す。
    """
    space = getattr(context, "space_data", None)
    if not space or space.type != 'VIEW_3D':
        return

    # 対象オブジェクトの決定
    if target_objects is not None:
        objs = [o for o in target_objects if o and getattr(o, "name", None) in context.scene.objects]
    else:
        active_obj = getattr(context, "active_object", None)
        if active_obj and hasattr(active_obj, "taremin_pattern") and active_obj.taremin_pattern.is_pattern_piece:
            selected_patterns = [
                o for o in context.selected_objects
                if hasattr(o, "taremin_pattern") and o.taremin_pattern.is_pattern_piece
            ]
            objs = selected_patterns if selected_patterns else [active_obj]
        else:
            objs = get_pattern_objects(context.scene)

    if not objs:
        return

    # すでにローカルビューに入っている場合は一度解除して入り直す
    if getattr(space, "local_view", None) is not None:
        exit_pattern_local_view(context)

    # 対象オブジェクトのみを選択状態にする
    try:
        for o in context.selected_objects:
            o.select_set(False)
        for p in objs:
            p.select_set(True)
        context.view_layer.objects.active = objs[0]

        # 3Dビューエリアで localview を実行
        if hasattr(context, "area") and context.area and context.area.type == 'VIEW_3D':
            bpy.ops.view3d.localview(frame_selected=True)
        else:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                    override_kwargs = {"area": area}
                    if region:
                        override_kwargs["region"] = region
                    with context.temp_override(**override_kwargs):
                        bpy.ops.view3d.localview(frame_selected=True)
                    break
    except Exception:
        pass


def exit_pattern_local_view(context: bpy.types.Context) -> None:
    """ローカルビューを解除し、素体や他のオブジェクトの通常表示に復帰する."""
    space = getattr(context, "space_data", None)
    if not space or space.type != 'VIEW_3D':
        return

    if getattr(space, "local_view", None) is None:
        return

    try:
        if hasattr(context, "area") and context.area and context.area.type == 'VIEW_3D':
            bpy.ops.view3d.localview()
        else:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                    override_kwargs = {"area": area}
                    if region:
                        override_kwargs["region"] = region
                    with context.temp_override(**override_kwargs):
                        bpy.ops.view3d.localview()
                    break
    except Exception:
        pass


def setup_2d_pattern_viewport(
    context: bpy.types.Context,
    *,
    target_objects: list[bpy.types.Object] | None = None,
) -> None:
    """3Dビューポートを2D型紙製図用（ローカルビュー・Top直交・回転ロック・グリッド最適化）に切り替える."""
    space = getattr(context, "space_data", None)
    if not space or space.type != 'VIEW_3D':
        return

    # 1. 型紙のみのローカルビューへ移行（指定オブジェクト単体または型紙パーツ）
    enter_pattern_local_view(context, target_objects=target_objects)

    # 2. Top直交および視点回転ロック
    rv3d = getattr(space, "region_3d", None)
    if rv3d:
        rv3d.view_perspective = 'ORTHO'
        rv3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
        rv3d.lock_rotation = True

    # 3. 2D製図用オーバーレイ調整
    overlay = getattr(space, "overlay", None)
    if overlay:
        overlay.show_floor = False
        overlay.show_axis_z = False
        overlay.show_cursor = False



def restore_3d_viewport(context: bpy.types.Context) -> None:
    """3Dビューポートの回転ロックとローカルビューを解除し、標準の3D視点に復帰する."""
    space = getattr(context, "space_data", None)
    if not space or space.type != 'VIEW_3D':
        return

    # 1. ローカルビュー解除
    exit_pattern_local_view(context)

    # 2. 回転ロック解除・パースペクティブ復帰
    rv3d = getattr(space, "region_3d", None)
    if rv3d:
        rv3d.lock_rotation = False
        rv3d.view_perspective = 'PERSP'

    # 3. 通常オーバーレイ復帰
    overlay = getattr(space, "overlay", None)
    if overlay:
        overlay.show_floor = True
        overlay.show_axis_x = True
        overlay.show_axis_y = True
        overlay.show_cursor = True


def is_pattern_view_active(context: bpy.types.Context) -> bool:
    """現在アクティブな3Dビューが2D型紙視点（回転ロック・Top直交）になっているか."""
    space = getattr(context, "space_data", None)
    if not space or space.type != 'VIEW_3D':
        return False
    rv3d = getattr(space, "region_3d", None)
    if not rv3d:
        return False
    return bool(rv3d.lock_rotation and rv3d.view_perspective == 'ORTHO')


_pending_restore_timer = False


def _do_restore_3d_viewport_timer():
    """タイマーによる安全な3Dビュー復帰処理"""
    global _pending_restore_timer
    _pending_restore_timer = False

    context = getattr(bpy, "context", None)
    if not context:
        return None

    wm = getattr(context, "window_manager", None)
    if not wm:
        return None

    for window in wm.windows:
        screen = getattr(window, "screen", None)
        if not screen:
            continue
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                rv3d = getattr(space, "region_3d", None)
                if rv3d and rv3d.lock_rotation and rv3d.view_perspective == 'ORTHO':
                    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                    override_kwargs = {"window": window, "screen": screen, "area": area}
                    if region:
                        override_kwargs["region"] = region
                    try:
                        with context.temp_override(**override_kwargs):
                            restore_3d_viewport(context)
                    except Exception:
                        pass
    return None


@_persistent
def on_depsgraph_update_pattern_view(scene, depsgraph):
    """2D製図モード中に型紙以外のオブジェクトがアクティブになった場合、自動的に2Dモードを解除するハンドラー."""
    global _pending_restore_timer
    if _pending_restore_timer:
        return

    context = getattr(bpy, "context", None)
    if not context:
        return

    view_layer = getattr(context, "view_layer", None)
    if not view_layer:
        return

    active_obj = getattr(view_layer.objects, "active", None)
    if not active_obj:
        return

    # 型紙オブジェクトがアクティブな場合は2Dモードを継続
    if hasattr(active_obj, "taremin_pattern") and active_obj.taremin_pattern.is_pattern_piece:
        return

    # 3Dビューポートが現在2Dモードであるか確認
    wm = getattr(context, "window_manager", None)
    if not wm:
        return

    has_active_2d_view = False
    for window in wm.windows:
        screen = getattr(window, "screen", None)
        if not screen:
            continue
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                rv3d = getattr(space, "region_3d", None)
                if rv3d and rv3d.lock_rotation and rv3d.view_perspective == 'ORTHO':
                    has_active_2d_view = True
                    break
        if has_active_2d_view:
            break

    if has_active_2d_view:
        _pending_restore_timer = True
        timers = getattr(getattr(bpy, "app", None), "timers", None)
        if timers and hasattr(timers, "register"):
            timers.register(_do_restore_3d_viewport_timer, first_interval=0.01)
        else:
            _do_restore_3d_viewport_timer()




def register_pattern_view_handlers():
    """ハンドラーの登録"""
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    post_handlers = getattr(handlers, "depsgraph_update_post", None)
    if post_handlers is not None and on_depsgraph_update_pattern_view not in post_handlers:
        post_handlers.append(on_depsgraph_update_pattern_view)


def unregister_pattern_view_handlers():
    """ハンドラーの登録解除"""
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    post_handlers = getattr(handlers, "depsgraph_update_post", None)
    if post_handlers is not None and on_depsgraph_update_pattern_view in post_handlers:
        post_handlers.remove(on_depsgraph_update_pattern_view)


def register():
    """モジュール登録"""
    register_pattern_view_handlers()


def unregister():
    """モジュール登録解除"""
    unregister_pattern_view_handlers()

