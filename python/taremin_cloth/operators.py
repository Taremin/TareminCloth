"""
taremin_cloth operators ファサードモジュール
後方互換性のために、engine および ops パッケージの全機能・クラスを再エクスポートします。
"""

import bpy
from .engine.cache import (
    _simulators,
    _rest_positions,
    _prev_coords_cache,
    _mesh_char_len_cache,
    _effective_substeps_cache,
    _collider_prev_locs_cache,
    _prev_elastic_scales,
    _timeline_frame_cache,
    _buffered_start_frames,
    _get_mesh_topology_signature,
    cache_rest_positions,
    restore_rest_positions,
    clear_simulator_for_object,
    clear_simulators,
)
from .engine.collider import (
    _collider_cache,
    clear_collider_cache,
    sync_colliders,
)
from .engine.params import (
    sync_cloth_parameters,
    sync_elastic_groups,
    sync_attachment_pins,
)
from .engine.runner import (
    _fast_playback_saved_mods,
    apply_fast_playback,
    restore_fast_playback,
    get_or_create_simulator,
    get_effective_substeps,
    cloth_frame_handler,
)
from .ops import (
    OPERATOR_CLASSES,
    TAREMIN_CLOTH_OT_toggle_cloth,
    TAREMIN_CLOTH_OT_reset_selected,
    TAREMIN_CLOTH_OT_reset_all,
    TAREMIN_CLOTH_OT_reset_simulation,
    TAREMIN_CLOTH_OT_apply_rest_shape,
    TAREMIN_CLOTH_OT_clear_cache,
    TAREMIN_CLOTH_OT_apply_gpu_settings,
    TAREMIN_CLOTH_OT_interactive,
    TAREMIN_CLOTH_OT_create_seam,
    TAREMIN_CLOTH_OT_add_elastic_group,
    TAREMIN_CLOTH_OT_remove_elastic_group,
    TAREMIN_CLOTH_OT_assign_elastic_edges,
    TAREMIN_CLOTH_OT_select_elastic_edges,
    TAREMIN_CLOTH_OT_apply_cross_subdivision,
    TAREMIN_CLOTH_OT_apply_post_process,
    TAREMIN_CLOTH_OT_apply_dynamic_diagonal,
    TAREMIN_CLOTH_OT_restore_quad_topology,
    TAREMIN_CLOTH_OT_record_pose,
    TAREMIN_CLOTH_OT_apply_pose_preview,
    TAREMIN_CLOTH_OT_select_object,
    TAREMIN_CLOTH_OT_auto_fit_thickness,
    FPSCounter,
    is_interactive_running,
    stop_interactive_if_running,
    resolve_debug_filepath,
)

# 従来互換用のクラス一覧
classes = OPERATOR_CLASSES


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers and hasattr(handlers, "frame_change_post") and handlers.frame_change_post is not None:
        if cloth_frame_handler not in handlers.frame_change_post:
            handlers.frame_change_post.append(cloth_frame_handler)


def unregister():
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers and hasattr(handlers, "frame_change_post") and handlers.frame_change_post is not None:
        if cloth_frame_handler in handlers.frame_change_post:
            handlers.frame_change_post.remove(cloth_frame_handler)
    clear_simulators()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
