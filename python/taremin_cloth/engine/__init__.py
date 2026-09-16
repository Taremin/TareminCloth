"""
taremin_cloth engine パッケージ
シミュレーションの実行、同期、キャッシュ管理などのコア機能を提供する。
"""

from .cache import (
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
from .collider import (
    _collider_cache,
    clear_collider_cache,
    sync_colliders,
)
from .params import (
    sync_cloth_parameters,
    sync_elastic_groups,
    sync_attachment_pins,
)
from .runner import (
    _fast_playback_saved_mods,
    apply_fast_playback,
    restore_fast_playback,
    get_or_create_simulator,
    get_effective_substeps,
    step_cloth_object,
    step_cloth_scene,
    cloth_frame_handler,
)

__all__ = [
    "_simulators",
    "_rest_positions",
    "_prev_coords_cache",
    "_mesh_char_len_cache",
    "_effective_substeps_cache",
    "_collider_prev_locs_cache",
    "_prev_elastic_scales",
    "_timeline_frame_cache",
    "_buffered_start_frames",
    "_get_mesh_topology_signature",
    "cache_rest_positions",
    "restore_rest_positions",
    "clear_simulator_for_object",
    "clear_simulators",
    "_collider_cache",
    "clear_collider_cache",
    "sync_colliders",
    "sync_cloth_parameters",
    "sync_elastic_groups",
    "sync_attachment_pins",
    "_fast_playback_saved_mods",
    "apply_fast_playback",
    "restore_fast_playback",
    "get_or_create_simulator",
    "get_effective_substeps",
    "step_cloth_object",
    "step_cloth_scene",
    "cloth_frame_handler",
]

