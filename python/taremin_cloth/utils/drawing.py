"""
3DビューポートGPUプレビュー描画ユーティリティ
ピン留め頂点や縫合線の視覚的オーバーレイ描画を行う。
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .. import i18n

_draw_handler = None
_draw_handler_2d = None
_interactive_active = False
_active_grabbed_info = None  # {"obj_name": str, "vert_idx": int, "target_world_pos": Vector or None}
_interactive_fps_info = None  # {"fps": float, "frame_ms": float, "show_overlay": bool, "position": str, "show_help": bool, "tool_text": str or None}
_brush_info = None  # {"x": float, "y": float, "radius_px": float}
_pin_indices_cache = {}  # {(obj_name, vg_name): (indices_list, vert_count)}
_sewing_edge_indices_cache = {}  # {obj_name: (edge_pairs_list, vert_count, edge_count)}
_bake_overlay_info = None  # {"current_frame": int, "end_frame": int, "pct": int}

_cached_point_shader = None
_cached_line_shader = None
_cached_3d_uniform_shader = None
_cached_2d_shader = None
_cached_smooth_line_shader = None


def clear_pin_cache():
    """ピン留め頂点インデックスのキャッシュをクリアする"""
    global _pin_indices_cache
    _pin_indices_cache.clear()


def clear_sewing_cache():
    """縫合エッジインデックスのキャッシュをクリアする"""
    global _sewing_edge_indices_cache
    _sewing_edge_indices_cache.clear()


def clear_overlay_caches():
    """すべてのオーバーレイ用キャッシュをクリアする"""
    clear_pin_cache()
    clear_sewing_cache()


def set_interactive_active(active: bool):
    """インタラクティブモードの実行状態を設定する"""
    global _interactive_active
    _interactive_active = active
    if not active:
        clear_overlay_caches()
        clear_brush_info()


def is_interactive_active() -> bool:
    """インタラクティブモードが実行中かどうかを取得する"""
    return _interactive_active


def _setup_font(font_size: int, font_id: int = 0):
    """blf.size のバージョン差を吸収してフォントサイズを設定する（docs/hud.md:4）"""
    import blf
    try:
        blf.size(font_id, font_size)
    except (TypeError, ValueError):
        try:
            blf.size(font_id, font_size, 72)
        except Exception:
            pass


def _measure_text(text: str, font_id: int = 0) -> float:
    """blf.dimensions の例外を吸収してテキスト幅を返す。失敗時は0.0。"""
    import blf
    try:
        dims = blf.dimensions(font_id, text)
        if dims and dims[0] > 0:
            return float(dims[0])
    except Exception:
        pass
    return 0.0


def set_interactive_fps_info(
    fps: float,
    frame_ms: float,
    show_overlay: bool = True,
    position: str = 'TOP_CENTER',
    show_help: bool = True,
    tool_text=None,
    guide_line1_suffix=None,
    guide_line2=None,
    sewing_text=None,
    sewing_active: bool = False,
):
    """インタラクティブモード中のFPSおよび操作ヘルプ計測情報を更新する"""
    global _interactive_fps_info
    _interactive_fps_info = {
        "fps": float(fps),
        "frame_ms": float(frame_ms),
        "show_overlay": bool(show_overlay),
        "position": str(position),
        "show_help": bool(show_help),
        "tool_text": str(tool_text) if tool_text else None,
        "guide_line1_suffix": str(guide_line1_suffix) if guide_line1_suffix else None,
        "guide_line2": str(guide_line2) if guide_line2 else None,
        "sewing_text": str(sewing_text) if sewing_text else None,
        "sewing_active": bool(sewing_active),
    }


def clear_interactive_fps_info():
    """インタラクティブモード中のFPS計測情報をクリアする"""
    global _interactive_fps_info
    _interactive_fps_info = None


def get_interactive_fps_info():
    """現在のFPS計測情報を取得する（テストまたはUI用）"""
    return _interactive_fps_info


def set_bake_overlay_info(current_frame: int, end_frame: int, pct: int, sewing_text=None):
    """ベイク中のオーバーレイ描画情報を更新する"""
    global _bake_overlay_info
    _bake_overlay_info = {
        "current_frame": int(current_frame),
        "end_frame": int(end_frame),
        "pct": int(pct),
        "sewing_text": str(sewing_text) if sewing_text else None,
    }


def clear_bake_overlay_info():
    """ベイク中のオーバーレイ描画情報をクリアする"""
    global _bake_overlay_info
    _bake_overlay_info = None


def get_bake_overlay_info():
    """現在のベイク中オーバーレイ描画情報を取得する（テストまたはUI用）"""
    return _bake_overlay_info


def is_bake_overlay_active() -> bool:
    """ベイク中のオーバーレイ描画が有効かどうかを取得する"""
    return _bake_overlay_info is not None


_init_overlay_info = None  # {"message": str, "current": int, "total": int}


def set_init_overlay_info(message: str, current: int, total: int):
    """インタラクティブ起動準備中のオーバーレイ描画情報を更新する"""
    global _init_overlay_info
    _init_overlay_info = {
        "message": str(message),
        "current": int(current),
        "total": int(total),
    }


def clear_init_overlay_info():
    """起動準備中のオーバーレイ描画情報をクリアする"""
    global _init_overlay_info
    _init_overlay_info = None


def get_init_overlay_info():
    """現在の起動準備中オーバーレイ描画情報を取得する（テストまたはUI用）"""
    return _init_overlay_info


def is_init_overlay_active() -> bool:
    """起動準備中のオーバーレイ描画が有効かどうかを取得する"""
    return _init_overlay_info is not None


def set_active_grabbed_vertex(obj_name: str, vert_idx: int, target_world_pos=None):
    """現在ドラッグ中の頂点情報を設定する"""
    global _active_grabbed_info
    _active_grabbed_info = {
        "obj_name": obj_name,
        "vert_idx": vert_idx,
        "target_world_pos": target_world_pos,
    }


def clear_active_grabbed_vertex():
    """現在ドラッグ中の頂点情報をクリアする"""
    global _active_grabbed_info
    _active_grabbed_info = None


def set_brush_info(x: float, y: float, radius_px: float):
    """範囲グラブブラシカーソル円の表示情報を更新する"""
    global _brush_info
    _brush_info = {
        "x": float(x),
        "y": float(y),
        "radius_px": max(float(radius_px), 2.0),
    }


def clear_brush_info():
    """範囲グラブブラシカーソル円の表示情報をクリアする"""
    global _brush_info
    _brush_info = None


def get_brush_info():
    """現在のブラシカーソル円情報を取得する（テストまたはUI用）"""
    return _brush_info


def apply_view_depth_bias(coords, region_3d=None):
    """
    深度テスト有効時のZ-fighting（埋没・本来の辺や面と色が混ざって薄くなる現象）を解消するため、
    描画座標をカメラ（視点）手前方向にわずかにオフセットする。
    物陰（裏面や他オブジェクト）にある頂点・エッジは引き続き確実に遮蔽される。
    """
    if coords is None or len(coords) == 0 or region_3d is None or not hasattr(region_3d, "view_matrix"):
        return coords

    try:
        coords_arr = np.asarray(coords, dtype=np.float32)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 3 or len(coords_arr) == 0:
            return coords

        inv = region_3d.view_matrix.inverted()
        is_persp = getattr(region_3d, "is_perspective", True)

        if is_persp:
            cam_p = np.array([inv.translation.x, inv.translation.y, inv.translation.z], dtype=np.float32)
            diff = cam_p - coords_arr
            dists = np.linalg.norm(diff, axis=1, keepdims=True)
            dists_safe = np.maximum(dists, 1e-6)
            dirs = diff / dists_safe
            # 距離に応じた微小バイアス (距離の0.2%、下限0.8mm、上限2cm)
            biases = np.clip(dists * 0.002, 0.0008, 0.02)
            offset_coords = coords_arr + dirs * biases
        else:
            forward = -inv.col[2].to_3d().normalized()
            bias_dir = np.array([-forward.x, -forward.y, -forward.z], dtype=np.float32)
            offset_coords = coords_arr + bias_dir * 0.002

        return offset_coords
    except Exception:
        return coords


def get_3d_uniform_color_shader():
    global _cached_3d_uniform_shader
    if _cached_3d_uniform_shader is not None:
        return _cached_3d_uniform_shader
    for name in ('UNIFORM_COLOR', '3D_UNIFORM_COLOR'):
        try:
            _cached_3d_uniform_shader = gpu.shader.from_builtin(name)
            return _cached_3d_uniform_shader
        except Exception:
            pass
    return None


def get_point_shader():
    global _cached_point_shader
    if _cached_point_shader is not None:
        return _cached_point_shader
    for name in ('POINT_UNIFORM_COLOR', 'UNIFORM_COLOR', '3D_UNIFORM_COLOR'):
        try:
            _cached_point_shader = gpu.shader.from_builtin(name)
            return _cached_point_shader
        except Exception:
            pass
    return None


def get_line_shader():
    global _cached_line_shader
    if _cached_line_shader is not None:
        return _cached_line_shader
    for name in ('UNIFORM_COLOR', '3D_UNIFORM_COLOR', 'POLYLINE_UNIFORM_COLOR'):
        try:
            _cached_line_shader = gpu.shader.from_builtin(name)
            return _cached_line_shader
        except Exception:
            pass
    return None


def get_2d_uniform_color_shader():
    global _cached_2d_shader
    if _cached_2d_shader is not None:
        return _cached_2d_shader
    for name in ('2D_UNIFORM_COLOR', 'UNIFORM_COLOR'):
        try:
            _cached_2d_shader = gpu.shader.from_builtin(name)
            return _cached_2d_shader
        except Exception:
            pass
    return None


def get_smooth_color_line_shader():
    global _cached_smooth_line_shader
    if _cached_smooth_line_shader is not None:
        return _cached_smooth_line_shader
    for name in ('POLYLINE_SMOOTH_COLOR', '3D_SMOOTH_COLOR'):
        try:
            _cached_smooth_line_shader = gpu.shader.from_builtin(name)
            return _cached_smooth_line_shader
        except Exception:
            pass
    return None


def get_edge_initial_length(obj, v0_idx, v1_idx):
    """オブジェクトの初期座標からエッジの初期自然長を算出する"""
    if "_taremin_cloth_rest_positions" in obj:
        raw = np.frombuffer(obj["_taremin_cloth_rest_positions"], dtype=np.float32)
        if len(raw) >= max(v0_idx, v1_idx) * 3 + 3:
            p0 = raw[v0_idx * 3 : v0_idx * 3 + 3]
            p1 = raw[v1_idx * 3 : v1_idx * 3 + 3]
            return float(np.linalg.norm(p1 - p0))
    v0 = obj.data.vertices[v0_idx].co
    v1 = obj.data.vertices[v1_idx].co
    return (v1 - v0).length


def draw_callback_3d():
    """3Dビューポートのオーバーレイ描画コールバック (インタラクティブシミュレーション実行中のみ)"""
    if not _interactive_active:
        return

    context = bpy.context
    if not context or not context.scene:
        return

    region_3d = getattr(context, "region_data", None)
    if region_3d is None and hasattr(context, "space_data") and hasattr(context.space_data, "region_3d"):
        region_3d = context.space_data.region_3d

    point_shader = get_point_shader()
    line_shader = get_line_shader()
    smooth_line_shader = get_smooth_color_line_shader()

    for obj in context.scene.objects:
        if obj.type != 'MESH' or not hasattr(obj, "taremin_cloth"):
            continue

        settings = obj.taremin_cloth
        if not settings.is_cloth:
            continue

        # 非表示オブジェクト（hide_viewport、コレクション非表示、ローカルビュー非表示等）はスキップ
        try:
            if not obj.visible_get():
                continue
        except Exception:
            if getattr(obj, "hide_viewport", False) or (hasattr(obj, "hide_get") and obj.hide_get()):
                continue

        # インタラクティブシミュレーション実行中は、シミュレータに登録されているアクティブな布オブジェクトのみ描画
        if _interactive_active:
            from ..engine.cache import _simulators
            if obj.name not in _simulators:
                continue

        mesh = obj.data
        world_mat = obj.matrix_world

        # 深度テスト（物陰オクルージョン判定）の設定
        use_depth_test = getattr(settings, "overlay_depth_test", False)
        orig_depth_test = gpu.state.depth_test_get()
        orig_depth_mask = gpu.state.depth_mask_get()

        try:
            if use_depth_test:
                gpu.state.depth_test_set('LESS_EQUAL')
                gpu.state.depth_mask_set(False)
            else:
                gpu.state.depth_test_set('NONE')

            # 1. ピン留め頂点の描画（設定色、Interactive Only対応）
            show_pin = True
            if getattr(settings, "pin_overlay_interactive_only", True) and not _interactive_active:
                show_pin = False

            vg_name = settings.pin_vertex_group or "Pin"
            vg = obj.vertex_groups.get(vg_name)
            if show_pin and vg and point_shader:
                cache_key = (obj.name, vg_name)
                vert_count = len(mesh.vertices)
                cached = _pin_indices_cache.get(cache_key)
                if cached is not None and cached[1] == vert_count:
                    pin_indices = cached[0]
                else:
                    vg_idx = vg.index
                    pin_indices = [
                        v.index
                        for v in mesh.vertices
                        for g in v.groups
                        if g.group == vg_idx and g.weight > 0.0
                    ]
                    _pin_indices_cache[cache_key] = (pin_indices, vert_count)

                if pin_indices:
                    if len(pin_indices) > 32:
                        coords = np.empty((vert_count, 3), dtype=np.float32)
                        mesh.vertices.foreach_get("co", coords.ravel())
                        coords_sub = coords[pin_indices]
                        mat_np = np.asarray(world_mat, dtype=np.float32)
                        pin_coords = coords_sub @ mat_np[:3, :3].T + mat_np[:3, 3]
                    else:
                        verts = mesh.vertices
                        pin_coords = [[*(world_mat @ verts[i].co)] for i in pin_indices]
                    draw_pin_coords = apply_view_depth_bias(pin_coords, region_3d) if use_depth_test else pin_coords
                    gpu.state.point_size_set(8.0)
                    gpu.state.blend_set('ALPHA')
                    batch = batch_for_shader(point_shader, 'POINTS', {"pos": draw_pin_coords})
                    point_shader.bind()
                    pin_col = tuple(getattr(settings, "pin_color", (1.0, 0.45, 0.0, 0.9)))
                    point_shader.uniform_float("color", pin_col)
                    batch.draw(point_shader)
                    gpu.state.blend_set('NONE')

            # 2. 現在掴んでいる頂点のハイライト描画（大サイズ・ライムグリーン色・コア発光・ポインター線）
            if _interactive_active and _active_grabbed_info and _active_grabbed_info.get("obj_name") == obj.name:
                v_idx = _active_grabbed_info.get("vert_idx")
                if v_idx is not None and 0 <= v_idx < len(mesh.vertices) and point_shader:
                    w_pos = world_mat @ mesh.vertices[v_idx].co
                    target_pos = _active_grabbed_info.get("target_world_pos")
                    grab_coords = [[w_pos.x, w_pos.y, w_pos.z]]
                    draw_grab_coords = apply_view_depth_bias(grab_coords, region_3d) if use_depth_test else grab_coords

                    # 外側リング（14px, 鮮やかなライムグリーン）
                    gpu.state.point_size_set(14.0)
                    gpu.state.blend_set('ALPHA')
                    batch = batch_for_shader(point_shader, 'POINTS', {"pos": draw_grab_coords})
                    point_shader.bind()
                    point_shader.uniform_float("color", (0.2, 1.0, 0.35, 0.95))
                    batch.draw(point_shader)

                    # 内側コア（7px, ホワイト）
                    gpu.state.point_size_set(7.0)
                    point_shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
                    batch.draw(point_shader)
                    gpu.state.blend_set('NONE')

                    # 目標位置との間にポインター線を描画
                    if target_pos is not None and line_shader:
                        dist = (target_pos - w_pos).length
                        if dist > 0.002:
                            line_pts = [
                                [w_pos.x, w_pos.y, w_pos.z],
                                [target_pos.x, target_pos.y, target_pos.z]
                            ]
                            draw_line_pts = apply_view_depth_bias(line_pts, region_3d) if use_depth_test else line_pts
                            gpu.state.line_width_set(2.0)
                            gpu.state.blend_set('ALPHA')
                            line_batch = batch_for_shader(line_shader, 'LINES', {"pos": draw_line_pts})
                            line_shader.bind()
                            line_shader.uniform_float("color", (0.3, 1.0, 0.4, 0.8))
                            line_batch.draw(line_shader)
                            gpu.state.blend_set('NONE')

            # 3. 縫合エッジ（Loose Edge）の描画（水色ライン）
            if settings.enable_sewing and line_shader:
                cache_key = obj.name
                vert_count = len(mesh.vertices)
                edge_count = len(mesh.edges)
                cached = _sewing_edge_indices_cache.get(cache_key)

                if cached is not None and cached[1] == vert_count and cached[2] == edge_count:
                    loose_edge_pairs = cached[0]
                else:
                    mesh.calc_loop_triangles()
                    face_edge_set = set()
                    for tri in mesh.loop_triangles:
                        face_edge_set.add((min(tri.vertices[0], tri.vertices[1]), max(tri.vertices[0], tri.vertices[1])))
                        face_edge_set.add((min(tri.vertices[1], tri.vertices[2]), max(tri.vertices[1], tri.vertices[2])))
                        face_edge_set.add((min(tri.vertices[2], tri.vertices[0]), max(tri.vertices[2], tri.vertices[0])))

                    loose_edge_pairs = []
                    for e in mesh.edges:
                        pair = (min(e.vertices[0], e.vertices[1]), max(e.vertices[0], e.vertices[1]))
                        if pair not in face_edge_set:
                            loose_edge_pairs.append((e.vertices[0], e.vertices[1]))
                    _sewing_edge_indices_cache[cache_key] = (loose_edge_pairs, vert_count, edge_count)

                if loose_edge_pairs:
                    sew_lines = []
                    # シミュレータから最新の現在座標 (coords) を直接取得してリアルタイム追従させる
                    curr_coords_2d = None
                    try:
                        from ..engine.cache import _simulators
                        sim_entry = _simulators.get(obj.name)
                        if sim_entry is not None and sim_entry[1] is not None:
                            curr_coords_2d = sim_entry[1].reshape(-1, 3)
                    except Exception:
                        pass

                    if curr_coords_2d is not None and len(curr_coords_2d) >= vert_count:
                        # シミュレータの現在座標はすでにワールド座標系
                        coords_world = curr_coords_2d
                        for v0_idx, v1_idx in loose_edge_pairs:
                            sew_lines.append(coords_world[v0_idx].tolist())
                            sew_lines.append(coords_world[v1_idx].tolist())
                    else:
                        # フォールバック: メッシュ座標から抽出
                        verts = mesh.vertices
                        for v0_idx, v1_idx in loose_edge_pairs:
                            p0 = world_mat @ verts[v0_idx].co
                            p1 = world_mat @ verts[v1_idx].co
                            sew_lines.append([p0.x, p0.y, p0.z])
                            sew_lines.append([p1.x, p1.y, p1.z])

                    if sew_lines:
                        draw_sew_lines = apply_view_depth_bias(sew_lines, region_3d) if use_depth_test else sew_lines
                        gpu.state.line_width_set(2.0)
                        gpu.state.blend_set('ALPHA')
                        batch = batch_for_shader(line_shader, 'LINES', {"pos": draw_sew_lines})
                        line_shader.bind()
                        line_shader.uniform_float("color", (0.0, 0.8, 1.0, 0.85))
                        batch.draw(line_shader)
                        gpu.state.blend_set('NONE')

            # 4. 伸縮エッジ（Elastic Bands）の描画（収縮=青、伸長=赤、目標収束=フェード）
            show_elastic = getattr(settings, "show_elastic_overlay", True)
            if getattr(settings, "elastic_overlay_interactive_only", True) and not _interactive_active:
                show_elastic = False

            if show_elastic and settings.elastic_groups:
                elastic_lines = []
                elastic_colors = []

                for grp in settings.elastic_groups:
                    if not grp.enabled:
                        continue
                    edge_indices = grp.get_edge_indices()
                    if not edge_indices:
                        continue

                    scale = grp.scale
                    # 基本色: 収縮(scale < 1.0)なら青系、伸長(scale > 1.0)なら赤・オレンジ系
                    if scale < 0.999:
                        base_rgb = (0.1, 0.65, 1.0) # シアンブルー
                    elif scale > 1.001:
                        base_rgb = (1.0, 0.35, 0.1) # オレンジレッド
                    else:
                        base_rgb = (0.6, 0.6, 0.6) # 変化なし（グレー）

                    for e_idx in edge_indices:
                        if e_idx >= len(mesh.edges):
                            continue
                        edge = mesh.edges[e_idx]
                        v0_idx = edge.vertices[0]
                        v1_idx = edge.vertices[1]

                        p0_w = world_mat @ mesh.vertices[v0_idx].co
                        p1_w = world_mat @ mesh.vertices[v1_idx].co
                        curr_len = (p1_w - p0_w).length

                        l0 = get_edge_initial_length(obj, v0_idx, v1_idx)
                        if l0 <= 1e-6:
                            continue

                        target_len = l0 * scale

                        # 目標収束度合い（テンション誤差の評価）
                        err = abs(curr_len - target_len) / (l0 * 0.25 + 1e-5)
                        fade = min(1.0, max(0.12, err)) # 最小アルファ 0.12 (位置がわかるように薄く残す)

                        alpha = fade * 0.9

                        c = (base_rgb[0], base_rgb[1], base_rgb[2], alpha)
                        elastic_lines.append([p0_w.x, p0_w.y, p0_w.z])
                        elastic_lines.append([p1_w.x, p1_w.y, p1_w.z])
                        elastic_colors.append(c)
                        elastic_colors.append(c)

                if elastic_lines:
                    draw_elastic_lines = apply_view_depth_bias(elastic_lines, region_3d) if use_depth_test else elastic_lines
                    gpu.state.line_width_set(3.0)
                    gpu.state.blend_set('ALPHA')
                    if smooth_line_shader:
                        batch = batch_for_shader(smooth_line_shader, 'LINES', {"pos": draw_elastic_lines, "color": elastic_colors})
                        smooth_line_shader.bind()
                        batch.draw(smooth_line_shader)
                    elif line_shader:
                        batch = batch_for_shader(line_shader, 'LINES', {"pos": draw_elastic_lines})
                        line_shader.bind()
                        line_shader.uniform_float("color", (0.2, 0.7, 1.0, 0.8))
                        batch.draw(line_shader)
                    gpu.state.blend_set('NONE')

            # --- 3. ドレープガイドの影響範囲の半透明描画 ---
            # use_wrinkle_field が無効・実験フラグOFFのときはプレビューも描画しない。
            try:
                try:
                    from ..preferences import is_wrinkle_field_enabled as _is_wf_enabled
                    _exp_enabled = bool(_is_wf_enabled())
                except Exception:
                    _exp_enabled = True
                preview_mode = getattr(settings, "wrinkle_preview_mode", "BOTH") if settings else "BOTH"
                if _exp_enabled and preview_mode in ("RANGE", "BOTH") and getattr(settings, "use_wrinkle_field", False):
                    # 実験フラグONかつ use_wrinkle_field 有効時のみ描画
                    wrinkle_curves_data = []
                    processed_objs = set()

                    target_col = None
                    if settings.wrinkle_collection:
                        target_col = settings.wrinkle_collection
                    elif bpy.data.collections.get("TareminCloth_Wrinkles"):
                        target_col = bpy.data.collections.get("TareminCloth_Wrinkles")

                    def _collect_curve_obj(c_obj):
                        tw = getattr(c_obj, "taremin_wrinkle", None)
                        w_type = tw.curve_type if tw else c_obj.get("wrinkle_type", "crest")
                        inf_r = tw.influence_radius if tw else float(c_obj.get("wrinkle_influence_radius", 0.03))
                        tgt_r = 0.0
                        if tw:
                            try:
                                tgt_r = float(getattr(tw, "target_radius", 0.0) or 0.0)
                            except Exception:
                                tgt_r = 0.0
                        else:
                            try:
                                tgt_r = float(c_obj.get("wrinkle_target_radius", 0.0) or 0.0)
                            except Exception:
                                tgt_r = 0.0
                        try:
                            curve_bone = str(c_obj.get("wrinkle_target_bone", "") or "")
                        except Exception:
                            curve_bone = ""
                        w_mat = c_obj.matrix_world
                        for sp in c_obj.data.splines:
                            pts = [(w_mat @ (p.co.xyz if hasattr(p.co, "xyz") else mathutils.Vector(p.co[:3]))) for p in sp.points]
                            if len(pts) >= 2:
                                wrinkle_curves_data.append((w_type, [list(p) for p in pts], inf_r, tgt_r, curve_bone))

                    if target_col:
                        for c_obj in target_col.objects:
                            if c_obj.type != 'CURVE' or c_obj.name in processed_objs:
                                continue
                            try:
                                if not c_obj.visible_get():
                                    continue
                            except Exception:
                                if getattr(c_obj, "hide_viewport", False) or (hasattr(c_obj, "hide_get") and c_obj.hide_get()):
                                    continue
                            processed_objs.add(c_obj.name)
                            _collect_curve_obj(c_obj)

                    active_c = context.active_object
                    if active_c and active_c.type == 'CURVE' and active_c.name not in processed_objs:
                        is_c_vis = True
                        try:
                            is_c_vis = active_c.visible_get()
                        except Exception:
                            is_c_vis = not (getattr(active_c, "hide_viewport", False) or (hasattr(active_c, "hide_get") and active_c.hide_get()))
                        if is_c_vis and (hasattr(active_c, "taremin_wrinkle") or "wrinkle_type" in active_c):
                            processed_objs.add(active_c.name)
                            _collect_curve_obj(active_c)

                    if wrinkle_curves_data:
                        # 各カーブの所属ボーン座標系で描画する。設定ボーン単一座標系では
                        # 別ボーン配置カーブが垂直化して見えるため、ボーン毎に分類する。
                        # (params.sync_wrinkle_field のベイク座標系と同一手順で解決)
                        arm_obj = getattr(settings, "wrinkle_armature", None) if settings else None
                        if arm_obj is None:
                            for o in bpy.data.objects:
                                if o.type == 'ARMATURE':
                                    arm_obj = o
                                    break
                        settings_bone = getattr(settings, "wrinkle_bone_name", "") if settings else ""
                        if arm_obj is not None and getattr(arm_obj, "pose", None):
                            from ..engine.wrinkle_field import build_orthonormal_basis, resolve_bone_radius
                            from collections import defaultdict
                            pbs = arm_obj.pose.bones
                            if settings_bone not in pbs:
                                settings_bone = pbs[0].name if len(pbs) else ""
                            groups = defaultdict(list)
                            for item in wrinkle_curves_data:
                                b_name = item[4] if len(item) > 4 and item[4] in pbs else settings_bone
                                groups[b_name].append(item[:4])
                            explicit_r = float(getattr(settings, "wrinkle_bone_radius", 0.0) or 0.0) if settings else 0.0
                            try:
                                v_depth = float(getattr(settings, "wrinkle_valley_depth", 0.015) or 0.015)
                            except Exception:
                                v_depth = 0.015
                            try:
                                c_depth = float(getattr(settings, "wrinkle_crest_depth", 0.020) or 0.020)
                            except Exception:
                                c_depth = 0.020
                            v_depth = max(float(v_depth), 0.001)
                            c_depth = max(float(c_depth), 0.001)
                            for b_name, items in groups.items():
                                if not b_name:
                                    continue
                                pb = pbs[b_name]
                                w_mat = arm_obj.matrix_world
                                b_head = np.array((w_mat @ pb.head)[:3], dtype=np.float64)
                                b_tail = np.array((w_mat @ pb.tail)[:3], dtype=np.float64)
                                axis_raw = b_tail - b_head
                                if float(np.linalg.norm(axis_raw)) < 1e-6:
                                    continue
                                axis_n, n_vec, _bn = build_orthonormal_basis(axis_raw)
                                bone_rad = resolve_bone_radius(
                                    explicit_r,
                                    float(getattr(pb.bone, "head_radius", 0.0) or 0.0),
                                    float(getattr(pb.bone, "tail_radius", 0.0) or 0.0),
                                    float(getattr(pb, "length", 0.0) or 0.0),
                                )
                                meshes = build_wrinkle_influence_meshes(
                                    items, b_head, axis_n, n_vec, bone_rad,
                                    valley_depth=v_depth, crest_depth=c_depth,
                                )
                                if meshes:
                                    draw_wrinkle_influence_meshes(meshes)
                            if preview_mode == "BOTH":
                                for proxy_bone in groups.keys():
                                    if not proxy_bone or proxy_bone not in pbs:
                                        continue
                                    pb = pbs[proxy_bone]
                                    w_mat = arm_obj.matrix_world
                                    b_head = np.array((w_mat @ pb.head)[:3], dtype=np.float64)
                                    b_tail = np.array((w_mat @ pb.tail)[:3], dtype=np.float64)
                                    axis_raw = b_tail - b_head
                                    if float(np.linalg.norm(axis_raw)) >= 1e-6:
                                        axis_n, n_vec, _bn = build_orthonormal_basis(axis_raw)
                                        bone_rad = resolve_bone_radius(
                                            explicit_r,
                                            float(getattr(pb.bone, "head_radius", 0.0) or 0.0),
                                            float(getattr(pb.bone, "tail_radius", 0.0) or 0.0),
                                            float(getattr(pb, "length", 0.0) or 0.0),
                                        )
                                        _draw_bone_proxy_cylinder(
                                            b_head, axis_n, n_vec, bone_rad,
                                            float(np.linalg.norm(axis_raw)),
                                        )
            except Exception:
                pass

        finally:
            gpu.state.depth_test_set(orig_depth_test)
            gpu.state.depth_mask_set(orig_depth_mask)


def _draw_bottom_center_progress(region, shader_2d, text: str, pct: int):
    """画面下部中央に進捗バー付きバッジを描画する（ベイク/起動準備の共通ヘルパー）"""
    pct = max(0, min(100, int(pct)))
    guide_h = 32.0
    pad_x = 18.0
    text_w = 400.0

    import blf
    font_id = 0
    font_size = 12

    try:
        blf.size(font_id, font_size)
    except (TypeError, ValueError):
        try:
            blf.size(font_id, font_size, 72)
        except Exception:
            pass

    try:
        dims = blf.dimensions(font_id, text)
        if dims and dims[0] > 0:
            text_w = dims[0]
    except Exception:
        pass

    guide_w = text_w + pad_x * 2.0
    guide_margin_x = max(10.0, (region.width - guide_w) / 2.0)
    guide_y_bottom = 24.0
    guide_y_top = guide_y_bottom + guide_h

    if shader_2d:
        orig_blend = gpu.state.blend_get()
        try:
            gpu.state.blend_set('ALPHA')
            # 1. メイン半透明背景ボックス
            vertices = [
                (guide_margin_x, guide_y_bottom),
                (guide_margin_x + guide_w, guide_y_bottom),
                (guide_margin_x + guide_w, guide_y_top),
                (guide_margin_x, guide_y_bottom),
                (guide_margin_x + guide_w, guide_y_top),
                (guide_margin_x, guide_y_top),
            ]
            batch = batch_for_shader(shader_2d, 'TRIS', {"pos": vertices})
            if batch:
                shader_2d.bind()
                shader_2d.uniform_float("color", (0.10, 0.10, 0.13, 0.85))
                batch.draw(shader_2d)

            # 2. 進捗プログレスバー（底辺3px）
            bar_h = 3.0
            bar_w = guide_w * (pct / 100.0)
            if bar_w > 0.0:
                bar_verts = [
                    (guide_margin_x, guide_y_bottom),
                    (guide_margin_x + bar_w, guide_y_bottom),
                    (guide_margin_x + bar_w, guide_y_bottom + bar_h),
                    (guide_margin_x, guide_y_bottom),
                    (guide_margin_x + bar_w, guide_y_bottom + bar_h),
                    (guide_margin_x, guide_y_bottom + bar_h),
                ]
                bar_batch = batch_for_shader(shader_2d, 'TRIS', {"pos": bar_verts})
                if bar_batch:
                    shader_2d.bind()
                    shader_2d.uniform_float("color", (0.2, 0.7, 1.0, 0.9))
                    bar_batch.draw(shader_2d)
        finally:
            gpu.state.blend_set(orig_blend)

    try:
        text_x = guide_margin_x + pad_x
        text_y = guide_y_bottom + 10.0
        blf.position(font_id, text_x, text_y, 0.0)
        blf.color(font_id, 0.95, 0.95, 0.98, 1.0)
        blf.draw(font_id, text)
    except Exception:
        pass


def _draw_top_badge(region, shader_2d, margin_x: float, y_top: float, text: str,
                    text_color=(0.95, 0.95, 0.98, 1.0), font_size: int = 13, min_w: float = 200.0):
    """画面上部バッジ（背景＋1行テキスト）を描画する。テキスト幅に合わせて自動拡幅する。"""
    import blf
    font_id = 0
    box_h = 28.0
    pad_x = 12.0

    try:
        blf.size(font_id, font_size)
    except (TypeError, ValueError):
        try:
            blf.size(font_id, font_size, 72)
        except Exception:
            pass

    text_w = min_w - pad_x * 2.0
    try:
        dims = blf.dimensions(font_id, text)
        if dims and dims[0] > 0:
            text_w = dims[0]
    except Exception:
        pass

    box_w = max(min_w, text_w + pad_x * 2.0)
    margin_x = max(10.0, min(margin_x, max(10.0, region.width - box_w - 10.0)))
    y_bottom = y_top - box_h

    if shader_2d:
        orig_blend = gpu.state.blend_get()
        try:
            gpu.state.blend_set('ALPHA')
            vertices = [
                (margin_x, y_bottom),
                (margin_x + box_w, y_bottom),
                (margin_x + box_w, y_top),
                (margin_x, y_bottom),
                (margin_x + box_w, y_top),
                (margin_x, y_top),
            ]
            batch = batch_for_shader(shader_2d, 'TRIS', {"pos": vertices})
            if batch:
                shader_2d.bind()
                shader_2d.uniform_float("color", (0.12, 0.12, 0.14, 0.75))
                batch.draw(shader_2d)
        finally:
            gpu.state.blend_set(orig_blend)

    try:
        blf.position(font_id, margin_x + pad_x, y_bottom + 8.0, 0.0)
        blf.color(font_id, text_color[0], text_color[1], text_color[2], text_color[3])
        blf.draw(font_id, text)
    except Exception:
        pass


def _draw_bottom_badge_2row(region, shader_2d, line1: str, line2=None,
                            font_size: int = 12):
    """画面下部中央のガイダンスバッジ（最大2行）を描画する（docs/hud.md:4）。

    line1=操作キー（下段・常に表示）、line2=ツール状態（上段・ある時のみ）。
    単行時は従来と同一見た目（box_h=28、y_bottom=20）を維持する。
    """
    import blf
    font_id = 0
    pad_x = 14.0
    guide_y_bottom = 20.0
    _setup_font(font_size, font_id)
    w1 = _measure_text(line1, font_id) or 340.0
    if line2:
        w2 = _measure_text(line2, font_id)
        text_w = max(w1, w2)
        box_h = 50.0
    else:
        box_h = 28.0
        text_w = w1
    guide_w = text_w + pad_x * 2.0
    guide_margin_x = max(10.0, (region.width - guide_w) / 2.0)
    guide_y_top = guide_y_bottom + box_h

    if shader_2d:
        orig_blend = gpu.state.blend_get()
        try:
            gpu.state.blend_set('ALPHA')
            vertices = [
                (guide_margin_x, guide_y_bottom),
                (guide_margin_x + guide_w, guide_y_bottom),
                (guide_margin_x + guide_w, guide_y_top),
                (guide_margin_x, guide_y_bottom),
                (guide_margin_x + guide_w, guide_y_top),
                (guide_margin_x, guide_y_top),
            ]
            batch = batch_for_shader(shader_2d, 'TRIS', {"pos": vertices})
            if batch:
                shader_2d.bind()
                shader_2d.uniform_float("color", (0.12, 0.12, 0.14, 0.75))
                batch.draw(shader_2d)
        finally:
            gpu.state.blend_set(orig_blend)

    try:
        blf.position(font_id, guide_margin_x + pad_x, guide_y_bottom + 8.0, 0.0)
        blf.color(font_id, 0.92, 0.92, 0.95, 0.95)
        blf.draw(font_id, line1)
        if line2:
            blf.position(font_id, guide_margin_x + pad_x, guide_y_bottom + 30.0, 0.0)
            blf.color(font_id, 0.35, 0.85, 1.0, 0.95)
            blf.draw(font_id, line2)
    except Exception:
        pass


def draw_wrinkle_slide_hud(region, line1: str, line2: str = ""):
    """シワカーブのモーダルスライド操作用HUDバッジを描画する（docs/hud.md準拠）"""
    shader_2d = get_2d_uniform_color_shader()
    _draw_bottom_badge_2row(region, shader_2d, line1, line2 if line2 else None)


def compute_octahedron_mesh(head, tail, radius=None):
    """
    ボーンのhead, tailから八面体（Octahedral）の三角形頂点リスト（TRIS: 24頂点）と
    ワイヤーフレーム線分頂点リスト（LINES: 24頂点）を算出する。
    太さはボーン長によらず一定（均一）のプロポーションを保つ。
    """
    h = np.asarray(head, dtype=np.float32)
    t = np.asarray(tail, dtype=np.float32)
    v = t - h
    length = float(np.linalg.norm(v))
    if length < 1e-4:
        return [], []

    a = v / length
    # 軸 a に直交する基底ベクトルを安定して計算
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u1 = np.cross(a, ref)
    norm_u1 = float(np.linalg.norm(u1))
    if norm_u1 < 1e-5:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u1 = np.cross(a, ref)
        norm_u1 = float(np.linalg.norm(u1))
    u1 = u1 / norm_u1
    u2 = np.cross(a, u1)

    # ボーンの太さはボーン長によらず一定（均一）にする。
    # 基準半径を 0.016m (16mm) とし、極短ボーン（指など）のみ突き抜けないよう上限クランプ。
    base_r = float(radius) if (radius is not None and float(radius) > 1e-4) else 0.016
    r = min(base_r, length * 0.25)

    # 八面体の最大膨らみリング位置：長いボーンで間延びしないよう基部から固定距離（最大 3.5cm、または長さの15%）
    ring_dist = min(0.035, length * 0.15)
    m = h + a * ring_dist

    p0 = m + u1 * r
    p1 = m + u2 * r
    p2 = m - u1 * r
    p3 = m - u2 * r

    tris = [
        h, p0, p1,  h, p1, p2,  h, p2, p3,  h, p3, p0,
        t, p1, p0,  t, p2, p1,  t, p3, p2,  t, p0, p3,
    ]
    lines = [
        h, p0,  h, p1,  h, p2,  h, p3,
        p0, t,  p1, t,  p2, t,  p3, t,
        p0, p1, p1, p2, p2, p3, p3, p0,
    ]
    return [tuple(x) for x in tris], [tuple(x) for x in lines]


def compute_wrinkle_influence_mesh(
    origin: Sequence[float],
    axis: Sequence[float],
    normal: Sequence[float],
    theta_min: float,
    theta_max: float,
    z_min: float,
    z_max: float,
    target_r: float,
    thick_in: float,
    thick_out: float,
    n_theta: int = 48,
    n_z: int = 8,
    with_lines: bool = True,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """
    目標半径上の影響範囲メッシュ（内外厚を持つ曲面パッチ）を生成する。
    - 谷: [target, target+valley_depth]（目標より外のみに力が働く）
    - 山: [target-crest_depth, target]（目標より内のみに力が働く）
    厚み窓は呼び出し側指定（GPUシェーダのグラデーション窓と一致させること）。
    角度・長さ方向の帯を示し、軸まで埋めない。Blender非依存の純粋関数。
    - with_lines=False の場合は面のみ生成する（フェード分割シェル用）。
    - 戻り値: (tris_pos, lines_pos)
    """
    orig = np.array(origin[:3], dtype=np.float64)
    ax = np.array(axis[:3], dtype=np.float64)
    ax_len = float(np.linalg.norm(ax))
    if ax_len < 1e-8 or target_r <= 1e-6:
        return [], []
    ax = ax / ax_len
    n_vec = np.array(normal[:3], dtype=np.float64)
    n_vec = n_vec - ax * float(np.dot(n_vec, ax))
    n_len = float(np.linalg.norm(n_vec))
    if n_len < 1e-8:
        return [], []
    n_vec = n_vec / n_len
    b_vec = np.cross(ax, n_vec)

    span = float(theta_max - theta_min)
    if span < 1e-6:
        return [], []
    is_full = span >= 2.0 * math.pi - 1e-3
    n_theta = max(4, int(n_theta))
    n_z = max(1, int(n_z))

    r_in = max(float(target_r) - float(thick_in), 0.0)
    r_out = float(target_r) + float(thick_out)
    if r_out <= r_in:
        return [], []

    def _surf_point(theta: float, z: float, r: float):
        direction = math.cos(theta) * n_vec + math.sin(theta) * b_vec
        p = orig + ax * z + direction * r
        return (float(p[0]), float(p[1]), float(p[2]))

    # 外面・内面グリッド
    outer_grid = []
    inner_grid = []
    for iz in range(n_z + 1):
        z = z_min + (z_max - z_min) * (iz / n_z)
        row_o = []
        row_i = []
        for it in range(n_theta + 1):
            th = theta_min + span * (it / n_theta)
            row_o.append(_surf_point(th, z, r_out))
            row_i.append(_surf_point(th, z, r_in))
        outer_grid.append(row_o)
        inner_grid.append(row_i)

    tris: List[Tuple[float, float, float]] = []
    lines: List[Tuple[float, float, float]] = []
    n_seg = n_theta if is_full else n_theta
    for iz in range(n_z):
        for it in range(n_seg):
            jt = (it + 1) % (n_theta + 1) if is_full else it + 1
            # 外面
            o00, o01 = outer_grid[iz][it], outer_grid[iz][jt]
            o10, o11 = outer_grid[iz + 1][it], outer_grid[iz + 1][jt]
            tris.extend([o00, o10, o11])
            tris.extend([o00, o11, o01])
            # 内面（逆巻き）
            i00, i01 = inner_grid[iz][it], inner_grid[iz][jt]
            i10, i11 = inner_grid[iz + 1][it], inner_grid[iz + 1][jt]
            tris.extend([i00, i11, i10])
            tris.extend([i00, i01, i11])

    # 外周ライン
    if with_lines:
        for it in range(n_seg):
            jt = (it + 1) % (n_theta + 1) if is_full else it + 1
            lines.extend([outer_grid[0][it], outer_grid[0][jt]])
            lines.extend([outer_grid[n_z][it], outer_grid[n_z][jt]])
        for iz in range(n_z):
            lines.extend([outer_grid[iz][0], outer_grid[iz + 1][0]])
            if not is_full:
                lines.extend([outer_grid[iz][n_theta], outer_grid[iz + 1][n_theta]])
        # 目標ライン（z両端）
        for it in range(n_seg):
            jt = (it + 1) % (n_theta + 1) if is_full else it + 1
            t0 = _surf_point(theta_min + span * (it / n_theta), z_min, target_r)
            t1 = _surf_point(theta_min + span * (jt / n_theta), z_min, target_r)
            lines.extend([t0, t1])
            t0 = _surf_point(theta_min + span * (it / n_theta), z_max, target_r)
            t1 = _surf_point(theta_min + span * (jt / n_theta), z_max, target_r)
            lines.extend([t0, t1])

    return tris, lines


def build_wrinkle_influence_meshes(
    curves_data: Sequence[Tuple[str, Sequence[Sequence[float]], float, float]],
    origin: Sequence[float],
    axis: Sequence[float],
    normal: Sequence[float],
    bone_radius: float,
    valley_depth: float = 0.015,
    crest_depth: float = 0.020,
    fade_shells: int = 3,
) -> List[Tuple[str, List[Tuple[float, float, float]], List[Tuple[float, float, float]], float]]:
    """カーブ群から影響範囲メッシュ群を構築する (Blender非依存の幾何部)。

    - curves_data: (wrinkle_type, points, influence_radius, target_radius_or_0) のリスト
    - 谷 (root/valley): [target, target+valley_depth]、山 (crest/ridge): [target-crest_depth, target]
    - 角度・長さ方向には influence_radius 分だけ帯を拡張する。
    - 動径方向に fade_shells 分割し、GPUシェーダの dist_factor に比例した
      alpha を付与する（影響力のない目標側ほど淡く）。戻りの各要素は
      (wrinkle_type, tris, lines, alpha)。lines は外縁シェルにのみ付属。
    動径遠方には遮断がないため、メッシュは立ち上がり域のみを示すことに注意。
    """
    from ..engine.wrinkle_field import derive_wrinkle_influence_spans

    out = []
    for item in curves_data:
        w_type = item[0]
        pts = item[1]
        inf_r = float(item[2]) if len(item) > 2 else 0.03
        explicit_tr = float(item[3]) if len(item) > 3 else 0.0
        span = derive_wrinkle_influence_spans(pts, origin, axis, normal, explicit_tr)
        if span is None:
            continue
        is_crest = str(w_type).lower() in ("crest", "ridge")
        t_min = span["theta_min"]
        t_max = span["theta_max"]
        z0 = span["z_min"] - max(inf_r, 0.0)
        z1 = span["z_max"] + max(inf_r, 0.0)
        if not span["is_closed"]:
            dtheta = max(inf_r, 0.0) / max(float(bone_radius), 1e-4)
            t_min -= dtheta
            t_max += dtheta
        depth = max(float(crest_depth if is_crest else valley_depth), 1e-4)
        n_sh = max(int(fade_shells), 1)
        # ラインは従来単一帯と同一のフルバンド表示を維持する
        _, band_lines = compute_wrinkle_influence_mesh(
            origin, axis, normal, t_min, t_max, z0, z1,
            span["target_r"],
            depth if is_crest else 0.0,
            0.0 if is_crest else depth,
        )
        for i in range(n_sh):
            # alpha はシェーダの dist_factor（目標で0・縁で最大）に比例させる。
            # 谷: 目標側ほど淡く / 山: 目標側ほど淡く（縁＝最大影響が濃い）。
            frac_in = i / n_sh
            frac_out = (i + 1) / n_sh
            mid = (frac_in + frac_out) * 0.5
            if is_crest:
                # i=0 が最内縁（最大影響）
                r_hi = span["target_r"] - depth * frac_in
                r_lo = span["target_r"] - depth * frac_out
                alpha = 1.0 - mid
            else:
                # i=n_sh-1 が最外縁（最大影響）
                r_lo = span["target_r"] + depth * frac_in
                r_hi = span["target_r"] + depth * frac_out
                alpha = mid
            tris, _ = compute_wrinkle_influence_mesh(
                origin, axis, normal, t_min, t_max, z0, z1,
                r_hi, r_hi - r_lo, 0.0,
                n_theta=32, n_z=4, with_lines=False,
            )
            if tris:
                # 最外縁シェルにのみラインを付属させる
                out.append((w_type, tris, band_lines if i == n_sh - 1 else [],
                            max(0.12, min(1.0, alpha))))
    return out


def _draw_bone_proxy_cylinder(b_head, axis_n, n_vec, bone_rad: float, bone_len: float):
    """骨半径代理の仮想円柱（先端〜末端）を薄い灰色で表示する。"""
    if bone_rad <= 1e-6 or bone_len <= 1e-6:
        return
    tris, lines = compute_wrinkle_influence_mesh(
        b_head, axis_n, n_vec, 0.0, 2.0 * math.pi, 0.0, float(bone_len),
        float(bone_rad), 0.0, 0.002, n_theta=32, n_z=1,
    )
    if tris:
        draw_wrinkle_influence_meshes([("proxy", tris, lines)])


def draw_wrinkle_influence_meshes(
    meshes_data: Sequence[tuple],
    alpha_mul: float = 1.0,
):
    """
    シワカーブ群の影響範囲（角度・長さ帯と目標前後の厚み）を3D空間上に半透明表示する。
    - meshes_data: (wrinkle_type, tris, lines[, alpha]) のリスト。
      alpha 省略時は 1.0。フェード分割シェルは dist_factor 比例の alpha を持つ。
    ドーナツ管表示の置換であり、軸まで埋めない。
    """
    if not meshes_data:
        return

    shader = get_3d_uniform_color_shader() or get_line_shader()
    if not shader:
        return

    orig_depth_test = gpu.state.depth_test_get()
    orig_blend = gpu.state.blend_get()
    orig_line_width = gpu.state.line_width_get()

    try:
        # 最前面（X-Ray）で半透明ボリュームを描画（衣装・コライダーに隠れない）
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('ALPHA')

        for entry in meshes_data:
            if len(entry) == 4:
                w_type, tris, lines, entry_alpha = entry
            else:
                w_type, tris, lines = entry
                entry_alpha = 1.0
            if not tris:
                continue
            eff = max(0.0, min(1.0, float(entry_alpha))) * alpha_mul

            is_crest = str(w_type).lower() in ("crest", "ridge")
            is_proxy = str(w_type).lower() == "proxy"
            if is_proxy:
                face_color = (0.75, 0.8, 0.85, 0.10 * alpha_mul)
                line_color = (0.8, 0.85, 0.9, 0.40 * alpha_mul)
            elif is_crest:
                face_color = (1.0, 0.42, 0.08, 0.15 * eff)
                line_color = (1.0, 0.55, 0.15, 0.65 * (0.35 + 0.65 * eff))
            else:
                face_color = (0.08, 0.72, 1.0, 0.15 * eff)
                line_color = (0.15, 0.85, 1.0, 0.65 * (0.35 + 0.65 * eff))

            # 面の描画 (TRIS)
            batch_tris = batch_for_shader(shader, 'TRIS', {"pos": tris})
            if batch_tris:
                shader.bind()
                shader.uniform_float("color", face_color)
                batch_tris.draw(shader)

            # 外周・目標ラインの描画 (LINES)
            if lines:
                try:
                    gpu.state.line_width_set(1.5)
                except Exception:
                    pass
                batch_lines = batch_for_shader(shader, 'LINES', {"pos": lines})
                if batch_lines:
                    shader.bind()
                    shader.uniform_float("color", line_color)
                    batch_lines.draw(shader)

    finally:
        gpu.state.depth_test_set(orig_depth_test)
        gpu.state.blend_set(orig_blend)
        try:
            gpu.state.line_width_set(orig_line_width)
        except Exception:
            pass


def draw_wrinkle_armature_overlay(
    all_bones_data: list,
    active_bone_name: str = "",
    chain_bone_names: list = None,
):
    """
    シワモーダル実行中に、アーマチュアの全ボーンおよびアクティブボーンを3D空間上に最前面（X-Ray）で強調表示する。
    アーマチュア自体の表示・非表示設定に関わらず、gpuモジュールで直接八面体メッシュとして描画する。
    - all_bones_data: 各ボーンの (name, head_world, tail_world, radius) のタプルリスト
    - active_bone_name: 現在カーソル最近傍またはスライド対象のボーン名
    - chain_bone_names: 現在のボーンチェーンに含まれるボーン名のリスト（オプション）
    """
    if not all_bones_data:
        return

    shader = get_3d_uniform_color_shader() or get_line_shader()
    point_shader = get_point_shader()
    if not shader:
        return

    chain_set = set(chain_bone_names) if chain_bone_names else set()

    orig_depth_test = gpu.state.depth_test_get()
    orig_line_width = gpu.state.line_width_get()
    orig_blend = gpu.state.blend_get()

    try:
        # 深度テストを無効化（最前面・X-Ray表示）
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('ALPHA')

        # 頂点バッチの構築
        other_tris, other_lines = [], []
        chain_tris, chain_lines = [], []
        active_tris, active_lines = [], []
        active_points = []

        for item in all_bones_data:
            b_name = item[0]
            head = item[1]
            tail = item[2]
            # ボーン長に比例するエンベロープ半径は使わず、一定の基準太さ（16mm）で統一
            rad = 0.016

            tris, lines = compute_octahedron_mesh(head, tail, rad)
            if not tris:
                continue

            if b_name == active_bone_name:
                active_tris.extend(tris)
                active_lines.extend(lines)
                active_points.extend([head, tail])
            elif b_name in chain_set:
                chain_tris.extend(tris)
                chain_lines.extend(lines)
            else:
                other_tris.extend(tris)
                other_lines.extend(lines)

        # 1. その他全ボーン（非アクティブ）: 半透明グレー・ホワイト
        if other_tris:
            batch = batch_for_shader(shader, 'TRIS', {"pos": other_tris})
            shader.bind()
            shader.uniform_float("color", (0.8, 0.85, 0.9, 0.12))
            batch.draw(shader)

        if other_lines:
            try:
                gpu.state.line_width_set(1.5)
            except Exception:
                pass
            batch = batch_for_shader(shader, 'LINES', {"pos": other_lines})
            shader.bind()
            shader.uniform_float("color", (0.85, 0.9, 1.0, 0.35))
            batch.draw(shader)

        # 2. チェーン内ボーン（スライド可能範囲）: 半透明ライトシアン
        if chain_tris:
            batch = batch_for_shader(shader, 'TRIS', {"pos": chain_tris})
            shader.bind()
            shader.uniform_float("color", (0.2, 0.75, 0.95, 0.25))
            batch.draw(shader)

        if chain_lines:
            try:
                gpu.state.line_width_set(2.5)
            except Exception:
                pass
            batch = batch_for_shader(shader, 'LINES', {"pos": chain_lines})
            shader.bind()
            shader.uniform_float("color", (0.3, 0.85, 1.0, 0.65))
            batch.draw(shader)

        # 3. アクティブボーン（現在選択中・カーソル最近傍）: 鮮やかなシアン＋太線
        if active_tris:
            batch = batch_for_shader(shader, 'TRIS', {"pos": active_tris})
            shader.bind()
            shader.uniform_float("color", (0.1, 0.85, 1.0, 0.45))
            batch.draw(shader)

        if active_lines:
            try:
                gpu.state.line_width_set(4.0)
            except Exception:
                pass
            batch = batch_for_shader(shader, 'LINES', {"pos": active_lines})
            shader.bind()
            shader.uniform_float("color", (0.35, 1.0, 1.0, 0.95))
            batch.draw(shader)

        # 4. アクティブボーンの Head / Tail マーカー
        if point_shader and active_points:
            orig_point_size = gpu.state.point_size_get()
            try:
                gpu.state.point_size_set(9.0)
                batch_pts = batch_for_shader(point_shader, 'POINTS', {"pos": active_points})
                point_shader.bind()
                point_shader.uniform_float("color", (1.0, 0.85, 0.2, 0.95))
                batch_pts.draw(point_shader)
            finally:
                gpu.state.point_size_set(orig_point_size)

    except Exception:
        pass
    finally:
        gpu.state.depth_test_set(orig_depth_test)
        gpu.state.line_width_set(orig_line_width)
        gpu.state.blend_set(orig_blend)


def draw_wrinkle_bone_overlay(chain_bones_coords, active_bone_idx: int = 0):
    """
    シワスライドモーダル中のボーンチェーンを3D空間上に最前面で強調表示する（後方互換ラッパー）。
    - chain_bones_coords: 各ボーンの (head_world, tail_world, name) のタプルリスト
    - active_bone_idx: 現在操作対象となっているボーンのインデックス
    """
    if not chain_bones_coords:
        return
    active_name = ""
    all_bones_data = []
    chain_names = []
    for idx, item in enumerate(chain_bones_coords):
        head = item[0]
        tail = item[1]
        name = item[2] if len(item) > 2 else f"bone_{idx}"
        rad = item[3] if len(item) > 3 else 0.05
        if isinstance(rad, (tuple, list)):
            rad = max(rad[0], rad[1])
        all_bones_data.append((name, head, tail, rad))
        chain_names.append(name)
        if idx == active_bone_idx:
            active_name = name

    draw_wrinkle_armature_overlay(all_bones_data, active_bone_name=active_name, chain_bone_names=chain_names)



def draw_callback_2d():
    """3Dビューポートの2D（POST_PIXEL）HUD描画コールバック"""
    if not _interactive_active and not _bake_overlay_info and not _init_overlay_info:
        return

    context = bpy.context
    region = getattr(context, "region", None)
    if not region:
        return

    shader_2d = get_2d_uniform_color_shader()

    # --- 1. インタラクティブモード中のHUD描画 ---
    if _interactive_active and _interactive_fps_info:
        show_fps = _interactive_fps_info.get("show_overlay", True)
        show_help = _interactive_fps_info.get("show_help", True)

        # 1. FPSバッジの描画
        if show_fps:
            fps = _interactive_fps_info.get("fps", 0.0)
            frame_ms = _interactive_fps_info.get("frame_ms", 0.0)
            position = _interactive_fps_info.get("position", 'TOP_CENTER')

            # バッジのサイズ設定
            box_w = 175.0
            box_h = 28.0

            # 配置座標の算出
            if position == 'TOP_CENTER':
                # 画面中央上部（ヘッダーと重ならないよう、上端から40px下げ、左右中央に配置）
                margin_x = (region.width - box_w) / 2.0
                y_top = region.height - 40.0
            elif position == 'TOP_RIGHT':
                # 画面右上（ナビゲーションギズモの左側を想定）
                margin_x = region.width - box_w - 90.0
                y_top = region.height - 40.0
            elif position == 'BOTTOM_RIGHT':
                # 画面右下
                margin_x = region.width - box_w - 24.0
                y_top = 24.0 + box_h
            elif position == 'BOTTOM_LEFT':
                # 画面左下（ツールバーの右下など）
                margin_x = 80.0
                y_top = 24.0 + box_h
            else:  # TOP_LEFT
                # 左上（ツールバー幅70pxの右側にオフセット）
                margin_x = 80.0
                y_top = region.height - 50.0

            y_bottom = y_top - box_h

            # 半透明ダーク背景（クアッド）の描画
            if shader_2d:
                orig_blend = gpu.state.blend_get()
                try:
                    gpu.state.blend_set('ALPHA')
                    vertices = [
                        (margin_x, y_bottom),
                        (margin_x + box_w, y_bottom),
                        (margin_x + box_w, y_top),
                        (margin_x, y_bottom),
                        (margin_x + box_w, y_top),
                        (margin_x, y_top),
                    ]
                    batch = batch_for_shader(shader_2d, 'TRIS', {"pos": vertices})
                    if batch:
                        shader_2d.bind()
                        shader_2d.uniform_float("color", (0.12, 0.12, 0.14, 0.75))
                        batch.draw(shader_2d)
                finally:
                    gpu.state.blend_set(orig_blend)

            # テキスト描画 (blf)
            try:
                import blf
                font_id = 0
                font_size = 13

                _setup_font(font_size, font_id)

                # FPSステータスに応じた文字色判定
                if fps >= 50.0:
                    text_color = (0.3, 0.95, 0.4, 1.0)   # 鮮やかなグリーン
                elif fps >= 30.0:
                    text_color = (1.0, 0.85, 0.2, 1.0)   # イエロー
                elif fps > 0.0:
                    text_color = (1.0, 0.35, 0.25, 1.0)  # オレンジレッド
                else:
                    text_color = (0.7, 0.7, 0.7, 1.0)    # グレー（初期状態）

                text_str = f"FPS: {fps:5.1f} ({frame_ms:4.1f} ms)"
                text_x = margin_x + 12.0
                text_y = y_bottom + 8.0

                blf.position(font_id, text_x, text_y, 0.0)
                blf.color(font_id, text_color[0], text_color[1], text_color[2], text_color[3])
                blf.draw(font_id, text_str)
            except Exception:
                pass

        # 1b. 縫合優先フェーズバッジの描画 (FPSバッジ直下。FPS非表示時も単独表示)
        sewing_text = _interactive_fps_info.get("sewing_text")
        if sewing_text:
            if show_fps:
                sew_margin_x = margin_x
                sew_y_top = y_bottom - 8.0
            else:
                sew_margin_x = (region.width - 240.0) / 2.0
                sew_y_top = region.height - 40.0
            if _interactive_fps_info.get("sewing_active"):
                sew_color = (0.35, 0.85, 1.0, 1.0)   # 縫合中: シアン
            else:
                sew_color = (0.3, 0.95, 0.4, 1.0)    # 通常: グリーン
            _draw_top_badge(region, shader_2d, sew_margin_x, sew_y_top, sewing_text, sew_color)

        # 2. キー操作ガイドバッジの描画 (画面下部中央・最大2行、docs/hud.md:5)
        if show_help:
            base_guide = i18n.trans("[LMB Drag] Move  |  [P] Pin/Unpin  |  [Esc / RMB] Exit")
            suffix = _interactive_fps_info.get("guide_line1_suffix")
            line2 = _interactive_fps_info.get("guide_line2")
            if suffix:
                line1 = f"{base_guide}  |  {suffix}"
            else:
                # 後方互換: 旧tool_text単一行呼出し
                legacy = _interactive_fps_info.get("tool_text")
                if legacy:
                    line1 = f"{base_guide}  |  {legacy}"
                else:
                    line1 = base_guide
            _draw_bottom_badge_2row(region, shader_2d, line1, line2)

        # 3. ブラシカーソル円の描画 (2Dスクリーン空間)
        if _brush_info:
            try:
                import math

                cx = _brush_info.get("x", 0.0)
                cy = _brush_info.get("y", 0.0)
                pr = max(_brush_info.get("radius_px", 20.0), 2.0)
                segments = 48
                pts = [
                    (cx + pr * math.cos(2.0 * math.pi * i / segments),
                     cy + pr * math.sin(2.0 * math.pi * i / segments))
                    for i in range(segments + 1)
                ]
                orig_blend = gpu.state.blend_get()
                try:
                    gpu.state.blend_set('ALPHA')
                    try:
                        gpu.state.line_width_set(2.0)
                    except Exception:
                        pass
                    batch = batch_for_shader(shader_2d, 'LINE_STRIP', {"pos": pts})
                    if batch:
                        shader_2d.bind()
                        shader_2d.uniform_float("color", (0.2, 0.8, 1.0, 0.9))
                        batch.draw(shader_2d)
                finally:
                    gpu.state.blend_set(orig_blend)
            except Exception:
                pass

    # --- 2. ベイク中オーバーレイバッジの描画 (画面下部中央) ---
    if _bake_overlay_info:
        cur_f = _bake_overlay_info.get("current_frame", 1)
        end_f = _bake_overlay_info.get("end_frame", 250)
        pct = max(0, min(100, _bake_overlay_info.get("pct", 0)))

        msg = i18n.trans("Baking Simulation: Frame %d / %d (%d%%)  |  [Esc] Cancel")
        try:
            bake_text = msg % (cur_f, end_f, pct)
        except Exception:
            bake_text = f"Baking Simulation: Frame {cur_f} / {end_f} ({pct}%)  |  [Esc] Cancel"

        sewing_text = _bake_overlay_info.get("sewing_text")
        if sewing_text:
            bake_text = f"{bake_text}  |  {sewing_text}"

        _draw_bottom_center_progress(region, shader_2d, bake_text, pct)

    # --- 3. 起動準備中オーバーレイバッジの描画 (画面下部中央) ---
    if _init_overlay_info:
        message = _init_overlay_info.get("message", "")
        cur = _init_overlay_info.get("current", 0)
        total = _init_overlay_info.get("total", 1)
        pct = int(min(1.0, max(0.0, cur / max(1, total))) * 100)

        stage_text = i18n.trans(message) if message else ""
        cancel_text = i18n.trans("[Esc] Cancel")
        try:
            init_text = f"{stage_text} {cur}/{total} ({pct}%)  |  {cancel_text}"
        except Exception:
            init_text = f"{message} {cur}/{total} ({pct}%)"

        _draw_bottom_center_progress(region, shader_2d, init_text, pct)


def register_draw_handler():
    """描画ハンドラーを登録する (3D POST_VIEW および 2D POST_PIXEL)"""
    global _draw_handler, _draw_handler_2d
    space_view_3d = getattr(bpy.types, "SpaceView3D", None)
    if space_view_3d and hasattr(space_view_3d, "draw_handler_add"):
        if _draw_handler is None:
            _draw_handler = space_view_3d.draw_handler_add(
                draw_callback_3d, (), 'WINDOW', 'POST_VIEW'
            )
        if _draw_handler_2d is None:
            _draw_handler_2d = space_view_3d.draw_handler_add(
                draw_callback_2d, (), 'WINDOW', 'POST_PIXEL'
            )


def unregister_draw_handler():
    """描画ハンドラーを解除する"""
    global _draw_handler, _draw_handler_2d
    space_view_3d = getattr(bpy.types, "SpaceView3D", None)
    if space_view_3d and hasattr(space_view_3d, "draw_handler_remove"):
        if _draw_handler is not None:
            try:
                space_view_3d.draw_handler_remove(_draw_handler, 'WINDOW')
            except Exception:
                pass
            _draw_handler = None
        if _draw_handler_2d is not None:
            try:
                space_view_3d.draw_handler_remove(_draw_handler_2d, 'WINDOW')
            except Exception:
                pass
            _draw_handler_2d = None
