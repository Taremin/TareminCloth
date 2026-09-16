"""
3DビューポートGPUプレビュー描画ユーティリティ
ピン留め頂点や縫合線の視覚的オーバーレイ描画を行う。
"""

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .. import i18n

_draw_handler = None
_draw_handler_2d = None
_interactive_active = False
_active_grabbed_info = None  # {"obj_name": str, "vert_idx": int, "target_world_pos": Vector or None}
_interactive_fps_info = None  # {"fps": float, "frame_ms": float, "show_overlay": bool, "position": str, "show_help": bool}
_pin_indices_cache = {}  # {(obj_name, vg_name): (indices_list, vert_count)}


def clear_pin_cache():
    """ピン留め頂点インデックスのキャッシュをクリアする"""
    global _pin_indices_cache
    _pin_indices_cache.clear()


def set_interactive_active(active: bool):
    """インタラクティブモードの実行状態を設定する"""
    global _interactive_active
    _interactive_active = active
    if not active:
        clear_pin_cache()


def is_interactive_active() -> bool:
    """インタラクティブモードが実行中かどうかを取得する"""
    return _interactive_active


def set_interactive_fps_info(
    fps: float,
    frame_ms: float,
    show_overlay: bool = True,
    position: str = 'TOP_CENTER',
    show_help: bool = True,
):
    """インタラクティブモード中のFPSおよび操作ヘルプ計測情報を更新する"""
    global _interactive_fps_info
    _interactive_fps_info = {
        "fps": float(fps),
        "frame_ms": float(frame_ms),
        "show_overlay": bool(show_overlay),
        "position": str(position),
        "show_help": bool(show_help),
    }


def clear_interactive_fps_info():
    """インタラクティブモード中のFPS計測情報をクリアする"""
    global _interactive_fps_info
    _interactive_fps_info = None


def get_interactive_fps_info():
    """現在のFPS計測情報を取得する（テストまたはUI用）"""
    return _interactive_fps_info


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


def apply_view_depth_bias(coords, region_3d=None):
    """
    深度テスト有効時のZ-fighting（埋没・本来の辺や面と色が混ざって薄くなる現象）を解消するため、
    描画座標をカメラ（視点）手前方向にわずかにオフセットする。
    物陰（裏面や他オブジェクト）にある頂点・エッジは引き続き確実に遮蔽される。
    """
    if not coords or region_3d is None or not hasattr(region_3d, "view_matrix"):
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

        return offset_coords.tolist()
    except Exception:
        return coords


def get_point_shader():
    for name in ('POINT_UNIFORM_COLOR', '3D_UNIFORM_COLOR'):
        try:
            return gpu.shader.from_builtin(name)
        except Exception:
            pass
    return None


def get_line_shader():
    for name in ('POLYLINE_UNIFORM_COLOR', '3D_UNIFORM_COLOR'):
        try:
            return gpu.shader.from_builtin(name)
        except Exception:
            pass
    return None


def get_2d_uniform_color_shader():
    for name in ('2D_UNIFORM_COLOR', 'UNIFORM_COLOR'):
        try:
            return gpu.shader.from_builtin(name)
        except Exception:
            pass
    return None


def get_smooth_color_line_shader():
    for name in ('POLYLINE_SMOOTH_COLOR', '3D_SMOOTH_COLOR'):
        try:
            return gpu.shader.from_builtin(name)
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
    """3Dビューポートのオーバーレイ描画コールバック"""
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
                    pin_coords = [[*(world_mat @ mesh.vertices[i].co)] for i in pin_indices]
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
                mesh.calc_loop_triangles()
                face_edge_set = set()
                for tri in mesh.loop_triangles:
                    face_edge_set.add((min(tri.vertices[0], tri.vertices[1]), max(tri.vertices[0], tri.vertices[1])))
                    face_edge_set.add((min(tri.vertices[1], tri.vertices[2]), max(tri.vertices[1], tri.vertices[2])))
                    face_edge_set.add((min(tri.vertices[2], tri.vertices[0]), max(tri.vertices[2], tri.vertices[0])))

                sew_lines = []
                for e in mesh.edges:
                    pair = (min(e.vertices[0], e.vertices[1]), max(e.vertices[0], e.vertices[1]))
                    if pair not in face_edge_set:
                        p0 = world_mat @ mesh.vertices[e.vertices[0]].co
                        p1 = world_mat @ mesh.vertices[e.vertices[1]].co
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

        finally:
            gpu.state.depth_test_set(orig_depth_test)
            gpu.state.depth_mask_set(orig_depth_mask)


def draw_callback_2d():
    """3Dビューポートの2D（POST_PIXEL）HUD描画コールバック"""
    if not _interactive_active or not _interactive_fps_info:
        return

    show_fps = _interactive_fps_info.get("show_overlay", True)
    show_help = _interactive_fps_info.get("show_help", True)

    if not show_fps and not show_help:
        return

    context = bpy.context
    region = getattr(context, "region", None)
    if not region:
        return

    shader_2d = get_2d_uniform_color_shader()

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

            # blf.size の互換性対応 (Blenderバージョン差対応)
            try:
                blf.size(font_id, font_size)
            except (TypeError, ValueError):
                try:
                    blf.size(font_id, font_size, 72)
                except Exception:
                    pass

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

    # 2. キー操作ガイドバッジの描画 (画面下部中央)
    if show_help:
        guide_text = i18n.trans("[LMB Drag] Move  |  [P] Pin/Unpin  |  [Esc / RMB] Exit")
        guide_h = 28.0
        pad_x = 14.0
        text_w = 340.0

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
            dims = blf.dimensions(font_id, guide_text)
            if dims and dims[0] > 0:
                text_w = dims[0]
        except Exception:
            pass

        guide_w = text_w + pad_x * 2.0
        guide_margin_x = max(10.0, (region.width - guide_w) / 2.0)
        guide_y_bottom = 20.0
        guide_y_top = guide_y_bottom + guide_h

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
            text_x = guide_margin_x + pad_x
            text_y = guide_y_bottom + 8.0
            blf.position(font_id, text_x, text_y, 0.0)
            blf.color(font_id, 0.92, 0.92, 0.95, 0.95)
            blf.draw(font_id, guide_text)
        except Exception:
            pass


def register_draw_handler():
    """描画ハンドラーを登録する (3D POST_VIEW および 2D POST_PIXEL)"""
    global _draw_handler, _draw_handler_2d
    if _draw_handler is None and hasattr(bpy.types, "SpaceView3D"):
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_3d, (), 'WINDOW', 'POST_VIEW'
        )
    if _draw_handler_2d is None and hasattr(bpy.types, "SpaceView3D"):
        _draw_handler_2d = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_2d, (), 'WINDOW', 'POST_PIXEL'
        )


def unregister_draw_handler():
    """描画ハンドラーを解除する"""
    global _draw_handler, _draw_handler_2d
    if _draw_handler is not None and hasattr(bpy.types, "SpaceView3D"):
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None
    if _draw_handler_2d is not None and hasattr(bpy.types, "SpaceView3D"):
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler_2d, 'WINDOW')
        _draw_handler_2d = None
