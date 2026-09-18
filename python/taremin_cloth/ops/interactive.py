"""
taremin_cloth インタラクティブ・モーダルオペレーター
3Dビューポート上でのリアルタイム布シミュレーション、マウスドラッグによる頂点操作、
ピン留めトグル、FPS計測・描画オーバーレイ、およびデバッグレコーダー統合を提供する。
"""

import os
import re
import time
from datetime import datetime
import bpy
import mathutils
from bpy_extras import view3d_utils
from ..engine.cache import (
    _simulators,
    cache_rest_positions,
    clear_simulator_for_object,
)
from ..engine.runner import (
    get_or_create_simulator,
    get_effective_substeps,
    step_cloth_scene,
    step_cloth_object,
    restore_fast_playback,
)
from ..preferences import get_preferences
from ..utils import drawing, topology, anim_driver
from ..utils.logger import logger
from ..utils.view3d import tag_redraw_view3d
from .. import i18n

_interactive_running = False
_interactive_operator_instance = None
_last_benchmark_summary = {}


def is_interactive_running():
    """インタラクティブモードが実行中かどうかを判定する"""
    global _interactive_running
    return _interactive_running


def get_last_benchmark_summary():
    """直近のインタラクティブモードのFPSサンプリング計測結果を取得する"""
    global _last_benchmark_summary
    return _last_benchmark_summary


def stop_interactive_if_running():
    """インタラクティブモードが実行中であれば安全に停止要求を行う"""
    global _interactive_running, _interactive_operator_instance
    if _interactive_running and _interactive_operator_instance is not None:
        _interactive_operator_instance._stop_requested = True


def resolve_debug_filepath(prefs, obj_name: str, frame_count: int, ext: str = "jsonl.gz") -> str:
    """プレファレンスの設定値と現在のオブジェクト名・日時からデバッグ出力先ファイルパスを解決・生成する"""
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    datetime_str = now.strftime("%Y%m%d_%H%M%S")

    # オブジェクト名のサニタイズ（Windows/POSIXでファイル名に使えない文字を _ に置換）
    safe_obj_name = re.sub(r'[\\/*?:"<>|]', '_', obj_name)

    # テンプレートの取得
    template = getattr(prefs, "debug_filename_template", "cloth_debug_{datetime}_{object}.{ext}") if prefs else "cloth_debug_{datetime}_{object}.{ext}"
    if not template or not template.strip():
        template = "cloth_debug_{datetime}_{object}.{ext}"

    # 拡張子の置換・付与
    filename = template.format(
        date=date_str,
        time=time_str,
        datetime=datetime_str,
        object=safe_obj_name,
        frames=frame_count,
        ext=ext,
    )

    if not filename.endswith(f".{ext}") and not filename.endswith(".gz"):
        filename = f"{filename}.{ext}"

    # 出力ディレクトリの解決
    out_dir = getattr(prefs, "debug_output_dir", "").strip() if prefs else ""
    dir_path = None
    if out_dir:
        try:
            res = bpy.path.abspath(out_dir)
            if isinstance(res, str) and not res.startswith("<MagicMock"):
                dir_path = res
        except Exception:
            pass
        if not dir_path:
            dir_path = os.path.abspath(out_dir)

    if not dir_path:
        # デフォルト(空欄)時はアドオンルート直下の frame_logs ディレクトリを使用
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # ops -> taremin_cloth -> python -> addon_root
        parent = os.path.dirname(os.path.dirname(current_dir))
        if os.path.basename(parent) == "python":
            addon_dir = os.path.dirname(parent)
        else:
            addon_dir = parent

        default_frame_logs = os.path.join(addon_dir, "frame_logs")
        try:
            os.makedirs(default_frame_logs, exist_ok=True)
            dir_path = default_frame_logs
        except Exception:
            import tempfile
            dir_path = getattr(getattr(bpy, "app", None), "tempdir", None) or tempfile.gettempdir()

    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, filename)


class FPSCounter:
    """フレーム間隔から指数移動平均（EMA）を用いて安定したFPSおよびフレーム時間を算出する軽量カウンター"""
    def __init__(self, ema_alpha: float = 0.15):
        self.ema_alpha = ema_alpha
        self.last_time = None
        self.fps = 0.0
        self.frame_ms = 0.0

    def tick(self, current_time: float = None):
        if current_time is None:
            current_time = time.perf_counter()

        if self.last_time is None:
            self.last_time = current_time
            return self.fps, self.frame_ms

        dt = current_time - self.last_time
        self.last_time = current_time

        if dt <= 0.0:
            return self.fps, self.frame_ms

        instant_fps = 1.0 / dt
        instant_ms = dt * 1000.0

        if self.fps <= 0.0:
            self.fps = instant_fps
            self.frame_ms = instant_ms
        else:
            self.fps = self.ema_alpha * instant_fps + (1.0 - self.ema_alpha) * self.fps
            self.frame_ms = self.ema_alpha * instant_ms + (1.0 - self.ema_alpha) * self.frame_ms

        return self.fps, self.frame_ms

    def reset(self):
        self.last_time = None
        self.fps = 0.0
        self.frame_ms = 0.0


def enter_isolated_view(context, cloth_obj):
    """シミュレーション関連オブジェクト（布、コライダー、アーマチュア）のみを対象にローカルビューに入る"""
    saved_selection = [o.name for o in context.selected_objects if o]
    saved_active = context.active_object.name if context.active_object else None
    isolated_areas = []

    try:
        # 1. 関連オブジェクトの収集（布 + コライダー + 関連アーマチュア/親）
        target_objs = {cloth_obj}
        for o in context.scene.objects:
            c_set = getattr(o, "taremin_cloth_collider", None)
            if c_set and c_set.is_collider and getattr(c_set, "enabled", True):
                target_objs.add(o)
                p = o.parent
                while p:
                    target_objs.add(p)
                    p = p.parent
                for mod in o.modifiers:
                    if mod.type == 'ARMATURE' and getattr(mod, "object", None):
                        target_objs.add(mod.object)

        # 布オブジェクトの親およびアーマチュアも含める
        p = cloth_obj.parent
        while p:
            target_objs.add(p)
            p = p.parent
        for mod in cloth_obj.modifiers:
            if mod.type == 'ARMATURE' and getattr(mod, "object", None):
                target_objs.add(mod.object)

        # 2. 選択状態の切り替え
        bpy.ops.object.select_all(action='DESELECT')
        for o in target_objs:
            try:
                o.select_set(True)
            except Exception:
                pass

        # 3. VIEW_3D エリアで localview をトグル
        if context.screen:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    space = area.spaces.active
                    if space and getattr(space, "local_view", None) is None:
                        try:
                            with context.temp_override(area=area):
                                bpy.ops.view3d.localview(frame_selected=False)
                            isolated_areas.append(area)
                        except Exception as e:
                            logger.debug(f"[Interactive] Failed to enter localview for area: {e}")

        # 4. 選択状態を布オブジェクト単独に復元（アクティブに設定）
        bpy.ops.object.select_all(action='DESELECT')
        cloth_obj.select_set(True)
        context.view_layer.objects.active = cloth_obj
    except Exception as e:
        logger.warning(f"[Interactive] Failed to isolate viewport view: {e}")

    return isolated_areas, saved_selection, saved_active


def exit_isolated_view(context, isolated_areas, saved_selection, saved_active):
    """ローカルビューから通常表示に復帰し、選択状態を復元する"""
    try:
        # 1. 記録されたエリアのローカルビューを解除
        for area in list(isolated_areas):
            space = area.spaces.active if area else None
            if space and getattr(space, "local_view", None) is not None:
                try:
                    with context.temp_override(area=area):
                        bpy.ops.view3d.localview(frame_selected=False)
                except Exception as e:
                    logger.debug(f"[Interactive] Failed to exit localview for area: {e}")

        # 2. 元の選択状態の復元
        bpy.ops.object.select_all(action='DESELECT')
        for name in saved_selection:
            o = context.scene.objects.get(name)
            if o:
                try:
                    o.select_set(True)
                except Exception:
                    pass
        if saved_active:
            act = context.scene.objects.get(saved_active)
            if act:
                context.view_layer.objects.active = act
    except Exception as e:
        logger.warning(f"[Interactive] Failed to restore viewport view: {e}")


class TAREMIN_CLOTH_OT_interactive(bpy.types.Operator):
    """3Dビューポート上でリアルタイムに布を掴んで動かすモーダルオペレーター"""
    bl_idname = "taremin_cloth.interactive"
    bl_label = "Interactive Cloth Simulation"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER'}

    _timer = None
    _fps_counter = None
    _grabbed_vert = None
    _grab_initial_pos_local = None
    _grab_plane_point = None
    _grab_plane_normal = None
    _grab_initial_plane_hit = None
    _grab_current_target_local = None
    _stop_requested = False
    _pinned_verts = set()
    _anim_frame_counter = 0
    _accumulator = 0.0
    _isolated_areas = []
    _saved_selection = []
    _saved_active = None
    _perf_samples = []

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj
            and obj.type == 'MESH'
            and getattr(obj, "taremin_cloth", None)
            and obj.taremin_cloth.is_cloth
            and getattr(obj.taremin_cloth, "enabled", True)
        )

    def step_simulation(self, context):
        """シミュレーションを実時間同期で進め、メッシュとオーバーレイを更新する"""
        obj = context.active_object
        if not obj or not obj.taremin_cloth.is_cloth:
            return

        sim, coords = get_or_create_simulator(obj)
        settings = getattr(obj, "taremin_cloth", None)

        t_start = time.perf_counter()
        now = t_start
        delta_time = now - (self._last_step_time or now)
        self._last_step_time = now

        # 固定時間刻み (60 FPS基準)
        FIXED_DT = 1.0 / 60.0

        realtime_sync = getattr(settings, "interactive_realtime_sync", True) if settings else True
        max_steps = getattr(settings, "interactive_max_steps", 4) if settings else 4

        # インタラクティブモードでは描画レスポンスと実時間同期の両立を図る。
        # 最大2ステップ/フレームに制限し、最大蓄積を3ステップ分 (約50ms) に抑えて遅延スパイラルを防止。
        step_limit = min(max_steps, 2) if realtime_sync else 1
        if realtime_sync:
            self._accumulator += delta_time
            self._accumulator = min(self._accumulator, FIXED_DT * 3.0)
        else:
            self._accumulator = FIXED_DT

        step_count = 0
        while self._accumulator >= FIXED_DT and step_count < step_limit:
            self._accumulator -= FIXED_DT
            step_count += 1

        t_step_total = 0.0
        t_get = 0.0
        t_mesh = 0.0

        # アキュムレータが 1/60 秒以上蓄積されていた場合、共通の step_cloth_scene を呼び出して実行
        if step_count > 0:
            t0 = time.perf_counter()
            step_cloth_scene(
                context.scene,
                dt=FIXED_DT,
                anim_frame=self._anim_frame_counter,
                target_objs=[obj],
                update_mesh=True,
                steps_per_frame=step_count,
            )
            self._anim_frame_counter += 1
            t_step_total = (time.perf_counter() - t0) * 1000.0

            # 3Dビューポートの再描画要求
            tag_redraw_view3d(context)

        # FPS計測とオーバーレイ更新
        if self._fps_counter is not None:
            fps, frame_ms = self._fps_counter.tick()
            show_overlay = getattr(settings, "show_fps_overlay", True) if settings else True
            position = getattr(settings, "fps_overlay_position", 'TOP_CENTER') if settings else 'TOP_CENTER'
            show_help = getattr(settings, "show_hud_help", True) if settings else True
            drawing.set_interactive_fps_info(fps, frame_ms, show_overlay=show_overlay, position=position, show_help=show_help)

        # パフォーマンスサンプリング記録
        if delta_time > 0:
            instant_fps = 1.0 / delta_time
            t_work = (time.perf_counter() - t_start) * 1000.0
            render_ms = max(0.0, delta_time * 1000.0 - t_work)
            self._perf_samples.append({
                "fps": instant_fps,
                "dt_ms": delta_time * 1000.0,
                "sim_ms": t_step_total,
                "get_ms": t_get,
                "mesh_ms": t_mesh,
                "work_ms": t_work,
                "render_ms": render_ms,
                "step_count": step_count,
            })

    def modal(self, context, event):
        global _interactive_running, _interactive_operator_instance
        if self._stop_requested:
            self.cancel(context)
            return {'FINISHED'}

        obj = context.active_object
        if not obj or not obj.taremin_cloth.is_cloth:
            self.cancel(context)
            return {'CANCELLED'}

        sim, coords = get_or_create_simulator(obj)

        if event.type == 'TIMER':
            self.step_simulation(context)
            return {'PASS_THROUGH'}

        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                region = context.region
                rv3d = context.region_data
                if region and rv3d:
                    mouse_pos = (event.mouse_region_x, event.mouse_region_y)
                    pos_2d = coords.reshape((-1, 3))
                    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_pos)
                    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_pos)

                    best_idx = None

                    # 1. 視線レイキャストによる最前面ポリゴンの交差判定
                    if origin and direction:
                        matrix_world = obj.matrix_world
                        matrix_inv = matrix_world.inverted()
                        ray_origin_local = matrix_inv @ origin
                        ray_dir_local = (matrix_inv.to_3x3() @ direction).normalized()

                        hit, hit_loc, hit_normal, face_idx = obj.ray_cast(ray_origin_local, ray_dir_local)
                        if hit and face_idx is not None and 0 <= face_idx < len(obj.data.polygons):
                            polygon = obj.data.polygons[face_idx]
                            min_dist_sq = float('inf')
                            # ヒットしたポリゴンの構成頂点から交点に最も近い頂点を選択
                            for v_idx in polygon.vertices:
                                v_co = obj.data.vertices[v_idx].co
                                dist_sq = (v_co - hit_loc).length_squared
                                if dist_sq < min_dist_sq:
                                    min_dist_sq = dist_sq
                                    best_idx = v_idx

                    # 2. 面に直接ヒットしなかった場合のフォールバック（画面最近傍探索）
                    if best_idx is None:
                        best_dist = float('inf')
                        for i, p in enumerate(pos_2d):
                            world_p = obj.matrix_world @ mathutils.Vector(p)
                            screen_co = view3d_utils.location_3d_to_region_2d(region, rv3d, world_p)
                            if screen_co:
                                dist = (screen_co.x - mouse_pos[0]) ** 2 + (screen_co.y - mouse_pos[1]) ** 2
                                if dist < best_dist and dist < 2500:
                                    best_dist = dist
                                    best_idx = i

                    if best_idx is not None and origin and direction:
                        self._grabbed_vert = best_idx
                        p = pos_2d[best_idx]
                        v_initial_local = mathutils.Vector((p[0], p[1], p[2]))
                        v_initial_world = obj.matrix_world @ v_initial_local

                        # カメラの視線前方向ベクトルを取得（ビュー平面の法線）
                        view_inv = rv3d.view_matrix.inverted()
                        camera_forward = -view_inv.to_3x3().col[2].normalized()

                        # ビュー平面の通過点は「選択された頂点自身のワールド位置」
                        self._grab_plane_point = v_initial_world
                        self._grab_plane_normal = camera_forward
                        self._grab_initial_pos_local = v_initial_local
                        self._grab_current_target_local = v_initial_local

                        # クリック時のレイとビュー平面との初期交点を計算・記録
                        denom = direction.dot(camera_forward)
                        if abs(denom) > 1e-6:
                            t0 = (v_initial_world - origin).dot(camera_forward) / denom
                            self._grab_initial_plane_hit = origin + direction * t0
                        else:
                            self._grab_initial_plane_hit = v_initial_world

                        sim.set_pin(best_idx, [p[0], p[1], p[2]], 1.0)
                        drawing.set_active_grabbed_vertex(obj.name, best_idx, v_initial_world)
                        if context.area:
                            context.area.tag_redraw()
                        return {'RUNNING_MODAL'}

            elif event.value == 'RELEASE':
                if self._grabbed_vert is not None:
                    # ピン留めされていない頂点のみ物理解放
                    if self._grabbed_vert not in self._pinned_verts:
                        sim.release_pin(self._grabbed_vert)
                    drawing.clear_active_grabbed_vertex()
                    self._grabbed_vert = None
                    self._grab_initial_pos_local = None
                    self._grab_plane_point = None
                    self._grab_plane_normal = None
                    self._grab_initial_plane_hit = None
                    self._grab_current_target_local = None
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

        elif event.type == 'MOUSEMOVE' and self._grabbed_vert is not None:
            region = context.region
            rv3d = context.region_data
            if (region and rv3d and 
                self._grab_plane_point is not None and 
                self._grab_plane_normal is not None and 
                self._grab_initial_plane_hit is not None and 
                self._grab_initial_pos_local is not None):

                mouse_pos = (event.mouse_region_x, event.mouse_region_y)
                origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_pos)
                direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_pos)
                if origin and direction:
                    denom = direction.dot(self._grab_plane_normal)
                    if abs(denom) > 1e-6:
                        # 現在のマウスレイとビュー平面の交点を計算
                        t = (self._grab_plane_point - origin).dot(self._grab_plane_normal) / denom
                        current_plane_hit = origin + direction * t

                        # ビュー平面上のワールド移動差分（オフセット）
                        delta_world = current_plane_hit - self._grab_initial_plane_hit

                        # オブジェクトローカル空間の移動差分に変換
                        matrix_inv = obj.matrix_world.inverted()
                        delta_local = matrix_inv.to_3x3() @ delta_world

                        # 初期のローカル頂点位置にオフセットを加算
                        target_pos_local = self._grab_initial_pos_local + delta_local
                        self._grab_current_target_local = target_pos_local
                        sim.set_pin(self._grabbed_vert, [target_pos_local.x, target_pos_local.y, target_pos_local.z], 1.0)
                        target_world = obj.matrix_world @ target_pos_local
                        drawing.set_active_grabbed_vertex(obj.name, self._grabbed_vert, target_world)

        elif event.type == 'P' and event.value == 'PRESS':
            # Pキーで掴んでいる頂点（またはカーソル下の頂点）をピン留め／解除（トグル）
            target_v_idx = self._grabbed_vert
            if target_v_idx is None:
                # ドラッグしていない場合はマウスカーソル直下の頂点を探索
                region = context.region
                rv3d = context.region_data
                if region and rv3d:
                    mouse_pos = (event.mouse_region_x, event.mouse_region_y)
                    pos_2d = coords.reshape((-1, 3))
                    best_dist = float('inf')
                    for i, p in enumerate(pos_2d):
                        world_p = obj.matrix_world @ mathutils.Vector(p)
                        screen_co = view3d_utils.location_3d_to_region_2d(region, rv3d, world_p)
                        if screen_co:
                            dist = (screen_co.x - mouse_pos[0]) ** 2 + (screen_co.y - mouse_pos[1]) ** 2
                            if dist < best_dist and dist < 2500:
                                best_dist = dist
                                target_v_idx = i

            if target_v_idx is not None:
                settings = obj.taremin_cloth
                vg_name = settings.pin_vertex_group or "Pin"
                vg = obj.vertex_groups.get(vg_name)
                if not vg:
                    vg = obj.vertex_groups.new(name="Pin")
                    settings.pin_vertex_group = "Pin"

                # すでにピン留めされているか判定
                is_pinned = (target_v_idx in self._pinned_verts)
                if not is_pinned:
                    try:
                        w = vg.weight(target_v_idx)
                        if w > 0.0:
                            is_pinned = True
                    except RuntimeError:
                        is_pinned = False

                if is_pinned:
                    # ピン解除 (Unpin)
                    try:
                        vg.remove([target_v_idx])
                    except RuntimeError:
                        pass
                    self._pinned_verts.discard(target_v_idx)
                    # ドラッグ中でなければ物理ピンも解除
                    if self._grabbed_vert != target_v_idx:
                        sim.release_pin(target_v_idx)
                    self.report({'INFO'}, i18n.trans("Unpinned vertex #%d") % target_v_idx)
                else:
                    # ピン留め (Pin)
                    vg.add([target_v_idx], 1.0, 'REPLACE')
                    self._pinned_verts.add(target_v_idx)
                    # 現在のローカル目標位置（または頂点現在位置）で固定
                    if self._grabbed_vert == target_v_idx and self._grab_current_target_local is not None:
                        t_pos = self._grab_current_target_local
                    else:
                        p = coords.reshape((-1, 3))[target_v_idx]
                        t_pos = mathutils.Vector((p[0], p[1], p[2]))
                    sim.set_pin(target_v_idx, [t_pos.x, t_pos.y, t_pos.z], 1.0)
                    self.report({'INFO'}, i18n.trans("Pinned vertex #%d") % target_v_idx)

                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cancel(context)
            return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        global _interactive_running, _interactive_operator_instance
        # トグル動作: すでに実行中の場合は停止を要求
        if _interactive_running:
            if _interactive_operator_instance is not None:
                _interactive_operator_instance._stop_requested = True
            return {'FINISHED'}

        obj = context.active_object
        self._stop_requested = False
        self._pinned_verts = set()
        self._fps_counter = FPSCounter(ema_alpha=0.15)
        drawing.clear_interactive_fps_info()

        # 十字分割の自動適用（未適用の四角面がある場合、かつCROSS_SUBDIVモード時のみ）
        if obj and obj.type == 'MESH':
            settings = getattr(obj, "taremin_cloth", None)
            if settings and settings.triangulation_mode == 'CROSS_SUBDIV' and not topology.is_cross_subdivided(obj):
                cache_rest_positions(obj, force=True)
                if topology.apply_cross_subdivision(obj):
                    clear_simulator_for_object(obj.name)
                    cache_rest_positions(obj, force=True)

        # 既存のピン留め頂点を事前ロード
        if obj and obj.type == 'MESH':
            settings = getattr(obj, "taremin_cloth", None)
            if settings:
                vg_name = settings.pin_vertex_group or "Pin"
                vg = obj.vertex_groups.get(vg_name)
                if vg:
                    vg_idx = vg.index
                    for v in obj.data.vertices:
                        for g in v.groups:
                            if g.group == vg_idx and g.weight > 0.0:
                                self._pinned_verts.add(v.index)
                                break

        self._anim_frame_counter = 0
        self._accumulator = 0.0
        self._last_step_time = time.perf_counter()

        # デバッグ状態記録の開始判定 (Console Log Level が DEBUG のとき、かつデバッグ記録設定有効時)
        prefs = get_preferences(context)
        should_debug_record = prefs and (prefs.log_level == 'DEBUG') and getattr(prefs, "enable_debug_recording", True)
        if should_debug_record and obj and obj.type == 'MESH':
            sim, _ = get_or_create_simulator(obj)
            if sim and hasattr(sim, "start_debug_recording"):
                max_frames = getattr(prefs, "debug_max_frames", 3600)
                sim.start_debug_recording(obj.name, max_frames=max_frames)
                logger.debug(f"[DebugRecorder] Started recording simulation states for '{obj.name}' (max_frames={max_frames})")

        # パフォーマンスサンプラー初期化
        self._perf_samples = []

        # ローカルビュー（描画隔離）の適用
        self._isolated_areas = []
        self._saved_selection = []
        self._saved_active = None
        settings = getattr(obj, "taremin_cloth", None)
        if settings and getattr(settings, "isolate_viewport_view", True):
            self._isolated_areas, self._saved_selection, self._saved_active = enter_isolated_view(context, obj)

        wm = context.window_manager
        # 60 FPS相当 (約16.6ms) の高精度タイマーを登録
        self._timer = wm.event_timer_add(0.016, window=context.window)
        wm.modal_handler_add(self)
        _interactive_running = True
        _interactive_operator_instance = self

        drawing.set_interactive_active(True)

        if hasattr(context.workspace, "status_text_set"):
            status_guide = i18n.trans(
                "Taremin Cloth: [Left Drag] Move Vertex | [P] Toggle Pin | [Right Click / ESC] Exit"
            )
            context.workspace.status_text_set(status_guide)

        self.report({'INFO'}, i18n.trans("Interactive Simulation Started (Press ESC / RightClick or Click Stop to exit)"))
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        global _interactive_running, _interactive_operator_instance, _last_benchmark_summary
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None

        drawing.clear_active_grabbed_vertex()
        drawing.clear_interactive_fps_info()
        drawing.set_interactive_active(False)
        self._fps_counter = None

        if self._grabbed_vert is not None:
            if self._grabbed_vert not in self._pinned_verts:
                obj = context.active_object
                if obj and obj.name in _simulators:
                    sim, _ = _simulators[obj.name]
                    sim.release_pin(self._grabbed_vert)
            self._grabbed_vert = None

        self._grab_initial_pos_local = None
        self._grab_plane_point = None
        self._grab_plane_normal = None
        self._grab_initial_plane_hit = None
        self._grab_current_target_local = None
        self._pinned_verts.clear()
        _interactive_running = False
        _interactive_operator_instance = None

        if hasattr(context.workspace, "status_text_set"):
            context.workspace.status_text_set(None)

        tag_redraw_view3d(context)

        # デバッグ状態記録のファイル保存とクリーンアップ
        obj = context.active_object
        if obj and obj.name in _simulators:
            sim, _ = _simulators[obj.name]
            if sim and hasattr(sim, "is_debug_recording") and sim.is_debug_recording():
                frame_count = sim.get_debug_frame_count()
                if frame_count > 0:
                    prefs = get_preferences(context)
                    filepath = resolve_debug_filepath(prefs, obj.name, frame_count, ext="jsonl.gz")
                    try:
                        saved_path = sim.save_debug_recording(filepath)
                        file_size = os.path.getsize(saved_path) if os.path.exists(saved_path) else 0
                        logger.info(f"[DebugRecorder] Successfully saved debug recording ({frame_count} frames, {file_size:,} bytes) to: {saved_path}")
                        self.report({'INFO'}, i18n.trans("Debug recording saved: %s (%d frames)") % (os.path.basename(saved_path), frame_count))
                    except Exception as e:
                        logger.error(f"[DebugRecorder] Failed to save debug recording: {e}")
                sim.stop_debug_recording()

        # ローカルビューの復帰
        if self._isolated_areas:
            exit_isolated_view(context, self._isolated_areas, self._saved_selection, self._saved_active)
            self._isolated_areas.clear()

        # Fast Playback の復元
        restore_fast_playback(context.scene)

        # パフォーマンスサンプリング結果の集計とレポート出力
        summary_msg = ""
        if self._perf_samples:
            valid_samples = self._perf_samples[5:] if len(self._perf_samples) > 10 else self._perf_samples
            if valid_samples:
                fps_vals = [s["fps"] for s in valid_samples]
                dt_vals = [s["dt_ms"] for s in valid_samples]
                sim_vals = [s["sim_ms"] for s in valid_samples]
                get_vals = [s["get_ms"] for s in valid_samples]
                mesh_vals = [s["mesh_ms"] for s in valid_samples]
                render_vals = [s["render_ms"] for s in valid_samples]

                avg_fps = sum(fps_vals) / len(fps_vals)
                min_fps = min(fps_vals)
                max_fps = max(fps_vals)
                avg_dt = sum(dt_vals) / len(dt_vals)
                avg_sim = sum(sim_vals) / len(sim_vals)
                avg_get = sum(get_vals) / len(get_vals)
                avg_mesh = sum(mesh_vals) / len(mesh_vals)
                avg_render = sum(render_vals) / len(render_vals)

                _last_benchmark_summary = {
                    "total_frames": len(self._perf_samples),
                    "sampled_frames": len(valid_samples),
                    "avg_fps": avg_fps,
                    "min_fps": min_fps,
                    "max_fps": max_fps,
                    "avg_dt_ms": avg_dt,
                    "avg_sim_ms": avg_sim,
                    "avg_get_ms": avg_get,
                    "avg_mesh_ms": avg_mesh,
                    "avg_render_ms": avg_render,
                }

                logger.info(
                    f"\n"
                    f"========================================================\n"
                    f"  [TareminCloth] Interactive Performance Summary\n"
                    f"========================================================\n"
                    f"  Sampled Frames:     {len(valid_samples)} frames\n"
                    f"  Average FPS:        {avg_fps:5.1f} FPS ({avg_dt:4.1f} ms/frame)\n"
                    f"  Min / Max FPS:      {min_fps:5.1f} / {max_fps:5.1f} FPS\n"
                    f"  Time Breakdown:\n"
                    f"    - Physics Sim:    {avg_sim:4.2f} ms ({avg_sim/avg_dt*100:4.1f}%)\n"
                    f"    - GPU Readback:   {avg_get:4.2f} ms ({avg_get/avg_dt*100:4.1f}%)\n"
                    f"    - Mesh Update:    {avg_mesh:4.2f} ms ({avg_mesh/avg_dt*100:4.1f}%)\n"
                    f"    - Viewport/Wait:  {avg_render:4.2f} ms ({avg_render/avg_dt*100:4.1f}%)\n"
                    f"========================================================"
                )
                summary_msg = f" | Avg FPS: {avg_fps:.1f} (Sim: {avg_sim:.1f}ms, Render: {avg_render:.1f}ms)"
        self._perf_samples = []

        # インタラクティブモード停止時は、次回スムーズに停止位置から再開（Warm Resume）できるよう
        # シミュレータインスタンスおよびQuadトポロジーをそのまま保持する
        stopped_msg = i18n.trans("Interactive Simulation Stopped (Paused)")
        self.report({'INFO'}, f"{stopped_msg}{summary_msg}")


class TAREMIN_CLOTH_OT_benchmark_fps(bpy.types.Operator):
    """Run interactive simulation for ~2 seconds to benchmark actual FPS and timing breakdown"""
    bl_idname = "taremin_cloth.benchmark_fps"
    bl_label = "Benchmark FPS"
    bl_description = "Automatically run interactive simulation for ~2 seconds to benchmark actual FPS and timing breakdown"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER'}

    _frame_counter = 0

    @classmethod
    def poll(cls, context):
        return TAREMIN_CLOTH_OT_interactive.poll(context) and not is_interactive_running()

    def execute(self, context):
        self._frame_counter = 0

        # インタラクティブシミュレーションを開始
        bpy.ops.taremin_cloth.interactive()

        # 約80ステップ（約1.5〜2秒）後に自動停止するタイマーを登録
        def check_auto_finish():
            if not is_interactive_running():
                return None
            self._frame_counter += 1
            if self._frame_counter >= 80:
                stop_interactive_if_running()
                return None
            return 0.02

        bpy.app.timers.register(check_auto_finish, first_interval=0.05)
        self.report({'INFO'}, i18n.trans("Benchmarking FPS... (will complete in ~2 seconds)"))
        return {'FINISHED'}
