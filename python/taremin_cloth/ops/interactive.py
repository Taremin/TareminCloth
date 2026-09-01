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
)
from ..engine.collider import sync_colliders
from ..engine.params import sync_cloth_parameters
from ..preferences import get_preferences
from ..utils import drawing, topology, anim_driver
from ..utils.logger import logger

_interactive_running = False
_interactive_operator_instance = None


def is_interactive_running():
    """インタラクティブモードが実行中かどうかを判定する"""
    global _interactive_running
    return _interactive_running


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
            dir_path = bpy.path.abspath(out_dir)
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


class TAREMIN_CLOTH_OT_interactive(bpy.types.Operator):
    """3Dビューポート上でリアルタイムに布を掴んで動かすモーダルオペレーター"""
    bl_idname = "taremin_cloth.interactive"
    bl_label = "Interactive Cloth Simulation"
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
    _last_step_time = 0.0

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

        if realtime_sync:
            # スパイラル・オブ・デス（処理落ち累積の無限ループ）を防ぐため最大0.1秒蓄積
            self._accumulator += min(delta_time, 0.1)
            step_limit = max_steps
        else:
            self._accumulator = FIXED_DT
            step_limit = 1

        step_count = 0
        t_anim_total = 0.0
        t_col_total = 0.0
        t_param_total = 0.0
        t_step_total = 0.0

        # アキュムレータが 1/60 秒以上蓄積されている間、必要な分だけ物理ステップを進める
        while self._accumulator >= FIXED_DT and step_count < step_limit:
            # コライダーのアニメーション駆動ステップ
            t0 = time.perf_counter()
            any_collider_deformed = False
            for col_o in context.scene.objects:
                c_set = getattr(col_o, "taremin_collider", None)
                if c_set and c_set.is_collider and getattr(c_set, "enabled", True) and getattr(c_set, "anim", None) and c_set.anim.enabled:
                    _, deformed = anim_driver.step_collider_animation(col_o, self._anim_frame_counter)
                    if deformed:
                        any_collider_deformed = True
            self._anim_frame_counter += 1

            depsgraph = context.evaluated_depsgraph_get()
            if any_collider_deformed:
                context.view_layer.update()
                depsgraph = context.evaluated_depsgraph_get()
            t_anim_total += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            sync_colliders(sim, context.scene, depsgraph=depsgraph, force=any_collider_deformed, cloth_obj=obj)
            t_col_total += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            sync_cloth_parameters(sim, obj, context.scene)
            actual_substeps = get_effective_substeps(obj, coords, FIXED_DT, scene=context.scene)
            t_param_total += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            sim.step(dt=FIXED_DT, substeps=actual_substeps, solver_iterations=settings.solver_iterations if settings else 1)
            t_step_total += (time.perf_counter() - t0) * 1000.0

            self._accumulator -= FIXED_DT
            step_count += 1

        # 実時間同期が上限に達した場合、残余アキュムレータをリセットして遅延蓄積を防止
        if step_count >= step_limit:
            self._accumulator = min(self._accumulator, FIXED_DT * 0.5)

        # 全ステップ完了後、1回だけGPUリードバックとメッシュ頂点更新を実行
        t0 = time.perf_counter()
        sim.get_positions(coords)
        t_get = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        obj.data.vertices.foreach_set("co", coords)
        obj.data.update()
        obj["_taremin_is_deformed"] = True
        t_mesh = (time.perf_counter() - t0) * 1000.0

        # FPS計測とオーバーレイ更新
        if self._fps_counter is not None:
            fps, frame_ms = self._fps_counter.tick()
            show_overlay = getattr(settings, "show_fps_overlay", True) if settings else True
            position = getattr(settings, "fps_overlay_position", 'TOP_CENTER') if settings else 'TOP_CENTER'
            drawing.set_interactive_fps_info(fps, frame_ms, show_overlay=show_overlay, position=position)

            # 5フレームごとにコンソールへ詳細内訳ログを出力
            if self._anim_frame_counter % 5 == 0:
                t_work = (time.perf_counter() - t_start) * 1000.0
                logger.info(
                    f"[Interactive Details] Render FPS: {fps:4.1f} ({frame_ms:5.1f}ms) | Sim Steps: {step_count}x | Work: {t_work:4.1f}ms (step:{t_step_total:3.1f}, get:{t_get:3.1f}, mesh:{t_mesh:3.1f})"
                )

        # 3Dビューポートの再描画要求
        has_redrawn = False
        if context.screen:
            for a in context.screen.areas:
                if a.type == 'VIEW_3D':
                    a.tag_redraw()
                    has_redrawn = True
        if not has_redrawn and context.area:
            context.area.tag_redraw()

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
                    self.report({'INFO'}, f"頂点 #{target_v_idx} のピン留めを解除しました")
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
                    self.report({'INFO'}, f"頂点 #{target_v_idx} をピン留めしました")

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

        wm = context.window_manager
        # 60 FPS相当 (約16.6ms) の高精度タイマーを登録
        self._timer = wm.event_timer_add(0.016, window=context.window)
        wm.modal_handler_add(self)
        _interactive_running = True
        _interactive_operator_instance = self

        drawing.set_interactive_active(True)

        if hasattr(context.workspace, "status_text_set"):
            context.workspace.status_text_set(
                "Taremin Cloth: [左ドラッグ] 頂点移動 | [P] ピン留め/解除 | [右クリック / ESC] 終了"
            )

        self.report({'INFO'}, "Interactive Simulation Started (Press ESC / RightClick or Click Stop to exit)")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        global _interactive_running, _interactive_operator_instance
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

        if context.screen:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

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
                        self.report({'INFO'}, f"Debug recording saved: {os.path.basename(saved_path)} ({frame_count} frames)")
                    except Exception as e:
                        logger.error(f"[DebugRecorder] Failed to save debug recording: {e}")
                sim.stop_debug_recording()

        # インタラクティブモード停止時は、次回スムーズに停止位置から再開（Warm Resume）できるよう
        # シミュレータインスタンスおよびQuadトポロジーをそのまま保持する
        self.report({'INFO'}, "Interactive Simulation Stopped (Paused)")
