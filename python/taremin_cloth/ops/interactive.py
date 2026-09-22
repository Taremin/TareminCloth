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
    begin_simulator_init,
    create_simulator_from_state,
    finalize_simulator_init,
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
from ..brush import create_tools, active_tool_name
from ..brush.base import BrushContext, RadialController, init_brush_circle_at, nudge_brush_radius

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
    _stop_requested = False
    _pinned_verts = set()
    _brush_tools = {}
    _brush_dragging = None
    _radial = None
    _anim_frame_counter = 0
    _accumulator = 0.0
    _isolated_areas = []
    _saved_selection = []
    _saved_active = None
    _perf_samples = []
    _init_stage = None
    _init_total = 6
    _modal_handler_added = False
    _progress_active = False
    _cursor_wait_set = False

    # 起動準備の段階メッセージ (i18n辞書キーと1:1対応すること)
    _INIT_STAGE_MESSAGES = (
        "Preparing cloth...",
        "Loading pins...",
        "Extracting mesh...",
        "Initializing GPU...",
        "Syncing colliders...",
        "Setting up viewport...",
    )

    # 低速起動とみなしてINFOサマリーを出す合計秒数の閾値
    _SLOW_STARTUP_SEC = 3.0

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
            tool_text = None
            try:
                brush = self._brush_settings_of(obj)
                mode = getattr(brush, "tool_mode", 'GRAB') if brush is not None else 'GRAB'
                if brush is not None and mode in ('RANGE_GRAB', 'SMOOTH'):
                    _r = float(getattr(brush, "radius", 0.05))
                    _label = {'RANGE_GRAB': 'Range Grab', 'SMOOTH': 'Smooth'}.get(mode, mode)
                    tool_text = i18n.trans("Tool: %s") % f"{i18n.trans(_label)} r={_r * 1000.0:.0f}mm"
                elif brush is not None:
                    tool_text = f"[E] {i18n.trans('Range Grab')} / {i18n.trans('Smooth')}"
            except Exception:
                tool_text = None
            drawing.set_interactive_fps_info(fps, frame_ms, show_overlay=show_overlay, position=position, show_help=show_help, tool_text=tool_text)

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

    def _brush_settings_of(self, obj):
        settings = getattr(obj, "taremin_cloth", None)
        return getattr(settings, "brush", None) if settings else None

    def _brush_ctx(self, context, obj, sim, coords, event, mouse_pos):
        return BrushContext(self, context, obj, sim, coords, event, mouse_pos)

    def _active_brush_tool(self, brush):
        return self._brush_tools.get(active_tool_name(brush))

    def _reset_run_state(self):
        """1回の起動試行に先立ち、実行時ステートを初期化する"""
        self._stop_requested = False
        self._pinned_verts = set()
        self._brush_tools = create_tools()
        self._brush_dragging = None
        self._radial = RadialController()
        self._fps_counter = FPSCounter(ema_alpha=0.15)
        self._anim_frame_counter = 0
        self._accumulator = 0.0
        self._last_step_time = time.perf_counter()
        self._perf_samples = []
        self._isolated_areas = []
        self._saved_selection = []
        self._saved_active = None
        self._init_stage = 0
        self._modal_handler_added = False
        self._progress_active = False
        self._cursor_wait_set = False
        self._init_timings = []
        self._init_partial = None
        self._init_sim = None
        drawing.clear_interactive_fps_info()
        drawing.clear_init_overlay_info()

    def _init_progress_total(self):
        return len(self._INIT_STAGE_MESSAGES)

    def _update_init_progress(self, context):
        """現在の準備段階をプログレス・ステータス・HUDに反映する"""
        total = self._init_progress_total()
        idx = min(max(self._init_stage or 0, 0), total)
        if idx < total:
            message = self._INIT_STAGE_MESSAGES[idx]
        else:
            message = self._INIT_STAGE_MESSAGES[-1]
        try:
            wm = context.window_manager
            wm.progress_update(idx)
        except Exception:
            pass
        drawing.set_init_overlay_info(message, idx, total)
        try:
            ws = getattr(context, "workspace", None)
            if ws and hasattr(ws, "status_text_set"):
                cancel_txt = i18n.trans("[Esc] Cancel")
                ws.status_text_set(f"{i18n.trans(message)} {idx}/{total}  |  {cancel_txt}")
        except Exception:
            pass
        try:
            tag_redraw_view3d(context)
        except Exception:
            pass

    def _begin_init_ui(self, context):
        """準備開始時のプログレス・カーソル・HUD初期表示を行う"""
        try:
            wm = context.window_manager
            wm.progress_begin(0, self._init_progress_total())
            wm.progress_update(0)
            self._progress_active = True
        except Exception:
            pass
        try:
            wm = context.window_manager
            cursor_set = getattr(wm, "cursor_modal_set", None)
            if callable(cursor_set):
                cursor_set('WAIT')
                self._cursor_wait_set = True
        except Exception:
            pass
        self._update_init_progress(context)

    def _end_init_ui(self, context, success: bool):
        """準備終了・中断時のプログレス・カーソル・HUD後片付けを行う"""
        if self._progress_active:
            try:
                wm = context.window_manager
                wm.progress_end()
            except Exception:
                pass
            self._progress_active = False
        if self._cursor_wait_set:
            try:
                wm = context.window_manager
                cursor_restore = getattr(wm, "cursor_modal_restore", None)
                if callable(cursor_restore):
                    cursor_restore()
            except Exception:
                pass
            self._cursor_wait_set = False
        drawing.clear_init_overlay_info()
        if not success:
            try:
                ws = getattr(context, "workspace", None)
                if ws and hasattr(ws, "status_text_set"):
                    ws.status_text_set(None)
            except Exception:
                pass
            try:
                tag_redraw_view3d(context)
            except Exception:
                pass

    def _run_init_stage(self, context):
        """現在の準備段階を1つだけ実行し、次段階へ進める（段階所要時間も記録する）"""
        obj = context.active_object
        stage = self._init_stage or 0
        t0 = time.perf_counter()
        if stage == 0:
            self._init_stage_topo(obj)
        elif stage == 1:
            self._init_stage_pins(obj)
        elif stage == 2:
            self._init_stage_sim_extract(obj)
        elif stage == 3:
            self._init_stage_sim_gpu(obj)
        elif stage == 4:
            self._init_stage_sim_collider(context, obj)
        elif stage == 5:
            self._init_stage_setup(context, obj)
        else:
            return False
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._init_timings.append((self._INIT_STAGE_MESSAGES[stage], elapsed_ms))
        self._init_stage = stage + 1
        return self._init_stage < self._init_progress_total()

    def _init_stage_topo(self, obj):
        """段階0: 十字分割の自動適用判定"""
        if obj and obj.type == 'MESH':
            settings = getattr(obj, "taremin_cloth", None)
            if settings and settings.triangulation_mode == 'CROSS_SUBDIV' and not topology.is_cross_subdivided(obj):
                cache_rest_positions(obj, force=True)
                if topology.apply_cross_subdivision(obj):
                    clear_simulator_for_object(obj.name)
                    cache_rest_positions(obj, force=True)

    def _init_stage_pins(self, obj):
        """段階1: 既存ピン留め頂点の事前ロード"""
        self._pinned_verts = set()
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

    def _init_stage_sim_extract(self, obj):
        """段階2: メッシュ抽出と生成パラメータの準備（Warm Resume時は即時復帰）"""
        if obj.name in _simulators:
            self._init_partial = {"cached": True}
            return
        self._init_partial = begin_simulator_init(obj)

    def _init_stage_sim_gpu(self, obj):
        """段階3: GPU初期化 (ClothSimulator生成 + Cold Resume)"""
        state = self._init_partial
        if not state or state.get("cached"):
            return
        sim = create_simulator_from_state(obj, state)
        state["sim"] = sim
        self._init_sim = sim

    def _init_stage_sim_collider(self, context, obj):
        """段階4: 特性長・レスト・コライダー/SDF同期・パラメータ同期・登録（最重量段階）"""
        state = self._init_partial
        if not state or state.get("cached"):
            return
        sim = state.get("sim")
        if sim is None:
            sim = create_simulator_from_state(obj, state)
            state["sim"] = sim
        finalize_simulator_init(obj, sim, state, scene=context.scene)
        self._init_sim = sim
        prefs = get_preferences(context)
        should_debug_record = prefs and (prefs.log_level == 'DEBUG') and getattr(prefs, "enable_debug_recording", True)
        if should_debug_record and obj and obj.type == 'MESH':
            if sim and hasattr(sim, "start_debug_recording"):
                max_frames = getattr(prefs, "debug_max_frames", 3600)
                sim.start_debug_recording(obj.name, max_frames=max_frames)
                logger.debug(f"[DebugRecorder] Started recording simulation states for '{obj.name}' (max_frames={max_frames})")

    def _init_stage_setup(self, context, obj):
        """段階3: 隔離ビュー等の仕上げ"""
        self._perf_samples = []
        self._isolated_areas = []
        self._saved_selection = []
        self._saved_active = None
        settings = getattr(obj, "taremin_cloth", None)
        if settings and getattr(settings, "isolate_viewport_view", True):
            self._isolated_areas, self._saved_selection, self._saved_active = enter_isolated_view(context, obj)

    def _report_startup_summary(self, context):
        """起動準備の段階別サマリーをログレベルに応じて出力する"""
        timings = list(getattr(self, "_init_timings", None) or [])
        if not timings:
            return
        total_ms = sum(ms for _, ms in timings)
        total_s = total_ms / 1000.0

        sim = getattr(self, "_init_sim", None)
        if sim is None:
            sdf_source = "warm"
        else:
            try:
                from ..engine.collider import _last_sync_info
                sdf_source = _last_sync_info.get(id(sim), {}).get("sdf_source", "none")
            except Exception:
                sdf_source = "unknown"

        prefs = get_preferences(context)
        level = getattr(prefs, "log_level", "INFO") if prefs else "INFO"

        breakdown = ", ".join(f"{msg} {ms / 1000.0:.2f}s" for msg, ms in timings)
        if level == 'DEBUG':
            logger.debug(
                f"[Interactive Startup] total {total_s:.2f}s (SDF: {sdf_source}) "
                f"[{breakdown}]"
            )
        elif total_s >= self._SLOW_STARTUP_SEC:
            logger.info(
                f"[Interactive Startup] Slow startup: total {total_s:.2f}s "
                f"(SDF: {sdf_source}) [{breakdown}]"
            )

    def _start_sim_loop(self, context):
        """準備完了後に本番シミュレーションループを開始する"""
        global _interactive_running, _interactive_operator_instance
        wm = context.window_manager
        if self._timer:
            try:
                wm.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        try:
            self._timer = wm.event_timer_add(0.016, window=context.window)
        except Exception:
            self._timer = None
        # invoke() ですでにモーダルハンドラ登録済みの場合は二重登録しない
        # （同一インスタンスの二重登録は終了時のクラッシュ要因となる）
        if not self._modal_handler_added:
            try:
                wm.modal_handler_add(self)
                self._modal_handler_added = True
            except Exception:
                pass
        _interactive_running = True
        _interactive_operator_instance = self
        drawing.set_interactive_active(True)
        if hasattr(context.workspace, "status_text_set"):
            status_guide = i18n.trans(
                "Taremin Cloth: [Left Drag] Move Vertex | [P] Toggle Pin | [Right Click / ESC] Exit"
            )
            try:
                status_guide = f"{status_guide} | [E] {i18n.trans('Range Grab')}"
            except Exception:
                pass
            context.workspace.status_text_set(status_guide)
        self._report_startup_summary(context)
        self.report({'INFO'}, i18n.trans("Interactive Simulation Started (Press ESC / RightClick or Click Stop to exit)"))

    def _abort_init(self, context):
        """準備中断時の後片付けを行う"""
        global _interactive_running, _interactive_operator_instance
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        self._end_init_ui(context, success=False)
        self._init_stage = None
        _interactive_running = False
        _interactive_operator_instance = None
        drawing.set_interactive_active(False)

    def _modal_init_step(self, context, event):
        """準備中のモーダルイベント処理（TIMERで1段階ずつ進行、ESC/RMBで中断）"""
        if self._stop_requested:
            self._abort_init(context)
            return {'CANCELLED'}
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._abort_init(context)
            self.report({'INFO'}, i18n.trans("[Esc] Cancel"))
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        try:
            has_more = self._run_init_stage(context)
        except Exception as e:
            logger.error(f"[Interactive] Init failed at stage {self._init_stage}: {e}", exc_info=True)
            self._abort_init(context)
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if has_more:
            self._update_init_progress(context)
            return {'RUNNING_MODAL'}
        self._init_stage = None
        self._end_init_ui(context, success=True)
        self._start_sim_loop(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        global _interactive_running, _interactive_operator_instance
        if self._init_stage is not None:
            return self._modal_init_step(context, event)
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
                # ラジアル調整中は確定として消費する
                if self._radial is not None and self._radial.active:
                    self._radial.confirm()
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                mouse_pos = (event.mouse_region_x, event.mouse_region_y)
                ctx = self._brush_ctx(context, obj, sim, coords, event, mouse_pos)
                tool = self._active_brush_tool(self._brush_settings_of(obj))
                if tool is not None and tool.on_press(ctx):
                    self._brush_dragging = tool.name
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

            elif event.value == 'RELEASE':
                if self._brush_dragging:
                    mouse_pos = (event.mouse_region_x, event.mouse_region_y)
                    ctx = self._brush_ctx(context, obj, sim, coords, event, mouse_pos)
                    tool = self._brush_tools.get(self._brush_dragging)
                    if tool is not None:
                        tool.on_release(ctx, self._pinned_verts)
                    self._brush_dragging = None
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

        elif event.type == 'MOUSEMOVE':
            mouse_pos = (event.mouse_region_x, event.mouse_region_y)
            ctx = self._brush_ctx(context, obj, sim, coords, event, mouse_pos)
            if self._radial is not None and self._radial.active:
                # ラジアル調整: 開始値×(1+dx/200px)
                try:
                    if self._radial.update(ctx):
                        if context.area:
                            context.area.tag_redraw()
                except Exception:
                    pass
                return {'RUNNING_MODAL'}
            if self._brush_dragging:
                tool = self._brush_tools.get(self._brush_dragging)
                if tool is not None and tool.on_move(ctx):
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                return {'PASS_THROUGH'}
            # ホバー時は円だけ追従 (パネル操作はPASS_THROUGHで継続可能)
            tool = self._active_brush_tool(self._brush_settings_of(obj))
            if tool is not None and tool.on_hover(ctx):
                if context.area:
                    context.area.tag_redraw()
            return {'PASS_THROUGH'}

        elif event.type == 'E' and event.value == 'PRESS':
            # Eキーでブラシツールを順方向に切替 (TOOLS登録順に循環)
            try:
                brush = self._brush_settings_of(obj)
                if brush is not None:
                    mouse_pos = (event.mouse_region_x, event.mouse_region_y)
                    ctx = self._brush_ctx(context, obj, sim, coords, event, mouse_pos)
                    order = list(self._brush_tools.keys())
                    cur = getattr(brush, "tool_mode", 'GRAB')
                    if cur not in order:
                        cur = order[0] if order else 'GRAB'
                    new_mode = order[(order.index(cur) + 1) % len(order)] if order else cur
                    old_tool = self._brush_tools.get(cur)
                    if old_tool is not None:
                        old_tool.on_deactivate(ctx, self._pinned_verts)
                    brush.tool_mode = new_mode
                    if self._brush_dragging == cur:
                        self._brush_dragging = None
                    new_tool = self._brush_tools.get(new_mode)
                    if new_tool is not None:
                        new_tool.on_activate(ctx)
                    self.report({'INFO'}, i18n.trans("Tool: %s") % new_mode)
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
            except Exception:
                pass

        elif event.type == 'F' and event.value == 'PRESS':
            # F / Shift+F でラジアル調整 (Blender準拠: 半径 / 強度)
            try:
                brush = self._brush_settings_of(obj)
                if brush is not None and self._radial is not None:
                    mouse_pos = (event.mouse_region_x, event.mouse_region_y)
                    if bool(getattr(event, 'shift', False)):
                        self._radial.enter('STRENGTH', event.mouse_region_x, brush)
                    else:
                        self._radial.enter('RADIUS', event.mouse_region_x, brush)
                    # 円未表示の場合はこの場で初期化する
                    if drawing.get_brush_info() is None:
                        ctx = self._brush_ctx(context, obj, sim, coords, event, mouse_pos)
                        init_brush_circle_at(ctx, max(float(getattr(brush, "radius", 0.05)), 1e-4))
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
            except Exception:
                pass

        elif event.type == 'RET' and event.value == 'PRESS' and self._radial is not None and self._radial.active:
            self._radial.confirm()
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        elif event.type == 'LEFT_BRACKET' and event.value == 'PRESS':
            mouse_pos = (event.mouse_region_x, event.mouse_region_y)
            nudge_brush_radius(obj, 0.9, mouse_pos)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        elif event.type == 'RIGHT_BRACKET' and event.value == 'PRESS':
            mouse_pos = (event.mouse_region_x, event.mouse_region_y)
            nudge_brush_radius(obj, 1.1, mouse_pos)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        elif event.type == 'P' and event.value == 'PRESS':
            # Pキーで掴んでいる頂点（またはカーソル下の頂点）をピン留め／解除（トグル）
            _sg = self._brush_tools.get('GRAB')
            target_v_idx = _sg.grabbed_vert if _sg is not None else None
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
                    if _sg is None or _sg.grabbed_vert != target_v_idx:
                        sim.release_pin(target_v_idx)
                    self.report({'INFO'}, i18n.trans("Unpinned vertex #%d") % target_v_idx)
                else:
                    # ピン留め (Pin)
                    vg.add([target_v_idx], 1.0, 'REPLACE')
                    self._pinned_verts.add(target_v_idx)
                    # 現在のローカル目標位置（または頂点現在位置）で固定
                    _grab_cur = _sg.grab_current_target_local if _sg is not None else None
                    if _sg is not None and _sg.grabbed_vert == target_v_idx and _grab_cur is not None:
                        t_pos = _grab_cur
                    else:
                        p = coords.reshape((-1, 3))[target_v_idx]
                        t_pos = mathutils.Vector((p[0], p[1], p[2]))
                    sim.set_pin(target_v_idx, [t_pos.x, t_pos.y, t_pos.z], 1.0)
                    self.report({'INFO'}, i18n.trans("Pinned vertex #%d") % target_v_idx)

                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            # ラジアル調整中は開始値へ復元してモード終了 (モーダル自体は継続)
            if self._radial is not None and self._radial.active:
                try:
                    mouse_pos = (event.mouse_region_x, event.mouse_region_y)
                    ctx = self._brush_ctx(context, obj, sim, coords, event, mouse_pos)
                    self._radial.cancel(ctx)
                except Exception:
                    pass
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            self.cancel(context)
            return {'FINISHED'}

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        global _interactive_running, _interactive_operator_instance
        # トグル動作: すでに実行中の場合は停止を要求
        if _interactive_running:
            if _interactive_operator_instance is not None:
                _interactive_operator_instance._stop_requested = True
            return {'FINISHED'}
        # ヘッドレス/バックグラウンド時は同期 execute に委譲
        try:
            if getattr(getattr(bpy, "app", None), "background", False):
                return self.execute(context)
        except Exception:
            pass
        self._reset_run_state()
        self._begin_init_ui(context)
        wm = context.window_manager
        try:
            self._timer = wm.event_timer_add(0.001, window=context.window)
        except Exception:
            self._timer = None
            return self.execute(context)
        try:
            wm.modal_handler_add(self)
            self._modal_handler_added = True
        except Exception:
            pass
        _interactive_running = True
        _interactive_operator_instance = self
        return {'RUNNING_MODAL'}

    def execute(self, context):
        global _interactive_running, _interactive_operator_instance
        # トグル動作: すでに実行中の場合は停止を要求
        if _interactive_running:
            if _interactive_operator_instance is not None:
                _interactive_operator_instance._stop_requested = True
            return {'FINISHED'}

        obj = context.active_object
        self._reset_run_state()
        # 同期パス（バックグラウンド/テスト用）: 全段階を一括実行
        try:
            while self._init_stage is not None and self._init_stage < self._init_progress_total():
                self._update_init_progress(context)
                has_more = self._run_init_stage(context)
                if not has_more:
                    break
        except Exception as e:
            logger.error(f"[Interactive] Init failed: {e}", exc_info=True)
            self._end_init_ui(context, success=False)
            self._init_stage = None
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self._init_stage = None
        self._end_init_ui(context, success=True)
        self._start_sim_loop(context)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        global _interactive_running, _interactive_operator_instance, _last_benchmark_summary
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None

        drawing.clear_active_grabbed_vertex()
        drawing.clear_interactive_fps_info()
        drawing.clear_init_overlay_info()
        drawing.set_interactive_active(False)
        self._fps_counter = None
        self._init_stage = None
        self._modal_handler_added = False
        # 準備UIが開始済みの場合のみ終了処理を行う（二重終了の不整合を防止）
        if self._progress_active:
            try:
                wm.progress_end()
            except Exception:
                pass
            self._progress_active = False
        if self._cursor_wait_set:
            try:
                cursor_restore = getattr(wm, "cursor_modal_restore", None)
                if callable(cursor_restore):
                    cursor_restore()
            except Exception:
                pass
            self._cursor_wait_set = False

        try:
            obj = context.active_object
            sim = _simulators[obj.name][0] if obj is not None and obj.name in _simulators else None
            if sim is not None:
                for tool in self._brush_tools.values():
                    try:
                        tool.abort(sim, self._pinned_verts)
                    except Exception:
                        pass
        except Exception:
            pass
        self._brush_dragging = None
        self._radial = RadialController()

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
