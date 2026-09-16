"""
taremin_cloth GUI 連携オペレーター
Taremin Cloth GUI の自動起動、接続、非同期プレビュー更新、および確定形状適用を提供する。
"""

import os
import subprocess
import time
from typing import Optional
import bpy
import numpy as np

from ..engine.gui_client import get_gui_client
from ..engine.cache import cache_rest_positions
from ..utils.logger import logger
from .. import i18n

_gui_process = None
_gui_preview_running = False
_gui_preview_instance = None
_last_blender_preview_fps = 0.0
_last_gui_physics_fps = 0.0


def is_gui_preview_running() -> bool:
    return _gui_preview_running


def get_gui_fps_stats():
    return _last_blender_preview_fps, _last_gui_physics_fps


def resolve_gui_binary_path() -> Optional[str]:
    """taremin_cloth_gui.exe のパスを解決する"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # ops -> taremin_cloth -> python -> addon_root
    addon_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

    candidates = [
        os.path.join(addon_root, "target", "release", "taremin_cloth_gui.exe"),
        os.path.join(addon_root, "target", "debug", "taremin_cloth_gui.exe"),
        os.path.join(addon_root, "bin", "taremin_cloth_gui.exe"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


class TAREMIN_CLOTH_OT_launch_gui(bpy.types.Operator):
    """Launch Taremin Cloth GUI, initialize selected object mesh and connect"""
    bl_idname = "taremin_cloth.launch_gui"
    bl_label = "Launch Taremin Cloth GUI"
    bl_description = "Launch an independent high-speed GUI process and start asynchronous real-time sync with Blender"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj
            and obj.type == 'MESH'
            and getattr(obj, "taremin_cloth", None)
            and obj.taremin_cloth.is_cloth
        )

    def execute(self, context):
        global _gui_process
        obj = context.active_object
        if obj and obj.type == 'MESH':
            cache_rest_positions(obj, force=True)

        exe_path = resolve_gui_binary_path()
        if not exe_path:
            self.report({'ERROR'}, "taremin_cloth_gui.exe が見つかりません。cargo build --release -p cloth_gui を実行してください。")
            return {'CANCELLED'}

        client = get_gui_client()

        # プロセスが未起動なら起動
        if _gui_process is None or _gui_process.poll() is not None:
            logger.info(f"[GuiOp] Spawning GUI process: {exe_path}")
            try:
                _gui_process = subprocess.Popen([exe_path, "--port", "9055"])
                # サーバーのバインド待機
                time.sleep(0.4)
            except Exception as e:
                self.report({'ERROR'}, f"GUIプロセスの起動に失敗しました: {e}")
                return {'CANCELLED'}

        # 接続
        if not client.connect(timeout=2.0):
            # 少し待って再試行
            time.sleep(0.5)
            if not client.connect(timeout=2.0):
                self.report({'ERROR'}, "Taremin Cloth GUI へのソケット接続に失敗しました。")
                return {'CANCELLED'}

        # シーンメッシュ送信
        if not client.send_init_scene(obj):
            self.report({'ERROR'}, "メッシュデータの初期化送信に失敗しました。")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Taremin Cloth GUI 起動・初期化完了: '{obj.name}'")

        # 自動的に非同期プレビューを開始 (ヘッドレス時はUIイベントループがないためスキップ)
        if not bpy.app.background and not is_gui_preview_running():
            bpy.ops.taremin_cloth.gui_preview('INVOKE_DEFAULT')

        return {'FINISHED'}


class TAREMIN_CLOTH_OT_connect_gui(bpy.types.Operator):
    """起動済みの Taremin Cloth GUI に接続してメッシュを送信する"""
    bl_idname = "taremin_cloth.connect_gui"
    bl_label = "Connect to GUI"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(obj and obj.type == 'MESH' and getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth)

    def execute(self, context):
        obj = context.active_object
        if obj and obj.type == 'MESH':
            cache_rest_positions(obj, force=True)

        client = get_gui_client()

        if not client.connect(timeout=1.0):
            self.report({'WARNING'}, "GUIサーバーが見つかりません。先に Launch GUI を実行してください。")
            return {'CANCELLED'}

        if not client.send_init_scene(obj):
            self.report({'ERROR'}, "メッシュデータの初期化送信に失敗しました。")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Connected and initialized '{obj.name}'")

        if not is_gui_preview_running():
            bpy.ops.taremin_cloth.gui_preview('INVOKE_DEFAULT')

        return {'FINISHED'}


class TAREMIN_CLOTH_OT_gui_preview(bpy.types.Operator):
    """Taremin Cloth GUI から最新フレームを非同期に受信してプレビュー表示するモーダルオペレーター"""
    bl_idname = "taremin_cloth.gui_preview"
    bl_label = "GUI Live Preview"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER'}

    _timer = None
    _stop_requested = False
    _last_rendered_seq = 0
    _frame_times = []
    _last_frame_time = 0.0

    def modal(self, context, event):
        global _gui_preview_running, _last_blender_preview_fps, _last_gui_physics_fps

        if self._stop_requested:
            self.cancel(context)
            return {'FINISHED'}

        if event.type in {'ESC'}:
            self.cancel(context)
            self.report({'INFO'}, "Live Preview 停止")
            return {'FINISHED'}

        if event.type == 'TIMER':
            client = get_gui_client()
            if not client.is_connected:
                self.cancel(context)
                return {'CANCELLED'}

            obj = context.active_object
            if not obj or not getattr(obj, "taremin_cloth", None) or not obj.taremin_cloth.is_cloth:
                self.cancel(context)
                return {'CANCELLED'}

            # 最新フレームのノンブロッキング取得
            res = client.get_latest_coords()
            if res is not None:
                seq, flat_coords, physics_fps, step_ms = res
                _last_gui_physics_fps = physics_fps

                if seq > self._last_rendered_seq:
                    self._last_rendered_seq = seq

                    # Blenderメッシュ頂点更新 (Blenderの余力がある時だけ反映)
                    try:
                        obj.data.vertices.foreach_set("co", flat_coords)
                        obj.data.update()
                        obj["_taremin_cloth_is_deformed"] = True

                        now = time.perf_counter()
                        if self._last_frame_time > 0.0:
                            dt = now - self._last_frame_time
                            if dt > 0.0:
                                instant_fps = 1.0 / dt
                                _last_blender_preview_fps = 0.9 * _last_blender_preview_fps + 0.1 * instant_fps
                        self._last_frame_time = now

                        # ビューポート再描画要求
                        if context.screen:
                            for a in context.screen.areas:
                                if a.type == 'VIEW_3D':
                                    a.tag_redraw()
                    except Exception as e:
                        logger.warning(f"[GuiPreview] Mesh update error: {e}")

            # コライダー・ボーン・伸縮グループ・物理パラメータの定期同期 (約0.5秒に1回)
            self._sync_count = getattr(self, "_sync_count", 0) + 1
            if self._sync_count % 8 == 0:
                client.send_update_colliders(context.scene)
                client.send_update_bone_transforms(context.scene)
                client.send_update_elastic_scales(obj)
                client.send_update_pins(obj, context.scene)
                client.send_update_params_from_settings(obj, context.scene)

            return {'PASS_THROUGH'}


        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        global _gui_preview_running, _gui_preview_instance
        if _gui_preview_running:
            self.report({'INFO'}, "既に Live Preview が実行中です")
            return {'CANCELLED'}

        client = get_gui_client()
        if not client.is_connected:
            self.report({'WARNING'}, "GUIサーバーに接続されていません")
            return {'CANCELLED'}

        self._stop_requested = False
        self._last_rendered_seq = 0
        self._last_frame_time = time.perf_counter()

        # 約 60 FPS (16ms) 刻みでタイマーをセット
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.016, window=context.window)
        wm.modal_handler_add(self)

        _gui_preview_running = True
        _gui_preview_instance = self
        logger.info("[GuiOp] GUI Live Preview Modal Operator started")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        global _gui_preview_running, _gui_preview_instance
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _gui_preview_running = False
        _gui_preview_instance = None
        logger.info("[GuiOp] GUI Live Preview Modal Operator stopped")


class TAREMIN_CLOTH_OT_stop_gui_preview(bpy.types.Operator):
    """実行中の GUI Live Preview を停止する"""
    bl_idname = "taremin_cloth.stop_gui_preview"
    bl_label = "Stop Preview"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER'}

    def execute(self, context):
        global _gui_preview_instance
        if _gui_preview_instance is not None:
            _gui_preview_instance._stop_requested = True
        self.report({'INFO'}, "GUI Live Preview 停止要求")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_apply_gui_pose(bpy.types.Operator):
    """Bake current deformed shape from Taremin Cloth GUI as new rest shape for Blender mesh"""
    bl_idname = "taremin_cloth.apply_gui_pose"
    bl_label = "Apply GUI Pose"
    bl_description = "Apply the current deformation from the Taremin Cloth GUI to the Blender mesh"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(obj and obj.type == 'MESH' and getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth)

    def execute(self, context):
        obj = context.active_object
        client = get_gui_client()

        if not client.is_connected:
            self.report({'WARNING'}, "GUIサーバーに接続されていません")
            return {'CANCELLED'}

        pose = client.apply_pose()
        if pose is None:
            self.report({'ERROR'}, "ポーズの取得に失敗しました")
            return {'CANCELLED'}

        obj.data.vertices.foreach_set("co", pose)
        obj.data.update()
        cache_rest_positions(obj, force=True)
        obj["_taremin_cloth_is_deformed"] = False

        if context.screen:
            for a in context.screen.areas:
                if a.type == 'VIEW_3D':
                    a.tag_redraw()

        self.report({'INFO'}, f"GUIの形状を '{obj.name}' に確定適用しました")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_sync_gui_colliders(bpy.types.Operator):
    """Sync scene colliders (spheres, capsules, planes, meshes) to Taremin Cloth GUI"""
    bl_idname = "taremin_cloth.sync_gui_colliders"
    bl_label = "Sync Colliders"
    bl_description = "Sync scene colliders (spheres, capsules, planes, meshes) to the Taremin Cloth GUI"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER'}

    def execute(self, context):
        client = get_gui_client()
        if not client.is_connected:
            self.report({'WARNING'}, "GUIサーバーに接続されていません")
            return {'CANCELLED'}

        success = client.send_update_colliders(context.scene)
        if success:
            self.report({'INFO'}, "コライダーを Taremin Cloth GUI に同期しました")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "コライダーの同期に失敗しました")
            return {'CANCELLED'}

