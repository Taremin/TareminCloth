import bpy
import logging

from ..engine import (
    bake_init_simulation,
    bake_step_frame,
    bake_finalize_simulation,
    bake_cloth_simulation,
    free_scene_bake,
    free_object_bake,
    is_scene_baked,
    is_object_baked,
)
from ..engine import runner
from ..utils.drawing import set_bake_overlay_info, clear_bake_overlay_info
from ..utils.view3d import tag_redraw_view3d
from .. import i18n

logger = logging.getLogger("taremin_cloth")


class TAREMIN_CLOTH_OT_bake(bpy.types.Operator):
    """シーン内の布シミュレーションを開始フレームから終了フレームまで一括計算してベイクする"""
    bl_idname = "taremin_cloth.bake"
    bl_label = "Bake Simulation"
    bl_description = "Bake cloth simulation for the entire timeline frame range"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER'}

    _timer = None
    bake_ctx = None
    current_frame = 0
    end_frame = 0
    total_steps = 0

    @classmethod
    def poll(cls, context):
        scene = context.scene
        if not scene:
            return False
        return any(
            getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth and getattr(obj.taremin_cloth, "enabled", True)
            for obj in scene.objects
        )

    def cleanup(self, context):
        """モーダルベイクの終了・クリーンアップ処理"""
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None

        clear_bake_overlay_info()
        runner._is_baking = False
        bake_finalize_simulation(self.bake_ctx)

        try:
            wm.progress_end()
        except Exception:
            pass

        if hasattr(context, "workspace") and context.workspace:
            context.workspace.status_text_set(None)

        tag_redraw_view3d(context, extra_area_types=('TIMELINE', 'DOPESHEET_EDITOR'))

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            logger.info(f"[Bake] Cancelled by user at frame {self.current_frame}")
            self.cleanup(context)
            self.report({'WARNING'}, "シミュレーションのベイクを中断しました")
            return {'CANCELLED'}

        if event.type == 'TIMER':
            if self.current_frame <= self.end_frame:
                try:
                    bake_step_frame(self.bake_ctx, self.current_frame)
                    baked_cnt = self.bake_ctx["baked_count"]
                    wm = context.window_manager
                    wm.progress_update(baked_cnt)

                    pct = int(min(1.0, baked_cnt / max(1, self.total_steps + 1)) * 100)
                    set_bake_overlay_info(self.current_frame, self.end_frame, pct)
                    status_str = f"Taremin Cloth: Baking frame {self.current_frame}/{self.end_frame} ({pct}%) - ESC to Cancel"
                    if hasattr(context, "workspace") and context.workspace:
                        context.workspace.status_text_set(status_str)

                    tag_redraw_view3d(context, extra_area_types=('TIMELINE',))

                    self.current_frame += 1
                except Exception as e:
                    logger.error(f"[Bake] Error during bake step at frame {self.current_frame}: {e}", exc_info=True)
                    self.cleanup(context)
                    self.report({'ERROR'}, f"ベイク実行中にエラーが発生しました: {e}")
                    return {'CANCELLED'}
            else:
                # ベイク完了
                baked_total = self.bake_ctx["baked_count"]
                self.cleanup(context)
                self.report({'INFO'}, f"シミュレーションのベイクが完了しました ({baked_total} フレーム)")
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        # ヘッドレスモード（CLI/テスト）時は同期的に execute を実行
        if getattr(bpy.app, "background", False):
            return self.execute(context)

        scene = context.scene
        wm = context.window_manager

        self.bake_ctx = bake_init_simulation(scene, start_frame=scene.frame_start, end_frame=scene.frame_end)
        if not self.bake_ctx:
            self.report({'WARNING'}, "有効な布オブジェクトが存在しません")
            return {'CANCELLED'}

        start_f = self.bake_ctx["start_frame"]
        self.end_frame = self.bake_ctx["end_frame"]
        self.total_steps = self.bake_ctx["total_steps"]
        self.current_frame = start_f + 1

        runner._is_baking = True
        set_bake_overlay_info(start_f, self.end_frame, 0)
        wm.progress_begin(0, self.total_steps + 1)
        wm.progress_update(1)

        self._timer = wm.event_timer_add(0.001, window=context.window)
        wm.modal_handler_add(self)

        logger.info(f"[Bake] Modal bake started: frames {start_f} to {self.end_frame} (total {self.total_steps + 1}f)")
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene = context.scene
        wm = context.window_manager

        start_f = scene.frame_start
        end_f = scene.frame_end
        total_steps = max(1, end_f - start_f + 1)

        wm.progress_begin(0, total_steps)

        def progress_cb(current_step, total):
            wm.progress_update(current_step)

        try:
            logger.info(f"[Bake] Synchronous bake triggered: frames {start_f} to {end_f}")
            baked = bake_cloth_simulation(
                scene,
                start_frame=start_f,
                end_frame=end_f,
                progress_callback=progress_cb,
            )
            msg = f"シミュレーションのベイクが完了しました ({baked} フレーム)"
            self.report({'INFO'}, msg)
        except Exception as e:
            err_msg = f"ベイク実行中にエラーが発生しました: {e}"
            logger.error(f"[Bake] {err_msg}", exc_info=True)
            self.report({'ERROR'}, err_msg)
            return {'CANCELLED'}
        finally:
            wm.progress_end()

        tag_redraw_view3d(context, extra_area_types=('TIMELINE', 'DOPESHEET_EDITOR'))

        return {'FINISHED'}


class TAREMIN_CLOTH_OT_free_bake(bpy.types.Operator):
    """ベイク済みシミュレーションキャッシュを破棄し、初期レスト形状に復元する"""
    bl_idname = "taremin_cloth.free_bake"
    bl_label = "Free Bake"
    bl_description = "Discard simulation bake cache and restore mesh to initial rest shape"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        if not scene:
            return False
        return is_scene_baked(scene) or any(
            getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth
            for obj in scene.objects
        )

    def execute(self, context):
        scene = context.scene
        if not scene:
            return {'CANCELLED'}

        free_scene_bake(scene)

        # タイムラインを開始フレームに巻き戻す
        scene.frame_set(scene.frame_start)

        tag_redraw_view3d(context, extra_area_types=('TIMELINE', 'DOPESHEET_EDITOR'))

        self.report({'INFO'}, "シミュレーションベイクを破棄し、初期形状に復元しました")
        return {'FINISHED'}


classes = (
    TAREMIN_CLOTH_OT_bake,
    TAREMIN_CLOTH_OT_free_bake,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
