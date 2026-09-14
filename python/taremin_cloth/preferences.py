import json
from pathlib import Path
import bpy
from bpy.props import EnumProperty, StringProperty, BoolProperty, IntProperty

from .utils.logger import logger, set_log_level

# デバイス列挙キャッシュ（動的EnumPropertyのGC保護用）
_device_items_cache = []

# 復元中フラグ（復元時の値代入に伴う不要な再保存を抑制）
_is_restoring = False
_has_restored_preferences = False
_restore_retry_count = 0


def get_preferences_filepath() -> Path:
    """設定ファイルの保存先パスを取得"""
    try:
        config_dir = bpy.utils.user_resource('CONFIG', path="taremin_cloth", create=True)
        if config_dir:
            return Path(config_dir) / "preferences.json"
    except Exception:
        pass
    return Path(__file__).parent / "preferences.json"


def save_preferences_to_disk(prefs=None) -> bool:
    """現在の設定（log_level, gpu_backend, gpu_device）をJSONファイルに保存する"""
    global _is_restoring
    if _is_restoring:
        return False
    if prefs is None:
        prefs = get_preferences()
    if prefs is None:
        return False

    data = {
        "log_level": getattr(prefs, "log_level", "INFO"),
        "gpu_backend": getattr(prefs, "gpu_backend", "AUTO"),
        "gpu_device": getattr(prefs, "gpu_device", "AUTO"),
        "enable_debug_recording": getattr(prefs, "enable_debug_recording", True),
        "debug_output_dir": getattr(prefs, "debug_output_dir", ""),
        "debug_filename_template": getattr(prefs, "debug_filename_template", "cloth_debug_{datetime}_{object}.{ext}"),
        "debug_max_frames": getattr(prefs, "debug_max_frames", 3600),
        "enable_standalone_gui": getattr(prefs, "enable_standalone_gui", False),
    }
    filepath = get_preferences_filepath()
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.warning(f"アドオン設定の保存に失敗しました ({filepath}): {e}")
        return False


def load_preferences_from_disk() -> dict:
    """JSONファイルから保存された設定を読み込む"""
    filepath = get_preferences_filepath()
    if not filepath.is_file():
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as e:
        logger.warning(f"アドオン設定の読み込みに失敗しました ({filepath}): {e}")
    return {}


def apply_saved_preferences(prefs=None) -> bool:
    """保存された設定をプレファレンスに適用する"""
    global _is_restoring
    if prefs is None:
        prefs = get_preferences()
    if prefs is None:
        return False

    data = load_preferences_from_disk()
    if not data:
        # 設定ファイルが存在しない場合でも、現在のログレベルとGPU状態を同期
        set_log_level(prefs.log_level)
        refresh_active_gpu_info(prefs)
        return False

    _is_restoring = True
    try:
        if "log_level" in data and hasattr(prefs, "log_level"):
            prefs.log_level = data["log_level"]
        if "gpu_backend" in data and hasattr(prefs, "gpu_backend"):
            prefs.gpu_backend = data["gpu_backend"]
        if "gpu_device" in data and hasattr(prefs, "gpu_device"):
            prefs.gpu_device = data["gpu_device"]
        if "enable_debug_recording" in data and hasattr(prefs, "enable_debug_recording"):
            prefs.enable_debug_recording = data["enable_debug_recording"]
        if "debug_output_dir" in data and hasattr(prefs, "debug_output_dir"):
            prefs.debug_output_dir = data["debug_output_dir"]
        if "debug_filename_template" in data and hasattr(prefs, "debug_filename_template"):
            prefs.debug_filename_template = data["debug_filename_template"]
        if "debug_max_frames" in data and hasattr(prefs, "debug_max_frames"):
            prefs.debug_max_frames = data["debug_max_frames"]
        if "enable_standalone_gui" in data and hasattr(prefs, "enable_standalone_gui"):
            prefs.enable_standalone_gui = bool(data["enable_standalone_gui"])
    finally:
        _is_restoring = False

    # ログレベル適用 & GPU状態反映
    set_log_level(prefs.log_level)
    refresh_active_gpu_info(prefs)
    return True


def _on_log_level_update(self, context):
    set_log_level(self.log_level)
    logger.info(f"Log level changed to: {self.log_level}")
    save_preferences_to_disk(self)


def _on_gpu_backend_update(self, context):
    """Backend変更時にDeviceをAUTOにリセットして安全性を担保"""
    self.gpu_device = 'AUTO'
    save_preferences_to_disk(self)


def _on_gpu_device_update(self, context):
    """Device変更時に保存"""
    save_preferences_to_disk(self)


def _on_debug_pref_update(self, context):
    """デバッグ設定変更時に保存"""
    save_preferences_to_disk(self)


def _on_standalone_gui_update(self, context):
    save_preferences_to_disk(self)


def _get_gpu_device_items(self, context):
    """選択中のBackendに対応する物理GPUデバイス一覧を動的生成する（案A: 純粋な物理GPU名のみ）"""
    global _device_items_cache
    items = [
        ('AUTO', "Auto (High Performance)", "最適な高パフォーマンスGPUを自動選択", 'AUTO', 0),
    ]
    try:
        import taremin_cloth_core
        backend_filter = None if self.gpu_backend == 'AUTO' else self.gpu_backend.lower()
        devs = taremin_cloth_core.get_available_gpu_devices(backend_filter)

        # 同名GPUがある場合のカウント用
        name_counts = {}
        for d in devs:
            n = d.get('name', 'Unknown')
            name_counts[n] = name_counts.get(n, 0) + 1

        seen_counts = {}
        for i, d in enumerate(devs, start=1):
            name = d.get('name', 'Unknown')
            dev_type = d.get('device_type', '')
            identifier = str(d.get('index', i - 1))

            # 同名GPUが複数台ある場合のみサフィックス (#1, #2) を付与
            if name_counts.get(name, 0) > 1:
                idx_num = seen_counts.get(name, 0) + 1
                seen_counts[name] = idx_num
                label = f"{name} #{idx_num}"
            else:
                label = name

            desc = f"{name} ({dev_type})"
            icon = 'DESKTOP' if dev_type == 'DiscreteGPU' else 'COMMUNITY'
            items.append((identifier, label, desc, icon, i))
    except Exception as e:
        logger.warning(f"GPUデバイス一覧取得エラー: {e}")

    _device_items_cache = items
    return _device_items_cache


def apply_gpu_settings_from_prefs(prefs):
    """プレファレンスの設定値に基づいてGPUコンテキストを再初期化する"""
    import taremin_cloth_core
    backend = None if prefs.gpu_backend == 'AUTO' else prefs.gpu_backend.lower()
    device_idx = None if prefs.gpu_device == 'AUTO' else int(prefs.gpu_device)

    taremin_cloth_core.set_gpu_device(backend, device_idx)
    refresh_active_gpu_info(prefs)


def refresh_active_gpu_info(prefs):
    """現在アクティブなGPUデバイス情報をプレファレンスに反映する"""
    try:
        import taremin_cloth_core
        info = taremin_cloth_core.get_current_gpu_device()
        prefs.active_device_name = info.get('name', 'Unknown')
        prefs.active_backend_name = info.get('backend', 'Unknown')
    except Exception as e:
        logger.warning(f"現在のアクティブGPU情報取得エラー: {e}")


class TareminClothPreferences(bpy.types.AddonPreferences):
    """Taremin Cloth アドオン設定"""
    bl_idname = __package__.split('.')[0] if __package__ else "taremin_cloth"

    log_level: EnumProperty(
        name="Log Level",
        description="システムコンソールに出力するログの詳細度",
        items=[
            ('DEBUG', "DEBUG", "詳細なデバッグログ（頂点数、キャッシュ判定、ステップ情報など）"),
            ('INFO', "INFO", "通常の情報ログ（シミュレーション開始・停止、プリセット適用など）"),
            ('WARNING', "WARNING", "警告メッセージのみ"),
            ('ERROR', "ERROR", "エラーメッセージのみ"),
        ],
        default='INFO',
        update=_on_log_level_update,
    )

    gpu_backend: EnumProperty(
        name="GPU Backend",
        description="GPU計算バックエンドの選択",
        items=[
            ('AUTO', "Auto (推奨)", "OSに最適なバックエンドを自動選択（WindowsはDirectX 12優先）"),
            ('DX12', "DirectX 12", "Windows標準のDirectX 12（Blenderとの競合が少なく最も安定）"),
            ('VULKAN', "Vulkan", "クロスプラットフォーム対応の高速API Vulkan"),
        ],
        default='AUTO',
        update=_on_gpu_backend_update,
    )

    gpu_device: EnumProperty(
        name="GPU Device",
        description="シミュレーション計算に使用するGPUデバイス",
        items=_get_gpu_device_items,
        update=_on_gpu_device_update,
    )

    active_device_name: StringProperty(
        name="Active Device",
        default="Unknown",
    )

    active_backend_name: StringProperty(
        name="Active Backend",
        default="Unknown",
    )

    enable_debug_recording: BoolProperty(
        name="Record Simulation Debug States",
        description="シミュレーション（インタラクティブモード）中の各フレーム状態をRust側で記録し、終了時にファイル出力する（※Console Log LevelがDEBUGの時に実行されます）",
        default=True,
        update=_on_debug_pref_update,
    )

    debug_output_dir: StringProperty(
        name="Output Directory",
        description="デバッグ記録ファイルの保存先ディレクトリ（空欄時はアドオン内の frame_logs フォルダを使用）",
        default="",
        subtype='DIR_PATH',
        update=_on_debug_pref_update,
    )

    debug_filename_template: StringProperty(
        name="Filename Template",
        description="デバッグファイル名テンプレート。利用可能な変数: {date}, {time}, {datetime}, {object}, {frames}, {ext}",
        default="cloth_debug_{datetime}_{object}.{ext}",
        update=_on_debug_pref_update,
    )

    debug_max_frames: IntProperty(
        name="Max Recording Frames",
        description="メモリ保護のための最大記録フレーム数（3600フレーム = 60FPSで約1分間）",
        default=3600,
        min=60,
        max=100000,
        update=_on_debug_pref_update,
    )

    enable_standalone_gui: BoolProperty(
        name="Enable Standalone GUI Client",
        description="外部プロセスの独立GPUシミュレーションGUIクライアントを有効化します（実験的機能）",
        default=False,
        update=_on_standalone_gui_update,
    )

    def draw(self, context):
        global _has_restored_preferences
        if not _has_restored_preferences:
            apply_saved_preferences(self)
            _has_restored_preferences = True

        layout = self.layout

        # GPU Settings
        box_gpu = layout.box()
        box_gpu.label(text="GPU & Hardware Settings", icon='PREFERENCES')
        col_gpu = box_gpu.column(align=True)
        col_gpu.prop(self, "gpu_backend", text="Backend")
        col_gpu.prop(self, "gpu_device", text="Device")

        row_act = box_gpu.row(align=True)
        row_act.label(
            text=f"Active: {self.active_device_name} ({self.active_backend_name})",
            icon='CHECKMARK' if self.active_device_name != "Unknown" else 'INFO',
        )
        box_gpu.operator("taremin_cloth.apply_gpu_settings", text="Apply GPU Settings", icon='FILE_REFRESH')

        # Logging & Diagnostics
        box_log = layout.box()
        box_log.label(text="Logging & Diagnostics", icon='CONSOLE')
        row_log = box_log.row()
        row_log.prop(self, "log_level", text="Console Log Level")

        # Simulation Debug State Recording
        box_debug = box_log.box()
        box_debug.label(text="Simulation Frame Debug Recorder", icon='FILE_CACHE')
        box_debug.prop(self, "enable_debug_recording", text="Enable Debug Recording (when DEBUG level)")

        col_debug = box_debug.column()
        col_debug.enabled = self.enable_debug_recording
        col_debug.prop(self, "debug_output_dir", text="Output Folder")
        col_debug.prop(self, "debug_filename_template", text="Template")
        col_debug.prop(self, "debug_max_frames", text="Max Frames")

        row_prev = col_debug.row()
        row_prev.alignment = 'RIGHT'
        row_prev.label(text="Format: gzip JSON Lines (.jsonl.gz)", icon='INFO')

        # Experimental Features
        box_exp = layout.box()
        box_exp.label(text="Experimental Features", icon='EXPERIMENTAL')
        box_exp.prop(self, "enable_standalone_gui", text="Enable Standalone GUI Client")



def get_preferences(context=None):
    """アドオンのプレファレンスインスタンスを取得する"""
    if context is None:
        context = getattr(bpy, "context", None)
    if not context or not hasattr(context, "preferences") or not context.preferences:
        return None
    addon_name = __package__.split('.')[0] if __package__ else "taremin_cloth"
    addon = context.preferences.addons.get(addon_name)
    if addon:
        return addon.preferences
    return None


def _deferred_restore_preferences():
    """アドオン有効化直後のタイマーコールバック用復元処理"""
    global _has_restored_preferences, _restore_retry_count
    prefs = get_preferences()
    if prefs is not None:
        apply_saved_preferences(prefs)
        _has_restored_preferences = True
        return None
    _restore_retry_count += 1
    if _restore_retry_count > 10:
        return None
    return 0.05


classes = (
    TareminClothPreferences,
)


def register():
    global _has_restored_preferences, _restore_retry_count
    _has_restored_preferences = False
    _restore_retry_count = 0

    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            pass

    # 初期設定の復元および適用
    prefs = get_preferences()
    if prefs:
        apply_saved_preferences(prefs)
        _has_restored_preferences = True
    else:
        # アドオン有効化の過渡状態でまだ取得できない場合はタイマーで遅延適用
        try:
            bpy.app.timers.register(_deferred_restore_preferences, first_interval=0.0)
        except Exception:
            pass


def unregister():
    # 登録解除直前に最新設定をディスクに保存
    save_preferences_to_disk()

    # タイマーが残っていれば解除
    try:
        if bpy.app.timers.is_registered(_deferred_restore_preferences):
            bpy.app.timers.unregister(_deferred_restore_preferences)
    except Exception:
        pass

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
