import json
import re
from pathlib import Path
import bpy
from bpy.props import StringProperty, EnumProperty
from bpy.types import Operator, Menu

# -----------------------------------------------------------------------------
# ビルトインプリセット定義
# -----------------------------------------------------------------------------

BUILTIN_FABRIC_PRESETS = {
    "Silk (絹)": {
        "tension_stiffness": 800.0,
        "compression_stiffness": 50.0,
        "shear_stiffness": 50.0,
        "bending_stiffness": 1.0,
        "tension_damping": 2.0,
        "compression_damping": 2.0,
        "shear_damping": 2.0,
        "bending_damping": 0.2,
        "air_damping": 1.5,
        "thickness": 0.002,
    },
    "Cotton (木綿)": {
        "tension_stiffness": 1500.0,
        "compression_stiffness": 150.0,
        "shear_stiffness": 100.0,
        "bending_stiffness": 15.0,
        "tension_damping": 5.0,
        "compression_damping": 5.0,
        "shear_damping": 5.0,
        "bending_damping": 0.5,
        "air_damping": 1.0,
        "thickness": 0.005,
    },
    "Linen (麻)": {
        "tension_stiffness": 2000.0,
        "compression_stiffness": 300.0,
        "shear_stiffness": 200.0,
        "bending_stiffness": 40.0,
        "tension_damping": 8.0,
        "compression_damping": 8.0,
        "shear_damping": 8.0,
        "bending_damping": 1.0,
        "air_damping": 1.0,
        "thickness": 0.006,
    },
    "Denim (デニム)": {
        "tension_stiffness": 4000.0,
        "compression_stiffness": 800.0,
        "shear_stiffness": 600.0,
        "bending_stiffness": 80.0,
        "tension_damping": 10.0,
        "compression_damping": 10.0,
        "shear_damping": 10.0,
        "bending_damping": 2.0,
        "air_damping": 1.0,
        "thickness": 0.010,
    },
    "Leather (革)": {
        "tension_stiffness": 8000.0,
        "compression_stiffness": 2000.0,
        "shear_stiffness": 1500.0,
        "bending_stiffness": 200.0,
        "tension_damping": 15.0,
        "compression_damping": 15.0,
        "shear_damping": 15.0,
        "bending_damping": 5.0,
        "air_damping": 0.8,
        "thickness": 0.015,
    },
    "Rubber (ゴム)": {
        "tension_stiffness": 400.0,
        "compression_stiffness": 100.0,
        "shear_stiffness": 200.0,
        "bending_stiffness": 25.0,
        "tension_damping": 12.0,
        "compression_damping": 12.0,
        "shear_damping": 12.0,
        "bending_damping": 3.0,
        "air_damping": 0.5,
        "thickness": 0.008,
    },
}

BUILTIN_SIMULATION_PRESETS = {
    "Fast (Realtime)": {
        "substeps": 8,
        "min_substeps": 4,
        "enable_adaptive_substep": True,
        "solver_iterations": 1,
    },
    "Balanced (Default)": {
        "substeps": 20,
        "min_substeps": 4,
        "enable_adaptive_substep": False,
        "solver_iterations": 2,
    },
    "Best (High Res)": {
        "substeps": 40,
        "min_substeps": 8,
        "enable_adaptive_substep": False,
        "solver_iterations": 4,
    },
    "Stiff (Anti-Stretch)": {
        "substeps": 30,
        "min_substeps": 6,
        "enable_adaptive_substep": False,
        "solver_iterations": 6,
    },
}

BUILTIN_COLLIDER_PRESETS = {
    "Skin (Human)": {
        "friction": 0.6,
        "restitution": 0.0,
        "thickness": 0.005,
    },
    "Fabric (Cloth)": {
        "friction": 0.4,
        "restitution": 0.0,
        "thickness": 0.003,
    },
    "Smooth / Metal": {
        "friction": 0.1,
        "restitution": 0.05,
        "thickness": 0.002,
    },
    "Rubber (High Friction)": {
        "friction": 0.9,
        "restitution": 0.1,
        "thickness": 0.005,
    },
}

CATEGORY_MAP = {
    'fabric': {
        'builtin': BUILTIN_FABRIC_PRESETS,
        'target_prop': 'taremin_cloth',
        'subfolder': 'fabric',
        'display_name': 'Fabric',
    },
    'simulation': {
        'builtin': BUILTIN_SIMULATION_PRESETS,
        'target_prop': 'taremin_cloth',
        'subfolder': 'simulation',
        'display_name': 'Simulation',
    },
    'collider': {
        'builtin': BUILTIN_COLLIDER_PRESETS,
        'target_prop': 'taremin_cloth_collider',
        'subfolder': 'collider',
        'display_name': 'Collider',
    },
}

# -----------------------------------------------------------------------------
# ディレクトリ・ファイル管理
# -----------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """ファイル名として使用できない危険な文字を置換"""
    return re.sub(r'[\\/*?:"<>|]', '_', name.strip())


def get_preset_base_dir() -> Path:
    """プリセット保存用ベースディレクトリを取得"""
    try:
        user_presets = bpy.utils.user_resource('SCRIPTS', path="presets")
        if user_presets:
            p = Path(user_presets) / "taremin_cloth"
            p.mkdir(parents=True, exist_ok=True)
            return p
    except Exception:
        pass
    fallback = Path(__file__).parent / "presets"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def get_category_preset_dir(category: str) -> Path:
    """指定カテゴリのプリセットディレクトリを取得"""
    base = get_preset_base_dir()
    subfolder = CATEGORY_MAP.get(category, {}).get('subfolder', category)
    p = base / subfolder
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_custom_presets(category: str) -> dict[str, dict]:
    """保存されたカスタムプリセット辞書を取得"""
    cat_dir = get_category_preset_dir(category)
    presets = {}
    if not cat_dir.exists():
        return presets
    for json_file in sorted(cat_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                preset_name = data.get("name", json_file.stem)
                preset_values = data.get("values", {})
                presets[preset_name] = preset_values
        except Exception as e:
            print(f"[TareminCloth] プリセット読み込み失敗 ({json_file}): {e}")
    return presets


def get_all_presets(category: str) -> dict[str, dict]:
    """ビルトインとカスタムを統合したプリセット辞書を取得"""
    builtin = CATEGORY_MAP.get(category, {}).get('builtin', {})
    custom = get_custom_presets(category)
    result = dict(builtin)
    result.update(custom)
    return result


def extract_properties_for_category(obj: bpy.types.Object, category: str) -> dict:
    """オブジェクトからカテゴリに対応するプロパティ値を抽出"""
    info = CATEGORY_MAP.get(category)
    if not info:
        return {}
    target_attr = info['target_prop']
    settings = getattr(obj, target_attr, None)
    if not settings:
        return {}
    sample_builtin = next(iter(info['builtin'].values()))
    keys = sample_builtin.keys()
    data = {}
    for k in keys:
        if hasattr(settings, k):
            data[k] = getattr(settings, k)
    return data


def apply_preset_values(obj: bpy.types.Object, category: str, values: dict):
    """オブジェクトにプロパティ値を適用"""
    info = CATEGORY_MAP.get(category)
    if not info:
        return
    target_attr = info['target_prop']
    settings = getattr(obj, target_attr, None)
    if not settings:
        return
    for k, v in values.items():
        if hasattr(settings, k):
            try:
                setattr(settings, k, v)
            except Exception as e:
                print(f"[TareminCloth] プロパティ設定エラー ({k}={v}): {e}")

# -----------------------------------------------------------------------------
# オペレーター
# -----------------------------------------------------------------------------

class TAREMIN_CLOTH_OT_apply_preset(Operator):
    """指定したプリセットを適用します"""
    bl_idname = "taremin_cloth.apply_preset"
    bl_label = "Apply Preset"
    bl_description = "指定したプリセットを適用します"
    bl_options = {'REGISTER', 'UNDO'}

    category: EnumProperty(
        name="Category",
        items=[
            ('fabric', "Fabric", "布素材"),
            ('simulation', "Simulation", "シミュレーション品質"),
            ('collider', "Collider", "コライダー"),
        ],
    )
    preset_name: StringProperty(name="Preset Name")

    def execute(self, context):
        obj = context.active_object
        if not obj:
            self.report({'WARNING'}, "メッシュオブジェクトが選択されていません")
            return {'CANCELLED'}

        all_presets = get_all_presets(self.category)
        if self.preset_name not in all_presets:
            self.report({'ERROR'}, f"プリセット '{self.preset_name}' が見つかりません")
            return {'CANCELLED'}

        apply_preset_values(obj, self.category, all_presets[self.preset_name])

        # 直近の適用プリセット名を記録
        if self.category == 'fabric' and hasattr(obj, "taremin_cloth"):
            obj.taremin_cloth.last_fabric_preset = self.preset_name
        elif self.category == 'simulation' and hasattr(obj, "taremin_cloth"):
            obj.taremin_cloth.last_simulation_preset = self.preset_name
        elif self.category == 'collider' and hasattr(obj, "taremin_cloth_collider"):
            obj.taremin_cloth_collider.last_collider_preset = self.preset_name

        self.report({'INFO'}, f"プリセット '{self.preset_name}' を適用しました")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_save_preset(Operator):
    """現在のパラメータをカスタムプリセットとして保存します"""
    bl_idname = "taremin_cloth.save_preset"
    bl_label = "Save Preset"
    bl_description = "現在のパラメータをカスタムプリセットとして保存します"
    bl_options = {'REGISTER', 'UNDO'}

    category: EnumProperty(
        name="Category",
        items=[
            ('fabric', "Fabric", "布素材"),
            ('simulation', "Simulation", "シミュレーション品質"),
            ('collider', "Collider", "コライダー"),
        ],
    )
    preset_name: StringProperty(
        name="Preset Name",
        description="保存するプリセット名",
        default="My Preset",
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "preset_name")

    def execute(self, context):
        name = self.preset_name.strip()
        if not name:
            self.report({'ERROR'}, "プリセット名を入力してください")
            return {'CANCELLED'}

        builtin = CATEGORY_MAP.get(self.category, {}).get('builtin', {})
        if name in builtin:
            self.report({'ERROR'}, f"'{name}' はビルトインプリセットの名前と重複しています")
            return {'CANCELLED'}

        obj = context.active_object
        if not obj:
            self.report({'WARNING'}, "オブジェクトが選択されていません")
            return {'CANCELLED'}

        values = extract_properties_for_category(obj, self.category)
        if not values:
            self.report({'ERROR'}, "保存対象のパラメータが取得できませんでした")
            return {'CANCELLED'}

        cat_dir = get_category_preset_dir(self.category)
        safe_name = sanitize_filename(name)
        file_path = cat_dir / f"{safe_name}.json"

        data = {
            "name": name,
            "category": self.category,
            "values": values,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # 直近のプリセット名を更新
        if self.category == 'fabric' and hasattr(obj, "taremin_cloth"):
            obj.taremin_cloth.last_fabric_preset = name
        elif self.category == 'simulation' and hasattr(obj, "taremin_cloth"):
            obj.taremin_cloth.last_simulation_preset = name
        elif self.category == 'collider' and hasattr(obj, "taremin_cloth_collider"):
            obj.taremin_cloth_collider.last_collider_preset = name

        self.report({'INFO'}, f"プリセット '{name}' を保存しました")
        return {'FINISHED'}


def get_custom_preset_items(self, context):
    presets = get_custom_presets(self.category)
    if not presets:
        return [('NONE', "(カスタムプリセットなし)", "")]
    return [(name, name, "") for name in presets.keys()]


class TAREMIN_CLOTH_OT_delete_preset(Operator):
    """保存されたカスタムプリセットを削除します"""
    bl_idname = "taremin_cloth.delete_preset"
    bl_label = "Delete Preset"
    bl_description = "保存されたカスタムプリセットを削除します"
    bl_options = {'REGISTER', 'UNDO'}

    category: EnumProperty(
        name="Category",
        items=[
            ('fabric', "Fabric", "布素材"),
            ('simulation', "Simulation", "シミュレーション品質"),
            ('collider', "Collider", "コライダー"),
        ],
    )
    preset_name: EnumProperty(
        name="Preset to Delete",
        items=get_custom_preset_items,
    )

    def invoke(self, context, event):
        presets = get_custom_presets(self.category)
        if not presets:
            self.report({'INFO'}, "削除可能なカスタムプリセットがありません")
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "preset_name")

    def execute(self, context):
        target_name = self.preset_name
        if target_name == 'NONE' or not target_name:
            return {'CANCELLED'}

        cat_dir = get_category_preset_dir(self.category)
        safe_name = sanitize_filename(target_name)
        file_path = cat_dir / f"{safe_name}.json"
        if file_path.exists():
            file_path.unlink()

            # 削除対象がアクティブオブジェクトの直近プリセット名と一致していたらクリア
            obj = context.active_object
            if obj:
                if self.category == 'fabric' and hasattr(obj, "taremin_cloth") and obj.taremin_cloth.last_fabric_preset == target_name:
                    obj.taremin_cloth.last_fabric_preset = ""
                elif self.category == 'simulation' and hasattr(obj, "taremin_cloth") and obj.taremin_cloth.last_simulation_preset == target_name:
                    obj.taremin_cloth.last_simulation_preset = ""
                elif self.category == 'collider' and hasattr(obj, "taremin_cloth_collider") and obj.taremin_cloth_collider.last_collider_preset == target_name:
                    obj.taremin_cloth_collider.last_collider_preset = ""

            self.report({'INFO'}, f"プリセット '{target_name}' を削除しました")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, f"プリセットファイルが見つかりません: {file_path.name}")
            return {'CANCELLED'}

# -----------------------------------------------------------------------------
# メニュー (UI)
# -----------------------------------------------------------------------------

class TAREMIN_CLOTH_MT_fabric_presets(Menu):
    """布素材プリセットメニュー"""
    bl_label = "Fabric Presets"
    bl_idname = "TAREMIN_CLOTH_MT_fabric_presets"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Built-in Presets", icon='MATERIAL')
        for name in BUILTIN_FABRIC_PRESETS.keys():
            op = layout.operator("taremin_cloth.apply_preset", text=name)
            op.category = 'fabric'
            op.preset_name = name

        custom_presets = get_custom_presets('fabric')
        if custom_presets:
            layout.separator()
            layout.label(text="Custom Presets", icon='PRESET')
            for name in custom_presets.keys():
                op = layout.operator("taremin_cloth.apply_preset", text=name)
                op.category = 'fabric'
                op.preset_name = name


class TAREMIN_CLOTH_MT_simulation_presets(Menu):
    """シミュレーション品質プリセットメニュー"""
    bl_label = "Simulation Presets"
    bl_idname = "TAREMIN_CLOTH_MT_simulation_presets"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Built-in Presets", icon='SETTINGS')
        for name in BUILTIN_SIMULATION_PRESETS.keys():
            op = layout.operator("taremin_cloth.apply_preset", text=name)
            op.category = 'simulation'
            op.preset_name = name

        custom_presets = get_custom_presets('simulation')
        if custom_presets:
            layout.separator()
            layout.label(text="Custom Presets", icon='PRESET')
            for name in custom_presets.keys():
                op = layout.operator("taremin_cloth.apply_preset", text=name)
                op.category = 'simulation'
                op.preset_name = name


class TAREMIN_CLOTH_MT_collider_presets(Menu):
    """コライダープリセットメニュー"""
    bl_label = "Collider Presets"
    bl_idname = "TAREMIN_CLOTH_MT_collider_presets"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Built-in Presets", icon='PHYSICS')
        for name in BUILTIN_COLLIDER_PRESETS.keys():
            op = layout.operator("taremin_cloth.apply_preset", text=name)
            op.category = 'collider'
            op.preset_name = name

        custom_presets = get_custom_presets('collider')
        if custom_presets:
            layout.separator()
            layout.label(text="Custom Presets", icon='PRESET')
            for name in custom_presets.keys():
                op = layout.operator("taremin_cloth.apply_preset", text=name)
                op.category = 'collider'
                op.preset_name = name


BUILTIN_CONFIG_PRESETS = {
    "All Enabled (全て有効)": 'ALL_ON',
    "All Muted (全て一時停止)": 'ALL_OFF',
    "Cloths Only (布のみ有効)": 'CLOTHS_ONLY',
    "Colliders Only (コライダーのみ有効)": 'COLLIDERS_ONLY',
}


def get_scene_config_presets(scene) -> dict:
    """シーンに保存されたカスタム構成プリセット辞書を取得"""
    raw_json = getattr(scene, "taremin_cloth_config_presets_json", "{}")
    try:
        return json.loads(raw_json) if raw_json else {}
    except Exception:
        return {}


def save_scene_config_presets(scene, presets: dict):
    """シーンにカスタム構成プリセット辞書を保存"""
    scene.taremin_cloth_config_presets_json = json.dumps(presets, ensure_ascii=False)


class TAREMIN_CLOTH_OT_batch_simulation_state(Operator):
    """シーン内の布・コライダーのシミュレーション有効状態を一括操作します"""
    bl_idname = "taremin_cloth.batch_simulation_state"
    bl_label = "Batch Simulation State"
    bl_description = "シーン内の布・コライダーのシミュレーション計算有効状態を一括操作します"
    bl_options = {'REGISTER', 'UNDO'}

    action: EnumProperty(
        name="Action",
        items=[
            ('ALL_ON', "All On", "全ての布とコライダーを有効化"),
            ('ALL_OFF', "All Off", "全ての布とコライダーを一時停止（ミュート）"),
            ('SOLO', "Solo", "選択中のオブジェクトのみ有効化し、他を一時停止"),
            ('CLOTHS_ONLY', "Cloths Only", "布のみ有効化し、コライダーを一時停止"),
            ('COLLIDERS_ONLY', "Colliders Only", "コライダーのみ有効化し、布を一時停止"),
            ('INVERT', "Invert", "有効/無効状態を反転"),
        ],
    )

    def execute(self, context):
        scene = context.scene
        if not scene:
            return {'CANCELLED'}

        active_obj = context.active_object

        for obj in scene.objects:
            cloth_s = getattr(obj, "taremin_cloth", None)
            col_s = getattr(obj, "taremin_cloth_collider", None)

            if self.action == 'ALL_ON':
                if cloth_s and cloth_s.is_cloth:
                    cloth_s.enabled = True
                if col_s and col_s.is_collider:
                    col_s.enabled = True

            elif self.action == 'ALL_OFF':
                if cloth_s and cloth_s.is_cloth:
                    cloth_s.enabled = False
                if col_s and col_s.is_collider:
                    col_s.enabled = False

            elif self.action == 'SOLO':
                is_this_active = (obj == active_obj)
                if cloth_s and cloth_s.is_cloth:
                    cloth_s.enabled = is_this_active
                if col_s and col_s.is_collider:
                    col_s.enabled = is_this_active

            elif self.action == 'CLOTHS_ONLY':
                if cloth_s and cloth_s.is_cloth:
                    cloth_s.enabled = True
                if col_s and col_s.is_collider:
                    col_s.enabled = False

            elif self.action == 'COLLIDERS_ONLY':
                if cloth_s and cloth_s.is_cloth:
                    cloth_s.enabled = False
                if col_s and col_s.is_collider:
                    col_s.enabled = True

            elif self.action == 'INVERT':
                if cloth_s and cloth_s.is_cloth:
                    cloth_s.enabled = not cloth_s.enabled
                if col_s and col_s.is_collider:
                    col_s.enabled = not col_s.enabled

        scene.taremin_cloth_active_config_preset = ""
        self.report({'INFO'}, f"一括操作 '{self.action}' を実行しました")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_save_config_preset(Operator):
    """現在の布・コライダーのシミュレーション有効状態をプリセットとして保存します"""
    bl_idname = "taremin_cloth.save_config_preset"
    bl_label = "Save Config Preset"
    bl_description = "現在の布・コライダーのシミュレーション有効状態をプリセットとして保存します"
    bl_options = {'REGISTER', 'UNDO'}

    preset_name: StringProperty(
        name="Preset Name",
        description="保存する構成プリセット名",
        default="My Preset",
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "preset_name")

    def execute(self, context):
        name = self.preset_name.strip()
        if not name:
            self.report({'ERROR'}, "プリセット名を入力してください")
            return {'CANCELLED'}

        if name in BUILTIN_CONFIG_PRESETS:
            self.report({'ERROR'}, f"'{name}' はビルトインプリセットの名前と重複しています")
            return {'CANCELLED'}

        scene = context.scene
        cloth_states = {}
        collider_states = {}

        for obj in scene.objects:
            cloth_s = getattr(obj, "taremin_cloth", None)
            if cloth_s and cloth_s.is_cloth:
                cloth_states[obj.name] = bool(cloth_s.enabled)

            col_s = getattr(obj, "taremin_cloth_collider", None)
            if col_s and col_s.is_collider:
                collider_states[obj.name] = bool(col_s.enabled)

        presets = get_scene_config_presets(scene)
        presets[name] = {
            "cloths": cloth_states,
            "colliders": collider_states,
        }
        save_scene_config_presets(scene, presets)
        scene.taremin_cloth_active_config_preset = name
        self.report({'INFO'}, f"構成プリセット '{name}' を保存しました")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_apply_config_preset(Operator):
    """指定した構成プリセットを適用します"""
    bl_idname = "taremin_cloth.apply_config_preset"
    bl_label = "Apply Config Preset"
    bl_description = "指定した構成プリセットを適用します"
    bl_options = {'REGISTER', 'UNDO'}

    preset_name: StringProperty(name="Preset Name")

    def execute(self, context):
        scene = context.scene
        if not scene:
            return {'CANCELLED'}

        name = self.preset_name
        if name in BUILTIN_CONFIG_PRESETS:
            action = BUILTIN_CONFIG_PRESETS[name]
            bpy.ops.taremin_cloth.batch_simulation_state(action=action)
            scene.taremin_cloth_active_config_preset = name
            return {'FINISHED'}

        presets = get_scene_config_presets(scene)
        if name not in presets:
            self.report({'ERROR'}, f"構成プリセット '{name}' が見つかりません")
            return {'CANCELLED'}

        data = presets[name]
        cloth_states = data.get("cloths", {})
        collider_states = data.get("colliders", {})

        for obj in scene.objects:
            cloth_s = getattr(obj, "taremin_cloth", None)
            if cloth_s and cloth_s.is_cloth and obj.name in cloth_states:
                cloth_s.enabled = cloth_states[obj.name]

            col_s = getattr(obj, "taremin_cloth_collider", None)
            if col_s and col_s.is_collider and obj.name in collider_states:
                col_s.enabled = collider_states[obj.name]

        scene.taremin_cloth_active_config_preset = name
        self.report({'INFO'}, f"構成プリセット '{name}' を適用しました")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_delete_config_preset(Operator):
    """選択中の構成プリセットを削除します"""
    bl_idname = "taremin_cloth.delete_config_preset"
    bl_label = "Delete Config Preset"
    bl_description = "選択中の構成プリセットを削除します"
    bl_options = {'REGISTER', 'UNDO'}

    preset_name: StringProperty(name="Preset Name", default="")

    def execute(self, context):
        scene = context.scene
        name = self.preset_name or scene.taremin_cloth_active_config_preset
        if not name:
            self.report({'WARNING'}, "削除するプリセットが指定されていません")
            return {'CANCELLED'}

        if name in BUILTIN_CONFIG_PRESETS:
            self.report({'ERROR'}, "ビルトインプリセットは削除できません")
            return {'CANCELLED'}

        presets = get_scene_config_presets(scene)
        if name in presets:
            del presets[name]
            save_scene_config_presets(scene, presets)
            if scene.taremin_cloth_active_config_preset == name:
                scene.taremin_cloth_active_config_preset = ""
            self.report({'INFO'}, f"構成プリセット '{name}' を削除しました")
            return {'FINISHED'}

        self.report({'WARNING'}, f"構成プリセット '{name}' が見つかりません")
        return {'CANCELLED'}


class TAREMIN_CLOTH_MT_config_presets(Menu):
    """シミュレーション構成プリセットのメニュー"""
    bl_label = "Config Presets"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Built-in Presets", icon='PHYSICS')
        for name in BUILTIN_CONFIG_PRESETS.keys():
            op = layout.operator("taremin_cloth.apply_config_preset", text=name)
            op.preset_name = name

        scene = context.scene
        if scene:
            custom_presets = get_scene_config_presets(scene)
            if custom_presets:
                layout.separator()
                layout.label(text="Custom Presets", icon='PRESET')
                for name in custom_presets.keys():
                    op = layout.operator("taremin_cloth.apply_config_preset", text=name)
                    op.preset_name = name


classes = (
    TAREMIN_CLOTH_OT_apply_preset,
    TAREMIN_CLOTH_OT_save_preset,
    TAREMIN_CLOTH_OT_delete_preset,
    TAREMIN_CLOTH_MT_fabric_presets,
    TAREMIN_CLOTH_MT_simulation_presets,
    TAREMIN_CLOTH_MT_collider_presets,
    TAREMIN_CLOTH_OT_batch_simulation_state,
    TAREMIN_CLOTH_OT_save_config_preset,
    TAREMIN_CLOTH_OT_apply_config_preset,
    TAREMIN_CLOTH_OT_delete_config_preset,
    TAREMIN_CLOTH_MT_config_presets,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

