import functools
import bpy
from bpy.props import (
    FloatProperty as _FloatProperty,
    IntProperty as _IntProperty,
    BoolProperty as _BoolProperty,
    EnumProperty as _EnumProperty,
    PointerProperty,
    StringProperty as _StringProperty,
    CollectionProperty,
    FloatVectorProperty as _FloatVectorProperty,
)
from bpy.types import PropertyGroup

from . import i18n


def _wrap_prop(prop_func):
    @functools.wraps(prop_func)
    def wrapper(*args, **kwargs):
        if "translation_context" not in kwargs:
            kwargs["translation_context"] = i18n.CONTEXT
        return prop_func(*args, **kwargs)
    return wrapper


FloatProperty = _wrap_prop(_FloatProperty)
IntProperty = _wrap_prop(_IntProperty)
BoolProperty = _wrap_prop(_BoolProperty)
EnumProperty = _wrap_prop(_EnumProperty)
StringProperty = _wrap_prop(_StringProperty)
FloatVectorProperty = _wrap_prop(_FloatVectorProperty)


class TareminClothElasticGroup(PropertyGroup):
    """辺の自然長スケーリング・ゴム紐グループ設定"""
    name: StringProperty(
        name="Group Name",
        description="Name of the elastic band group",
        default="Elastic Group",
    )
    scale: FloatProperty(
        name="Scale",
        description="Rest length scaling factor (1.0=normal, 0.7=30% contract, 1.3=30% extend)",
        default=1.0,
        min=0.05,
        max=5.0,
    )
    stiffness_multiplier: FloatProperty(
        name="Stiffness Multiplier",
        description="Stiffness multiplier for stronger elastic effect",
        default=1.0,
        min=0.1,
        max=20.0,
    )
    edge_indices_str: StringProperty(
        name="Edge Indices",
        description="Comma-separated registered edge indices",
        default="",
    )
    enabled: BoolProperty(
        name="Enabled",
        description="Apply this elastic group to simulation",
        default=True,
    )
    color: FloatVectorProperty(
        name="Base Color",
        subtype='COLOR',
        size=4,
        default=(0.2, 0.7, 1.0, 1.0),
    )

    def get_edge_indices(self):
        """登録されたエッジインデックスのリストを取得する"""
        if not self.edge_indices_str:
            return []
        try:
            return [int(x.strip()) for x in self.edge_indices_str.split(",") if x.strip()]
        except ValueError:
            return []

    def set_edge_indices(self, indices):
        """エッジインデックスのリストを文字列として保存する"""
        self.edge_indices_str = ",".join(str(i) for i in indices)


def _on_self_collision_purpose_updated(self, context):
    """自己衝突の用途プリセットが変更された時にパラメータを自動更新する"""
    try:
        if getattr(self, "self_collision_purpose", 'STANDARD') == 'CUSTOM':
            return
        obj = getattr(self, "id_data", None) or getattr(context, "active_object", None)
        if obj:
            from .utils.self_collision_fit import fit_self_collision_for_object
            fit_self_collision_for_object(obj, self, self.self_collision_purpose)
    except Exception:
        pass


def _on_enable_self_collision_updated(self, context):
    """自己衝突が有効化された際に、初回自動フィッティングを実行する"""
    try:
        if getattr(self, "enable_self_collision", False) and getattr(self, "self_collision_purpose", 'STANDARD') != 'CUSTOM':
            obj = getattr(self, "id_data", None) or getattr(context, "active_object", None)
            if obj:
                from .utils.self_collision_fit import fit_self_collision_for_object
                fit_self_collision_for_object(obj, self, self.self_collision_purpose)
    except Exception:
        pass


class TareminClothObjectSettings(PropertyGroup):
    is_cloth: BoolProperty(
        name="Cloth Enabled",
        description="Enable cloth simulation for this object",
        default=False,
    )
    enabled: BoolProperty(
        name="Simulation Active",
        description="Enable simulation calculation for this cloth (OFF to pause/mute)",
        default=True,
    )
    last_fabric_preset: StringProperty(
        name="Fabric Preset",
        description="Last applied fabric material preset",
        default="",
    )
    last_simulation_preset: StringProperty(
        name="Simulation Preset",
        description="Last applied simulation quality preset",
        default="",
    )
    # 剛性 (Stiffness)
    tension_stiffness: FloatProperty(
        name="Tension",
        description="Resistance against stretching",
        default=1000.0,
        min=0.1,
        max=100000.0,
        soft_min=10.0,
        soft_max=10000.0,
    )
    compression_stiffness: FloatProperty(
        name="Compression",
        description="Resistance against compression and wrinkles",
        default=100.0,
        min=0.1,
        max=100000.0,
        soft_min=10.0,
        soft_max=10000.0,
    )
    shear_stiffness: FloatProperty(
        name="Shear",
        description="Resistance against shear distortion",
        default=100.0,
        min=0.1,
        max=100000.0,
        soft_min=10.0,
        soft_max=5000.0,
    )
    bending_stiffness: FloatProperty(
        name="Bending",
        description="Resistance against face bending",
        default=10.0,
        min=0.0,
        max=1000.0,
        soft_min=0.0,
        soft_max=200.0,
    )
    # 後方互換用エイリアス
    stiffness: FloatProperty(
        name="Tension",
        description="Tensile stiffness linked with tension_stiffness",
        default=1000.0,
        min=0.1,
        max=100000.0,
        soft_min=10.0,
        soft_max=10000.0,
    )

    # 減衰 (Damping)
    tension_damping: FloatProperty(
        name="Tension",
        description="Damping to absorb stretch vibrations",
        default=5.0,
        min=0.0,
        max=50.0,
        soft_min=0.0,
        soft_max=25.0,
    )
    compression_damping: FloatProperty(
        name="Compression",
        description="Damping to absorb compression and wrinkle vibrations",
        default=5.0,
        min=0.0,
        max=50.0,
        soft_min=0.0,
        soft_max=25.0,
    )
    shear_damping: FloatProperty(
        name="Shear",
        description="Damping to absorb shear vibrations",
        default=5.0,
        min=0.0,
        max=50.0,
        soft_min=0.0,
        soft_max=25.0,
    )
    bending_damping: FloatProperty(
        name="Bending",
        description="Damping to absorb bending vibrations",
        default=0.5,
        min=0.0,
        max=50.0,
        soft_min=0.0,
        soft_max=10.0,
    )
    air_damping: FloatProperty(
        name="Air",
        description="Air resistance and velocity damping",
        default=1.0,
        min=0.0,
        max=50.0,
        soft_min=0.0,
        soft_max=10.0,
    )

    # シミュレーション設定 (Settings)
    gravity: FloatProperty(
        name="Gravity Scale",
        description="Gravity multiplier (1.0=scene gravity, 0.0=zero gravity)",
        default=1.0,
        min=-10.0,
        max=10.0,
        soft_min=0.0,
        soft_max=2.0,
    )
    substeps: IntProperty(
        name="Quality Steps",
        description="Subdivision steps per frame",
        default=20,
        min=1,
        max=100,
    )
    enable_adaptive_substep: BoolProperty(
        name="Adaptive Steps",
        description="Dynamically adjust substeps based on cloth velocity to improve FPS",
        default=False,
    )
    min_substeps: IntProperty(
        name="Min Steps",
        description="Minimum substeps for adaptive stepping",
        default=4,
        min=1,
        max=50,
    )
    max_substeps: IntProperty(
        name="Max Steps",
        description="Maximum substeps for adaptive stepping (CFL emergency limit)",
        default=64,
        min=10,
        max=200,
    )
    solver_iterations: IntProperty(
        name="Solver Iterations",
        description="Constraint solve iterations per substep",
        default=2,
        min=1,
        max=10,
    )
    # パフォーマンスチューニング設定
    workgroup_size: EnumProperty(
        name="Workgroup Size",
        description="Compute shader workgroup size (AMD Wave32 / NVIDIA Warp32 optimal: 32)",
        items=[
            ('32', "32 (Optimal)", "AMD RDNA Wave32 / NVIDIA Warp32 optimized workgroup size (Recommended)"),
            ('64', "64 (Legacy)", "Legacy 64 threads workgroup"),
        ],
        default='32',
    )
    solver_mode: EnumProperty(
        name="Solver Mode",
        description="GPU parallel solver algorithm for distance constraints",
        items=[
            ('COLORING', "Coloring (Gauss-Seidel)", "Graph coloring high-stiffness sequential solver"),
            ('ATOMIC', "Atomic Jacobi (Single-Pass)", "Fixed-point atomic addition single-dispatch fast solver"),
        ],
        default='COLORING',
    )
    enable_async_readback: BoolProperty(
        name="Async Readback",
        description="1-frame delayed asynchronous readback to hide GPU wait time",
        default=False,
    )
    enable_compact_readback: BoolProperty(
        name="Compact Readback",
        description="Transfer only coordinate data (12B/vert) on GPU to reduce PCIe bandwidth",
        default=True,
    )
    enable_frame_buffering: BoolProperty(
        name="Frame Buffering",
        description="Buffer recent frames on GPU and batch transfer to reduce sync overhead",
        default=False,
    )
    frame_buffer_size: IntProperty(
        name="Buffer Size",
        description="Number of frames to buffer (2-5)",
        default=2,
        min=2,
        max=5,
    )
    layer_id: IntProperty(
        name="Layer ID",
        description="Cloth layer number (0=innermost)",
        default=0,
        min=0,
        max=10,
    )
    thickness: FloatProperty(
        name="Thickness",
        description="Cloth thickness in meters",
        default=0.005,
        min=0.0001,
        max=0.1,
        unit='LENGTH',
    )
    enable_self_collision: BoolProperty(
        name="Self Collision",
        description="Enable self and inter-layer collision using GPU spatial hash",
        default=False,
        update=_on_enable_self_collision_updated,
    )
    self_collision_purpose: EnumProperty(
        name="Purpose",
        description="Self-collision parameter preset for cloth usage",
        items=[
            ('STANDARD', "Standard (General Clothing)", "Standard settings for shirts, pants, dresses (balanced performance and stability)"),
            ('SKIRT', "Skirt / Folds (Pleats & Layers)", "High accuracy for dense overlapping cloth like skirts, frills, ribbons"),
            ('THIN', "Thin / Delicate (Silk & Light)", "Mild repulsion for thin fabrics, scarves, silk to avoid explosions"),
            ('CUSTOM', "Custom (Manual)", "Manual tuning mode for fine parameter adjustments"),
        ],
        default='STANDARD',
        update=_on_self_collision_purpose_updated,
    )
    self_collision_relief_factor: FloatProperty(
        name="Relief Factor",
        description="Penetration relief factor (1.0 for instant response, smaller for smooth resolution)",
        default=0.2,
        min=0.01,
        max=1.0,
        precision=2,
    )
    self_collision_max_displacement_ratio: FloatProperty(
        name="Max Step Ratio",
        description="Maximum correction displacement per substep as a ratio of edge length",
        default=0.2,
        min=0.01,
        max=1.0,
        precision=2,
    )
    self_collision_max_iterations: EnumProperty(
        name="Max Search Iterations",
        description="Maximum search iterations per cell in GPU spatial hash",
        items=[
            ('128', "128 (Fast)", "For fast preview and lightweight meshes (standard)"),
            ('256', "256 (Balanced)", "Balanced setting to suppress pass-through"),
            ('512', "512 (High Quality)", "High quality for folding and dense meshes"),
            ('1024', "1024 (Ultra)", "Ultra dense for complex wrinkle penetration prevention"),
            ('4096', "4096 (No Limit)", "Virtually unlimited to thoroughly prevent penetration"),
        ],
        default='256',
    )
    self_collision_exclude_neighbors: BoolProperty(
        name="Exclude Neighbors",
        description="Exclude mesh edge-connected vertices from self-collision",
        default=True,
    )
    enable_normal_untangling: BoolProperty(
        name="Normal Untangling",
        description="Use vertex normals to push penetrated vertices outward",
        default=True,
    )
    coupled_self_collision_mode: EnumProperty(
        name="Coupled Mode",
        description="Coupled convergence mode for self-collision and distance constraints",
        items=[
            ('OFF', "Off (Legacy)", "Traditional self-collision (executed outside loop, no relaxation)"),
            ('RELAXATION', "Relaxation (Balanced)", "Re-apply distance constraints twice right after collision"),
            ('FULL_COUPLED', "Full Coupled (High Quality)", "Coupled solve inside loop with finishing relaxation"),
        ],
        default='RELAXATION',
    )
    post_collision_relaxation_iters: IntProperty(
        name="Relaxation Steps",
        description="Distance relaxation steps after self-collision (0: auto, 1-8: manual)",
        default=2,
        min=0,
        max=8,
    )
    self_collision_substep_interval: IntProperty(
        name="Substep Interval",
        description="Substep frequency for self-collision detection (1: every substep, 2: every 2 substeps, etc.)",
        default=1,
        min=1,
        max=4,
    )
    enable_edge_collision: BoolProperty(
        name="Edge-Collider Collision",
        description="Enable detailed edge-collider contact detection",
        default=False,
    )
    edge_margin_scale: FloatProperty(
        name="Edge Margin Scale",
        description="Safety margin factor for edge contact detection",
        default=1.0,
        min=1.0,
        max=3.0,
        step=10,
        precision=2,
    )
    edge_margin_offset: FloatProperty(
        name="Margin Offset",
        description="Safety margin fixed offset for edge contact detection",
        default=0.0,
        min=0.0,
        max=0.1,
        step=0.1,
        precision=3,
        unit='LENGTH',
    )
    # 縫合（Sewing）
    enable_sewing: BoolProperty(
        name="Enable Sewing",
        description="Enable sewing constraints",
        default=False,
    )
    sewing_shrink_speed: FloatProperty(
        name="Shrink Speed",
        description="Sewing contraction speed (m/s)",
        default=1.0,
        min=0.01,
        max=50.0,
    )
    sewing_stiffness: FloatProperty(
        name="Sewing Stiffness",
        description="Stiffness of sewing constraints (>= 5000: rigid closure)",
        default=10000.0,
        min=1.0,
        max=50000.0,
        soft_max=10000.0,
    )
    enable_sewing_lock: BoolProperty(
        name="Lock When Closed",
        description="Lock sewing edges rigidly once fully contracted, preventing gap reopen",
        default=True,
    )
    # ピン留め・アタッチメント
    pin_target_object: PointerProperty(
        name="Pin Target",
        type=bpy.types.Object,
        description="Target object to follow for pinned vertices",
    )
    pin_target_bone: bpy.props.StringProperty(
        name="Pin Target Bone",
        description="Bone name to follow when target is an armature",
        default="",
    )
    pin_vertex_group: bpy.props.StringProperty(
        name="Pin Vertex Group",
        description="Vertex group name used for pinning and tracking",
        default="Pin",
    )
    pin_color: FloatVectorProperty(
        name="Pin Color",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.45, 0.0, 0.9),
        description="Highlight display color for pinned vertices",
    )
    pin_overlay_interactive_only: BoolProperty(
        name="Pin Overlay Interactive Only",
        description="Show pin highlight overlay only during interactive mode",
        default=True,
    )
    overlay_depth_test: BoolProperty(
        name="Depth Test (Z)",
        description="Apply depth test to overlays so they are occluded by meshes",
        default=False,
    )
    interactive_realtime_sync: BoolProperty(
        name="Real-time Sync",
        description="Advance multiple simulation steps based on elapsed time to maintain real-time speed",
        default=True,
    )
    interactive_max_steps: IntProperty(
        name="Max Steps / Frame",
        description="Maximum simulation steps advanced per rendered frame (1-8)",
        default=4,
        min=1,
        max=8,
    )
    isolate_viewport_view: BoolProperty(
        name="Isolate View (Local)",
        description="Isolate cloth and colliders to local view during simulation for maximum viewport FPS",
        default=True,
    )
    show_fps_overlay: BoolProperty(
        name="Show FPS Overlay",
        description="Display FPS and frame timing overlay in 3D viewport during simulation",
        default=True,
    )
    fps_overlay_position: EnumProperty(
        name="FPS Position",
        description="Screen corner to display FPS overlay",
        items=[
            ('TOP_CENTER', "Top Center", "Top center of screen"),
            ('TOP_RIGHT', "Top Right", "Top right of screen"),
            ('BOTTOM_RIGHT', "Bottom Right", "Bottom right of screen"),
            ('BOTTOM_LEFT', "Bottom Left", "Bottom left of screen"),
            ('TOP_LEFT', "Top Left", "Top left of screen"),
        ],
        default='TOP_CENTER',
    )
    show_hud_help: BoolProperty(
        name="Show HUD Help",
        description="Display interactive shortcut key guide overlay in 3D viewport",
        default=True,
    )
    # 伸縮グループ (Elastic Bands / Edge Scaling)
    elastic_groups: CollectionProperty(
        type=TareminClothElasticGroup,
    )
    active_elastic_group_index: IntProperty(
        name="Active Group Index",
        default=0,
    )
    show_elastic_overlay: BoolProperty(
        name="Show Elastic Lines",
        description="Show elastic line tension colors in 3D viewport",
        default=True,
    )
    elastic_overlay_interactive_only: BoolProperty(
        name="Elastic Overlay Interactive Only",
        description="Show elastic band highlight only during interactive mode",
        default=True,
    )
    # トポロジー・分割モード (Triangulation & Topology)
    triangulation_mode: EnumProperty(
        name="Triangulation Mode",
        description="Quad diagonal bias prevention and split mode",
        items=[
            ('DYNAMIC_DIAGONAL', "Dynamic Diagonal (Lightweight)", "Zero vertex overhead during sim; split quads along strain on stop (Recommended)"),
            ('CROSS_SUBDIV', "Cross Subdivision (Poke)", "Add center vertex to split quad into 4 triangles for isotropic wrinkling (High accuracy)"),
            ('NONE', "None (Original)", "Keep quads as-is (computed with default Blender diagonal)"),
        ],
        default='DYNAMIC_DIAGONAL',
    )
    # 動的対角線分割オプション
    dynamic_preserve_flat: BoolProperty(
        name="Preserve Flat Quads",
        description="Preserve flat quad faces without splitting into triangles",
        default=False,
    )
    dynamic_flatness_threshold: FloatProperty(
        name="Flatness Angle",
        description="Maximum angle in degrees to consider a face flat",
        default=5.0,
        min=0.5,
        max=45.0,
    )
    auto_triangulate_on_stop: BoolProperty(
        name="Auto Triangulate on Stop",
        description="Automatically perform optimal diagonal split when simulation stops",
        default=True,
    )
    # 十字分割用（互換性維持）
    def _update_cross_subdiv(self, context):
        if self.enable_cross_subdivision:
            self.triangulation_mode = 'CROSS_SUBDIV'
        elif self.triangulation_mode == 'CROSS_SUBDIV':
            self.triangulation_mode = 'DYNAMIC_DIAGONAL'

    enable_cross_subdivision: BoolProperty(
        name="Cross Subdivision",
        description="Subdivide quads into 4 triangles to eliminate diagonal bias",
        default=False,
        update=_update_cross_subdiv,
    )
    post_process_mode: EnumProperty(
        name="Post-Process",
        description="Mesh post-processing mode after simulation",
        items=[
            ('OPTIMAL_TRI', "Optimal 2 Triangles", "Split into 2 triangles along wrinkle ridge lines"),
            ('QUAD', "Restore Quad", "Remove center vertex and restore original quad"),
            ('ADAPTIVE', "Adaptive", "Automatically select quad for flat areas and 2 triangles for wrinkles"),
            ('KEEP', "Keep Cross (4 Triangles)", "Keep cross subdivision mesh"),
        ],
        default='OPTIMAL_TRI',
    )
    adaptive_flatness_threshold: FloatProperty(
        name="Flatness Threshold",
        description="Flatness threshold angle for adaptive post-processing",
        default=5.0,
        min=0.5,
        max=45.0,
    )
    auto_post_process: BoolProperty(
        name="Auto Post-Process on Stop",
        description="Automatically execute post-processing on simulation stop",
        default=True,
    )


def _on_anim_progress_updated(self, context):
    """手動プログレススライダー操作時にビューポート上のポーズ／シェイプキーを即座に更新するコールバック"""
    try:
        from .utils import anim_driver
        # self.id_data はプロパティを保持するオブジェクト
        obj = self.id_data
        if not obj or not context:
            return

        eased_t = anim_driver.apply_easing(self.progress, self.easing)
        if self.target_type == 'SHAPE_KEY':
            if obj.type == 'MESH' and obj.data and obj.data.shape_keys:
                kb = obj.data.shape_keys.key_blocks.get(self.shape_key_name)
                if kb:
                    kb.value = self.start_value + eased_t * (self.end_value - self.start_value)
        elif self.target_type == 'POSE_BLEND':
            armature = self.armature_obj or (obj if obj.type == 'ARMATURE' else obj.parent)
            if armature and self.start_pose_data and self.target_pose_data:
                anim_driver.apply_pose_blend(armature, self.start_pose_data, self.target_pose_data, eased_t)
        elif self.target_type == 'ACTION':
            armature = self.armature_obj or (obj if obj.type == 'ARMATURE' else obj.parent)
            if armature and self.action:
                eval_frame = self.frame_start + eased_t * (self.frame_end - self.frame_start)
                anim_driver.apply_action_frame(armature, self.action, eval_frame)

        context.view_layer.update()
    except Exception:
        pass


class TareminClothColliderAnimSettings(PropertyGroup):
    """コライダーアニメーション駆動設定"""
    enabled: BoolProperty(
        name="Animation Enabled",
        description="Enable collider animation driving",
        default=False,
    )
    target_type: EnumProperty(
        name="Target Type",
        description="Animation driver target type",
        items=[
            ('SHAPE_KEY', "Shape Key", "Mesh shape key value animation"),
            ('POSE_BLEND', "Pose Blend", "Blend between two pose snapshots (0.0 - 1.0)"),
            ('ACTION', "Action", "Play specified frame range of existing action"),
        ],
        default='POSE_BLEND',
    )
    # シェイプキー設定
    shape_key_name: StringProperty(
        name="Shape Key",
        description="Shape key name to drive",
        default="",
    )
    start_value: FloatProperty(
        name="Start Value",
        description="Shape key value at start",
        default=0.0,
    )
    end_value: FloatProperty(
        name="End Value",
        description="Shape key value at end",
        default=1.0,
    )
    # ポーズブレンド設定
    armature_obj: PointerProperty(
        name="Armature",
        type=bpy.types.Object,
        description="Target armature object for pose blend (auto-detected if empty)",
        poll=lambda s, o: o.type == 'ARMATURE',
    )
    start_pose_data: StringProperty(
        name="Start Pose (0.0)",
        description="Start pose snapshot data (JSON)",
        default="",
    )
    target_pose_data: StringProperty(
        name="Target Pose (1.0)",
        description="Target pose snapshot data (JSON)",
        default="",
    )
    # アクション設定
    action: PointerProperty(
        name="Action",
        type=bpy.types.Action,
        description="Animation action to play",
    )
    frame_start: IntProperty(
        name="Start Frame",
        description="Action playback start frame",
        default=1,
    )
    frame_end: IntProperty(
        name="End Frame",
        description="Action playback end frame",
        default=60,
    )
    # 再生・サイクル設定
    play_mode: EnumProperty(
        name="Play Mode",
        description="Animation playback mode",
        items=[
            ('ONCE', "Once", "Play once and hold last frame"),
            ('REPEAT', "Repeat", "Loop playback from beginning"),
            ('PINGPONG', "Ping-Pong", "Ping-pong forward and backward"),
        ],
        default='ONCE',
    )
    cycle_frames: IntProperty(
        name="Cycle Frames",
        description="Frames per cycle (e.g. 60 frames = 1 second)",
        default=60,
        min=1,
        max=10000,
    )
    loop_count: IntProperty(
        name="Loop Count",
        description="Number of repeats or round-trips",
        default=1,
        min=1,
        max=1000,
    )
    infinite_loop: BoolProperty(
        name="Infinite",
        description="Repeat animation indefinitely",
        default=False,
    )
    easing: EnumProperty(
        name="Easing",
        description="Interpolation easing curve",
        items=[
            ('SMOOTH', "Smooth (Ease In-Out)", "Smoothstep easing (Recommended: suppresses shocks)"),
            ('LINEAR', "Linear", "Linear constant speed interpolation"),
        ],
        default='SMOOTH',
    )
    progress: FloatProperty(
        name="Progress",
        description="Current progress (0.0 to 1.0). Manually scrubbable with slider",
        default=0.0,
        min=0.0,
        max=1.0,
        precision=3,
        update=_on_anim_progress_updated,
    )


def _on_collider_prop_updated(self, context):
    try:
        from .engine.collider import clear_collider_cache
        clear_collider_cache()
    except Exception:
        pass


def _on_collider_purpose_updated(self, context):
    try:
        from .utils.collider_detect import PURPOSE_TO_COLLIDER_TYPE, detect_collider_type
        if self.collider_purpose == 'AUTO':
            obj = getattr(self, "id_data", None) or getattr(context, "active_object", None)
            if obj:
                self.collider_type = detect_collider_type(obj)
        elif self.collider_purpose in PURPOSE_TO_COLLIDER_TYPE:
            self.collider_type = PURPOSE_TO_COLLIDER_TYPE[self.collider_purpose]
    except Exception:
        pass
    _on_collider_prop_updated(self, context)


def _on_is_collider_updated(self, context):
    if self.is_collider and getattr(self, "collider_purpose", 'AUTO') == 'AUTO':
        try:
            from .utils.collider_detect import detect_collider_type
            obj = getattr(self, "id_data", None) or getattr(context, "active_object", None)
            if obj:
                self.collider_type = detect_collider_type(obj)
        except Exception:
            pass
    _on_collider_prop_updated(self, context)


class TareminClothColliderSettings(PropertyGroup):
    is_collider: BoolProperty(
        name="Collider Enabled",
        description="Enable GPU collider on this object",
        default=False,
        update=_on_is_collider_updated,
    )
    enabled: BoolProperty(
        name="Collider Active",
        description="Enable collision detection for this collider (OFF to temporarily disable)",
        default=True,
        update=_on_collider_prop_updated,
    )
    collider_purpose: EnumProperty(
        name="Collider Purpose",
        description="Recommended collider setting preset",
        items=[
            ('AUTO', "Auto Detect", "Auto-select best collider type from object structure"),
            ('CHARACTER', "Character Body", "Humanoid character avatar (Bone SDF)"),
            ('MANNEQUIN', "Mannequin / Prop", "Mannequin or rigid furniture (Mesh SDF)"),
            ('FLOOR', "Floor / Ground", "Ground floor (Plane)"),
            ('SPHERE', "Sphere", "Sphere collider"),
            ('CUSTOM', "Custom / Simple", "Custom polygon mesh (Mesh)"),
        ],
        default='AUTO',
        update=_on_collider_purpose_updated,
    )
    last_collider_preset: StringProperty(
        name="Collider Preset",
        description="Last applied collider preset",
        default="",
    )
    collider_type: EnumProperty(
        name="Collider Shape",
        items=[
            ('SPHERE', "Sphere", "Sphere collider"),
            ('CAPSULE', "Capsule", "Capsule collider"),
            ('PLANE', "Plane", "Plane collider"),
            ('MESH', "Mesh", "Mesh collider (custom polygon mesh)"),
            ('BONE_SDF', "Bone SDF", "Bone-local SDF collider (Character body)"),
            ('MESH_SDF', "Mesh SDF", "Single mesh SDF collider (Mannequin / Rigid)"),
        ],
        default='MESH',
        update=_on_collider_prop_updated,
    )
    sdf_resolution: EnumProperty(
        name="SDF Resolution",
        description="Texture resolution for each bone-local SDF",
        items=[
            ('32', "32 (Low)", "Lightweight and fast (32x32x32, ~131KB/bone)"),
            ('64', "64 (Standard)", "Standard recommended (64x64x64, ~1MB/bone)"),
            ('128', "128 (High)", "High detail (128x128x128, ~8MB/bone)"),
            ('CUSTOM', "Custom", "Custom resolution specification"),
        ],
        default='64',
        update=_on_collider_prop_updated,
    )
    sdf_resolution_custom: IntProperty(
        name="Custom Resolution",
        description="Custom SDF resolution (16 to 512)",
        default=128,
        min=16,
        max=512,
        update=_on_collider_prop_updated,
    )
    sdf_margin: FloatProperty(
        name="SDF Margin",
        description="Bone local AABB margin ratio",
        default=0.2,
        min=0.05,
        max=1.0,
        update=_on_collider_prop_updated,
    )
    weight_threshold: FloatProperty(
        name="Weight Threshold",
        description="Minimum vertex weight threshold recognized as bone influence",
        default=0.02,
        min=0.001,
        max=0.5,
        update=_on_collider_prop_updated,
    )
    blend_k: FloatProperty(
        name="Blend Smoothness",
        description="SDF blend smoothness radius across bone joints in meters",
        default=0.05,
        min=0.001,
        max=0.5,
        unit='LENGTH',
        update=_on_collider_prop_updated,
    )
    sdf_update_mode: EnumProperty(
        name="SDF Update Mode",
        description="Bone SDF update mode",
        items=[
            ('STATIC', "Static (Fastest)", "Pre-baked rigid SDF in rest pose (Fastest, recommended)"),
            ('DYNAMIC_GPU', "Dynamic Full GPU", "Per-frame GPU LBS deformation and SDF update"),
        ],
        default='STATIC',
        update=_on_collider_prop_updated,
    )
    sdf_dynamic_update_interval: IntProperty(
        name="Update Interval",
        description="Dynamic SDF recalculation interval in frames (1=every frame)",
        default=1,
        min=1,
        max=10,
        update=_on_collider_prop_updated,
    )
    sdf_cache_enabled: BoolProperty(
        name="Cache SDF",
        description="Cache baked SDF to disk for instant reload",
        default=True,
    )
    enable_joint_mesh: BoolProperty(
        name="Joint Mesh Hybrid",
        description="Use hybrid partial mesh collider only at joint blend regions to eliminate sharp SDF corners",
        default=True,
        update=_on_collider_prop_updated,
    )
    joint_weight_threshold: FloatProperty(
        name="Joint Blend Threshold",
        description="Bone weight threshold below which faces are converted to partial mesh",
        default=0.85,
        min=0.5,
        max=0.99,
        precision=2,
        update=_on_collider_prop_updated,
    )
    joint_rotation_threshold: FloatProperty(
        name="Activation Angle",
        description="Minimum joint bend angle in degrees to activate dynamic mesh sync",
        default=2.0,
        min=0.0,
        max=45.0,
        precision=1,
        update=_on_collider_prop_updated,
    )
    mesh_sdf_voxel_size: FloatProperty(
        name="Voxel Size",
        description="Mesh SDF voxel size in meters (smaller is higher detail, e.g. 0.004 = 4mm)",
        default=0.004,
        min=0.0005,
        max=0.05,
        step=0.1,
        precision=3,
        unit='LENGTH',
        update=_on_collider_prop_updated,
    )
    mesh_sdf_margin: FloatProperty(
        name="SDF Margin",
        description="SDF margin around mesh in meters",
        default=0.02,
        min=0.005,
        max=0.2,
        step=0.5,
        precision=3,
        unit='LENGTH',
        update=_on_collider_prop_updated,
    )
    mesh_sdf_max_vram_mb: IntProperty(
        name="Max VRAM (MB)",
        description="Maximum VRAM budget for mesh SDF texture (MB)",
        default=256,
        min=64,
        max=4096,
        step=64,
        update=_on_collider_prop_updated,
    )
    mesh_sdf_auto_scale: BoolProperty(
        name="Auto Fit VRAM",
        description="Automatically adjust voxel size if estimated VRAM exceeds budget",
        default=True,
        update=_on_collider_prop_updated,
    )
    mesh_sdf_cache_enabled: BoolProperty(
        name="Cache SDF",
        description="Cache mesh SDF to disk for instant reload",
        default=True,
        update=_on_collider_prop_updated,
    )
    radius: FloatProperty(
        name="Radius",
        description="Collider radius in meters",
        default=0.5,
        min=0.001,
        max=10.0,
        unit='LENGTH',
        update=_on_collider_prop_updated,
    )
    thickness: FloatProperty(
        name="Thickness",
        description="Mesh collider surface thickness in meters",
        default=0.005,
        min=0.0001,
        max=0.5,
        unit='LENGTH',
        update=_on_collider_prop_updated,
    )
    single_sided: BoolProperty(
        name="Single Sided",
        description="Enable single-sided collision to prevent penetration from backside",
        default=True,
        update=_on_collider_prop_updated,
    )
    enable_single_sided_recovery: BoolProperty(
        name="Single-Sided Recovery",
        description="Safely recover vertices that penetrated behind single-sided collider",
        default=True,
        update=_on_collider_prop_updated,
    )
    enable_cluster_culling: BoolProperty(
        name="Linear BVH Culling",
        description="Cluster BVH culling for fast collision against dense meshes",
        default=False,
        update=_on_collider_prop_updated,
    )
    sweep_margin_offset: FloatProperty(
        name="Sweep Margin Offset",
        description="Additional safety margin for fast moving colliders in meters",
        default=0.05,
        min=0.001,
        max=0.5,
        unit='LENGTH',
        update=_on_collider_prop_updated,
    )
    friction: FloatProperty(
        name="Friction",
        description="Surface friction coefficient",
        default=0.5,
        min=0.0,
        max=1.0,
        update=_on_collider_prop_updated,
    )
    restitution: FloatProperty(
        name="Restitution",
        description="Collision restitution coefficient (0.0: inelastic, 1.0: fully elastic)",
        default=0.0,
        min=0.0,
        max=1.0,
        update=_on_collider_prop_updated,
    )
    anim: PointerProperty(type=TareminClothColliderAnimSettings)


def register():
    for cls in (TareminClothElasticGroup, TareminClothObjectSettings, TareminClothColliderAnimSettings, TareminClothColliderSettings):
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            pass
    bpy.types.Object.taremin_cloth = PointerProperty(type=TareminClothObjectSettings)
    bpy.types.Object.taremin_cloth_collider = PointerProperty(type=TareminClothColliderSettings)
    bpy.types.Scene.taremin_cloth_fast_playback = BoolProperty(
        name="Fast Playback",
        description="Bypass modifiers of non-simulated objects during playback to reduce Depsgraph overhead",
        default=False,
    )
    bpy.types.Scene.taremin_cloth_config_presets_json = StringProperty(
        name="Config Presets JSON",
        description="Saved simulation configuration presets (JSON)",
        default="{}",
    )
    bpy.types.Scene.taremin_cloth_active_config_preset = StringProperty(
        name="Active Config Preset",
        description="Currently applied configuration preset name",
        default="",
    )
    bpy.types.Scene.taremin_cloth_ui_mode = EnumProperty(
        name="UI Mode",
        description="Taremin Cloth UI display mode (Simple / Advanced)",
        items=[
            ('SIMPLE', "Simple", "Simplified UI for beginners with key presets and basic settings", 'PLAY', 0),
            ('ADVANCED', "Advanced", "Detailed UI allowing full access to all physics and internal settings", 'PREFERENCES', 1),
        ],
        default='SIMPLE',
    )


def unregister():
    if hasattr(bpy.types.Scene, "taremin_cloth_ui_mode"):
        del bpy.types.Scene.taremin_cloth_ui_mode
    if hasattr(bpy.types.Scene, "taremin_cloth_active_config_preset"):
        del bpy.types.Scene.taremin_cloth_active_config_preset
    if hasattr(bpy.types.Scene, "taremin_cloth_config_presets_json"):
        del bpy.types.Scene.taremin_cloth_config_presets_json
    if hasattr(bpy.types.Scene, "taremin_cloth_fast_playback"):
        del bpy.types.Scene.taremin_cloth_fast_playback
    if hasattr(bpy.types.Object, "taremin_cloth"):
        del bpy.types.Object.taremin_cloth
    if hasattr(bpy.types.Object, "taremin_cloth_collider"):
        del bpy.types.Object.taremin_cloth_collider
    for cls in (TareminClothColliderSettings, TareminClothColliderAnimSettings, TareminClothObjectSettings, TareminClothElasticGroup):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


