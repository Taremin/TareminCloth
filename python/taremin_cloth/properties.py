import bpy
from bpy.props import (
    FloatProperty,
    IntProperty,
    BoolProperty,
    EnumProperty,
    PointerProperty,
    StringProperty,
    CollectionProperty,
    FloatVectorProperty,
)
from bpy.types import PropertyGroup


class TareminClothElasticGroup(PropertyGroup):
    """辺の自然長スケーリング・ゴム紐グループ設定"""
    name: StringProperty(
        name="Group Name",
        description="グループ名",
        default="Elastic Group",
    )
    scale: FloatProperty(
        name="Scale",
        description="自然長の倍率 (1.0=等倍, 0.7=30%収縮, 1.3=30%伸長)",
        default=1.0,
        min=0.05,
        max=5.0,
    )
    stiffness_multiplier: FloatProperty(
        name="Stiffness Multiplier",
        description="剛性倍率 (強いゴム紐にする場合の硬さ倍率)",
        default=1.0,
        min=0.1,
        max=20.0,
    )
    edge_indices_str: StringProperty(
        name="Edge Indices",
        description="登録されたエッジインデックス（カンマ区切り）",
        default="",
    )
    enabled: BoolProperty(
        name="Enabled",
        description="この伸縮グループをシミュレーションに適用する",
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


class TareminClothObjectSettings(PropertyGroup):
    is_cloth: BoolProperty(
        name="Cloth Enabled",
        description="このオブジェクトでClothシミュレーションを有効にする",
        default=False,
    )
    enabled: BoolProperty(
        name="Simulation Active",
        description="この布オブジェクトのシミュレーション計算を有効にする（OFFで一時停止・計算除外）",
        default=True,
    )
    last_fabric_preset: StringProperty(
        name="Fabric Preset",
        description="直近に適用された布素材プリセット",
        default="",
    )
    last_simulation_preset: StringProperty(
        name="Simulation Preset",
        description="直近に適用されたシミュレーション品質プリセット",
        default="",
    )
    # 剛性 (Stiffness)
    tension_stiffness: FloatProperty(
        name="Tension",
        description="伸びに対する抵抗力 (引張剛性)",
        default=1000.0,
        min=0.1,
        max=100000.0,
    )
    compression_stiffness: FloatProperty(
        name="Compression",
        description="縮み・シワに対する抵抗力 (圧縮剛性)",
        default=100.0,
        min=0.1,
        max=100000.0,
    )
    shear_stiffness: FloatProperty(
        name="Shear",
        description="斜め歪みに対する抵抗力 (せん断剛性)",
        default=100.0,
        min=0.1,
        max=100000.0,
    )
    bending_stiffness: FloatProperty(
        name="Bending",
        description="面同士の折れ曲がりに対する抵抗力 (曲げ剛性)",
        default=10.0,
        min=0.0,
        max=1000.0,
    )
    # 後方互換用エイリアス
    stiffness: FloatProperty(
        name="Tension",
        description="伸縮剛性 (Stiffness) - tension_stiffness と連動",
        default=1000.0,
        min=0.1,
        max=100000.0,
    )

    # 減衰 (Damping)
    tension_damping: FloatProperty(
        name="Tension",
        description="伸び縮みの反発振動を吸収する減衰量",
        default=5.0,
        min=0.0,
        max=50.0,
    )
    compression_damping: FloatProperty(
        name="Compression",
        description="シワが寄る際の振動を吸収する減衰量",
        default=5.0,
        min=0.0,
        max=50.0,
    )
    shear_damping: FloatProperty(
        name="Shear",
        description="斜めの歪み振動を吸収する減衰量",
        default=5.0,
        min=0.0,
        max=50.0,
    )
    bending_damping: FloatProperty(
        name="Bending",
        description="折り曲がり・ヒラヒラ振動を吸収する減衰量",
        default=0.5,
        min=0.0,
        max=50.0,
    )
    air_damping: FloatProperty(
        name="Air",
        description="空気抵抗・速度減衰 (値が大きいほど揺れが早く収まる)",
        default=1.0,
        min=0.0,
        max=50.0,
    )

    # シミュレーション設定 (Settings)
    gravity: FloatProperty(
        name="Gravity",
        description="重力倍率 (1.0=標準重力、0.0=無重力。Blenderのシーン重力と連動)",
        default=1.0,
        min=0.0,
        max=10.0,
    )
    substeps: IntProperty(
        name="Quality Steps",
        description="品質のステップ数 (1フレームあたりの細分化ステップ数)",
        default=20,
        min=1,
        max=100,
    )
    enable_adaptive_substep: BoolProperty(
        name="Adaptive Steps",
        description="布の動きの大きさに応じてサブステップ数を自動調整し、静止時・微動時のFPSを向上させます",
        default=False,
    )
    min_substeps: IntProperty(
        name="Min Steps",
        description="適応型ステップにおける最小細分化ステップ数",
        default=4,
        min=1,
        max=50,
    )
    max_substeps: IntProperty(
        name="Max Steps",
        description="適応型ステップにおける最大細分化ステップ数（激突・高速移動時のCFL緊急上限）",
        default=64,
        min=10,
        max=200,
    )
    solver_iterations: IntProperty(
        name="Solver Iterations",
        description="1サブステップあたりの拘束反復回数。肩紐など細長いパーツの伸びを抑制します",
        default=2,
        min=1,
        max=10,
    )
    # パフォーマンスチューニング設定
    workgroup_size: EnumProperty(
        name="Workgroup Size",
        description="GPUコンピュートシェーダーのワークグループサイズ (AMD Wave32 / NVIDIA Warp32 最適値: 32)",
        items=[
            ('32', "32 (Optimal)", "AMD RDNA Wave32 / NVIDIA Warp32 に最適化されたワークグループサイズ (推奨・標準)"),
            ('64', "64 (Legacy)", "従来の64スレッドワークグループ"),
        ],
        default='32',
    )
    solver_mode: EnumProperty(
        name="Solver Mode",
        description="距離拘束のGPU並列解決方式",
        items=[
            ('COLORING', "Coloring (Gauss-Seidel)", "グラフ彩色による高剛性・順次伝播ソルバー"),
            ('ATOMIC', "Atomic Jacobi (Single-Pass)", "固定小数点アトミック加算による全拘束単一ディスパッチ高速ソルバー"),
        ],
        default='COLORING',
    )
    enable_async_readback: BoolProperty(
        name="Async Readback",
        description="1フレーム遅延の非同期リードバック（ダブルバッファリング）によりGPU完了待機時間を隠蔽・短縮します",
        default=False,
    )
    enable_compact_readback: BoolProperty(
        name="Compact Readback",
        description="GPU上で座標データ(12B/頂点)のみを抽出転送し、PCIeバス帯域とCPU負荷を1/4に削減します",
        default=True,
    )
    enable_frame_buffering: BoolProperty(
        name="Frame Buffering",
        description="GPU上で直近Nフレームの計算結果をバッファリングし、まとめて転送・キャッシュすることで描画負荷とGPU同期待機を大幅に削減します",
        default=False,
    )
    frame_buffer_size: IntProperty(
        name="Buffer Size",
        description="バッファリングするフレーム数 (2〜5)。このフレーム数ごとにまとめてBlenderに転送し、中間フレームも欠落なくキャッシュに保存します",
        default=2,
        min=2,
        max=5,
    )
    layer_id: IntProperty(
        name="Layer ID",
        description="布のレイヤー番号 (0=最内層)",
        default=0,
        min=0,
        max=10,
    )
    thickness: FloatProperty(
        name="Thickness",
        description="布の厚み (m)",
        default=0.005,
        min=0.0001,
        max=0.1,
        unit='LENGTH',
    )
    enable_self_collision: BoolProperty(
        name="Self Collision",
        description="自己衝突およびレイヤー衝突を有効化する（GPU空間ハッシュを使用）",
        default=False,
    )
    self_collision_relief_factor: FloatProperty(
        name="Relief Factor",
        description="自己衝突・貫通時の緩和係数。1.0で即時反発、小さい値で数フレームかけて滑らかに解消し破裂を防止します",
        default=0.2,
        min=0.01,
        max=1.0,
        precision=2,
    )
    self_collision_max_displacement_ratio: FloatProperty(
        name="Max Step Ratio",
        description="1サブステップあたりの最大補正変位割合（周囲の辺の長さに対する割合）。急激な跳ね上がりや布の破裂を防止します",
        default=0.2,
        min=0.01,
        max=1.0,
        precision=2,
    )
    self_collision_max_iterations: EnumProperty(
        name="Max Search Iterations",
        description="GPU空間ハッシュでの1セルあたりの最大探索反復回数。密集時の貫通・すり抜けを防ぐには大きな値を指定します",
        items=[
            ('128', "128 (Fast)", "高速プレビュー・軽量メッシュ向け (標準)"),
            ('256', "256 (Balanced)", "バランス設定。衝突時のすり抜けを抑制"),
            ('512', "512 (High Quality)", "高品質設定。折り畳みや高密度メッシュ向け"),
            ('1024', "1024 (Ultra)", "超高密度・複雑なシワの貫通防止"),
            ('4096', "4096 (No Limit)", "実質無制限。時間をかけて確実に貫通を防ぎます"),
        ],
        default='256',
    )
    self_collision_exclude_neighbors: BoolProperty(
        name="Exclude Neighbors",
        description="メッシュのエッジで直接接続された隣接頂点を自己衝突から除外し、安静時の自縄自縛やシワ・縮みを防止します",
        default=True,
    )
    enable_normal_untangling: BoolProperty(
        name="Normal Untangling",
        description="頂点法線を用いて裏抜けした頂点を表側へ押し戻し、自己交差からの自律的な脱出を可能にします",
        default=True,
    )
    enable_edge_collision: BoolProperty(
        name="Edge-Collider Collision",
        description="布のエッジ（線分）と外部コライダーの詳細接触判定を有効化し、尖ったコライダーの角抜け・線分貫通を防止します",
        default=False,
    )
    edge_margin_scale: FloatProperty(
        name="Edge Margin Scale",
        description="エッジ詳細接触判定時の安全マージン倍率。通常の厚みは小さいままで、エッジ・面の突き抜け補正に余裕を持たせます",
        default=1.0,
        min=1.0,
        max=3.0,
        step=10,
        precision=2,
    )
    edge_margin_offset: FloatProperty(
        name="Margin Offset",
        description="エッジ詳細接触判定時の安全マージン固定加算値。薄い布でも確実にクリアランスを確保します",
        default=0.0,
        min=0.0,
        max=0.1,
        step=0.1,
        precision=3,
        unit='LENGTH',
    )
    # コライダー最適化 & リカバリー設定
    enable_collider_cluster_culling: BoolProperty(
        name="Linear BVH Culling",
        description="大量のメッシュコライダー面を16面クラスタ単位でGPU階層カリングし、大幅に高速化します（複雑な衣服・人体向け）",
        default=False,
    )
    enable_single_sided_recovery: BoolProperty(
        name="Single-Sided Recovery",
        description="片面メッシュコライダーの裏側に侵入した頂点を、安全ガード（面内部判定かつ直近表側接触なし）を満たす最近傍面から表側へ脱出させます",
        default=True,
    )
    collider_sweep_margin_offset: FloatProperty(
        name="Sweep Margin Offset",
        description="高速移動するコライダー判定時の追加安全マージン (m)",
        default=0.05,
        min=0.001,
        max=0.5,
        unit='LENGTH',
    )
    # 縫合（Sewing）
    enable_sewing: BoolProperty(
        name="Enable Sewing",
        description="縫合線（Sewing Constraints）を有効にする",
        default=False,
    )
    sewing_shrink_speed: FloatProperty(
        name="Shrink Speed",
        description="縫合収縮速度 (m/s)",
        default=1.0,
        min=0.01,
        max=50.0,
    )
    # ピン留め・アタッチメント
    pin_target_object: PointerProperty(
        name="Pin Target",
        type=bpy.types.Object,
        description="ピン留め頂点を追従させるターゲットオブジェクト",
    )
    pin_target_bone: bpy.props.StringProperty(
        name="Pin Target Bone",
        description="ターゲットがアーマチュアの場合に追従させるボーン名",
        default="",
    )
    pin_vertex_group: bpy.props.StringProperty(
        name="Pin Vertex Group",
        description="ピン留め・追従に使用する頂点グループ名",
        default="Pin",
    )
    pin_color: FloatVectorProperty(
        name="Pin Color",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.45, 0.0, 0.9),
        description="ピン留め頂点のハイライト表示色",
    )
    pin_overlay_interactive_only: BoolProperty(
        name="Pin Overlay Interactive Only",
        description="ピン頂点のハイライトをインタラクティブモード中のみ表示する",
        default=True,
    )
    overlay_depth_test: BoolProperty(
        name="Depth Test (Z)",
        description="ハイライト表示（ピン留め・ドラッグ頂点・伸縮ライン等）に深度テストを適用し、メッシュや物陰に隠れるようにします",
        default=False,
    )
    interactive_realtime_sync: BoolProperty(
        name="Real-time Sync",
        description="描画フレームレートが低下しても、経過時間に応じてシミュレーションステップをまとめて進め、布の動きを実時間通りの自然な速度に保ちます",
        default=True,
    )
    interactive_max_steps: IntProperty(
        name="Max Steps / Frame",
        description="1描画フレームあたりに進める最大シミュレーションステップ数 (1〜8)。急激な負荷スパイク時の無限ループを防止します",
        default=4,
        min=1,
        max=8,
    )
    show_fps_overlay: BoolProperty(
        name="Show FPS Overlay",
        description="インタラクティブシミュレーション実行中に3DビューポートにFPSとフレーム時間を表示します",
        default=True,
    )
    fps_overlay_position: EnumProperty(
        name="FPS Position",
        description="3Dビューポート内のFPS表示位置",
        items=[
            ('TOP_CENTER', "Top Center", "画面中央上部"),
            ('TOP_RIGHT', "Top Right", "画面右上"),
            ('BOTTOM_RIGHT', "Bottom Right", "画面右下"),
            ('BOTTOM_LEFT', "Bottom Left", "画面左下"),
            ('TOP_LEFT', "Top Left", "画面左上"),
        ],
        default='TOP_CENTER',
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
        description="3Dビューポートに伸縮ラインのテンション色を表示する",
        default=True,
    )
    elastic_overlay_interactive_only: BoolProperty(
        name="Elastic Overlay Interactive Only",
        description="伸縮ラインのハイライトをインタラクティブモード中のみ表示する",
        default=True,
    )
    # トポロジー・分割モード (Triangulation & Topology)
    triangulation_mode: EnumProperty(
        name="Triangulation Mode",
        description="四角面（Quad）メッシュの対角線バイアス対策と分割モード",
        items=[
            ('DYNAMIC_DIAGONAL', "Dynamic Diagonal (Lightweight)", "【推奨】シミュレーション中は頂点数増加ゼロ。終了時に歪み（Strain）から最適な対角線で2分割"),
            ('CROSS_SUBDIV', "Cross Subdivision (Poke)", "【高精度】中心頂点を追加して4分割。ドーム状の強い突起も表現可能だが負荷は高め"),
            ('NONE', "None (Original)", "四角面のまま（Blenderの標準対角線で計算）"),
        ],
        default='DYNAMIC_DIAGONAL',
    )
    # 動的対角線分割オプション
    dynamic_preserve_flat: BoolProperty(
        name="Preserve Flat Quads",
        description="平坦な四角面は2分割せず四角面のまま保持する",
        default=False,
    )
    dynamic_flatness_threshold: FloatProperty(
        name="Flatness Angle",
        description="平坦と判定する最大角度（度）",
        default=5.0,
        min=0.5,
        max=45.0,
    )
    auto_triangulate_on_stop: BoolProperty(
        name="Auto Triangulate on Stop",
        description="シミュレーション停止時に自動で最適対角線分割を実行する",
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
        description="四角面を中心点で4分割し、対角線バイアスを解消して等方的なシワを表現する",
        default=False,
        update=_update_cross_subdiv,
    )
    post_process_mode: EnumProperty(
        name="Post-Process",
        description="シミュレーション後のメッシュ後処理モード",
        items=[
            ('OPTIMAL_TRI', "Optimal 2 Triangles", "シワの稜線に沿った最適な対角線で2分割"),
            ('QUAD', "Restore Quad", "中心頂点を削除して元の四角面に戻す"),
            ('ADAPTIVE', "Adaptive", "平坦部はQuad、シワ部は最適2三角面に自動判定"),
            ('KEEP', "Keep Cross (4 Triangles)", "十字分割のまま保持"),
        ],
        default='OPTIMAL_TRI',
    )
    adaptive_flatness_threshold: FloatProperty(
        name="Flatness Threshold",
        description="アダプティブ判定時の平坦度閾値（度）",
        default=5.0,
        min=0.5,
        max=45.0,
    )
    auto_post_process: BoolProperty(
        name="Auto Post-Process on Stop",
        description="シミュレーション停止時に自動で後処理を実行する",
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


class TareminColliderAnimSettings(PropertyGroup):
    """コライダーアニメーション駆動設定"""
    enabled: BoolProperty(
        name="Animation Enabled",
        description="コライダーのアニメーション変形駆動を有効にする",
        default=False,
    )
    target_type: EnumProperty(
        name="Target Type",
        description="アニメーション対象の種類",
        items=[
            ('SHAPE_KEY', "Shape Key", "メッシュのシェイプキー値アニメーション"),
            ('POSE_BLEND', "Pose Blend", "2つのポーズスナップショット間のブレンド (0.0〜1.0)"),
            ('ACTION', "Action", "既存アクションの指定フレーム区間再生"),
        ],
        default='POSE_BLEND',
    )
    # シェイプキー設定
    shape_key_name: StringProperty(
        name="Shape Key",
        description="駆動するシェイプキーの名前",
        default="",
    )
    start_value: FloatProperty(
        name="Start Value",
        description="アニメーション開始時のシェイプキー値",
        default=0.0,
    )
    end_value: FloatProperty(
        name="End Value",
        description="アニメーション終了時のシェイプキー値",
        default=1.0,
    )
    # ポーズブレンド設定
    armature_obj: PointerProperty(
        name="Armature",
        type=bpy.types.Object,
        description="ポーズを適用する対象アーマチュアオブジェクト（未指定時は自動探索）",
        poll=lambda s, o: o.type == 'ARMATURE',
    )
    start_pose_data: StringProperty(
        name="Start Pose (0.0)",
        description="初期姿勢スナップショットデータ (JSON)",
        default="",
    )
    target_pose_data: StringProperty(
        name="Target Pose (1.0)",
        description="目標姿勢スナップショットデータ (JSON)",
        default="",
    )
    # アクション設定
    action: PointerProperty(
        name="Action",
        type=bpy.types.Action,
        description="再生するアニメーションアクション",
    )
    frame_start: IntProperty(
        name="Start Frame",
        description="アクション再生開始フレーム",
        default=1,
    )
    frame_end: IntProperty(
        name="End Frame",
        description="アクション再生終了フレーム",
        default=60,
    )
    # 再生・サイクル設定
    play_mode: EnumProperty(
        name="Play Mode",
        description="アニメーション再生方式",
        items=[
            ('ONCE', "Once", "一度だけ再生して終了姿勢で停止"),
            ('REPEAT', "Repeat", "再生後、先頭に戻って繰り返し"),
            ('PINGPONG', "Ping-Pong", "再生後、逆再生して往復"),
        ],
        default='ONCE',
    )
    cycle_frames: IntProperty(
        name="Cycle Frames",
        description="1サイクルにかけるフレーム数 (例: 60フレーム = 1秒)",
        default=60,
        min=1,
        max=10000,
    )
    loop_count: IntProperty(
        name="Loop Count",
        description="リピート / 往復の繰り返し回数",
        default=1,
        min=1,
        max=1000,
    )
    infinite_loop: BoolProperty(
        name="Infinite",
        description="無限にアニメーションを繰り返す",
        default=False,
    )
    easing: EnumProperty(
        name="Easing",
        description="補間加減速方式 (Smoothは始点・終点で滑らかに減速し布の暴れを防ぐ)",
        items=[
            ('SMOOTH', "Smooth (Ease In-Out)", "スムーズステップ加減速 (推奨: 衝撃を抑える)"),
            ('LINEAR', "Linear", "等速直線補間"),
        ],
        default='SMOOTH',
    )
    progress: FloatProperty(
        name="Progress",
        description="現在の進行度 (0.0〜1.0)。スライダー操作で手動スクラブ可能",
        default=0.0,
        min=0.0,
        max=1.0,
        precision=3,
        update=_on_anim_progress_updated,
    )


class TareminColliderSettings(PropertyGroup):
    is_collider: BoolProperty(
        name="Collider Enabled",
        description="このオブジェクトを剛体コライダーとして登録する",
        default=False,
    )
    enabled: BoolProperty(
        name="Collider Active",
        description="このコライダーの衝突判定を有効にする（OFFで一時的に無効化）",
        default=True,
    )
    last_collider_preset: StringProperty(
        name="Collider Preset",
        description="直近に適用されたコライダープリセット",
        default="",
    )
    collider_type: EnumProperty(
        name="Collider Shape",
        items=[
            ('SPHERE', "Sphere", "球コライダー"),
            ('CAPSULE', "Capsule", "カプセルコライダー"),
            ('PLANE', "Plane", "平面コライダー"),
            ('MESH', "Mesh", "メッシュコライダー (カスタムポリゴンメッシュ)"),
            ('BONE_SDF', "Bone SDF", "ボーン局所SDFコライダー (素体・キャラクタ向け)"),
        ],
        default='MESH',
    )
    sdf_resolution: EnumProperty(
        name="SDF Resolution",
        description="各ボーンローカルSDFのテクスチャ解像度",
        items=[
            ('32', "32 (Low)", "軽量・高速 (32x32x32, 約131KB/ボーン)"),
            ('64', "64 (Standard)", "標準・推奨 (64x64x64, 約1MB/ボーン)"),
            ('128', "128 (High)", "高精細 (128x128x128, 約8MB/ボーン)"),
            ('CUSTOM', "Custom", "カスタム解像度指定"),
        ],
        default='64',
    )
    sdf_resolution_custom: IntProperty(
        name="Custom Resolution",
        description="カスタムSDF解像度 (16〜512)",
        default=128,
        min=16,
        max=512,
    )
    sdf_margin: FloatProperty(
        name="SDF Margin",
        description="ボーンローカルAABBのマージン比率",
        default=0.2,
        min=0.05,
        max=1.0,
    )
    weight_threshold: FloatProperty(
        name="Weight Threshold",
        description="ボーン影響度として認識する最小ウェイト閾値（ゴミウェイトの除外）",
        default=0.02,
        min=0.001,
        max=0.5,
    )
    blend_k: FloatProperty(
        name="Blend Smoothness",
        description="関節部での複数ボーンSDF合成の滑らかさ・ブレンド半径 (m)",
        default=0.05,
        min=0.001,
        max=0.5,
        unit='LENGTH',
    )
    sdf_update_mode: EnumProperty(
        name="SDF Update Mode",
        description="ボーンSDFの更新方式",
        items=[
            ('STATIC', "Static (Fastest)", "静止ポーズで事前ベイクした剛体SDFを使用（最高速・通常推奨）"),
            ('DYNAMIC_GPU', "Dynamic Full GPU", "GPUスキニング(LBS)とGPU内SDF更新によりアニメーション変形に毎フレーム追従"),
        ],
        default='STATIC',
    )
    sdf_dynamic_update_interval: IntProperty(
        name="Update Interval",
        description="動的SDFの再計算間隔（フレーム数）。1で毎フレーム、2で2フレームごと（負荷軽減）",
        default=1,
        min=1,
        max=10,
    )
    sdf_cache_enabled: BoolProperty(
        name="Cache SDF",
        description="SDFベイク結果をディスクキャッシュし、次回以降即時ロードする",
        default=True,
    )
    enable_joint_mesh: BoolProperty(
        name="Joint Mesh Hybrid",
        description="関節部（腰・背骨・首など）の複数ボーンブレンド領域のみ部分メッシュコライダーを併用し、剛体SDFの角ばり突出を解消する（激しい屈曲アニメーション時に推奨）",
        default=True,
    )
    joint_weight_threshold: FloatProperty(
        name="Joint Blend Threshold",
        description="関節部と判定する最大ボーンウェイトの閾値（これ未満のブレンド頂点を含む面を部分メッシュ化）",
        default=0.85,
        min=0.5,
        max=0.99,
        precision=2,
    )
    joint_rotation_threshold: FloatProperty(
        name="Activation Angle",
        description="関節メッシュを動的同期する最小屈曲角（度）。0で常時全同期、2度程度で曲がった関節のみ同期し高速化",
        default=2.0,
        min=0.0,
        max=45.0,
        precision=1,
    )
    radius: FloatProperty(
        name="Radius",
        description="コライダー半径 (m)",
        default=0.5,
        min=0.001,
        max=10.0,
        unit='LENGTH',
    )
    thickness: FloatProperty(
        name="Thickness",
        description="メッシュコライダーの表面厚み (m)",
        default=0.005,
        min=0.0001,
        max=0.5,
        unit='LENGTH',
    )
    single_sided: BoolProperty(
        name="Single Sided",
        description="片面衝突判定を有効化。メッシュ表面（法線方向）からの侵入を遮断し、裏側へめり込んだ場合も法線方向の表面へ押し戻して貫通を防止します",
        default=True,
    )
    friction: FloatProperty(
        name="Friction",
        description="摩擦係数",
        default=0.5,
        min=0.0,
        max=1.0,
    )
    restitution: FloatProperty(
        name="Restitution",
        description="衝突時の反発係数 (0.0: 完全非弾性・跳ね返りなし, 1.0: 完全弾性)",
        default=0.0,
        min=0.0,
        max=1.0,
    )
    anim: PointerProperty(type=TareminColliderAnimSettings)


def register():
    for cls in (TareminClothElasticGroup, TareminClothObjectSettings, TareminColliderAnimSettings, TareminColliderSettings):
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            pass
    bpy.types.Object.taremin_cloth = PointerProperty(type=TareminClothObjectSettings)
    bpy.types.Object.taremin_collider = PointerProperty(type=TareminColliderSettings)
    bpy.types.Scene.taremin_cloth_fast_playback = BoolProperty(
        name="Fast Playback",
        description="タイムライン再生中にシミュレーション対象外オブジェクトのモディファイアを一時バイパスしてBlenderのDepsgraph負荷を軽減します",
        default=False,
    )
    bpy.types.Scene.taremin_cloth_config_presets_json = StringProperty(
        name="Config Presets JSON",
        description="保存されたシミュレーション構成プリセット（JSON）",
        default="{}",
    )
    bpy.types.Scene.taremin_cloth_active_config_preset = StringProperty(
        name="Active Config Preset",
        description="現在適用されている構成プリセット名",
        default="",
    )


def unregister():
    if hasattr(bpy.types.Scene, "taremin_cloth_active_config_preset"):
        del bpy.types.Scene.taremin_cloth_active_config_preset
    if hasattr(bpy.types.Scene, "taremin_cloth_config_presets_json"):
        del bpy.types.Scene.taremin_cloth_config_presets_json
    if hasattr(bpy.types.Scene, "taremin_cloth_fast_playback"):
        del bpy.types.Scene.taremin_cloth_fast_playback
    if hasattr(bpy.types.Object, "taremin_cloth"):
        del bpy.types.Object.taremin_cloth
    if hasattr(bpy.types.Object, "taremin_collider"):
        del bpy.types.Object.taremin_collider
    for cls in (TareminColliderSettings, TareminColliderAnimSettings, TareminClothObjectSettings, TareminClothElasticGroup):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass

