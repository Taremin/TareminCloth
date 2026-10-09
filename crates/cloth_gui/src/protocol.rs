use serde::{Deserialize, Serialize};

/// BlenderとTaremin Cloth GUI間で送受信するコマンドメッセージ (JSON)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum GuiCommand {
    /// シーンメッシュと拘束の初期化
    InitScene(SceneInitData),
    /// コライダーの動的更新
    UpdateColliders {
        colliders: Vec<GuiColliderData>,
        mesh_triangles: Vec<GuiMeshTriangleData>,
    },
    /// ボーン変換行列の動的更新
    UpdateBoneTransforms {
        transforms: Vec<[[f32; 4]; 4]>,
    },
    /// 伸縮グループ（ゴム紐）スケールの動的更新
    UpdateElasticScales {
        edge_indices: Vec<u32>,
        scales: Vec<f32>,
    },
    /// 動的アタッチメントピンの更新
    UpdatePins {
        vertex_indices: Vec<u32>,
        positions: Vec<[f32; 3]>,
        weights: Vec<f32>,
    },
    /// シミュレーション進行の開始
    Play,
    /// シミュレーションの一時停止
    Pause,
    /// シミュレーションを初期状態（レスト位置）にリセット
    Reset,
    /// 1ステップ（dt, substeps）だけ前進
    Step {
        dt: f32,
        substeps: u32,
        solver_iterations: u32,
    },
    /// 物理パラメータのリアルタイム更新
    SetParams(GuiParamsUpdate),
    /// 最新の変形頂点座標を要求
    GetLatestCoords,
    /// シミュレーションステータス（FPS、フレーム番号等）を要求（GPUリードバックなし）
    GetStatus,
    /// 現在の変形ポーズを確定データとして要求
    ApplyPose,
    /// アプリケーション終了要求
    Quit,
}

/// 解析コライダー定義 (球, カプセル, 平面)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GuiColliderData {
    pub collider_type: u32, // 0: Sphere, 1: Capsule, 2: Plane
    pub friction: f32,
    pub restitution: f32,
    pub point_a: [f32; 3],  // Sphere中心, Capsule始点, Plane通過点
    pub radius: f32,
    pub point_b: [f32; 3],  // Capsule終点, Plane法線
}

/// メッシュコライダー三角形定義
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GuiMeshTriangleData {
    pub p0: [f32; 3],
    pub p1: [f32; 3],
    pub p2: [f32; 3],
    pub friction: f32,
    pub thickness: f32,
    pub restitution: f32,
    pub flags: u32,
}

/// 自己衝突オプション
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GuiSelfCollisionData {
    pub enabled: bool,
    #[serde(default = "default_coupled_mode")]
    pub coupled_mode: u32, // 0: OFF, 1: RELAXATION, 2: PER_ITERATION, 3: FULL_COUPLED
    #[serde(default = "default_post_relax")]
    pub post_relaxation_iters: u32,
    #[serde(default = "default_relief_factor")]
    pub relief_factor: f32,
    #[serde(default = "default_max_disp")]
    pub max_displacement_ratio: f32,
    #[serde(default = "default_true")]
    pub exclude_neighbors: bool,
    #[serde(default = "default_true")]
    pub enable_normal_untangling: bool,
    #[serde(default = "default_max_search_iters")]
    pub max_iterations: u32,
    #[serde(default = "default_substep_interval")]
    pub substep_interval: u32,
    #[serde(default)]
    pub enable_edge_collision: bool,
    #[serde(default = "default_one")]
    pub edge_margin_scale: f32,
    #[serde(default)]
    pub edge_margin_offset: f32,
    #[serde(default)]
    pub enable_pair_cache: bool,
    #[serde(default = "default_pair_margin_mode")]
    pub pair_cache_margin_mode: u32,
    #[serde(default = "default_pair_margin")]
    pub pair_cache_safety_margin: f32,
    #[serde(default = "default_horizon_scale")]
    pub pair_cache_horizon_scale: f32,
    #[serde(default = "default_max_horizon")]
    pub pair_cache_max_horizon: f32,
    #[serde(default = "default_pair_max")]
    pub pair_cache_max_pairs: u32,
    #[serde(default = "default_true")]
    pub enable_pair_cache_final_fallback: bool,
    #[serde(default)]
    pub self_collision_algorithm: u32,
    #[serde(default)]
    pub coupled_collider: bool,
    #[serde(default = "default_substep_interval")]
    pub ee_substep_interval: u32,
}

fn default_pair_margin_mode() -> u32 { 1 }
fn default_pair_margin() -> f32 { 0.005 }
fn default_horizon_scale() -> f32 { 1.3 }
fn default_max_horizon() -> f32 { 0.02 }
fn default_pair_max() -> u32 { 65536 }

fn default_coupled_mode() -> u32 { 1 }
fn default_post_relax() -> u32 { 1 }
fn default_max_search_iters() -> u32 { 256 }
fn default_substep_interval() -> u32 { 1 }
fn default_relief_factor() -> f32 { 0.2 }
fn default_max_disp() -> f32 { 0.2 }
fn default_true() -> bool { true }
fn default_one() -> f32 { 1.0 }

/// ボーンSDFコライダーデータ
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GuiBoneSdfData {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub texture_base64: String,
    pub bone_infos: Vec<[f32; 20]>,
    #[serde(default)]
    pub bone_transforms: Vec<[[f32; 4]; 4]>,
}

/// 伸縮グループデータ
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GuiElasticBandData {
    pub edge_indices: Vec<u32>,
    pub scales: Vec<f32>,
}

/// シーン初期化データ
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneInitData {
    pub object_name: String,
    pub positions: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub edges: Vec<[u32; 2]>,
    pub sewing_springs: Option<Vec<[u32; 2]>>,
    pub inv_masses: Vec<f32>,
    #[serde(default)]
    pub layer_ids: Option<Vec<u32>>,
    pub layer_id: u32,
    pub thickness: f32,
    #[serde(default = "default_areal_density")]
    pub areal_density: f32,
    /// 長距離拘束 (2ホップ) の有効化。メッシュ構築時のみ有効 (init-only)。
    #[serde(default)]
    pub enable_coarse_constraints: bool,
    pub stiffness: f32,
    pub compression_stiffness: f32,
    pub shear_stiffness: f32,
    pub bending_stiffness: f32,
    #[serde(default)]
    pub air_damping: f32,
    #[serde(default)]
    pub tension_damping: f32,
    #[serde(default)]
    pub compression_damping: f32,
    #[serde(default)]
    pub shear_damping: f32,
    #[serde(default)]
    pub bending_damping: f32,
    #[serde(default = "default_gravity")]
    pub gravity: [f32; 3],
    #[serde(default = "default_one")]
    pub gravity_scale: f32,
    pub sewing_shrink_speed: f32,
    #[serde(default = "default_sewing_stiffness")]
    pub sewing_stiffness: f32,
    #[serde(default = "default_true")]
    pub enable_sewing_lock: bool,
    #[serde(default = "default_sewing_lock_distance")]
    pub sewing_lock_distance: f32,
    #[serde(default)]
    pub sewing_priority_enabled: bool,
    #[serde(default = "default_sewing_priority_threshold")]
    pub sewing_priority_threshold: f32,
    #[serde(default = "default_sewing_priority_merge_dist")]
    pub sewing_priority_merge_dist: f32,
    #[serde(default = "default_sewing_priority_ramp_frames")]
    pub sewing_priority_ramp_frames: u32,
    #[serde(default = "default_sewing_priority_max_frames")]
    pub sewing_priority_max_frames: u32,
    pub workgroup_size: u32,
    pub solver_mode: u32,
    #[serde(default = "default_solver_iters")]
    pub solver_iterations: u32,
    #[serde(default = "default_substeps")]
    pub substeps: u32,
    #[serde(default)]
    pub enable_adaptive_substep: bool,
    #[serde(default = "default_min_substeps")]
    pub min_substeps: u32,
    #[serde(default = "default_max_substeps")]
    pub max_substeps: u32,
    #[serde(default = "default_true")]
    pub auto_coupled_on_low_substeps: bool,
    #[serde(default = "default_true")]
    pub auto_compensate_iterations: bool,
    #[serde(default = "default_true")]
    pub enable_strain_adaptive: bool,
    #[serde(default = "default_strain_tolerance")]
    pub strain_tolerance: f32,
    pub self_collision: Option<GuiSelfCollisionData>,
    pub bone_sdf: Option<GuiBoneSdfData>,
    #[serde(default)]
    pub colliders: Vec<GuiColliderData>,
    #[serde(default)]
    pub mesh_triangles: Vec<GuiMeshTriangleData>,
    pub elastic_bands: Option<GuiElasticBandData>,
    #[serde(default)]
    pub fps: Option<f32>,
    #[serde(default)]
    pub voxels: Vec<[f32; 7]>,      // [x, y, z, size, r, g, b]
    #[serde(default)]
    pub extra_lines: Vec<[f32; 9]>, // [p0x..z, p1x..z, r, g, b]
}

fn default_solver_iters() -> u32 { 10 }
fn default_sewing_stiffness() -> f32 { 10000.0 }
fn default_sewing_lock_distance() -> f32 { 0.02 }
fn default_areal_density() -> f32 { 0.15 }
fn default_substeps() -> u32 { 20 }
fn default_min_substeps() -> u32 { 4 }
fn default_max_substeps() -> u32 { 64 }
fn default_gravity() -> [f32; 3] { [0.0, 0.0, -9.81] }
fn default_sewing_priority_threshold() -> f32 { 0.9 }
fn default_sewing_priority_merge_dist() -> f32 { 0.005 }
fn default_sewing_priority_ramp_frames() -> u32 { 3 }
fn default_sewing_priority_max_frames() -> u32 { 600 }
fn default_strain_tolerance() -> f32 { 0.008 }

/// 物理パラメータの更新
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GuiParamsUpdate {
    pub gravity: Option<[f32; 3]>,
    pub gravity_scale: Option<f32>,
    pub air_damping: Option<f32>,
    pub tension_damping: Option<f32>,
    pub compression_damping: Option<f32>,
    pub shear_damping: Option<f32>,
    pub bending_damping: Option<f32>,
    pub stiffness: Option<f32>,
    #[serde(default)]
    pub tension_stiffness: Option<f32>,
    pub compression_stiffness: Option<f32>,
    pub shear_stiffness: Option<f32>,
    pub bending_stiffness: Option<f32>,
    #[serde(default)]
    pub substeps: Option<u32>,
    #[serde(default)]
    pub enable_adaptive_substep: Option<bool>,
    #[serde(default)]
    pub min_substeps: Option<u32>,
    #[serde(default)]
    pub max_substeps: Option<u32>,
    pub solver_iterations: Option<u32>,
    #[serde(default)]
    pub target_fps: Option<f32>,
    #[serde(default)]
    pub sewing_priority_enabled: Option<bool>,
    #[serde(default)]
    pub sewing_priority_threshold: Option<f32>,
    #[serde(default)]
    pub sewing_priority_merge_dist: Option<f32>,
    #[serde(default)]
    pub sewing_priority_ramp_frames: Option<u32>,
    #[serde(default)]
    pub sewing_priority_max_frames: Option<u32>,
    #[serde(default)]
    pub sewing_lock_distance: Option<f32>,

    // 自己衝突系オプション
    #[serde(default)]
    pub enable_self_collision: Option<bool>,
    #[serde(default)]
    pub self_collision_algorithm: Option<u32>,
    #[serde(default)]
    pub coupled_mode: Option<u32>,
    #[serde(default)]
    pub coupled_collider: Option<bool>,
    #[serde(default)]
    pub post_relaxation_iters: Option<u32>,
    #[serde(default)]
    pub relief_factor: Option<f32>,
    #[serde(default)]
    pub max_displacement_ratio: Option<f32>,
    #[serde(default)]
    pub exclude_neighbors: Option<bool>,
    #[serde(default)]
    pub enable_normal_untangling: Option<bool>,
    #[serde(default)]
    pub self_collision_max_iterations: Option<u32>,
    #[serde(default)]
    pub substep_interval: Option<u32>,
    #[serde(default)]
    pub ee_substep_interval: Option<u32>,
    #[serde(default)]
    pub enable_edge_collision: Option<bool>,
    #[serde(default)]
    pub edge_margin_scale: Option<f32>,
    #[serde(default)]
    pub edge_margin_offset: Option<f32>,
    #[serde(default)]
    pub enable_pair_cache: Option<bool>,
    #[serde(default)]
    pub pair_margin_mode: Option<u32>,
    #[serde(default)]
    pub pair_safety_margin: Option<f32>,
    #[serde(default)]
    pub pair_horizon_scale: Option<f32>,
    #[serde(default)]
    pub pair_max_horizon: Option<f32>,
    #[serde(default)]
    pub pair_max_pairs: Option<u32>,
    #[serde(default)]
    pub enable_pair_final_fallback: Option<bool>,

    // ソルバー & ひずみ適応 & その他
    #[serde(default)]
    pub solver_mode: Option<u32>,
    #[serde(default)]
    pub workgroup_size: Option<u32>,
    #[serde(default)]
    pub auto_coupled_on_low_substeps: Option<bool>,
    #[serde(default)]
    pub auto_compensate_iterations: Option<bool>,
    #[serde(default)]
    pub enable_strain_adaptive: Option<bool>,
    #[serde(default)]
    pub strain_tolerance: Option<f32>,
    #[serde(default)]
    pub sewing_stiffness: Option<f32>,
    #[serde(default)]
    pub enable_sewing_lock: Option<bool>,
    #[serde(default)]
    pub areal_density: Option<f32>,
}

/// GUIからBlenderへ返送するレスポンスメッセージ (JSON)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum GuiResponse {
    /// 正常終了レスポンス
    Ack { message: String },
    /// エラーレスポンス
    Error { message: String },
    /// シミュレーションステータス
    Status {
        running: bool,
        current_frame: u64,
        num_vertices: u32,
        physics_fps: f32,
        #[serde(default)]
        render_fps: f32,
        step_time_ms: f32,
    },
    /// 変形頂点座標データ (最新フレーム)
    Coords {
        frame_seq: u64,
        num_vertices: u32,
        physics_fps: f32,
        #[serde(default)]
        render_fps: f32,
        step_time_ms: f32,
        positions: Vec<[f32; 3]>,
    },
    /// 確定ポーズデータ
    AppliedPose {
        num_vertices: u32,
        positions: Vec<[f32; 3]>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_client_json_without_new_fields_parses_with_defaults() {
        // 新規キー (max_iterations/substep_interval/sewing_stiffness/enable_sewing_lock)
        // を送らない旧Blenderクライアントでも受信できること
        let sc_json = r#"{"enabled": true, "coupled_mode": 1}"#;
        let sc: GuiSelfCollisionData = serde_json::from_str(sc_json).unwrap();
        assert_eq!(sc.max_iterations, 256);
        assert_eq!(sc.substep_interval, 1);
        let init_json = r#"{
            "object_name": "O", "positions": [], "faces": [], "edges": [],
            "sewing_springs": null, "inv_masses": [], "layer_id": 0,
            "thickness": 0.005, "stiffness": 100.0,
            "compression_stiffness": 100.0, "shear_stiffness": 50.0,
            "bending_stiffness": 1.0, "gravity": [0.0, 0.0, -9.81],
            "sewing_shrink_speed": 0.5, "workgroup_size": 32,
            "solver_mode": 0, "solver_iterations": 2,
            "self_collision": {"enabled": false},
            "bone_sdf": null
        }"#;
        let init: SceneInitData = serde_json::from_str(init_json).unwrap();
        assert_eq!(init.sewing_stiffness, 10000.0);
        assert!(init.enable_sewing_lock);
        assert_eq!(init.sewing_lock_distance, 0.02);
        assert!(!init.enable_adaptive_substep);
        assert_eq!(init.min_substeps, 4);
        assert_eq!(init.max_substeps, 64);
        assert!(init.enable_strain_adaptive);
    }

    #[test]
    fn new_keys_roundtrip() {
        let sc = GuiSelfCollisionData {
            enabled: true,
            coupled_mode: 3,
            post_relaxation_iters: 1,
            relief_factor: 0.2,
            max_displacement_ratio: 0.2,
            exclude_neighbors: true,
            enable_normal_untangling: true,
            max_iterations: 512,
            substep_interval: 2,
            enable_edge_collision: false,
            edge_margin_scale: 1.0,
            edge_margin_offset: 0.0,
            enable_pair_cache: true,
            pair_cache_margin_mode: 1,
            pair_cache_safety_margin: 0.005,
            pair_cache_horizon_scale: 1.3,
            pair_cache_max_horizon: 0.02,
            pair_cache_max_pairs: 32768,
            enable_pair_cache_final_fallback: true,
            self_collision_algorithm: 2,
            coupled_collider: false,
            ee_substep_interval: 1,
        };
        let s = serde_json::to_string(&sc).unwrap();
        let d: GuiSelfCollisionData = serde_json::from_str(&s).unwrap();
        assert_eq!(d.max_iterations, 512);
        assert_eq!(d.substep_interval, 2);
        assert_eq!(d.self_collision_algorithm, 2);
        assert!(!d.coupled_collider);
        assert_eq!(d.ee_substep_interval, 1);
        assert_eq!(d.coupled_mode, 3);

        // mode 2 (PER_ITERATION) が往復で欠落・変換されないこと
        let mut sc2 = d.clone();
        sc2.coupled_mode = 2;
        let s2 = serde_json::to_string(&sc2).unwrap();
        let d2: GuiSelfCollisionData = serde_json::from_str(&s2).unwrap();
        assert_eq!(d2.coupled_mode, 2);
    }
}

