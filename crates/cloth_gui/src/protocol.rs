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
    pub coupled_mode: u32, // 0: OFF, 1: RELAXATION, 3: FULL_COUPLED
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
    #[serde(default)]
    pub enable_edge_collision: bool,
    #[serde(default = "default_one")]
    pub edge_margin_scale: f32,
    #[serde(default)]
    pub edge_margin_offset: f32,
}

fn default_coupled_mode() -> u32 { 1 }
fn default_post_relax() -> u32 { 2 }
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
    pub layer_id: u32,
    pub thickness: f32,
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
    pub workgroup_size: u32,
    pub solver_mode: u32,
    #[serde(default = "default_solver_iters")]
    pub solver_iterations: u32,
    #[serde(default = "default_substeps")]
    pub substeps: u32,
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
fn default_substeps() -> u32 { 20 }
fn default_gravity() -> [f32; 3] { [0.0, 0.0, -9.81] }

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
    pub solver_iterations: Option<u32>,
    #[serde(default)]
    pub target_fps: Option<f32>,
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

