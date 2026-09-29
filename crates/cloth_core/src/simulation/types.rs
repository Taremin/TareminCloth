#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DispatchInfo {
    pub color_offset: u32,
    pub color_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PinParams {
    pub num_pins: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CollisionParams {
    pub num_vertices: u32,
    pub num_colliders: u32,
    pub num_mesh_triangles: u32,
    pub num_clusters: u32,
    pub dt: f32,
    pub edge_margin_scale: f32,
    pub edge_margin_offset: f32,
    pub enable_cluster_culling: u32,
    pub enable_single_sided_recovery: u32,
    pub sweep_margin_offset: f32,
    pub num_bones: u32,
    pub enable_bone_sdf: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NormalParams {
    pub num_vertices: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// ボーンSDFコライダーの静的設定情報
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBoneInfo {
    pub aabb_min: [f32; 4],
    pub aabb_max: [f32; 4],
    pub uvw_scale: [f32; 4],
    pub uvw_offset: [f32; 4],
    pub params: [f32; 4], // x: friction, y: thickness, z: restitution, w: blend_k
}

/// 毎フレーム更新されるボーン変換行列
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBoneTransform {
    pub world_matrix: [[f32; 4]; 4],
    pub inv_world_matrix: [[f32; 4]; 4],
}

/// GPU LBS スキニング用の静止頂点データ（サイズ: 64バイト）
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSkinningVertex {
    pub pos: [f32; 3],
    pub _pad0: f32,
    pub normal: [f32; 3],
    pub _pad1: f32,
    pub bone_indices: [u32; 4],
    pub bone_weights: [f32; 4],
}

/// 動的SDFベイク用のボーン三角形ソース（サイズ: 32バイト）
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBoneTriangleSource {
    pub i0: u32,
    pub i1: u32,
    pub i2: u32,
    pub bone_idx: u32,
    pub w0: f32,
    pub w1: f32,
    pub w2: f32,
    pub _pad: f32,
}

/// シワフィールド旧1Dプロファイルサンプル（サイズ: 32バイト, 16Bアライメント準拠）
/// 注意: 旧1Dプロファイルパスの残骸であり、現行2D-SDFテクスチャ方式では未使用。
/// WGSL・BGL・dispatchからの参照はない。alignmentテストの回帰検出用に型のみ保持する。
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuWrinkleProfileSample {
    pub valley_z: f32,       // 谷の軸方向ターゲット位置 (m)
    pub valley_radius: f32,  // 谷の径方向ターゲット半径 (m, 通常はボーン表面)
    pub crest_z: f32,        // 山の軸方向ターゲット位置 (m)
    pub crest_radius: f32,   // 山の径方向ターゲット半径 (m, 外側)
    pub valley_weight: f32,  // 谷のメタボールポテンシャル強度 (0.0〜1.0)
    pub crest_weight: f32,   // 山のメタボールポテンシャル強度 (0.0〜1.0)
    pub _pad0: f32,
    pub _pad1: f32,
}

/// シワフィールド Uniform パラメータ（サイズ: 96バイト, 16Bアライメント準拠）
/// 注意: `bone_origin.w` および `use_texture` は現在WGSL未読の予約フィールドである。
/// サイズ変更（96B→80B）は WGSL・Rust・Python・alignmentテストの同時変更になるため行わない。
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WrinkleFieldParams {
    pub bone_origin: [f32; 4],     // xyz: origin, w: influence_radius
    pub bone_axis: [f32; 4],       // xyz: axis, w: bone_radius
    pub bone_normal: [f32; 4],     // xyz: normal, w: stiffness
    pub bone_binormal: [f32; 4],   // xyz: binormal, w: blend_weight
    pub z_range: [f32; 2],         // x: z_min, y: z_max (ボーン長手方向のUVマッピング範囲)
    pub r_range: [f32; 2],         // x: r_min, y: r_max (径方向のターゲット正規化範囲)
    pub enabled: u32,              // 有効フラグ (0 or 1)
    pub use_texture: u32,          // 2Dテクスチャモードフラグ (0 or 1)
    pub _pad0: u32,
    pub _pad1: u32,
}


