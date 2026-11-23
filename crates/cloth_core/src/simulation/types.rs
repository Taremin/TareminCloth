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

/// 距離拘束原子加算ソルバー用アキュムレータ（WGSL `AtomicAccum` と 1:1 対応・16 bytes / vertex）
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAtomicAccum {
    /// X変位合計 (固定小数点 i32)
    pub dx: i32,
    /// Y変位合計 (固定小数点 i32)
    pub dy: i32,
    /// Z変位合計 (固定小数点 i32)
    pub dz: i32,
    /// 加算回数
    pub count: u32,
}

/// ドレープガイド Uniform パラメータ（サイズ: 96バイト, 16Bアライメント準拠）
/// `bone_origin.w` は径方向カットオフ影響半径 (m, `wrinkle_field.wgsl` で使用)。
/// `valley_window` / `crest_window` は径方向グラデーション窓 (m)。
/// サイズ変更（96B→）は WGSL・Rust・Python・alignmentテストの同時変更になるため行わない。
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WrinkleFieldParams {
    pub bone_origin: [f32; 4],     // xyz: origin, w: influence_radius
    pub bone_axis: [f32; 4],       // xyz: axis, w: bone_radius
    pub bone_normal: [f32; 4],     // xyz: normal, w: stiffness
    pub bone_binormal: [f32; 4],   // xyz: binormal, w: blend_weight
    pub z_range: [f32; 2],         // x: z_min, y: z_max (ボーン長手方向のUVマッピング範囲)
    pub r_range: [f32; 2],         // x: r_min, y: r_max (径方向のターゲット正規化範囲)
    pub enabled: u32,              // 有効フラグ (0 or 1)
    pub use_texture: u32,          // 2Dテクスチャモードフラグ (0 or 1)
    pub valley_window: f32,        // 谷の径方向グラデーション窓 (m, 既定 0.015)
    pub crest_window: f32,         // 山の径方向グラデーション窓 (m, 既定 0.020)
}

impl Default for WrinkleFieldParams {
    fn default() -> Self {
        // bytemuck::Zeroable のゼロ埋めではなく、窓の既定値を明示する。
        // ゼロ窓は dist_factor の飽和（全域最大強度）を招くため避ける。
        Self {
            bone_origin: [0.0; 4],
            bone_axis: [0.0; 4],
            bone_normal: [0.0; 4],
            bone_binormal: [0.0; 4],
            z_range: [0.0; 2],
            r_range: [0.0; 2],
            enabled: 0,
            use_texture: 0,
            valley_window: 0.015,
            crest_window: 0.020,
        }
    }
}

/// 自己衝突アキュムレータバッファ（WGSL `SelfCollisionAccum` と 1:1 対応・64 bytes / vertex）
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSelfCollisionAccum {
    /// ペナルティ（近接離散反発）変位 X (固定小数点 i32)
    pub dx: i32,
    /// ペナルティ変位 Y (固定小数点 i32)
    pub dy: i32,
    /// ペナルティ変位 Z (固定小数点 i32)
    pub dz: i32,
    /// ペナルティ接触カウント
    pub count: u32,
    /// ピン固定相手との離散接触変位 X (固定小数点 i32)
    pub pin_dx: i32,
    /// ピン固定相手との離散接触変位 Y (固定小数点 i32)
    pub pin_dy: i32,
    /// ピン固定相手との離散接触変位 Z (固定小数点 i32)
    pub pin_dz: i32,
    /// ピン固定相手との離散接触カウント
    pub pin_count: u32,
    /// 真の CCD 正方向最大変位 X (固定小数点 i32, atomicMax)
    pub ccd_pos_x: i32,
    /// 真の CCD 正方向最大変位 Y (固定小数点 i32, atomicMax)
    pub ccd_pos_y: i32,
    /// 真の CCD 正方向最大変位 Z (固定小数点 i32, atomicMax)
    pub ccd_pos_z: i32,
    /// 真の CCD 負方向最小変位 X (固定小数点 i32, atomicMin)
    pub ccd_neg_x: i32,
    /// 真の CCD 負方向最小変位 Y (固定小数点 i32, atomicMin)
    pub ccd_neg_y: i32,
    /// 真の CCD 負方向最小変位 Z (固定小数点 i32, atomicMin)
    pub ccd_neg_z: i32,
    /// 多層布 Untangling 用の最近傍内層頂点候補 (`j + 1`, `0` = 候補なし)
    pub untangle_cand_v: u32,
    /// E-E CCD 非ゼロ変位書き込みカウント（高密度衝突時の協調減衰判定用）
    pub ee_ccd_count: u32,
}



