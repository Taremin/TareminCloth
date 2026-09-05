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

