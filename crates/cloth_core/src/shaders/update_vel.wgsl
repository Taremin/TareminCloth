struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct SimParams {
    gravity: vec4<f32>, // xyz: gravity vector, w: dt
    damping: f32,
    substeps: u32,
    num_vertices: u32,
    num_distance_constraints: u32,
    num_bending_constraints: u32,
    num_sewing_constraints: u32,
    sewing_compliance: f32,
    enable_sewing_lock: f32,
    sewing_lock_distance: f32,
    tension_damping: f32,
    compression_damping: f32,
    shear_damping: f32,
    bending_damping: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_vertices) {
        return;
    }

    var v = vertices[index];
    let dt = max(params.gravity.w, 1e-8);

    if (v.inv_mass > 0.0) {
        // 空気減衰は predict.wgsl で1回のみ適用済みのため、位置差分から純粋に速度を更新する
        v.velocity = (v.prev_pos - v.position) / dt;
        v.position = v.prev_pos;
    } else {
        v.velocity = vec3<f32>(0.0);
        v.position = v.prev_pos;
    }

    vertices[index] = v;
}
