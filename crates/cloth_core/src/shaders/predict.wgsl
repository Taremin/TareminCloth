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
    _pad0: f32,
    _pad1: f32,
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
    let dt = params.gravity.w;

    if (v.inv_mass <= 0.0) {
        // 固定ピン: 速度ゼロ、予測位置は現在位置
        v.velocity = vec3<f32>(0.0);
        v.prev_pos = v.position;
    } else {
        // 外力（重力）の適用と減衰
        let g = params.gravity.xyz;
        v.velocity = v.velocity * max(0.0, 1.0 - params.damping * dt) + g * dt;
        v.prev_pos = v.position + v.velocity * dt;
    }

    vertices[index] = v;
}
