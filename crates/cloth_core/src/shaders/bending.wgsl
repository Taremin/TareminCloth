struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct GpuBendingConstraint {
    v0: u32,
    v1: u32,
    v2: u32,
    v3: u32,
    rest_length: f32,
    compliance: f32,
    _pad0: f32,
    _pad1: f32,
};

struct SimParams {
    gravity: vec4<f32>, // xyz: gravity vector, w: dt
    damping: f32,
    substeps: u32,
    num_vertices: u32,
    num_distance_constraints: u32,
    num_bending_constraints: u32,
};

struct DispatchInfo {
    color_offset: u32,
    color_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> constraints: array<GpuBendingConstraint>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<uniform> dispatch_info: DispatchInfo;

const EPSILON: f32 = 1e-7;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let local_idx = global_id.x;
    if (local_idx >= dispatch_info.color_count) {
        return;
    }

    let constraint_idx = dispatch_info.color_offset + local_idx;
    let c = constraints[constraint_idx];

    // compliance が非常に大きい（無制限）または stiffness=0 の場合はスキップ
    if (c.compliance >= 1e9) {
        return;
    }

    let w2 = vertices[c.v2].inv_mass;
    let w3 = vertices[c.v3].inv_mass;
    let w_sum = w2 + w3;

    if (w_sum <= EPSILON) {
        return;
    }

    let p2 = vertices[c.v2].prev_pos;
    let p3 = vertices[c.v3].prev_pos;

    let delta = p2 - p3;
    let dist = length(delta);

    if (dist <= EPSILON) {
        return;
    }

    let dir = delta / dist;
    let dt = params.gravity.w;
    let alpha = c.compliance / (dt * dt);

    let c_val = dist - c.rest_length;
    let delta_lambda = -c_val / (w_sum + alpha);

    let corr2 = w2 * delta_lambda * dir;
    let corr3 = -w3 * delta_lambda * dir;

    vertices[c.v2].prev_pos += corr2;
    vertices[c.v3].prev_pos += corr3;
}
