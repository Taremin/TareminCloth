struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct GpuDistanceConstraint {
    v0: u32,
    v1: u32,
    rest_length: f32,
    tension_compliance: f32,
    compression_compliance: f32,
    constraint_type: u32, // 0: Stretch (Tension/Compression), 1: Shear
    _pad0: f32,
    _pad1: f32,
};


struct SimParams {
    gravity: vec4<f32>, // xyz: gravity vector, w: dt
    damping: f32,
    substeps: u32,
    num_vertices: u32,
    num_distance_constraints: u32,
};

struct DispatchInfo {
    color_offset: u32,
    color_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> constraints: array<GpuDistanceConstraint>;
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

    let w0 = vertices[c.v0].inv_mass;
    let w1 = vertices[c.v1].inv_mass;
    let w_sum = w0 + w1;

    if (w_sum <= EPSILON) {
        return;
    }

    let p0 = vertices[c.v0].prev_pos;
    let p1 = vertices[c.v1].prev_pos;

    let delta = p0 - p1;
    let dist = length(delta);

    if (dist <= EPSILON) {
        return;
    }

    let c_val = dist - c.rest_length;
    // 伸び(c_val >= 0)なら tension_compliance、縮み(c_val < 0)なら compression_compliance
    let compliance = select(c.compression_compliance, c.tension_compliance, c_val >= 0.0);

    if (compliance >= 1e9) {
        return;
    }

    let dir = delta / dist;
    let dt = params.gravity.w;
    let alpha = compliance / (dt * dt);

    let delta_lambda = -c_val / (w_sum + alpha);

    // 安定緩和係数: 制約競合時の発振を防ぎ安定収束
    const OVER_RELAXATION: f32 = 1.0;
    let corr0 = w0 * delta_lambda * dir * OVER_RELAXATION;
    let corr1 = -w1 * delta_lambda * dir * OVER_RELAXATION;

    vertices[c.v0].prev_pos += corr0;
    vertices[c.v1].prev_pos += corr1;
}
