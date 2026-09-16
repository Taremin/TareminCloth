struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct GpuSewingConstraint {
    v0: u32,
    v1: u32,
    current_rest_len: f32,
    target_rest_len: f32,
    shrink_speed: f32,
    compliance: f32,
    lock_on_close: f32,
    _pad1: f32,
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
};

struct DispatchInfo {
    color_offset: u32,
    color_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read_write> sewing_constraints: array<GpuSewingConstraint>;
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
    var c = sewing_constraints[constraint_idx];

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

    let dt = params.gravity.w;
    // 自然長の短縮 (shrink)
    c.current_rest_len = max(c.target_rest_len, c.current_rest_len - c.shrink_speed * dt);
    sewing_constraints[constraint_idx] = c;

    let dir = delta / dist;

    // 密着ロック (Lock When Closed)
    // 自然長が目標自然長 (0.0) に達し、かつ距離が布厚み程度以下に近接している場合、
    // enable_sewing_lock > 0.5 であれば実効コンプライアンスを 0.0 (完全非伸縮) として隙間の再開口を防止
    var effective_compliance = params.sewing_compliance;
    let lock_thresh = max(vertices[c.v0].thickness + vertices[c.v1].thickness, 0.005);
    if (params.enable_sewing_lock > 0.5 && c.current_rest_len <= c.target_rest_len + 1e-4 && dist <= lock_thresh) {
        effective_compliance = 0.0;
    }

    let alpha = effective_compliance / (dt * dt);
    let c_val = dist - c.current_rest_len;
    let delta_lambda = -c_val / (w_sum + alpha);

    vertices[c.v0].prev_pos += w0 * delta_lambda * dir;
    vertices[c.v1].prev_pos -= w1 * delta_lambda * dir;
}

