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
// XPBDラグランジュ乗数 (拘束毎、サブステップ先頭で零化)
@group(0) @binding(4) var<storage, read_write> lambdas: array<f32>;

const EPSILON: f32 = 1e-7;
// UI減衰係数 1.0 あたり 0.01ms の剛性比例 Rayleigh 緩和時間定数 (C = tau * K -> gamma = tau / dt)
const RAYLEIGH_TIME_SCALE: f32 = 1e-5;

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
    let dt = max(params.gravity.w, 1e-8);
    let alpha = compliance / (dt * dt);

    // 拘束種別・伸縮方向に応じた内部Rayleigh減衰係数を選択
    let damp_coeff = select(
        select(params.compression_damping, params.tension_damping, c_val >= 0.0),
        params.shear_damping,
        c.constraint_type == 1u
    );
    let tau = damp_coeff * RAYLEIGH_TIME_SCALE;
    let gamma = min(tau / dt, 1e4);
    // サブステップ内の変位から相対歪み速度成分を算出 (position=確定位置, prev_pos=予測・補正位置)
    let rel_disp = (p0 - vertices[c.v0].position) - (p1 - vertices[c.v1].position);
    let v_proj = dot(dir, rel_disp);

    // 正規XPBD + Rayleigh内部減衰: ラムダを反復跨ぎで蓄積し、時間刻み・反復数に非依存な剛性と減衰とする
    let lambda = lambdas[constraint_idx];
    let delta_lambda = (-c_val - alpha * lambda - gamma * v_proj) / ((1.0 + gamma) * w_sum + alpha);
    lambdas[constraint_idx] = lambda + delta_lambda;

    // 安定緩和係数: 制約競合時の発振を防ぎ安定収束
    const OVER_RELAXATION: f32 = 1.0;
    let corr0 = w0 * delta_lambda * dir * OVER_RELAXATION;
    let corr1 = -w1 * delta_lambda * dir * OVER_RELAXATION;

    vertices[c.v0].prev_pos += corr0;
    vertices[c.v1].prev_pos += corr1;
}
