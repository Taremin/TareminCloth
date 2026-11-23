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
@group(0) @binding(1) var<storage, read> constraints: array<GpuBendingConstraint>;
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

    // シーム跨ぎ曲げ拘束 (_pad0 ≈ 1.0; _pad0 > 1.5 は 1Dストランド曲げ) のゲート処理:
    // 1. 平坦展開距離 (rest_length) よりも開いている場合 (dist >= c.rest_length) は引っ張らず、
    //    折れ曲がり（圧縮方向: dist < c.rest_length）のみに復元力を働かせる
    // 2. 縫合ペア (v0, v1) がまだ閉じきっていない場合 (距離 > c._pad1) はスキップ
    if (c._pad0 > 0.5 && c._pad0 < 1.5) {
        if (dist >= c.rest_length) {
            return;
        }
        let seam_gap = length(vertices[c.v0].prev_pos - vertices[c.v1].prev_pos);
        if (seam_gap > c._pad1) {
            return;
        }
    }

    let dir = delta / dist;
    let dt = max(params.gravity.w, 1e-8);
    let alpha = c.compliance / (dt * dt);
    let tau = params.bending_damping * RAYLEIGH_TIME_SCALE;
    let gamma = min(tau / dt, 1e4);
    // サブステップ内の変位から相対曲げ歪み速度を算出 (position=確定位置, prev_pos=予測・補正位置)
    let rel_disp = (p2 - vertices[c.v2].position) - (p3 - vertices[c.v3].position);
    let v_proj = dot(dir, rel_disp);

    let c_val = dist - c.rest_length;
    // 正規XPBD + Rayleigh曲げ減衰: ラムダを反復跨ぎで蓄積する
    let lambda = lambdas[constraint_idx];
    let delta_lambda = (-c_val - alpha * lambda - gamma * v_proj) / ((1.0 + gamma) * w_sum + alpha);
    lambdas[constraint_idx] = lambda + delta_lambda;

    let corr2 = w2 * delta_lambda * dir;
    let corr3 = -w3 * delta_lambda * dir;

    vertices[c.v2].prev_pos += corr2;
    vertices[c.v3].prev_pos += corr3;
}
