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
    constraint_type: u32,
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
    _pad0: f32,
    _pad1: f32,
};

struct AtomicAccum {
    dx: atomic<i32>,
    dy: atomic<i32>,
    dz: atomic<i32>,
    count: atomic<u32>,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> constraints: array<GpuDistanceConstraint>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read_write> accum: array<AtomicAccum>;

const EPSILON: f32 = 1e-7;
const FIXED_SCALE: f32 = 1000000.0;
const INV_FIXED_SCALE: f32 = 1.0 / 1000000.0;

// 1. 全エッジを単一ディスパッチで一斉評価し、頂点ごとのアトミックバッファに変位を加算
@compute @workgroup_size(64)
fn solve_distance_atomic(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let constraint_idx = global_id.x;
    if (constraint_idx >= params.num_distance_constraints) {
        return;
    }

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
    let compliance = select(c.compression_compliance, c.tension_compliance, c_val >= 0.0);

    if (compliance >= 1e9) {
        return;
    }

    let dir = delta / dist;
    let dt = params.gravity.w;
    let alpha = compliance / (dt * dt);

    let delta_lambda = -c_val / (w_sum + alpha);

    let corr0 = w0 * delta_lambda * dir;
    let corr1 = -w1 * delta_lambda * dir;

    // 固定小数点に変換してアトミック加算
    let ix0 = i32(corr0.x * FIXED_SCALE);
    let iy0 = i32(corr0.y * FIXED_SCALE);
    let iz0 = i32(corr0.z * FIXED_SCALE);

    atomicAdd(&accum[c.v0].dx, ix0);
    atomicAdd(&accum[c.v0].dy, iy0);
    atomicAdd(&accum[c.v0].dz, iz0);
    atomicAdd(&accum[c.v0].count, 1u);

    let ix1 = i32(corr1.x * FIXED_SCALE);
    let iy1 = i32(corr1.y * FIXED_SCALE);
    let iz1 = i32(corr1.z * FIXED_SCALE);

    atomicAdd(&accum[c.v1].dx, ix1);
    atomicAdd(&accum[c.v1].dy, iy1);
    atomicAdd(&accum[c.v1].dz, iz1);
    atomicAdd(&accum[c.v1].count, 1u);
}

// 2. 各頂点に累積された平均変位量を適用し、次反復用にアトミックバッファをクリア
@compute @workgroup_size(64)
fn apply_atomic_accum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let v_idx = global_id.x;
    if (v_idx >= params.num_vertices) {
        return;
    }

    let count = atomicLoad(&accum[v_idx].count);
    if (count > 0u) {
        let f_count = f32(count);
        let avg_corr = vec3<f32>(
            f32(atomicLoad(&accum[v_idx].dx)),
            f32(atomicLoad(&accum[v_idx].dy)),
            f32(atomicLoad(&accum[v_idx].dz))
        ) * (INV_FIXED_SCALE / f_count);

        // Jacobi型ソルバーの収束速度を高めるための過緩和係数 (Over-relaxation)
        const OVER_RELAXATION: f32 = 1.25;
        vertices[v_idx].prev_pos += avg_corr * OVER_RELAXATION;

        // 次の反復用にゼロクリア
        atomicStore(&accum[v_idx].dx, 0);
        atomicStore(&accum[v_idx].dy, 0);
        atomicStore(&accum[v_idx].dz, 0);
        atomicStore(&accum[v_idx].count, 0u);
    }
}
