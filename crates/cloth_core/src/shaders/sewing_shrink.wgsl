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
    sewing_lock_distance: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read_write> sewing_constraints: array<GpuSewingConstraint>;
@group(0) @binding(1) var<uniform> params: SimParams;

// 縫合自然長の時間進行パス (サブステップ毎に1回のみ)。
// 収縮速度 (m/s) を時間刻みで進めるため、反復回数・自己衝突モードに非依存の
// 一定速度となる。拘束解決カーネル (sewing.wgsl) 側では収縮を行わない。
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_sewing_constraints) {
        return;
    }

    var c = sewing_constraints[idx];
    let dt = params.gravity.w;
    c.current_rest_len = max(c.target_rest_len, c.current_rest_len - c.shrink_speed * dt);
    sewing_constraints[idx] = c;
}
