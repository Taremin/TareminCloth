struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct SelfCollisionAccum {
    dx: atomic<i32>,
    dy: atomic<i32>,
    dz: atomic<i32>,
    count: atomic<u32>,
    ccd_dx: atomic<i32>,
    ccd_dy: atomic<i32>,
    ccd_dz: atomic<i32>,
    ccd_count: atomic<u32>,
};

struct SelfCollisionParams {
    cell_size: f32,
    table_size: u32,
    num_vertices: u32,
    _pad0: u32,
    relief_factor: f32,
    max_displacement_ratio: f32,
    enable_relief: u32,
    enable_normal_untangling: u32,
    exclude_neighbors: u32,
    max_search_iterations: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read_write> accum: array<SelfCollisionAccum>;
@group(0) @binding(2) var<uniform> params: SelfCollisionParams;
@group(0) @binding(3) var<storage, read> local_edge_lengths: array<f32>;

const FIXED_SCALE: f32 = 1000000.0;
const INV_FIXED_SCALE: f32 = 1.0 / FIXED_SCALE;
const EPSILON: f32 = 1e-7;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_vertices) {
        return;
    }

    var v = vertices[index];
    let is_pinned = (v.inv_mass <= 0.0);

    var final_disp = vec3<f32>(0.0);

    // 1. ペナルティ反発 (近接反発): 接触密度に応じた適切な剛性を維持し、緩和とステップクランプを適用 (非ピン頂点のみ)
    let count = atomicLoad(&accum[index].count);
    if (count > 0u) {
        if (!is_pinned) {
            let raw_dx = atomicLoad(&accum[index].dx);
            let raw_dy = atomicLoad(&accum[index].dy);
            let raw_dz = atomicLoad(&accum[index].dz);

            let total_disp = vec3<f32>(
                f32(raw_dx),
                f32(raw_dy),
                f32(raw_dz)
            ) * INV_FIXED_SCALE;

            var eff_disp = total_disp / max(1.0, sqrt(f32(count)));

            if (params.enable_relief != 0u) {
                eff_disp = eff_disp * params.relief_factor;
            }

            let p_len = length(eff_disp);
            let ratio = select(0.50, params.max_displacement_ratio, params.max_displacement_ratio > 0.0);
            let max_step = min(local_edge_lengths[index] * ratio, 0.10);
            if (p_len > max_step && p_len > EPSILON) {
                eff_disp = (eff_disp / p_len) * max_step;
            }

            final_disp += eff_disp;
        }

        atomicStore(&accum[index].dx, 0);
        atomicStore(&accum[index].dy, 0);
        atomicStore(&accum[index].dz, 0);
        atomicStore(&accum[index].count, 0u);
    }

    // 2. CCD (連続衝突判定 & 貫通阻止ハード制約): 減衰・クランプをかけずに100%確実に適用
    let ccd_count = atomicLoad(&accum[index].ccd_count);
    if (ccd_count > 0u) {
        let raw_ccd_x = atomicLoad(&accum[index].ccd_dx);
        let raw_ccd_y = atomicLoad(&accum[index].ccd_dy);
        let raw_ccd_z = atomicLoad(&accum[index].ccd_dz);

        let total_ccd = vec3<f32>(
            f32(raw_ccd_x),
            f32(raw_ccd_y),
            f32(raw_ccd_z)
        ) * INV_FIXED_SCALE;

        let avg_ccd = total_ccd / f32(ccd_count);
        final_disp += avg_ccd;

        atomicStore(&accum[index].ccd_dx, 0);
        atomicStore(&accum[index].ccd_dy, 0);
        atomicStore(&accum[index].ccd_dz, 0);
        atomicStore(&accum[index].ccd_count, 0u);
    }

    if (length(final_disp) > EPSILON) {
        v.prev_pos += final_disp;
        vertices[index] = v;
    }
}
