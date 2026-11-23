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
    pin_dx: atomic<i32>,
    pin_dy: atomic<i32>,
    pin_dz: atomic<i32>,
    pin_count: atomic<u32>,
    ccd_pos_x: atomic<i32>,
    ccd_pos_y: atomic<i32>,
    ccd_pos_z: atomic<i32>,
    ccd_neg_x: atomic<i32>,
    ccd_neg_y: atomic<i32>,
    ccd_neg_z: atomic<i32>,
    untangle_cand_v: atomic<u32>,
    ee_ccd_count: atomic<u32>,
};

struct SelfCollisionParams {
    cell_size: f32,
    table_size: u32,
    num_vertices: u32,
    num_edges: u32,
    relief_factor: f32,
    max_displacement_ratio: f32,
    enable_relief: u32,
    enable_normal_untangling: u32,
    exclude_neighbors: u32,
    max_search_iterations: u32,
    enable_ee: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read_write> accum: array<SelfCollisionAccum>;
@group(0) @binding(2) var<uniform> params: SelfCollisionParams;
@group(0) @binding(3) var<storage, read> local_edge_lengths: array<f32>;

const FIXED_SCALE: f32 = 1000000.0;
const INV_FIXED_SCALE: f32 = 1.0 / FIXED_SCALE;
const EPSILON: f32 = 1e-7;
// E-E CCD 高密度衝突時の協調減衰パラメータ（4件以上の同時衝突で 1/sqrt 則減衰、下限 0.5）
const EE_CCD_DAMP_THRESHOLD: u32 = 4u;
const EE_CCD_DAMP_NUMERATOR: f32 = 3.0;
const EE_CCD_DAMP_MIN_SCALE: f32 = 0.5;

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

    // 2. ピン固定相手との離散接触 (非ピン頂点のみ): 真のCCDと独立チャネルで平均化し、希釈を防止
    let pin_count = atomicLoad(&accum[index].pin_count);
    var avg_pin = vec3<f32>(0.0);
    if (pin_count > 0u) {
        if (!is_pinned) {
            let raw_px = atomicLoad(&accum[index].pin_dx);
            let raw_py = atomicLoad(&accum[index].pin_dy);
            let raw_pz = atomicLoad(&accum[index].pin_dz);

            avg_pin = vec3<f32>(
                f32(raw_px),
                f32(raw_py),
                f32(raw_pz)
            ) * INV_FIXED_SCALE / f32(pin_count);
        }

        atomicStore(&accum[index].pin_dx, 0);
        atomicStore(&accum[index].pin_dy, 0);
        atomicStore(&accum[index].pin_dz, 0);
        atomicStore(&accum[index].pin_count, 0u);
    }

    // 3. 真の CCD (連続衝突判定 & 貫通阻止ハード制約): 軸別符号付き最大包絡線で希釈と過剰加算を同時防止
    let raw_pos_x = atomicLoad(&accum[index].ccd_pos_x);
    let raw_pos_y = atomicLoad(&accum[index].ccd_pos_y);
    let raw_pos_z = atomicLoad(&accum[index].ccd_pos_z);
    let raw_neg_x = atomicLoad(&accum[index].ccd_neg_x);
    let raw_neg_y = atomicLoad(&accum[index].ccd_neg_y);
    let raw_neg_z = atomicLoad(&accum[index].ccd_neg_z);

    if (raw_pos_x != 0) { atomicStore(&accum[index].ccd_pos_x, 0); }
    if (raw_pos_y != 0) { atomicStore(&accum[index].ccd_pos_y, 0); }
    if (raw_pos_z != 0) { atomicStore(&accum[index].ccd_pos_z, 0); }
    if (raw_neg_x != 0) { atomicStore(&accum[index].ccd_neg_x, 0); }
    if (raw_neg_y != 0) { atomicStore(&accum[index].ccd_neg_y, 0); }
    if (raw_neg_z != 0) { atomicStore(&accum[index].ccd_neg_z, 0); }
    let ee_cnt = atomicLoad(&accum[index].ee_ccd_count);
    if (ee_cnt != 0u) { atomicStore(&accum[index].ee_ccd_count, 0u); }
    if (atomicLoad(&accum[index].untangle_cand_v) != 0u) {
        atomicStore(&accum[index].untangle_cand_v, 0u);
    }

    if (!is_pinned) {
        let ccd_pos = vec3<f32>(
            f32(raw_pos_x),
            f32(raw_pos_y),
            f32(raw_pos_z)
        ) * INV_FIXED_SCALE;
        let ccd_neg = vec3<f32>(
            f32(raw_neg_x),
            f32(raw_neg_y),
            f32(raw_neg_z)
        ) * INV_FIXED_SCALE;

        // 同一頂点で E-E CCD が4件以上同時発火した高密度折り畳み時は、協調減衰で一点凝縮を抑止する
        var ccd_env = ccd_pos + ccd_neg;
        if (ee_cnt >= EE_CCD_DAMP_THRESHOLD) {
            let scale = max(EE_CCD_DAMP_MIN_SCALE, sqrt(EE_CCD_DAMP_NUMERATOR / f32(ee_cnt)));
            ccd_env = ccd_env * scale;
        }

        // 同方向のピン離散反発と真のCCDが重なった場合の2倍跳ねを防ぎつつ、
        // 逆方向の押し合いは正負独立包絡で正しく釣り合わせる。
        let hard_pos = max(max(ccd_env, vec3<f32>(0.0)), max(avg_pin, vec3<f32>(0.0)));
        let hard_neg = min(min(ccd_env, vec3<f32>(0.0)), min(avg_pin, vec3<f32>(0.0)));
        final_disp += hard_pos + hard_neg;
    }

    if (length(final_disp) > EPSILON) {
        v.prev_pos += final_disp;
        vertices[index] = v;
    }
}
