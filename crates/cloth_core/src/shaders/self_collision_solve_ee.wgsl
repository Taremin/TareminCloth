// 接触候補ペア解決シェーダー (ナローフェーズ: E-E 辺対辺)
// 収集された E-E ペアに対してのみピンポイントで線分最短幾何反発を適用する。

struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct GpuEePair {
    v_i: u32,
    v_ui: u32,
    v_j: u32,
    v_vj: u32,
};

struct SolveParams {
    num_vertices: u32,
    max_vt_pairs: u32,
    max_ee_pairs: u32,
    enable_normal_untangling: u32,
};

struct PairCounters {
    vt_count: u32,
    ee_count: u32,
    _pad0: u32,
    _pad1: u32,
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

@group(0) @binding(0) var<storage, read> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> active_ee_pairs: array<GpuEePair>;
@group(0) @binding(2) var<storage, read> counters: PairCounters;
@group(0) @binding(3) var<storage, read_write> accum: array<SelfCollisionAccum>;
@group(0) @binding(4) var<uniform> params: SolveParams;
@group(0) @binding(5) var<storage, read> normals: array<vec4<f32>>;

const EPSILON: f32 = 1e-7;
const FIXED_SCALE: f32 = 1000000.0;

fn add_disp_atomic(v_idx: u32, disp: vec3<f32>) {
    if (v_idx >= params.num_vertices) {
        return;
    }
    let ix = i32(clamp(disp.x * FIXED_SCALE, -2e9, 2e9));
    let iy = i32(clamp(disp.y * FIXED_SCALE, -2e9, 2e9));
    let iz = i32(clamp(disp.z * FIXED_SCALE, -2e9, 2e9));
    atomicAdd(&accum[v_idx].dx, ix);
    atomicAdd(&accum[v_idx].dy, iy);
    atomicAdd(&accum[v_idx].dz, iz);
    atomicAdd(&accum[v_idx].count, 1u);
}

fn add_ccd_atomic(v_idx: u32, disp: vec3<f32>) {
    if (v_idx >= params.num_vertices) {
        return;
    }
    let ix = i32(clamp(disp.x * FIXED_SCALE, -2e9, 2e9));
    let iy = i32(clamp(disp.y * FIXED_SCALE, -2e9, 2e9));
    let iz = i32(clamp(disp.z * FIXED_SCALE, -2e9, 2e9));
    atomicAdd(&accum[v_idx].ccd_dx, ix);
    atomicAdd(&accum[v_idx].ccd_dy, iy);
    atomicAdd(&accum[v_idx].ccd_dz, iz);
    atomicAdd(&accum[v_idx].ccd_count, 1u);
}


// 2線分 (p1-q1) と (p2-q2) の最近傍パラメータ (s, t)
fn closest_points_segments(p1: vec3<f32>, q1: vec3<f32>, p2: vec3<f32>, q2: vec3<f32>) -> vec2<f32> {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);

    if (a <= EPSILON && e <= EPSILON) {
        return vec2<f32>(0.0, 0.0);
    }
    if (a <= EPSILON) {
        return vec2<f32>(0.0, clamp(f / e, 0.0, 1.0));
    }

    let c = dot(d1, r);
    if (e <= EPSILON) {
        return vec2<f32>(clamp(-c / a, 0.0, 1.0), 0.0);
    }

    let b = dot(d1, d2);
    let denom = a * e - b * b;
    var s = 0.0;
    if (abs(denom) > EPSILON) {
        s = clamp((b * f - c * e) / denom, 0.0, 1.0);
    }

    var t = (b * s + f) / e;
    if (t < 0.0) {
        t = 0.0;
        s = clamp(-c / a, 0.0, 1.0);
    } else if (t > 1.0) {
        t = 1.0;
        s = clamp((b - c) / a, 0.0, 1.0);
    }

    return vec2<f32>(s, t);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    let total_pairs = min(counters.ee_count, params.max_ee_pairs);
    if (pair_idx >= total_pairs) {
        return;
    }

    let pair = active_ee_pairs[pair_idx];
    let index = pair.v_i;
    let ui = pair.v_ui;
    let j = pair.v_j;
    let vj = pair.v_vj;

    let n_verts = params.num_vertices;
    if (index >= n_verts || ui >= n_verts || j >= n_verts || vj >= n_verts) {
        return;
    }
    if (index == ui || j == vj || index == j || index == vj || ui == j || ui == vj) {
        return;
    }

    let v_i = vertices[index];

    let v_ui = vertices[ui];
    let v_j = vertices[j];
    let v_vj = vertices[vj];

    let p_i = v_i.prev_pos;
    let p_ui = v_ui.prev_pos;
    let p_j = v_j.prev_pos;
    let p_vj = v_vj.prev_pos;

    let x_old_i = v_i.position;
    let p_j_old = v_j.position;
    let p_vj_old = v_vj.position;

    let effective_thick = v_i.thickness + v_j.thickness;

    let st = closest_points_segments(p_i, p_ui, p_j, p_vj);
    let pt1 = p_i + st.x * (p_ui - p_i);
    let pt2 = p_j + st.y * (p_vj - p_j);

    let delta_ee = pt1 - pt2;
    let dist_ee = length(delta_ee);

    let is_untangling_ee = (params.enable_normal_untangling != 0u)
        && (v_i.layer_id > v_j.layer_id)
        && (dot(delta_ee, normals[j].xyz) < 0.0);

    if (!is_untangling_ee && dist_ee < effective_thick && dist_ee > EPSILON) {
        let n_ee = delta_ee / dist_ee;
        let pen_ee = effective_thick - dist_ee;
        let s = st.x;
        let t = st.y;

        let edge_i_has_pin = (v_i.inv_mass <= 0.0) || (v_ui.inv_mass <= 0.0);
        let edge_j_has_pin = (v_j.inv_mass <= 0.0) || (v_vj.inv_mass <= 0.0);

        if (edge_j_has_pin) {
            let edge_move = ((p_j - p_j_old) + (p_vj - p_vj_old)) * 0.5;
            let col_disp = n_ee * (pen_ee * (1.0 - s)) + edge_move * (1.0 - s) * 0.5;
            add_ccd_atomic(index, col_disp);
            if (v_ui.inv_mass > 0.0) {
                let col_disp_ui = n_ee * (pen_ee * s) + edge_move * s * 0.5;
                add_ccd_atomic(ui, col_disp_ui);
            }
        } else if (edge_i_has_pin) {
            let edge_move = ((p_i - x_old_i) + (p_ui - v_ui.position)) * 0.5;
            let col_disp = -n_ee * (pen_ee * (1.0 - t)) + edge_move * (1.0 - t) * 0.5;
            add_ccd_atomic(j, col_disp);
            if (v_vj.inv_mass > 0.0) {
                let col_disp_vj = -n_ee * (pen_ee * t) + edge_move * t * 0.5;
                add_ccd_atomic(vj, col_disp_vj);
            }
        } else {
            let w_i = v_i.inv_mass;
            let w_ui = v_ui.inv_mass;
            let w_j = v_j.inv_mass;
            let w_vj = v_vj.inv_mass;
            let w_e1 = (1.0 - s) * (1.0 - s) * w_i + s * s * w_ui;
            let w_e2 = (1.0 - t) * (1.0 - t) * w_j + t * t * w_vj;
            let w_tot = w_e1 + w_e2;

            if (w_tot > EPSILON) {
                let total_disp = n_ee * (pen_ee * 0.6);
                // エッジ1（自側）
                if (w_i > 0.0) {
                    add_disp_atomic(index, total_disp * ((1.0 - s) * w_i / w_tot));
                }
                if (w_ui > 0.0) {
                    add_disp_atomic(ui, total_disp * (s * w_ui / w_tot));
                }
                // エッジ2（相手側、反作用）
                if (w_j > 0.0) {
                    add_disp_atomic(j, -total_disp * ((1.0 - t) * w_j / w_tot));
                }
                if (v_vj.inv_mass > 0.0) {
                    add_disp_atomic(vj, -total_disp * (t * w_vj / w_tot));
                }
            }
        }
    }
}
