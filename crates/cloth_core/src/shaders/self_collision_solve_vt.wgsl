// 接触候補ペア解決シェーダー (ナローフェーズ: V-T 頂点対面)
// 収集された V-T ペアに対してのみピンポイントで Möller-Trumbore CCD およびボロノイ幾何反発を適用する。

struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct GpuVtPair {
    vert_i: u32,
    vert_j: u32,
    v0: u32,
    v1: u32,
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

struct ClosestBaryResult {
    pt: vec3<f32>,
    bary: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> active_vt_pairs: array<GpuVtPair>;
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


// 空間上の点 p から三角形 (a, b, c) への最近傍点および重心座標
fn closest_point_on_triangle_bary(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> ClosestBaryResult {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return ClosestBaryResult(a, vec3<f32>(1.0, 0.0, 0.0));
    }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return ClosestBaryResult(b, vec3<f32>(0.0, 1.0, 0.0));
    }

    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return ClosestBaryResult(a + v * ab, vec3<f32>(1.0 - v, v, 0.0));
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return ClosestBaryResult(c, vec3<f32>(0.0, 0.0, 1.0));
    }

    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return ClosestBaryResult(a + w * ac, vec3<f32>(1.0 - w, 0.0, w));
    }

    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return ClosestBaryResult(b + w * (c - b), vec3<f32>(0.0, 1.0 - w, w));
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return ClosestBaryResult(a + ab * v + ac * w, vec3<f32>(1.0 - v - w, v, w));
}

// 線分 (p_old -> p_new) と 三角形 (a, b, c) の連続衝突判定 (CCD: Möller–Trumbore)
fn intersect_segment_triangle(
    p_old: vec3<f32>, p_new: vec3<f32>,
    a: vec3<f32>, b: vec3<f32>, c: vec3<f32>
) -> vec4<f32> {
    let d = p_new - p_old;
    let len_d = length(d);
    if (len_d < EPSILON) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let e1 = b - a;
    let e2 = c - a;
    let pvec = cross(d, e2);
    let det = dot(e1, pvec);

    if (abs(det) < EPSILON) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let inv_det = 1.0 / det;
    let tvec = p_old - a;
    let u = dot(tvec, pvec) * inv_det;
    if (u < -0.05 || u > 1.05) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let qvec = cross(tvec, e1);
    let v = dot(d, qvec) * inv_det;
    if (v < -0.05 || (u + v) > 1.05) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let t = dot(e2, qvec) * inv_det;
    if (t >= 0.0 && t <= 1.0) {
        return vec4<f32>(1.0, t, u, v);
    }

    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    let total_pairs = min(counters.vt_count, params.max_vt_pairs);
    if (pair_idx >= total_pairs) {
        return;
    }

    let pair = active_vt_pairs[pair_idx];
    let index = pair.vert_i;
    let j = pair.vert_j;
    let v0 = pair.v0;
    let v1 = pair.v1;

    let n_verts = params.num_vertices;
    if (index >= n_verts || j >= n_verts || v0 >= n_verts || v1 >= n_verts) {
        return;
    }
    if (index == j || index == v0 || index == v1 || v0 == v1 || j == v0 || j == v1) {
        return;
    }

    let v_i = vertices[index];

    let is_pinned_i = (v_i.inv_mass <= 0.0);
    let p_i = v_i.prev_pos;
    let x_old_i = v_i.position;

    let v_j = vertices[j];
    let v_v0 = vertices[v0];
    let v_v1 = vertices[v1];

    let p_j = v_j.prev_pos;
    let p_v0 = v_v0.prev_pos;
    let p_v1 = v_v1.prev_pos;

    let p_j_old = v_j.position;
    let p_v0_old = v_v0.position;
    let p_v1_old = v_v1.position;

    let effective_thick = v_i.thickness + v_j.thickness;

    // 相手三角形の移動ベクトル
    let tri_move = ((p_j - p_j_old) + (p_v0 - p_v0_old) + (p_v1 - p_v1_old)) / 3.0;
    let tri_has_pin = (v_j.inv_mass <= 0.0) || (v_v0.inv_mass <= 0.0) || (v_v1.inv_mass <= 0.0);

    // 相手の移動を考慮した相対始点（トンネリング検知用）
    let rel_old_i = x_old_i - tri_move;

    // 1. 軌道トポロジー貫通遮断 (CCD)
    let ccd_hit = intersect_segment_triangle(rel_old_i, p_i, p_j, p_v0, p_v1);
    if (ccd_hit.x > 0.5) {
        let t = ccd_hit.y;
        var push_disp = vec3<f32>(0.0);
        if (tri_has_pin) {
            push_disp = tri_move * (1.0 - t);
        }

        let move_vec = p_i - rel_old_i;
        let move_len = length(move_vec);
        let margin_t = min(effective_thick / max(move_len, 1e-4), 0.3);
        let t_safe = max(0.0, t - margin_t);
        let p_safe = rel_old_i + t_safe * move_vec;

        let ccd_disp = (p_safe - p_i) + push_disp;
        add_ccd_atomic(index, ccd_disp);
    } else {
        // 2. 最近傍点幾何反発（重心座標による作用反作用の対称分配）
        let res_vt = closest_point_on_triangle_bary(p_i, p_j, p_v0, p_v1);
        let delta_vt = p_i - res_vt.pt;
        let dist_vt = length(delta_vt);

        let tri_cross = cross(p_v0 - p_j, p_v1 - p_j);
        let is_untangling_vt = (params.enable_normal_untangling != 0u)
            && (v_i.layer_id > v_j.layer_id)
            && (dot(delta_vt, tri_cross) < 0.0);

        if (!is_pinned_i && !is_untangling_vt && dist_vt < effective_thick && dist_vt > EPSILON) {
            let n_vt = delta_vt / dist_vt;
            let pen_vt = effective_thick - dist_vt;
            if (tri_has_pin) {
                let col_disp = n_vt * pen_vt + tri_move * 0.5;
                add_ccd_atomic(index, col_disp);
            } else {
                let w_i = v_i.inv_mass;
                let bary = res_vt.bary;
                let w_j = v_j.inv_mass;
                let w_v0 = v_v0.inv_mass;
                let w_v1 = v_v1.inv_mass;
                let w_tri = bary.x * bary.x * w_j + bary.y * bary.y * w_v0 + bary.z * bary.z * w_v1;
                let w_tot = w_i + w_tri;

                if (w_tot > EPSILON) {
                    let total_pen = n_vt * (pen_vt * 0.6);
                    // 自頂点
                    add_disp_atomic(index, total_pen * (w_i / w_tot));
                    // 相手三角形の3頂点（反作用）
                    if (w_j > 0.0) {
                        add_disp_atomic(j, -total_pen * (bary.x * w_j / w_tot));
                    }
                    if (w_v0 > 0.0) {
                        add_disp_atomic(v0, -total_pen * (bary.y * w_v0 / w_tot));
                    }
                    if (w_v1 > 0.0) {
                        add_disp_atomic(v1, -total_pen * (bary.z * w_v1 / w_tot));
                    }
                }
            }
        }
    }
}
