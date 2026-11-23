struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct GpuStarPair {
    v0: u32,
    v1: u32,
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

struct ClosestBaryResult {
    pt: vec3<f32>,
    bary: vec3<f32>,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(3) var<uniform> params: SelfCollisionParams;
@group(0) @binding(4) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> local_edge_lengths: array<f32>;
@group(0) @binding(6) var<storage, read> adj_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> adj_indices: array<u32>;
@group(0) @binding(9) var<storage, read> star_offsets: array<u32>;
@group(0) @binding(10) var<storage, read> star_indices: array<GpuStarPair>;
@group(0) @binding(11) var<storage, read_write> accum: array<SelfCollisionAccum>;
@group(0) @binding(12) var<storage, read> two_hop_offsets: array<u32>;
@group(0) @binding(13) var<storage, read> two_hop_indices: array<u32>;

const EPSILON: f32 = 1e-7;
const FIXED_SCALE: f32 = 1000000.0;
const MAX_ATOMIC_FIXED: f32 = 50000000.0;

fn add_disp_atomic(v_idx: u32, disp: vec3<f32>) {
    let ix = i32(clamp(disp.x * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    let iy = i32(clamp(disp.y * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    let iz = i32(clamp(disp.z * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    atomicAdd(&accum[v_idx].dx, ix);
    atomicAdd(&accum[v_idx].dy, iy);
    atomicAdd(&accum[v_idx].dz, iz);
    atomicAdd(&accum[v_idx].count, 1u);
}

// ループ内軽量版: 空間上の点 p から三角形 (a, b, c) への最近傍座標のみを返す
fn closest_point_on_triangle_pos(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return a;
    }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return a + v * ab;
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return a + w * ac;
    }

    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return a + ab * v + ac * w;
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

// キャッシュ済み範囲 [start, end) による純粋な2ホップ近傍判定（Untangling 隣接三角形除外用）
fn is_within_two_hops_range(start: u32, end: u32, vert_b: u32) -> bool {
    if (start >= end) {
        return false;
    }
    var low = start;
    var high = end;
    while (low < high) {
        let mid = low + (high - low) / 2u;
        let val = two_hop_indices[mid];
        if (val == vert_b) {
            return true;
        } else if (val < vert_b) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_vertices || params.enable_normal_untangling == 0u) {
        return;
    }

    // sc_solve / pair_vt が記録した最近傍内層頂点候補を消費（候補なし・L0全頂点は 1 リードで即リターン）
    let cand = atomicLoad(&accum[index].untangle_cand_v);
    if (cand == 0u) {
        return;
    }
    atomicStore(&accum[index].untangle_cand_v, 0u);

    let v_i = vertices[index];
    if (v_i.inv_mass <= 0.0 || v_i.layer_id == 0u) {
        return;
    }

    let j1_encoded = cand & 0x3FFFFFu;
    if (j1_encoded == 0u) {
        return;
    }
    let j1 = j1_encoded - 1u;
    if (j1 >= params.num_vertices || j1 == index) {
        return;
    }

    let p_i = v_i.prev_pos;
    let thick_i = v_i.thickness;
    let local_len_i = local_edge_lengths[index];
    let th_start = two_hop_offsets[index];
    let th_end = two_hop_offsets[index + 1u];

    // j1 の 1-ring 隣接内層頂点から p_i に最も近い第2候補 j2 を選定（空間ハッシュ走査ゼロ）
    var j2 = 0xFFFFFFFFu;
    var best_nb_dist_sq = 1e20;
    let adj_start = adj_offsets[j1];
    let adj_end = adj_offsets[j1 + 1u];
    for (var a = adj_start; a < adj_end; a = a + 1u) {
        let nb = adj_indices[a];
        if (nb < params.num_vertices && nb != index && vertices[nb].layer_id < v_i.layer_id) {
            let dnb = p_i - vertices[nb].prev_pos;
            let dnb_sq = dot(dnb, dnb);
            if (dnb_sq < best_nb_dist_sq && !is_within_two_hops_range(th_start, th_end, nb)) {
                best_nb_dist_sq = dnb_sq;
                j2 = nb;
            }
        }
    }

    var best_dist_sq = 1e20;
    var best_h = 0.0;
    var best_j = 0xFFFFFFFFu;
    var best_v0 = 0u;
    var best_v1 = 0u;

    // j1 と j2 の star（共有三角形を除外し最大 ~10 三角形）のみ評価して最近傍内層三角形 1 件を選出
    for (var ci = 0u; ci < 2u; ci = ci + 1u) {
        let j = select(j2, j1, ci == 0u);
        if (j == 0xFFFFFFFFu) {
            continue;
        }
        let p_j = vertices[j].prev_pos;
        let thick_j = vertices[j].thickness;
        let local_len_j = local_edge_lengths[j];
        let effective_thick = max(thick_i + thick_j, min(local_len_i, local_len_j) * 0.06);
        let capture_thick = effective_thick * 2.5;
        let capture_sq = capture_thick * capture_thick;

        let star_start = star_offsets[j];
        let star_end = star_offsets[j + 1u];
        for (var s = star_start; s < star_end; s = s + 1u) {
            let pair = star_indices[s];
            let v0 = pair.v0;
            let v1 = pair.v1;
            if (v0 == index || v1 == index) {
                continue;
            }
            if (ci == 1u && (v0 == j1 || v1 == j1)) {
                continue;
            }

            let p_v0 = vertices[v0].prev_pos;
            let p_v1 = vertices[v1].prev_pos;
            let tri_min = min(min(p_j, p_v0), p_v1);
            let tri_max = max(max(p_j, p_v0), p_v1);
            let aabb_d = max(tri_min - p_i, vec3<f32>(0.0)) + max(p_i - tri_max, vec3<f32>(0.0));
            let limit_sq = min(capture_sq, best_dist_sq + 1e-10);
            if (dot(aabb_d, aabb_d) >= limit_sq) {
                continue;
            }

            let pt_ut = closest_point_on_triangle_pos(p_i, p_j, p_v0, p_v1);
            let delta_ut = p_i - pt_ut;
            let dist_sq_ut = dot(delta_ut, delta_ut);
            if (dist_sq_ut >= capture_sq) {
                continue;
            }
            if (best_j != 0xFFFFFFFFu && dist_sq_ut > best_dist_sq + 1e-10) {
                continue;
            }

            let tri_cross = cross(p_v0 - p_j, p_v1 - p_j);
            let tri_len = length(tri_cross);
            if (tri_len <= EPSILON) {
                continue;
            }
            let h_ut = dot(delta_ut, tri_cross / tri_len);
            let same_d = abs(dist_sq_ut - best_dist_sq) <= 1e-10;
            let better_tie = (h_ut >= 0.0 && best_h < 0.0)
                || (((h_ut >= 0.0) == (best_h >= 0.0)) && abs(h_ut) > abs(best_h));
            if (best_j == 0xFFFFFFFFu
                || dist_sq_ut < best_dist_sq - 1e-10
                || (same_d && better_tie)) {
                if (is_within_two_hops_range(th_start, th_end, j)
                    || is_within_two_hops_range(th_start, th_end, v0)
                    || is_within_two_hops_range(th_start, th_end, v1)) {
                    continue;
                }
                best_dist_sq = dist_sq_ut;
                best_h = h_ut;
                best_j = j;
                best_v0 = v0;
                best_v1 = v1;
            }
        }
    }

    // 最近傍内層三角形1件のみを再評価し、真裏円錐・面内はみ出し・擬似法線・同向き検証を満たす場合だけ表側へ脱出
    if (best_j != 0xFFFFFFFFu && best_h < 0.0) {
        let uj = best_j;
        let uv0 = best_v0;
        let uv1 = best_v1;
        let uv_j = vertices[uj];
        let uv_v0 = vertices[uv0];
        let uv_v1 = vertices[uv1];
        let up_j = uv_j.prev_pos;
        let up_v0 = uv_v0.prev_pos;
        let up_v1 = uv_v1.prev_pos;

        let res_ut = closest_point_on_triangle_bary(p_i, up_j, up_v0, up_v1);
        let delta_ut = p_i - res_ut.pt;
        let dist_ut = length(delta_ut);
        let tri_cross_ut = cross(up_v0 - up_j, up_v1 - up_j);
        let tri_len_ut = length(tri_cross_ut);
        if (tri_len_ut > EPSILON && dist_ut > EPSILON) {
            let tri_n_ut = tri_cross_ut / tri_len_ut;
            let h_ut = dot(delta_ut, tri_n_ut);
            if (h_ut < 0.0) {
                let n_i_raw = normals[index].xyz;
                let n_i_len = length(n_i_raw);
                let tri_has_pin_ut = (uv_j.inv_mass <= 0.0) || (uv_v0.inv_mass <= 0.0) || (uv_v1.inv_mass <= 0.0);
                let allow_untangle = tri_has_pin_ut
                    || (n_i_len <= 0.5)
                    || (dot(n_i_raw / max(n_i_len, EPSILON), tri_n_ut) > 0.0);
                let eff_thick_ut = max(thick_i + uv_j.thickness, min(local_len_i, local_edge_lengths[uj]) * 0.06);
                let cos_behind = -h_ut / dist_ut;
                let r_tan = sqrt(max(dist_ut * dist_ut - h_ut * h_ut, 0.0));
                let r_limit = max(eff_thick_ut * 0.5, EPSILON);
                if (allow_untangle && cos_behind > 0.707 && r_tan < r_limit) {
                    let bary = res_ut.bary;
                    let n_interp = bary.x * normals[uj].xyz
                        + bary.y * normals[uv0].xyz
                        + bary.z * normals[uv1].xyz;
                    let n_interp_len = length(n_interp);
                    let behind_pseudo = (n_interp_len <= 0.5) || (dot(delta_ut, n_interp) < 0.0);
                    if (behind_pseudo) {
                        let cone_w = clamp((cos_behind - 0.707) / 0.2, 0.0, 1.0);
                        let r_w = clamp(1.0 - r_tan / r_limit, 0.0, 1.0);
                        let pen = min(eff_thick_ut - h_ut, eff_thick_ut * 2.0) * cone_w * r_w;
                        if (pen > EPSILON) {
                            if (tri_has_pin_ut) {
                                add_disp_atomic(index, tri_n_ut * pen);
                            } else {
                                let w_i = v_i.inv_mass;
                                let w_j = uv_j.inv_mass;
                                let w_v0 = uv_v0.inv_mass;
                                let w_v1 = uv_v1.inv_mass;
                                let w_tri = bary.x * bary.x * w_j + bary.y * bary.y * w_v0 + bary.z * bary.z * w_v1;
                                let w_tot = w_i + w_tri;
                                if (w_tot > EPSILON) {
                                    let disp = tri_n_ut * (pen * 0.5);
                                    add_disp_atomic(index, disp * (w_i / w_tot));
                                    if (w_j > 0.0) { add_disp_atomic(uj, -disp * (bary.x * w_j / w_tot)); }
                                    if (w_v0 > 0.0) { add_disp_atomic(uv0, -disp * (bary.y * w_v0 / w_tot)); }
                                    if (w_v1 > 0.0) { add_disp_atomic(uv1, -disp * (bary.z * w_v1 / w_tot)); }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
