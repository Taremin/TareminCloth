// Edge-Centric Self Collision Solve (Direct E-E Pass)
// エッジスレッド駆動による自己衝突 Edge-Edge 幾何反発パス。
// 直積二重ループを排除し、エッジ辞書順序制約により全エッジペアの判定を GPU 全体で厳密に 1 回のみ実行する。

struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct GpuEdge {
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

@group(0) @binding(0) var<storage, read> edges: array<GpuEdge>;
@group(0) @binding(1) var<storage, read> vertices: array<GpuVertex>;
@group(0) @binding(2) var<storage, read> edge_cell_starts: array<u32>;
@group(0) @binding(3) var<storage, read> edge_sorted_indices: array<u32>;
@group(0) @binding(4) var<uniform> params: SelfCollisionParams;
@group(0) @binding(5) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> two_hop_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> two_hop_indices: array<u32>;
@group(0) @binding(8) var<storage, read_write> accum: array<SelfCollisionAccum>;
@group(0) @binding(9) var<storage, read> two_hop_rest_lengths: array<f32>;
@group(0) @binding(10) var<storage, read> edge_boundary_bits: array<u32>;

fn is_open_boundary_edge(e_idx: u32) -> bool {
    return ((edge_boundary_bits[e_idx >> 5u] >> (e_idx & 31u)) & 1u) != 0u;
}

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

fn add_pin_disp_atomic(v_idx: u32, disp: vec3<f32>) {
    let ix = i32(clamp(disp.x * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    let iy = i32(clamp(disp.y * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    let iz = i32(clamp(disp.z * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    atomicAdd(&accum[v_idx].pin_dx, ix);
    atomicAdd(&accum[v_idx].pin_dy, iy);
    atomicAdd(&accum[v_idx].pin_dz, iz);
    atomicAdd(&accum[v_idx].pin_count, 1u);
}

fn add_ccd_atomic(v_idx: u32, disp: vec3<f32>, count_ee: bool) {
    let ix = i32(clamp(disp.x * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    let iy = i32(clamp(disp.y * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    let iz = i32(clamp(disp.z * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    if (ix > 0) {
        atomicMax(&accum[v_idx].ccd_pos_x, ix);
    } else if (ix < 0) {
        atomicMin(&accum[v_idx].ccd_neg_x, ix);
    }
    if (iy > 0) {
        atomicMax(&accum[v_idx].ccd_pos_y, iy);
    } else if (iy < 0) {
        atomicMin(&accum[v_idx].ccd_neg_y, iy);
    }
    if (iz > 0) {
        atomicMax(&accum[v_idx].ccd_pos_z, iz);
    } else if (iz < 0) {
        atomicMin(&accum[v_idx].ccd_neg_z, iz);
    }
    // 高密度衝突時の協調減衰判定用に動的布同士の非ゼロ E-E CCD 書き込み回数を記録
    if (count_ee) {
        let len_fixed = u32(clamp(length(disp) * FIXED_SCALE, 0.0, 2000000000.0));
        if (len_fixed > 0u) {
            atomicAdd(&accum[v_idx].ee_ccd_count, 1u);
        }
    }
}

// 多層布の表層エッジが内層エッジの裏側に入り込んでいるかをエッジ順序に対称判定する
// delta_ab は Edge A -> Edge B 基準の分離ベクトル (pt_a - pt_b)
fn is_outer_edge_behind_inner(
    layer_a: u32,
    layer_b: u32,
    a0: u32,
    a1: u32,
    b0: u32,
    b1: u32,
    a_fixed: bool,
    b_fixed: bool,
    delta_ab: vec3<f32>,
) -> bool {
    if (layer_a == layer_b) {
        return false;
    }
    let na_raw = normals[a0].xyz + normals[a1].xyz;
    let nb_raw = normals[b0].xyz + normals[b1].xyz;
    let na_len = length(na_raw);
    let nb_len = length(nb_raw);
    if (layer_a > layer_b) {
        if (nb_len <= 0.5) { return false; }
        let n_inner = nb_raw / nb_len;
        let allow = b_fixed || (na_len <= 0.5) || (dot(na_raw / max(na_len, EPSILON), n_inner) > 0.0);
        return allow && (dot(delta_ab, n_inner) < 0.0);
    }
    if (na_len <= 0.5) { return false; }
    let n_inner = na_raw / na_len;
    let allow = a_fixed || (nb_len <= 0.5) || (dot(nb_raw / max(nb_len, EPSILON), n_inner) > 0.0);
    return allow && (dot(-delta_ab, n_inner) < 0.0);
}

fn hash_coords(coord: vec3<i32>, table_size: u32) -> u32 {
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    let n = (u32(coord.x) * p1) ^ (u32(coord.y) * p2) ^ (u32(coord.z) * p3);
    return n % table_size;
}

// 2線分 p1-q1 と p2-q2 間の最短距離パラメータ (s, t) in [0, 1]^2
fn closest_points_segments(p1: vec3<f32>, q1: vec3<f32>, p2: vec3<f32>, q2: vec3<f32>) -> vec2<f32> {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);

    if (a <= 1e-12 && e <= 1e-12) {
        return vec2<f32>(0.0, 0.0);
    }
    if (a <= 1e-12) {
        return vec2<f32>(0.0, clamp(f / e, 0.0, 1.0));
    }
    let c = dot(d1, r);
    if (e <= 1e-12) {
        return vec2<f32>(clamp(-c / a, 0.0, 1.0), 0.0);
    }

    let b = dot(d1, d2);
    let denom = a * e - b * b;

    var s = 0.0;
    // denom は m^4 次元であるため、a * e に対する相対閾値で平行判定を行う
    if (denom > a * e * 1e-7) {
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

struct SegmentEval {
    dist_sq: f32,
    st: vec2<f32>,
};

fn eval_segment_dist_sq(
    a0_old: vec3<f32>, a0_cur: vec3<f32>,
    a1_old: vec3<f32>, a1_cur: vec3<f32>,
    b0_old: vec3<f32>, b0_cur: vec3<f32>,
    b1_old: vec3<f32>, b1_cur: vec3<f32>,
    tau: f32
) -> SegmentEval {
    let a0 = mix(a0_old, a0_cur, tau);
    let a1 = mix(a1_old, a1_cur, tau);
    let b0 = mix(b0_old, b0_cur, tau);
    let b1 = mix(b1_old, b1_cur, tau);
    let st = closest_points_segments(a0, a1, b0, b1);
    let pa = mix(a0, a1, st.x);
    let pb = mix(b0, b1, st.y);
    let d = pa - pb;
    return SegmentEval(dot(d, d), st);
}

fn eval_segment_volume(
    a0_old: vec3<f32>, a0_cur: vec3<f32>,
    a1_old: vec3<f32>, a1_cur: vec3<f32>,
    b0_old: vec3<f32>, b0_cur: vec3<f32>,
    b1_old: vec3<f32>, b1_cur: vec3<f32>,
    tau: f32
) -> f32 {
    let a0 = mix(a0_old, a0_cur, tau);
    let a1 = mix(a1_old, a1_cur, tau);
    let b0 = mix(b0_old, b0_cur, tau);
    let b1 = mix(b1_old, b1_cur, tau);
    return dot(cross(a1 - a0, b1 - b0), b0 - a0);
}

// トポロジー2ホップ近傍判定（二乗距離版: sqrt不要）
fn is_topologically_near_folded_sq(vert_a: u32, vert_b: u32, current_dist_sq: f32) -> bool {
    if (vert_a == vert_b) {
        return true;
    }
    let start = two_hop_offsets[vert_a];
    let end = two_hop_offsets[vert_a + 1u];

    var low = start;
    var high = end;
    while (low < high) {
        let mid = low + (high - low) / 2u;
        let val = two_hop_indices[mid];
        if (val == vert_b) {
            let l0 = two_hop_rest_lengths[mid];
            if (current_dist_sq < (l0 * l0) * 0.25) {
                return false;
            }
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
    let edge_idx = global_id.x;
    if (edge_idx >= params.num_edges || params.enable_ee == 0u) {
        return;
    }

    let edge_a = edges[edge_idx];
    let u0 = edge_a.v0;
    let u1 = edge_a.v1;

    let v_u0 = vertices[u0];
    let v_u1 = vertices[u1];

    let is_pinned_u0 = (v_u0.inv_mass <= 0.0);
    let is_pinned_u1 = (v_u1.inv_mass <= 0.0);
    let edge_a_has_pin = is_pinned_u0 || is_pinned_u1;

    let p_u0 = v_u0.prev_pos;
    let p_u1 = v_u1.prev_pos;
    let p_u0_old = v_u0.position;
    let p_u1_old = v_u1.position;

    // 両端点静止ピンエッジも edge_idx < edge_b_idx の相手動的エッジを処理するため早期リターンしない

    let thick_u0 = v_u0.thickness;
    let thick_u1 = v_u1.thickness;
    let max_thick_a = max(thick_u0, thick_u1);

    // エッジ A の AABB（マージン付き。相手が境界エッジである場合も考慮し常に開始位置も含めた掃引 AABB）
    let is_boundary_a = is_open_boundary_edge(edge_idx);
    let aabb_a_min = min(min(p_u0, p_u1), min(p_u0_old, p_u1_old)) - vec3<f32>(max_thick_a * 2.0);
    let aabb_a_max = max(max(p_u0, p_u1), max(p_u0_old, p_u1_old)) + vec3<f32>(max_thick_a * 2.0);

    // エッジ中点の空間セル座標（周囲27セルを均一・定数走査）
    let p_mid = (p_u0 + p_u1) * 0.5;
    let base_cell = vec3<i32>(floor(p_mid / params.cell_size));

    var search_count = 0u;

    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let neighbor_cell = base_cell + vec3<i32>(dx, dy, dz);
                let cell_hash = hash_coords(neighbor_cell, params.table_size);
                let start = edge_cell_starts[cell_hash];
                let end = edge_cell_starts[cell_hash + 1u];
                if (start >= end) {
                    continue;
                }

                let count = end - start;
                for (var idx = 0u; idx < count; idx = idx + 1u) {
                    let edge_b_idx = edge_sorted_indices[start + idx];

                    // 【辞書順序制約】全世界で edge_idx < edge_b_idx のペアのみを厳密に 1 回処理
                    // 静止ピンエッジスレッドも有効であるためメモリロード前に 50% を即除外する
                    if (edge_idx >= edge_b_idx) {
                        continue;
                    }

                    let edge_b = edges[edge_b_idx];
                    let j = edge_b.v0;
                    let vj = edge_b.v1;

                    // 端点共有（トポロジー隣接）の除外
                    if (j == u0 || j == u1 || vj == u0 || vj == u1) {
                        continue;
                    }

                    let v_j = vertices[j];
                    let v_vj = vertices[vj];
                    let p_j = v_j.prev_pos;
                    let p_vj = v_vj.prev_pos;

                    // ハッシュ衝突排除（相手エッジの中点セル座標が neighbor_cell と一致するか）
                    let p_b_mid = (p_j + p_vj) * 0.5;
                    let b_cell = vec3<i32>(floor(p_b_mid / params.cell_size));
                    if (!all(b_cell == neighbor_cell)) {
                        continue;
                    }

                    // 【早期枝切り】エッジ掃引 AABB 重なり判定 (Edge-Edge Sweep AABB Overlap)
                    let p_j_old = v_j.position;
                    let p_vj_old = v_vj.position;
                    let aabb_b_min = min(min(p_j, p_vj), min(p_j_old, p_vj_old));
                    let aabb_b_max = max(max(p_j, p_vj), max(p_j_old, p_vj_old));
                    if (any(aabb_a_min > aabb_b_max) || any(aabb_a_max < aabb_b_min)) {
                        continue;
                    }

                    search_count = search_count + 1u;
                    if (search_count > params.max_search_iterations) {
                        return;
                    }

                    let is_boundary_b = is_open_boundary_edge(edge_b_idx);
                    let is_boundary_pair = is_boundary_a || is_boundary_b;

                    var run_ccd = is_boundary_pair;
                    var has_cached_st1 = false;
                    var st1_cached = vec2<f32>(0.0, 0.0);
                    var pt1_a_cached = vec3<f32>(0.0);
                    var pt1_b_cached = vec3<f32>(0.0);
                    var delta1_cached = vec3<f32>(0.0);
                    var dist_sq_ee_cached = 0.0;
                    var eff_thick_cached = 0.0;

                    if (!run_ccd) {
                        // 内部エッジペアに対する早期枝切りおよび選択的 CCD ゲート (G1: 高相対速度, G2: 近接クロスオーバー)
                        st1_cached = closest_points_segments(p_u0, p_u1, p_j, p_vj);
                        pt1_a_cached = mix(p_u0, p_u1, st1_cached.x);
                        pt1_b_cached = mix(p_j, p_vj, st1_cached.y);
                        delta1_cached = pt1_a_cached - pt1_b_cached;
                        dist_sq_ee_cached = dot(delta1_cached, delta1_cached);

                        let thick_j_g = v_j.thickness;
                        let thick_vj_g = v_vj.thickness;
                        eff_thick_cached = mix(thick_u0, thick_u1, st1_cached.x) + mix(thick_j_g, thick_vj_g, st1_cached.y);
                        let max_eff_thick_g = max_thick_a + max(thick_j_g, thick_vj_g);

                        let dv_u0_g = p_u0 - p_u0_old;
                        let dv_u1_g = p_u1 - p_u1_old;
                        let dv_j_g = p_j - p_j_old;
                        let dv_vj_g = p_vj - p_vj_old;
                        let va_avg_g = 0.5 * (dv_u0_g + dv_u1_g);
                        let vb_avg_g = 0.5 * (dv_j_g + dv_vj_g);
                        let rel_disp_g = length(va_avg_g - vb_avg_g) + 0.5 * (length(dv_u1_g - dv_u0_g) + length(dv_vj_g - dv_j_g));

                        let reach_bound_g = max_eff_thick_g + rel_disp_g;
                        if (dist_sq_ee_cached > reach_bound_g * reach_bound_g) {
                            continue;
                        }

                        has_cached_st1 = true;
                        let delta0_g = mix(p_u0_old, p_u1_old, st1_cached.x) - mix(p_j_old, p_vj_old, st1_cached.y);
                        let gate_g1 = rel_disp_g > eff_thick_cached * 0.5;
                        let gate_g2 = dot(delta0_g, delta1_cached) <= 0.0;
                        run_ccd = gate_g1 || gate_g2;
                    }

                    // 開放境界エッジおよびゲート通過内部エッジの CCD 軌道最接近探索およびクロスオーバー遮断
                    if (run_ccd) {
                        // 粗サンプル（9点: tau = 0/8 .. 8/8）による最小距離と体積符号反転の探索
                        var min_sample_dist_sq = 1e20;
                        var best_tau = 0.0;
                        var half_span = 0.125;
                        var min_st = vec2<f32>(0.0, 0.0);
                        var prev_vol = eval_segment_volume(p_u0_old, p_u0, p_u1_old, p_u1, p_j_old, p_j, p_vj_old, p_vj, 0.0);
                        var zero_cross_tau = -1.0;

                        for (var k = 0u; k <= 8u; k = k + 1u) {
                            let tau_k = f32(k) * 0.125;
                            let eval_res = eval_segment_dist_sq(p_u0_old, p_u0, p_u1_old, p_u1, p_j_old, p_j, p_vj_old, p_vj, tau_k);
                            if (eval_res.dist_sq < min_sample_dist_sq) {
                                min_sample_dist_sq = eval_res.dist_sq;
                                best_tau = tau_k;
                                min_st = eval_res.st;
                            }
                            if (k > 0u) {
                                let vol = eval_segment_volume(p_u0_old, p_u0, p_u1_old, p_u1, p_j_old, p_j, p_vj_old, p_vj, tau_k);
                                if (prev_vol * vol <= 0.0 && zero_cross_tau < 0.0) {
                                    zero_cross_tau = tau_k - 0.0625;
                                }
                                prev_vol = vol;
                            }
                        }

                        // 体積符号反転点（共面クロス候補）があれば評価
                        if (zero_cross_tau >= 0.0) {
                            let eval_zc = eval_segment_dist_sq(p_u0_old, p_u0, p_u1_old, p_u1, p_j_old, p_j, p_vj_old, p_vj, zero_cross_tau);
                            if (eval_zc.dist_sq < min_sample_dist_sq) {
                                min_sample_dist_sq = eval_zc.dist_sq;
                                best_tau = zero_cross_tau;
                                half_span = 0.0625;
                                min_st = eval_zc.st;
                            }
                        }

                        // 三分探索（6回）で最接近点 (tau*, s*, t*) を特定
                        var t_lo = max(0.0, best_tau - half_span);
                        var t_hi = min(1.0, best_tau + half_span);
                        var tau_star = best_tau;
                        var dist_sq_star = min_sample_dist_sq;
                        var st_star = min_st;

                        for (var iter = 0u; iter < 6u; iter = iter + 1u) {
                            let m1 = t_lo + (t_hi - t_lo) * (1.0 / 3.0);
                            let m2 = t_hi - (t_hi - t_lo) * (1.0 / 3.0);
                            let eval1 = eval_segment_dist_sq(p_u0_old, p_u0, p_u1_old, p_u1, p_j_old, p_j, p_vj_old, p_vj, m1);
                            let eval2 = eval_segment_dist_sq(p_u0_old, p_u0, p_u1_old, p_u1, p_j_old, p_j, p_vj_old, p_vj, m2);

                            if (eval1.dist_sq < dist_sq_star) {
                                dist_sq_star = eval1.dist_sq;
                                tau_star = m1;
                                st_star = eval1.st;
                            }
                            if (eval2.dist_sq < dist_sq_star) {
                                dist_sq_star = eval2.dist_sq;
                                tau_star = m2;
                                st_star = eval2.st;
                            }

                            if (eval1.dist_sq < eval2.dist_sq) {
                                t_hi = m2;
                            } else {
                                t_lo = m1;
                            }
                        }

                        let s_star = st_star.x;
                        let t_star = st_star.y;
                        let thick_j_ccd = v_j.thickness;
                        let thick_vj_ccd = v_vj.thickness;
                        let eff_thick_star = mix(thick_u0, thick_u1, s_star) + mix(thick_j_ccd, thick_vj_ccd, t_star);

                        // 軌道全体で厚み内に侵入しない場合は非接触
                        if (dist_sq_star >= eff_thick_star * eff_thick_star) {
                            continue;
                        }

                        // Step B: トポロジー2ホップ除外判定
                        if (params.exclude_neighbors != 0u) {
                            let a0_star = mix(p_u0_old, p_u0, tau_star);
                            let a1_star = mix(p_u1_old, p_u1, tau_star);
                            let b0_star = mix(p_j_old, p_j, tau_star);
                            let b1_star = mix(p_vj_old, p_vj, tau_star);
                            let td0 = a0_star - b0_star;
                            let td1 = a0_star - b1_star;
                            let td2 = a1_star - b0_star;
                            let td3 = a1_star - b1_star;
                            if (is_topologically_near_folded_sq(u0, j, dot(td0, td0)) ||
                                is_topologically_near_folded_sq(u0, vj, dot(td1, td1)) ||
                                is_topologically_near_folded_sq(u1, j, dot(td2, td2)) ||
                                is_topologically_near_folded_sq(u1, vj, dot(td3, td3))) {
                                continue;
                            }
                        }

                        // 多層布 untangling: 固定基準エッジまたは1Dストランド外層エッジが裏側から表側へ復帰する軌道のみCCD逆阻止を保護
                        var is_untangling_ccd = false;
                        if (params.enable_normal_untangling != 0u && v_u0.layer_id != v_j.layer_id) {
                            let edge_a_fixed = (v_u0.inv_mass == 0.0) && (v_u1.inv_mass == 0.0);
                            let edge_b_fixed = (v_j.inv_mass == 0.0) && (v_vj.inv_mass == 0.0);
                            let delta0_u = ((p_u0_old + p_u1_old) - (p_j_old + p_vj_old)) * 0.5;
                            let delta1_u = ((p_u0 + p_u1) - (p_j + p_vj)) * 0.5;
                            let move_u = delta1_u - delta0_u;
                            if (v_u0.layer_id > v_j.layer_id && !edge_a_fixed) {
                                let n_inner = normals[j].xyz;
                                let n_outer = normals[u0].xyz;
                                let allow_ee_escape = edge_b_fixed || (dot(n_outer, n_outer) <= 0.25);
                                is_untangling_ccd = allow_ee_escape && (dot(n_inner, n_inner) > 0.25) && (dot(delta0_u, n_inner) < 0.0) && (dot(move_u, n_inner) > 0.0);
                            } else if (v_j.layer_id > v_u0.layer_id && !edge_b_fixed) {
                                let n_inner = normals[u0].xyz;
                                let n_outer = normals[j].xyz;
                                let allow_ee_escape = edge_a_fixed || (dot(n_outer, n_outer) <= 0.25);
                                is_untangling_ccd = allow_ee_escape && (dot(n_inner, n_inner) > 0.25) && (dot(-delta0_u, n_inner) < 0.0) && (dot(-move_u, n_inner) > 0.0);
                            }
                        }
                        if (is_untangling_ccd) {
                            continue;
                        }

                        // Step C: 開始時分離法線 n0 の算出とクロスオーバー判定
                        let st0 = closest_points_segments(p_u0_old, p_u1_old, p_j_old, p_vj_old);
                        let pt0_a = mix(p_u0_old, p_u1_old, st0.x);
                        let pt0_b = mix(p_j_old, p_vj_old, st0.y);
                        let delta0 = pt0_a - pt0_b;
                        let dist0 = length(delta0);

                        if (dist0 > EPSILON) {
                            let n0 = delta0 / dist0;

                            let st1 = closest_points_segments(p_u0, p_u1, p_j, p_vj);
                            let pt1_a = mix(p_u0, p_u1, st1.x);
                            let pt1_b = mix(p_j, p_vj, st1.y);
                            let delta1 = pt1_a - pt1_b;
                            let dist1 = length(delta1);

                            let pt_end_a = mix(p_u0, p_u1, s_star);
                            let pt_end_b = mix(p_j, p_vj, t_star);
                            let delta_end = pt_end_a - pt_end_b;
                            let h_end = dot(delta_end, n0);

                            let is_crossover = (h_end <= 0.0) ||
                                               (dot(delta1, n0) <= 0.0) ||
                                               (is_boundary_pair && dist1 >= eff_thick_star && tau_star < 1.0);

                            if (h_end < eff_thick_star && is_crossover) {
                                // Step D: 運動量保存を満たす 100% ハード CCD 変位の分配
                                let pen_ccd = eff_thick_star - h_end;
                                let edge_b_has_pin = (v_j.inv_mass <= 0.0) || (v_vj.inv_mass <= 0.0);
                                if (edge_a_has_pin && edge_b_has_pin) {
                                    continue;
                                }

                                let edge_a_all_pin = (v_u0.inv_mass == 0.0) && (v_u1.inv_mass == 0.0);
                                let edge_b_all_pin = (v_j.inv_mass == 0.0) && (v_vj.inv_mass == 0.0);
                                let count_pin_ee = !(edge_a_all_pin || edge_b_all_pin);
                                if (edge_b_has_pin) {
                                    let edge_move = ((p_j - p_j_old) + (p_vj - p_vj_old)) * 0.5;
                                    let a = 1.0 - s_star;
                                    let b = s_star;
                                    let w0 = v_u0.inv_mass;
                                    let w1 = v_u1.inv_mass;
                                    let denom = a * a * w0 + b * b * w1;
                                    if (denom > EPSILON) {
                                        let push_base = n0 * pen_ccd + edge_move * 0.5;
                                        if (w0 > 0.0) {
                                            let f0 = min((a * w0) / denom, 2.0);
                                            add_ccd_atomic(u0, push_base * f0, count_pin_ee);
                                        }
                                        if (w1 > 0.0) {
                                            let f1 = min((b * w1) / denom, 2.0);
                                            add_ccd_atomic(u1, push_base * f1, count_pin_ee);
                                        }
                                    }
                                } else if (edge_a_has_pin) {
                                    let edge_move = ((p_u0 - p_u0_old) + (p_u1 - p_u1_old)) * 0.5;
                                    let a = 1.0 - t_star;
                                    let b = t_star;
                                    let w0 = v_j.inv_mass;
                                    let w1 = v_vj.inv_mass;
                                    let denom = a * a * w0 + b * b * w1;
                                    if (denom > EPSILON) {
                                        let push_base = -n0 * pen_ccd + edge_move * 0.5;
                                        if (w0 > 0.0) {
                                            let fj = min((a * w0) / denom, 2.0);
                                            add_ccd_atomic(j, push_base * fj, count_pin_ee);
                                        }
                                        if (w1 > 0.0) {
                                            let fvj = min((b * w1) / denom, 2.0);
                                            add_ccd_atomic(vj, push_base * fvj, count_pin_ee);
                                        }
                                    }
                                } else {
                                    let w_u0 = v_u0.inv_mass;
                                    let w_u1 = v_u1.inv_mass;
                                    let w_j = v_j.inv_mass;
                                    let w_vj = v_vj.inv_mass;
                                    let w_e1 = (1.0 - s_star) * (1.0 - s_star) * w_u0 + s_star * s_star * w_u1;
                                    let w_e2 = (1.0 - t_star) * (1.0 - t_star) * w_j + t_star * t_star * w_vj;
                                    let w_tot = w_e1 + w_e2;

                                    if (w_tot > EPSILON) {
                                        let total_ccd = n0 * pen_ccd;
                                        if (w_u0 > 0.0) {
                                            add_ccd_atomic(u0, total_ccd * ((1.0 - s_star) * w_u0 / w_tot), true);
                                        }
                                        if (w_u1 > 0.0) {
                                            add_ccd_atomic(u1, total_ccd * (s_star * w_u1 / w_tot), true);
                                        }
                                        if (w_j > 0.0) {
                                            add_ccd_atomic(j, -total_ccd * ((1.0 - t_star) * w_j / w_tot), true);
                                        }
                                        if (w_vj > 0.0) {
                                            add_ccd_atomic(vj, -total_ccd * (t_star * w_vj / w_tot), true);
                                        }
                                    }
                                }
                                continue;
                            }
                        }
                    }

                    // 【早期枝切り】線分間最短距離の二乗判定（内部エッジで先行計算済みの場合は再利用）
                    var st = vec2<f32>(0.0, 0.0);
                    var pt1 = vec3<f32>(0.0);
                    var pt2 = vec3<f32>(0.0);
                    var delta_ee = vec3<f32>(0.0);
                    var dist_sq_ee = 0.0;
                    var effective_thick = 0.0;
                    if (has_cached_st1) {
                        st = st1_cached;
                        pt1 = pt1_a_cached;
                        pt2 = pt1_b_cached;
                        delta_ee = delta1_cached;
                        dist_sq_ee = dist_sq_ee_cached;
                        effective_thick = eff_thick_cached;
                    } else {
                        st = closest_points_segments(p_u0, p_u1, p_j, p_vj);
                        pt1 = p_u0 + st.x * (p_u1 - p_u0);
                        pt2 = p_j + st.y * (p_vj - p_j);
                        delta_ee = pt1 - pt2;
                        dist_sq_ee = dot(delta_ee, delta_ee);

                        let thick_j = v_j.thickness;
                        let thick_vj = v_vj.thickness;
                        let thick_a = mix(thick_u0, thick_u1, st.x);
                        let thick_b = mix(thick_j, thick_vj, st.y);
                        effective_thick = thick_a + thick_b;
                    }

                    if (dist_sq_ee >= effective_thick * effective_thick || dist_sq_ee <= EPSILON * EPSILON) {
                        continue;
                    }

                    // 【遅延評価】実際に厚み内に侵入しているペア（0.1%未満）のみ 2ホップ二分探索×4回を実行
                    if (params.exclude_neighbors != 0u) {
                        let d0 = p_u0 - p_j;
                        let d1 = p_u0 - p_vj;
                        let d2 = p_u1 - p_j;
                        let d3 = p_u1 - p_vj;
                        if (is_topologically_near_folded_sq(u0, j, dot(d0, d0)) ||
                            is_topologically_near_folded_sq(u0, vj, dot(d1, d1)) ||
                            is_topologically_near_folded_sq(u1, j, dot(d2, d2)) ||
                            is_topologically_near_folded_sq(u1, vj, dot(d3, d3))) {
                            continue;
                        }
                    }

                    let dist_ee = sqrt(dist_sq_ee);
                    // 単一レイヤー時は normals バッファへのグローバルメモリアクセスを完全スキップ
                    var is_untangling_ee = false;
                    if (params.enable_normal_untangling != 0u) {
                        let edge_a_fixed_ee = (v_u0.inv_mass <= 0.0) && (v_u1.inv_mass <= 0.0);
                        let edge_b_fixed_ee = (v_j.inv_mass <= 0.0) && (v_vj.inv_mass <= 0.0);
                        is_untangling_ee = is_outer_edge_behind_inner(
                            v_u0.layer_id,
                            v_j.layer_id,
                            u0,
                            u1,
                            j,
                            vj,
                            edge_a_fixed_ee,
                            edge_b_fixed_ee,
                            delta_ee,
                        );
                    }

                    if (!is_untangling_ee) {
                        let n_ee = delta_ee / dist_ee;
                        let pen_ee = effective_thick - dist_ee;
                        let s = st.x;
                        let t = st.y;

                        let edge_b_has_pin = (v_j.inv_mass <= 0.0) || (v_vj.inv_mass <= 0.0);
                        if (edge_a_has_pin && edge_b_has_pin) {
                            continue;
                        }

                        if (edge_b_has_pin) {
                            // 相手エッジがピン -> 自エッジ側を変位（線分パラメータ正規化）
                            let edge_move = ((p_j - p_j_old) + (p_vj - p_vj_old)) * 0.5;
                            let a = 1.0 - s;
                            let b = s;
                            let w0 = v_u0.inv_mass;
                            let w1 = v_u1.inv_mass;
                            let denom = a * a * w0 + b * b * w1;
                            if (denom > EPSILON) {
                                let push_base = n_ee * pen_ee + edge_move * 0.5;
                                if (w0 > 0.0) {
                                    let f0 = min((a * w0) / denom, 2.0);
                                    add_pin_disp_atomic(u0, push_base * f0);
                                }
                                if (w1 > 0.0) {
                                    let f1 = min((b * w1) / denom, 2.0);
                                    add_pin_disp_atomic(u1, push_base * f1);
                                }
                            }
                        } else if (edge_a_has_pin) {
                            // 自エッジがピン -> 相手エッジ側を変位（線分パラメータ正規化）
                            let edge_move = ((p_u0 - p_u0_old) + (p_u1 - p_u1_old)) * 0.5;
                            let a = 1.0 - t;
                            let b = t;
                            let w0 = v_j.inv_mass;
                            let w1 = v_vj.inv_mass;
                            let denom = a * a * w0 + b * b * w1;
                            if (denom > EPSILON) {
                                let push_base = -n_ee * pen_ee + edge_move * 0.5;
                                if (w0 > 0.0) {
                                    let fj = min((a * w0) / denom, 2.0);
                                    add_pin_disp_atomic(j, push_base * fj);
                                }
                                if (w1 > 0.0) {
                                    let fvj = min((b * w1) / denom, 2.0);
                                    add_pin_disp_atomic(vj, push_base * fvj);
                                }
                            }
                        } else {
                            // 両エッジ動的 -> Coupled XPBD 対称質量分配
                            let w_u0 = v_u0.inv_mass;
                            let w_u1 = v_u1.inv_mass;
                            let w_j = v_j.inv_mass;
                            let w_vj = v_vj.inv_mass;
                            let w_e1 = (1.0 - s) * (1.0 - s) * w_u0 + s * s * w_u1;
                            let w_e2 = (1.0 - t) * (1.0 - t) * w_j + t * t * w_vj;
                            let w_tot = w_e1 + w_e2;

                            if (w_tot > EPSILON) {
                                let total_disp = n_ee * (pen_ee * 0.6);
                                if (w_u0 > 0.0) {
                                    add_disp_atomic(u0, total_disp * ((1.0 - s) * w_u0 / w_tot));
                                }
                                if (w_u1 > 0.0) {
                                    add_disp_atomic(u1, total_disp * (s * w_u1 / w_tot));
                                }
                                if (w_j > 0.0) {
                                    add_disp_atomic(j, -total_disp * ((1.0 - t) * w_j / w_tot));
                                }
                                if (w_vj > 0.0) {
                                    add_disp_atomic(vj, -total_disp * (t * w_vj / w_tot));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
