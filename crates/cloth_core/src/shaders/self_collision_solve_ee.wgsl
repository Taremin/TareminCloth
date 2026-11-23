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
    vt_dropped: u32,
    ee_dropped: u32,
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

@group(0) @binding(0) var<storage, read> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> active_ee_pairs: array<GpuEePair>;
@group(0) @binding(2) var<storage, read> counters: PairCounters;
@group(0) @binding(3) var<storage, read_write> accum: array<SelfCollisionAccum>;
@group(0) @binding(4) var<uniform> params: SolveParams;
@group(0) @binding(5) var<storage, read> normals: array<vec4<f32>>;

const EPSILON: f32 = 1e-7;
const FIXED_SCALE: f32 = 1000000.0;
const MAX_ATOMIC_FIXED: f32 = 50000000.0;

fn add_disp_atomic(v_idx: u32, disp: vec3<f32>) {
    if (v_idx >= params.num_vertices) {
        return;
    }
    let ix = i32(clamp(disp.x * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    let iy = i32(clamp(disp.y * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    let iz = i32(clamp(disp.z * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    atomicAdd(&accum[v_idx].dx, ix);
    atomicAdd(&accum[v_idx].dy, iy);
    atomicAdd(&accum[v_idx].dz, iz);
    atomicAdd(&accum[v_idx].count, 1u);
}

fn add_pin_disp_atomic(v_idx: u32, disp: vec3<f32>) {
    if (v_idx >= params.num_vertices) {
        return;
    }
    let ix = i32(clamp(disp.x * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    let iy = i32(clamp(disp.y * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    let iz = i32(clamp(disp.z * FIXED_SCALE, -MAX_ATOMIC_FIXED, MAX_ATOMIC_FIXED));
    atomicAdd(&accum[v_idx].pin_dx, ix);
    atomicAdd(&accum[v_idx].pin_dy, iy);
    atomicAdd(&accum[v_idx].pin_dz, iz);
    atomicAdd(&accum[v_idx].pin_count, 1u);
}

fn add_ccd_atomic(v_idx: u32, disp: vec3<f32>, count_ee: bool) {
    if (v_idx >= params.num_vertices) {
        return;
    }
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


// 2線分 (p1-q1) と (p2-q2) の最近傍パラメータ (s, t)
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    // 頂点メジャー固定スロット配置 (N*quota entries)。未使用枠は
    // センチネル (v_i == 0xFFFFFFFF) で埋められている。
    let n_verts = params.num_vertices;
    if (n_verts == 0u) {
        return;
    }
    if (pair_idx >= params.max_ee_pairs) {
        return;
    }

    let pair = active_ee_pairs[pair_idx];
    if (pair.v_i == 0xFFFFFFFFu) {
        return;
    }
    let index = pair.v_i;
    let ui = pair.v_ui;
    let j = pair.v_j;
    let vj = pair.v_vj;

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

    let edge_i_has_pin = (v_i.inv_mass <= 0.0) || (v_ui.inv_mass <= 0.0);
    let edge_j_has_pin = (v_j.inv_mass <= 0.0) || (v_vj.inv_mass <= 0.0);
    if (edge_i_has_pin && edge_j_has_pin) {
        return;
    }

    let p_i = v_i.prev_pos;
    let p_ui = v_ui.prev_pos;
    let p_j = v_j.prev_pos;
    let p_vj = v_vj.prev_pos;

    let x_old_i = v_i.position;
    let x_old_ui = v_ui.position;
    let p_j_old = v_j.position;
    let p_vj_old = v_vj.position;

    let thick_i = v_i.thickness;
    let thick_ui = v_ui.thickness;
    let thick_j = v_j.thickness;
    let thick_vj = v_vj.thickness;

    // 選択的 CCD ゲート兼 終了時刻最近接点キャッシュ
    let st1_cached = closest_points_segments(p_i, p_ui, p_j, p_vj);
    let pt1_a_cached = mix(p_i, p_ui, st1_cached.x);
    let pt1_b_cached = mix(p_j, p_vj, st1_cached.y);
    let delta1_cached = pt1_a_cached - pt1_b_cached;
    let dist_sq_ee_cached = dot(delta1_cached, delta1_cached);

    let effective_thick = mix(thick_i, thick_ui, st1_cached.x) + mix(thick_j, thick_vj, st1_cached.y);
    let max_eff_thick = max(thick_i, thick_ui) + max(thick_j, thick_vj);

    let dv_i = p_i - x_old_i;
    let dv_ui = p_ui - x_old_ui;
    let dv_j = p_j - p_j_old;
    let dv_vj = p_vj - p_vj_old;
    let va_avg = 0.5 * (dv_i + dv_ui);
    let vb_avg = 0.5 * (dv_j + dv_vj);
    let rel_disp = length(va_avg - vb_avg) + 0.5 * (length(dv_ui - dv_i) + length(dv_vj - dv_j));

    let reach_bound = max_eff_thick + rel_disp;
    if (dist_sq_ee_cached > reach_bound * reach_bound) {
        return;
    }

    let delta0_g = mix(x_old_i, x_old_ui, st1_cached.x) - mix(p_j_old, p_vj_old, st1_cached.y);
    let run_ccd = (rel_disp > effective_thick * 0.5) || (dot(delta0_g, delta1_cached) <= 0.0);

    if (run_ccd) {
        // 粗サンプル（9点: tau = 0/8 .. 8/8）による最小距離と体積符号反転の探索
        var min_sample_dist_sq = 1e20;
        var best_tau = 0.0;
        var half_span = 0.125;
        var min_st = vec2<f32>(0.0, 0.0);
        var prev_vol = eval_segment_volume(x_old_i, p_i, x_old_ui, p_ui, p_j_old, p_j, p_vj_old, p_vj, 0.0);
        var zero_cross_tau = -1.0;

        for (var k = 0u; k <= 8u; k = k + 1u) {
            let tau_k = f32(k) * 0.125;
            let eval_res = eval_segment_dist_sq(x_old_i, p_i, x_old_ui, p_ui, p_j_old, p_j, p_vj_old, p_vj, tau_k);
            if (eval_res.dist_sq < min_sample_dist_sq) {
                min_sample_dist_sq = eval_res.dist_sq;
                best_tau = tau_k;
                min_st = eval_res.st;
            }
            if (k > 0u) {
                let vol = eval_segment_volume(x_old_i, p_i, x_old_ui, p_ui, p_j_old, p_j, p_vj_old, p_vj, tau_k);
                if (prev_vol * vol <= 0.0 && zero_cross_tau < 0.0) {
                    zero_cross_tau = tau_k - 0.0625;
                }
                prev_vol = vol;
            }
        }

        if (zero_cross_tau >= 0.0) {
            let eval_zc = eval_segment_dist_sq(x_old_i, p_i, x_old_ui, p_ui, p_j_old, p_j, p_vj_old, p_vj, zero_cross_tau);
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
            let eval1 = eval_segment_dist_sq(x_old_i, p_i, x_old_ui, p_ui, p_j_old, p_j, p_vj_old, p_vj, m1);
            let eval2 = eval_segment_dist_sq(x_old_i, p_i, x_old_ui, p_ui, p_j_old, p_j, p_vj_old, p_vj, m2);

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
        let eff_thick_star = mix(thick_i, thick_ui, s_star) + mix(thick_j, thick_vj, t_star);

        if (dist_sq_star >= eff_thick_star * eff_thick_star) {
            return;
        }

        // 多層布 untangling: 固定基準エッジまたは1Dストランド外層エッジが裏側から表側へ復帰する軌道のみ保護
        var is_untangling_ccd = false;
        if (params.enable_normal_untangling != 0u && v_i.layer_id != v_j.layer_id) {
            let edge_i_fixed = (v_i.inv_mass == 0.0) && (v_ui.inv_mass == 0.0);
            let edge_j_fixed = (v_j.inv_mass == 0.0) && (v_vj.inv_mass == 0.0);
            let delta0_u = ((x_old_i + x_old_ui) - (p_j_old + p_vj_old)) * 0.5;
            let delta1_u = ((p_i + p_ui) - (p_j + p_vj)) * 0.5;
            let move_u = delta1_u - delta0_u;
            if (v_i.layer_id > v_j.layer_id && !edge_i_fixed) {
                let n_inner = normals[j].xyz;
                let n_outer = normals[index].xyz;
                let allow_ee_escape = edge_j_fixed || (dot(n_outer, n_outer) <= 0.25);
                is_untangling_ccd = allow_ee_escape && (dot(n_inner, n_inner) > 0.25) && (dot(delta0_u, n_inner) < 0.0) && (dot(move_u, n_inner) > 0.0);
            } else if (v_j.layer_id > v_i.layer_id && !edge_j_fixed) {
                let n_inner = normals[index].xyz;
                let n_outer = normals[j].xyz;
                let allow_ee_escape = edge_i_fixed || (dot(n_outer, n_outer) <= 0.25);
                is_untangling_ccd = allow_ee_escape && (dot(n_inner, n_inner) > 0.25) && (dot(-delta0_u, n_inner) < 0.0) && (dot(-move_u, n_inner) > 0.0);
            }
        }
        if (is_untangling_ccd) {
            return;
        }

        // 開始時分離法線 n0 の算出とクロスオーバー遮断
        let st0 = closest_points_segments(x_old_i, x_old_ui, p_j_old, p_vj_old);
        let pt0_a = mix(x_old_i, x_old_ui, st0.x);
        let pt0_b = mix(p_j_old, p_vj_old, st0.y);
        let delta0 = pt0_a - pt0_b;
        let dist0 = length(delta0);

        if (dist0 > EPSILON) {
            let n0 = delta0 / dist0;
            let pt_end_a = mix(p_i, p_ui, s_star);
            let pt_end_b = mix(p_j, p_vj, t_star);
            let h_end = dot(pt_end_a - pt_end_b, n0);
            let is_crossover = (h_end <= 0.0) || (dot(delta1_cached, n0) <= 0.0);

            if (h_end < eff_thick_star && is_crossover) {
                let pen_ccd = eff_thick_star - h_end;
                let edge_i_all_pin = (v_i.inv_mass == 0.0) && (v_ui.inv_mass == 0.0);
                let edge_j_all_pin = (v_j.inv_mass == 0.0) && (v_vj.inv_mass == 0.0);
                let count_pin_ee = !(edge_i_all_pin || edge_j_all_pin);

                if (edge_j_has_pin) {
                    let edge_move = ((p_j - p_j_old) + (p_vj - p_vj_old)) * 0.5;
                    let a = 1.0 - s_star;
                    let b = s_star;
                    let w0 = v_i.inv_mass;
                    let w1 = v_ui.inv_mass;
                    let denom = a * a * w0 + b * b * w1;
                    if (denom > EPSILON) {
                        let push_base = n0 * pen_ccd + edge_move * 0.5;
                        if (w0 > 0.0) {
                            let f0 = min((a * w0) / denom, 2.0);
                            add_ccd_atomic(index, push_base * f0, count_pin_ee);
                        }
                        if (w1 > 0.0) {
                            let f1 = min((b * w1) / denom, 2.0);
                            add_ccd_atomic(ui, push_base * f1, count_pin_ee);
                        }
                    }
                } else if (edge_i_has_pin) {
                    let edge_move = ((p_i - x_old_i) + (p_ui - x_old_ui)) * 0.5;
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
                    let w_i = v_i.inv_mass;
                    let w_ui = v_ui.inv_mass;
                    let w_j = v_j.inv_mass;
                    let w_vj = v_vj.inv_mass;
                    let w_e1 = (1.0 - s_star) * (1.0 - s_star) * w_i + s_star * s_star * w_ui;
                    let w_e2 = (1.0 - t_star) * (1.0 - t_star) * w_j + t_star * t_star * w_vj;
                    let w_tot = w_e1 + w_e2;

                    if (w_tot > EPSILON) {
                        let total_ccd = n0 * pen_ccd;
                        if (w_i > 0.0) {
                            add_ccd_atomic(index, total_ccd * ((1.0 - s_star) * w_i / w_tot), true);
                        }
                        if (w_ui > 0.0) {
                            add_ccd_atomic(ui, total_ccd * (s_star * w_ui / w_tot), true);
                        }
                        if (w_j > 0.0) {
                            add_ccd_atomic(j, -total_ccd * ((1.0 - t_star) * w_j / w_tot), true);
                        }
                        if (w_vj > 0.0) {
                            add_ccd_atomic(vj, -total_ccd * (t_star * w_vj / w_tot), true);
                        }
                    }
                }
                return;
            }
        }
    }

    // 離散線分最短幾何反発
    if (dist_sq_ee_cached >= effective_thick * effective_thick || dist_sq_ee_cached <= EPSILON * EPSILON) {
        return;
    }

    let dist_ee = sqrt(dist_sq_ee_cached);
    var is_untangling_ee = false;
    if (params.enable_normal_untangling != 0u) {
        let edge_i_fixed_ee = (v_i.inv_mass <= 0.0) && (v_ui.inv_mass <= 0.0);
        let edge_j_fixed_ee = (v_j.inv_mass <= 0.0) && (v_vj.inv_mass <= 0.0);
        is_untangling_ee = is_outer_edge_behind_inner(
            v_i.layer_id,
            v_j.layer_id,
            index,
            ui,
            j,
            vj,
            edge_i_fixed_ee,
            edge_j_fixed_ee,
            delta1_cached,
        );
    }

    if (!is_untangling_ee) {
        let n_ee = delta1_cached / dist_ee;
        let pen_ee = effective_thick - dist_ee;
        let s = st1_cached.x;
        let t = st1_cached.y;

        if (edge_j_has_pin) {
            let edge_move = ((p_j - p_j_old) + (p_vj - p_vj_old)) * 0.5;
            let a = 1.0 - s;
            let b = s;
            let w0 = v_i.inv_mass;
            let w1 = v_ui.inv_mass;
            let denom = a * a * w0 + b * b * w1;
            if (denom > EPSILON) {
                let push_base = n_ee * pen_ee + edge_move * 0.5;
                if (w0 > 0.0) {
                    let f0 = min((a * w0) / denom, 2.0);
                    add_pin_disp_atomic(index, push_base * f0);
                }
                if (w1 > 0.0) {
                    let f1 = min((b * w1) / denom, 2.0);
                    add_pin_disp_atomic(ui, push_base * f1);
                }
            }
        } else if (edge_i_has_pin) {
            let edge_move = ((p_i - x_old_i) + (p_ui - x_old_ui)) * 0.5;
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
                if (w_vj > 0.0) {
                    add_disp_atomic(vj, -total_disp * (t * w_vj / w_tot));
                }
            }
        }
    }
}
