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

struct AtomicAccum {
    dx: atomic<i32>,
    dy: atomic<i32>,
    dz: atomic<i32>,
    count: atomic<u32>,
};

struct ClosestBaryResult {
    pt: vec3<f32>,
    bary: vec3<f32>,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> cell_heads: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> vert_next: array<i32>;
@group(0) @binding(3) var<uniform> params: SelfCollisionParams;
@group(0) @binding(4) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> local_edge_lengths: array<f32>;
@group(0) @binding(6) var<storage, read> adj_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> adj_indices: array<u32>;
@group(0) @binding(8) var<storage, read> island_ids: array<u32>;
@group(0) @binding(9) var<storage, read> star_offsets: array<u32>;
@group(0) @binding(10) var<storage, read> star_indices: array<GpuStarPair>;
@group(0) @binding(11) var<storage, read_write> accum: array<AtomicAccum>;

const EPSILON: f32 = 1e-7;
const FIXED_SCALE: f32 = 1000000.0;
const INV_FIXED_SCALE: f32 = 1.0 / FIXED_SCALE;

fn add_disp_atomic(v_idx: u32, disp: vec3<f32>) {
    let ix = i32(clamp(disp.x * FIXED_SCALE, -2e9, 2e9));
    let iy = i32(clamp(disp.y * FIXED_SCALE, -2e9, 2e9));
    let iz = i32(clamp(disp.z * FIXED_SCALE, -2e9, 2e9));
    atomicAdd(&accum[v_idx].dx, ix);
    atomicAdd(&accum[v_idx].dy, iy);
    atomicAdd(&accum[v_idx].dz, iz);
    atomicAdd(&accum[v_idx].count, 1u);
}

fn hash_coords(coord: vec3<i32>, table_size: u32) -> u32 {
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    let n = (u32(coord.x) * p1) ^ (u32(coord.y) * p2) ^ (u32(coord.z) * p3);
    return n % table_size;
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
    if (u < 0.0 || u > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let qvec = cross(tvec, e1);
    let v = dot(d, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let t = dot(e2, qvec) * inv_det;
    if (t >= 0.0 && t <= 1.0) {
        return vec4<f32>(1.0, t, u, v);
    }

    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

// 2線分 p1-q1 と p2-q2 間の最短距離パラメータ (s, t) in [0, 1]^2
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
    if (denom > EPSILON) {
        s = clamp((b * f - c * e) / denom, 0.0, 1.0);
    } else {
        s = 0.0;
    }
    let t = clamp((b * s + f) / e, 0.0, 1.0);
    return vec2<f32>(s, t);
}

// トポロジー1ホップ直接隣接判定
fn is_adjacent_vert(vert_a: u32, vert_b: u32) -> bool {
    let start_a = adj_offsets[vert_a];
    let end_a = adj_offsets[vert_a + 1u];
    for (var k = start_a; k < end_a; k = k + 1u) {
        if (adj_indices[k] == vert_b) {
            return true;
        }
    }
    return false;
}

// トポロジー2ホップ近傍判定
fn is_topologically_near(vert_a: u32, vert_b: u32) -> bool {
    if (vert_a == vert_b) {
        return true;
    }
    let start_a = adj_offsets[vert_a];
    let end_a = adj_offsets[vert_a + 1u];

    for (var k = start_a; k < end_a; k = k + 1u) {
        let u = adj_indices[k];
        if (u == vert_b) {
            return true;
        }
        let start_u = adj_offsets[u];
        let end_u = adj_offsets[u + 1u];
        for (var ku = start_u; ku < end_u; ku = ku + 1u) {
            if (adj_indices[ku] == vert_b) {
                return true;
            }
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_vertices) {
        return;
    }

    var v_i = vertices[index];
    if (v_i.inv_mass <= 0.0) {
        return;
    }

    let x_old_i = v_i.position; // サブステップ開始時の確定位置
    let p_i = v_i.prev_pos;     // 現在の予測位置
    let thick_i = v_i.thickness;

    let cell = vec3<i32>(floor(p_i / params.cell_size));

    let thick_margin = vec3<f32>(thick_i * 1.5);
    let sweep_min = min(x_old_i, p_i) - thick_margin;
    let sweep_max = max(x_old_i, p_i) + thick_margin;

    let target_min_cell = vec3<i32>(floor(sweep_min / params.cell_size));
    let target_max_cell = vec3<i32>(floor(sweep_max / params.cell_size));

    // 移動量が過大でも計算爆発を防ぐため、基準セルから各軸最大 ±2 セル (最大 5x5x5) に制限
    let min_cell = max(cell - vec3<i32>(2), target_min_cell);
    let max_cell = min(cell + vec3<i32>(2), target_max_cell);

    // Sweep AABB 内の全セルを走査 (CCD による飛び越え遮断を保証)
    for (var cx = min_cell.x; cx <= max_cell.x; cx = cx + 1) {
        for (var cy = min_cell.y; cy <= max_cell.y; cy = cy + 1) {
            for (var cz = min_cell.z; cz <= max_cell.z; cz = cz + 1) {
                let neighbor_cell = vec3<i32>(cx, cy, cz);
                let h = hash_coords(neighbor_cell, params.table_size);

                var other_idx = atomicLoad(&cell_heads[h]);
                var iter_count = 0u;

                let max_iters = select(128u, params.max_search_iterations, params.max_search_iterations > 0u);
                while (other_idx >= 0 && iter_count < max_iters) {
                    let j = u32(other_idx);
                    let v_j = vertices[j];
                    let p_j = v_j.prev_pos;

                    // 枝切り: ハッシュ値が同じでも実際の空間セルが異なる場合はスキップ（ハッシュ衝突排除）
                    let j_cell = vec3<i32>(floor(p_j / params.cell_size));
                    if (all(j_cell == neighbor_cell) && index != j) {
                        let is_near_2hop = is_topologically_near(index, j);

                        if (!is_near_2hop) {
                            let p_j_old = v_j.position;
                            let thick_j = v_j.thickness;

                            let local_len_i = local_edge_lengths[index];
                            let local_len_j = local_edge_lengths[j];
                            let min_dist = thick_i + thick_j;
                            let edge_margin = min(local_len_i, local_len_j) * 0.06;
                            let effective_thick = max(min_dist, edge_margin);

                            // ==========================================
                            // 1. 球対球 (Vertex-Vertex) 幾何反発 (対称アトミック分配)
                            // ==========================================
                            // 相手がピン留め（v_j.inv_mass <= 0）の場合は相手スレッドが処理しないため、
                            // index < j の有無にかかわらず自スレッドが必ず処理して自頂点を退避させる。
                            let both_dynamic = (v_j.inv_mass > 0.0);
                            if (!both_dynamic || index < j) {
                                let delta_vv = p_i - p_j;
                                let dist_vv = length(delta_vv);

                                if (dist_vv < effective_thick && dist_vv > EPSILON) {
                                    let n = delta_vv / dist_vv;
                                    let pen = effective_thick - dist_vv;
                                    let w_i = v_i.inv_mass;
                                    let w_j = v_j.inv_mass;
                                    let w_sum = w_i + w_j;

                                    if (w_sum > EPSILON) {
                                        if (w_j <= 0.0) {
                                            // 相手がピン留め（相手スレッドは停止しているため、自頂点のみ退避）
                                            let v_j_move = p_j - p_j_old;
                                            let col_disp = n * pen + v_j_move * 0.5;
                                            add_disp_atomic(index, col_disp);
                                        } else {
                                            // 動的頂点同士：index < j により1回のみ評価され、質量比に応じて対称分配
                                            let disp = n * (pen * 0.5);
                                            add_disp_atomic(index, disp * (w_i / w_sum));
                                            add_disp_atomic(j, -disp * (w_j / w_sum));
                                        }
                                    }
                                }
                            }

                            // ==========================================
                            // 2. Vertex-Triangle (頂点 対 面) 相対CCD & 幾何反発
                            // ==========================================
                                let star_start = star_offsets[j];
                                let star_end = star_offsets[j + 1u];

                                for (var s = star_start; s < star_end; s = s + 1u) {
                                    let pair = star_indices[s];
                                    let v0 = pair.v0;
                                    let v1 = pair.v1;

                                    if (!is_topologically_near(index, v0) && !is_topologically_near(index, v1)) {
                                        let v_v0 = vertices[v0];
                                        let v_v1 = vertices[v1];
                                        let p_v0 = v_v0.prev_pos;
                                        let p_v1 = v_v1.prev_pos;
                                        let p_v0_old = v_v0.position;
                                        let p_v1_old = v_v1.position;

                                        // 相手三角形の移動ベクトル
                                        let tri_move = ((p_j - p_j_old) + (p_v0 - p_v0_old) + (p_v1 - p_v1_old)) / 3.0;
                                        let tri_has_pin = (v_j.inv_mass <= 0.0) || (v_v0.inv_mass <= 0.0) || (v_v1.inv_mass <= 0.0);

                                        // 相手の移動を考慮した相対始点（トンネリング検知用）
                                        let rel_old_i = x_old_i - tri_move;

                                        // A) 軌道トポロジー貫通遮断 (CCD)
                                        let ccd_hit = intersect_segment_triangle(rel_old_i, p_i, p_j, p_v0, p_v1);
                                        if (ccd_hit.x > 0.5) {
                                            let t = ccd_hit.y;
                                            var push_disp = vec3<f32>(0.0);
                                            if (tri_has_pin) {
                                                push_disp = tri_move * (1.0 - t);
                                            }

                                            // 面の手前への安全クリアランス
                                            let move_vec = p_i - rel_old_i;
                                            let move_len = length(move_vec);
                                            let margin_t = min(effective_thick / max(move_len, 1e-4), 0.3);
                                            let t_safe = max(0.0, t - margin_t);
                                            let p_safe = rel_old_i + t_safe * move_vec;
                                            
                                            let ccd_disp = (p_safe - p_i) + push_disp;
                                            add_disp_atomic(index, ccd_disp);
                                        } else {
                                            // B) 最近傍点幾何反発（重心座標による作用反作用の対称分配）
                                            let res_vt = closest_point_on_triangle_bary(p_i, p_j, p_v0, p_v1);
                                            let delta_vt = p_i - res_vt.pt;
                                            let dist_vt = length(delta_vt);

                                            if (dist_vt < effective_thick && dist_vt > EPSILON) {
                                                let n_vt = delta_vt / dist_vt;
                                                let pen_vt = effective_thick - dist_vt;
                                                if (tri_has_pin) {
                                                    // 相手三角形がピン留め頂点を含む場合は自頂点のみコライダー押し出し
                                                    let col_disp = n_vt * pen_vt + tri_move * 0.5;
                                                    add_disp_atomic(index, col_disp);
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
                                }

                            // ==========================================
                            // 3. Edge-Edge (辺 対 辺) 幾何反発 (対称アトミック分配)
                            // ==========================================
                            let adj_i_start = adj_offsets[index];
                                let adj_i_end = adj_offsets[index + 1u];
                                let adj_j_start = adj_offsets[j];
                                let adj_j_end = adj_offsets[j + 1u];

                                let num_edges_i = min(adj_i_end - adj_i_start, 8u);
                                let num_edges_j = min(adj_j_end - adj_j_start, 8u);

                                for (var ei = 0u; ei < num_edges_i; ei = ei + 1u) {
                                    let ui = adj_indices[adj_i_start + ei];
                                    let v_ui = vertices[ui];
                                    let p_ui = v_ui.prev_pos;

                                    for (var ej = 0u; ej < num_edges_j; ej = ej + 1u) {
                                        let vj = adj_indices[adj_j_start + ej];
                                        if (ui != j && ui != vj && index != vj) {
                                            let v_vj = vertices[vj];
                                            let p_vj = v_vj.prev_pos;
                                            let p_vj_old = v_vj.position;

                                            let edge_j_has_pin = (v_j.inv_mass <= 0.0) || (v_vj.inv_mass <= 0.0);
                                            // 相手エッジがピンの場合は自エッジ側が必ず処理。両方が動的エッジなら index < j で重複排除
                                            if (edge_j_has_pin || index < j) {
                                                let st = closest_points_segments(p_i, p_ui, p_j, p_vj);
                                                let pt1 = p_i + st.x * (p_ui - p_i);
                                                let pt2 = p_j + st.y * (p_vj - p_j);

                                                let delta_ee = pt1 - pt2;
                                                let dist_ee = length(delta_ee);

                                                if (dist_ee < effective_thick && dist_ee > EPSILON) {
                                                    let n_ee = delta_ee / dist_ee;
                                                    let pen_ee = effective_thick - dist_ee;
                                                    let s = st.x;
                                                    let t = st.y;

                                                    let edge_i_has_pin = (v_i.inv_mass <= 0.0) || (v_ui.inv_mass <= 0.0);

                                                    if (edge_j_has_pin) {
                                                        let edge_move = ((p_j - p_j_old) + (p_vj - p_vj_old)) * 0.5;
                                                        let col_disp = n_ee * (pen_ee * (1.0 - s)) + edge_move * (1.0 - s) * 0.5;
                                                        add_disp_atomic(index, col_disp);
                                                        if (v_ui.inv_mass > 0.0) {
                                                            let col_disp_ui = n_ee * (pen_ee * s) + edge_move * s * 0.5;
                                                            add_disp_atomic(ui, col_disp_ui);
                                                        }
                                                    } else if (edge_i_has_pin) {
                                                        let edge_move = ((p_i - x_old_i) + (p_ui - v_ui.position)) * 0.5;
                                                        let col_disp = -n_ee * (pen_ee * (1.0 - t)) + edge_move * (1.0 - t) * 0.5;
                                                        add_disp_atomic(j, col_disp);
                                                        if (v_vj.inv_mass > 0.0) {
                                                            let col_disp_vj = -n_ee * (pen_ee * t) + edge_move * t * 0.5;
                                                            add_disp_atomic(vj, col_disp_vj);
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
                                        }
                                    }
                                }
                        }
                    }
                    other_idx = vert_next[j];
                    iter_count = iter_count + 1u;
                }
            }
}
    }
}
