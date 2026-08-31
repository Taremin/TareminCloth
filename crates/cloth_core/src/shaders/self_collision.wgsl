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

const EPSILON: f32 = 1e-7;

fn hash_coords(coord: vec3<i32>, table_size: u32) -> u32 {
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    let n = (u32(coord.x) * p1) ^ (u32(coord.y) * p2) ^ (u32(coord.z) * p3);
    return n % table_size;
}

// 空間上の点 p から三角形 (a, b, c) への最近傍点
fn closest_point_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
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

// 2本の線分 (p0-p1) と (q0-q1) の最短距離パラメータ (s, t) in [0, 1]
fn closest_points_segments(
    p0: vec3<f32>, p1: vec3<f32>,
    q0: vec3<f32>, q1: vec3<f32>
) -> vec2<f32> {
    let d1 = p1 - p0;
    let d2 = q1 - q0;
    let r = p0 - q0;
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
    var p_i = v_i.prev_pos;     // 現在の予測位置
    let thick_i = v_i.thickness;

    let cell = vec3<i32>(floor(p_i / params.cell_size));

    var total_penalty_disp = vec3<f32>(0.0);
    var penalty_count = 0.0;
    var total_ccd_disp = vec3<f32>(0.0);
    var ccd_count = 0.0;

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
                        let is_direct_edge = is_adjacent_vert(index, j);

                        if (!is_direct_edge) {
                            let p_j_old = v_j.position;
                            let thick_j = v_j.thickness;

                            let local_len_i = local_edge_lengths[index];
                            let local_len_j = local_edge_lengths[j];
                            let min_dist = thick_i + thick_j;
                            let edge_margin = min(local_len_i, local_len_j) * 0.06;
                            let effective_thick = max(min_dist, edge_margin);

                            // ==========================================
                            // 1. 球対球 (Vertex-Vertex) 幾何反発
                            // ==========================================
                            let delta_vv = p_i - p_j;
                            let dist_vv = length(delta_vv);

                            if (dist_vv < effective_thick && dist_vv > EPSILON) {
                                let n = delta_vv / dist_vv;
                                let pen = effective_thick - dist_vv;
                                let w_sum = v_i.inv_mass + v_j.inv_mass;
                                let ratio = select(0.5, v_i.inv_mass / w_sum, w_sum > EPSILON);

                                if (v_j.inv_mass <= 0.0) {
                                    let v_j_move = p_j - p_j_old;
                                    let col_disp = n * pen + v_j_move * 0.5;
                                    total_ccd_disp = total_ccd_disp + col_disp;
                                    ccd_count = ccd_count + 1.0;
                                } else {
                                    total_penalty_disp = total_penalty_disp + n * (pen * ratio * 0.5);
                                    penalty_count = penalty_count + 1.0;
                                }
                            }

                            // ==========================================
                            // 2. Vertex-Triangle (頂点 対 面) 相対CCD & 幾何反発
                            // ==========================================
                            let is_near_2hop = is_topologically_near(index, j);
                            if (!is_near_2hop) {
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
                                            total_ccd_disp = total_ccd_disp + ccd_disp;
                                            ccd_count = ccd_count + 1.0;
                                        } else {
                                            // B) 最近傍点幾何反発
                                            let q = closest_point_on_triangle(p_i, p_j, p_v0, p_v1);
                                            let delta_vt = p_i - q;
                                            let dist_vt = length(delta_vt);

                                            if (dist_vt < effective_thick && dist_vt > EPSILON) {
                                                let n_vt = delta_vt / dist_vt;
                                                let pen_vt = effective_thick - dist_vt;
                                                if (tri_has_pin) {
                                                    // 相手三角形がピン留め頂点を含む場合はコライダー押し出し（100%退避）
                                                    let col_disp = n_vt * pen_vt + tri_move * 0.5;
                                                    total_ccd_disp = total_ccd_disp + col_disp;
                                                    ccd_count = ccd_count + 1.0;
                                                } else {
                                                    total_penalty_disp = total_penalty_disp + n_vt * (pen_vt * 0.6);
                                                    penalty_count = penalty_count + 1.0;
                                                }
                                            }
                                        }
                                    }
                                }
                            }



                            // ==========================================
                            // 3. Edge-Edge (辺 対 辺) 幾何反発 + コライダー押し出し
                            // ==========================================
                            if (!is_near_2hop) {
                                let adj_i_start = adj_offsets[index];
                                let adj_i_end = adj_offsets[index + 1u];
                                let adj_j_start = adj_offsets[j];
                                let adj_j_end = adj_offsets[j + 1u];

                                let num_edges_i = min(adj_i_end - adj_i_start, 8u);
                                let num_edges_j = min(adj_j_end - adj_j_start, 8u);

                                for (var ei = 0u; ei < num_edges_i; ei = ei + 1u) {
                                    let ui = adj_indices[adj_i_start + ei];
                                    let p_ui = vertices[ui].prev_pos;

                                    for (var ej = 0u; ej < num_edges_j; ej = ej + 1u) {
                                        let vj = adj_indices[adj_j_start + ej];
                                        if (ui != j && ui != vj && index != vj) {
                                            let v_vj = vertices[vj];
                                            let p_vj = v_vj.prev_pos;
                                            let p_vj_old = v_vj.position;

                                            let st = closest_points_segments(p_i, p_ui, p_j, p_vj);
                                            let pt1 = p_i + st.x * (p_ui - p_i);
                                            let pt2 = p_j + st.y * (p_vj - p_j);

                                            let delta_ee = pt1 - pt2;
                                            let dist_ee = length(delta_ee);

                                            if (dist_ee < effective_thick && dist_ee > EPSILON) {
                                                let n_ee = delta_ee / dist_ee;
                                                let pen_ee = effective_thick - dist_ee;
                                                let weight_i = 1.0 - st.x;

                                                let edge_has_pin = (v_j.inv_mass <= 0.0) || (v_vj.inv_mass <= 0.0);
                                                if (edge_has_pin) {
                                                    let edge_move = ((p_j - p_j_old) + (p_vj - p_vj_old)) * 0.5;
                                                    let col_disp = n_ee * (pen_ee * weight_i) + edge_move * weight_i * 0.5;
                                                    total_ccd_disp = total_ccd_disp + col_disp;
                                                    ccd_count = ccd_count + 1.0;
                                                } else {
                                                    total_penalty_disp = total_penalty_disp + n_ee * (pen_ee * weight_i * 0.6);
                                                    penalty_count = penalty_count + 1.0;
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

    // 変位の適用とクランプ
    var final_disp = vec3<f32>(0.0);

    // 1. ペナルティ反発 (近接反発): 多数の近接点がある場合に平均化で反発力が弱まる（希釈）のを防ぐ
    if (penalty_count > 0.0) {
        // 反発ベクトルの合成: 単純平均ではなく、接触密度に応じた適切な剛性を維持
        var avg_penalty = total_penalty_disp / max(1.0, sqrt(penalty_count));
        if (params.enable_relief != 0u) {
            avg_penalty = avg_penalty * params.relief_factor;
        }
        let p_len = length(avg_penalty);
        let ratio = select(0.50, params.max_displacement_ratio, params.max_displacement_ratio > 0.0);
        let max_penalty_step = min(local_edge_lengths[index] * ratio, 0.10);
        if (p_len > max_penalty_step && p_len > EPSILON) {
            avg_penalty = (avg_penalty / p_len) * max_penalty_step;
        }
        final_disp = final_disp + avg_penalty;
    }

    // 2. CCD (連続衝突判定): 貫通防止のハード制約なので、減衰・クランプをかけずに確実に面手前に留める
    if (ccd_count > 0.0) {
        let avg_ccd = total_ccd_disp / ccd_count;
        final_disp = final_disp + avg_ccd;
    }

    p_i = p_i + final_disp;
    v_i.prev_pos = p_i;
    vertices[index] = v_i;
}
