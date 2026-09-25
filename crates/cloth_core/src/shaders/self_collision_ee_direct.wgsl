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
    ccd_dx: atomic<i32>,
    ccd_dy: atomic<i32>,
    ccd_dz: atomic<i32>,
    ccd_count: atomic<u32>,
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

const EPSILON: f32 = 1e-7;
const FIXED_SCALE: f32 = 1000000.0;

fn add_disp_atomic(v_idx: u32, disp: vec3<f32>) {
    let ix = i32(clamp(disp.x * FIXED_SCALE, -2e9, 2e9));
    let iy = i32(clamp(disp.y * FIXED_SCALE, -2e9, 2e9));
    let iz = i32(clamp(disp.z * FIXED_SCALE, -2e9, 2e9));
    atomicAdd(&accum[v_idx].dx, ix);
    atomicAdd(&accum[v_idx].dy, iy);
    atomicAdd(&accum[v_idx].dz, iz);
    atomicAdd(&accum[v_idx].count, 1u);
}

fn add_ccd_atomic(v_idx: u32, disp: vec3<f32>) {
    let ix = i32(clamp(disp.x * FIXED_SCALE, -2e9, 2e9));
    let iy = i32(clamp(disp.y * FIXED_SCALE, -2e9, 2e9));
    let iz = i32(clamp(disp.z * FIXED_SCALE, -2e9, 2e9));
    atomicAdd(&accum[v_idx].ccd_dx, ix);
    atomicAdd(&accum[v_idx].ccd_dy, iy);
    atomicAdd(&accum[v_idx].ccd_dz, iz);
    atomicAdd(&accum[v_idx].ccd_count, 1u);
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

// トポロジー2ホップ近傍判定（CSRテーブル二分探索）
fn is_topologically_near(vert_a: u32, vert_b: u32) -> bool {
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

    // 自エッジが両端点ピン留めで静止している場合は移動がないため早期リターン可能
    if (is_pinned_u0 && is_pinned_u1 && length(p_u0 - p_u0_old) < EPSILON && length(p_u1 - p_u1_old) < EPSILON) {
        return;
    }

    let thick_u0 = v_u0.thickness;
    let thick_u1 = v_u1.thickness;
    let max_thick_a = max(thick_u0, thick_u1);

    // エッジ A の AABB（マージン付き）
    let aabb_a_min = min(p_u0, p_u1) - vec3<f32>(max_thick_a * 2.0);
    let aabb_a_max = max(p_u0, p_u1) + vec3<f32>(max_thick_a * 2.0);

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
                    search_count = search_count + 1u;
                    if (search_count > params.max_search_iterations) {
                        return;
                    }

                    let edge_b_idx = edge_sorted_indices[start + idx];

                    // 【辞書順序制約】全世界で edge_idx < edge_b_idx のペアのみを厳密に 1 回処理
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

                    // 【早期枝切り】エッジ AABB 重なり判定 (Edge-Edge AABB Overlap)
                    let aabb_b_min = min(p_j, p_vj);
                    let aabb_b_max = max(p_j, p_vj);
                    if (any(aabb_a_min > aabb_b_max) || any(aabb_a_max < aabb_b_min)) {
                        continue;
                    }

                    // 2ホップ近傍除外（AABB重なりを通過した候補ペアのみ二分探索を実行）
                    if (params.exclude_neighbors != 0u) {
                        if (is_topologically_near(u0, j) || is_topologically_near(u0, vj) ||
                            is_topologically_near(u1, j) || is_topologically_near(u1, vj)) {
                            continue;
                        }
                    }

                    let st = closest_points_segments(p_u0, p_u1, p_j, p_vj);
                    let pt1 = p_u0 + st.x * (p_u1 - p_u0);
                    let pt2 = p_j + st.y * (p_vj - p_j);

                    let delta_ee = pt1 - pt2;
                    let dist_ee = length(delta_ee);

                    let thick_j = v_j.thickness;
                    let thick_vj = v_vj.thickness;
                    let thick_a = mix(thick_u0, thick_u1, st.x);
                    let thick_b = mix(thick_j, thick_vj, st.y);
                    let effective_thick = thick_a + thick_b;

                    let is_untangling_ee = (params.enable_normal_untangling != 0u)
                        && (v_u0.layer_id > v_j.layer_id)
                        && (dot(delta_ee, normals[j].xyz) < 0.0);

                    if (!is_untangling_ee && dist_ee < effective_thick && dist_ee > EPSILON) {
                        let n_ee = delta_ee / dist_ee;
                        let pen_ee = effective_thick - dist_ee;
                        let s = st.x;
                        let t = st.y;

                        let edge_b_has_pin = (v_j.inv_mass <= 0.0) || (v_vj.inv_mass <= 0.0);
                        if (edge_a_has_pin && edge_b_has_pin) {
                            continue;
                        }

                        let p_j_old = v_j.position;
                        let p_vj_old = v_vj.position;

                        if (edge_b_has_pin) {
                            // 相手エッジがピン -> 自エッジ側を変位
                            let edge_move = ((p_j - p_j_old) + (p_vj - p_vj_old)) * 0.5;
                            if (v_u0.inv_mass > 0.0) {
                                let col_disp_0 = n_ee * (pen_ee * (1.0 - s)) + edge_move * (1.0 - s) * 0.5;
                                add_ccd_atomic(u0, col_disp_0);
                            }
                            if (v_u1.inv_mass > 0.0) {
                                let col_disp_1 = n_ee * (pen_ee * s) + edge_move * s * 0.5;
                                add_ccd_atomic(u1, col_disp_1);
                            }
                        } else if (edge_a_has_pin) {
                            // 自エッジがピン -> 相手エッジ側を変位
                            let edge_move = ((p_u0 - p_u0_old) + (p_u1 - p_u1_old)) * 0.5;
                            if (v_j.inv_mass > 0.0) {
                                let col_disp_j = -n_ee * (pen_ee * (1.0 - t)) + edge_move * (1.0 - t) * 0.5;
                                add_ccd_atomic(j, col_disp_j);
                            }
                            if (v_vj.inv_mass > 0.0) {
                                let col_disp_vj = -n_ee * (pen_ee * t) + edge_move * t * 0.5;
                                add_ccd_atomic(vj, col_disp_vj);
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
