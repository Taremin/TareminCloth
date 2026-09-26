// 接触候補ペア収集シェーダー (Broadphase Pair Collection)
// 空間グリッドから接触可能圏内にある V-T (頂点対面) および E-E (辺対辺) ペアを検出し、
// 頂点スレッドごとに距離上位 quota 件を固定スロット (頂点メジャー配置) に保持する。
// グローバルアトミックは統計用のみ (格納は競合なし・決定的)。

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

struct GpuVtPair {
    vert_i: u32,
    vert_j: u32,
    v0: u32,
    v1: u32,
};

struct GpuEePair {
    v_i: u32,
    v_ui: u32,
    v_j: u32,
    v_vj: u32,
};

struct PairCollectParams {
    cell_size: f32,
    table_size: u32,
    num_vertices: u32,
    max_vt_pairs: u32,
    max_ee_pairs: u32,
    safety_margin: f32,
    exclude_neighbors: u32,
    margin_mode: u32,
    dt_frame: f32,
    velocity_horizon_scale: f32,
    max_horizon: f32,
    quota_vt: u32,
    quota_ee: u32,
};

struct PairCounters {
    vt_count: atomic<u32>,
    ee_count: atomic<u32>,
    vt_dropped: atomic<u32>,
    ee_dropped: atomic<u32>,
};

struct ClosestBaryResult {
    pt: vec3<f32>,
    bary: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(2) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(3) var<uniform> params: PairCollectParams;
@group(0) @binding(4) var<storage, read> local_edge_lengths: array<f32>;
@group(0) @binding(5) var<storage, read> adj_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> adj_indices: array<u32>;
@group(0) @binding(7) var<storage, read> island_ids: array<u32>;
@group(0) @binding(8) var<storage, read> star_offsets: array<u32>;
@group(0) @binding(9) var<storage, read> star_indices: array<GpuStarPair>;
@group(0) @binding(10) var<storage, read> two_hop_offsets: array<u32>;
@group(0) @binding(11) var<storage, read> two_hop_indices: array<u32>;
@group(0) @binding(12) var<storage, read_write> counters: PairCounters;
@group(0) @binding(13) var<storage, read_write> active_vt_pairs: array<GpuVtPair>;
@group(0) @binding(14) var<storage, read_write> active_ee_pairs: array<GpuEePair>;
@group(0) @binding(15) var<storage, read> two_hop_rest_lengths: array<f32>;

const EPSILON: f32 = 1e-7;

fn hash_coords(coord: vec3<i32>, table_size: u32) -> u32 {
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    let n = (u32(coord.x) * p1) ^ (u32(coord.y) * p2) ^ (u32(coord.z) * p3);
    return n % table_size;
}

// 2ホップ近傍CSR配列に対する二分探索 (Bridson 2002: レスト長比率による動的折り畳み検出)
// 初期距離 L0 の 50% 未満に圧縮された場合、鋭角なシワ・折り畳みと判定してペア収集を許可（除外解除）
fn is_topologically_near_folded(vert_a: u32, vert_b: u32, current_dist: f32) -> bool {
    let start = two_hop_offsets[vert_a];
    let end = two_hop_offsets[vert_a + 1u];
    if (start >= end) {
        return false;
    }

    var low = start;
    var high = end - 1u;

    for (var step = 0u; step < 7u; step = step + 1u) {
        if (low > high) {
            break;
        }
        let mid = low + ((high - low) >> 1u);
        let val = two_hop_indices[mid];
        if (val == vert_b) {
            let l0 = two_hop_rest_lengths[mid];
            // 距離が初期レスト長の 50% 未満に潰れた場合は折り畳みとみなし、ペア収集を有効化 (false を返す)
            if (current_dist < l0 * 0.5) {
                return false;
            }
            return true;
        } else if (val < vert_b) {
            low = mid + 1u;
        } else {
            if (mid == 0u) {
                break;
            }
            high = mid - 1u;
        }
    }
    return false;
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
    let index = global_id.x;
    if (index >= params.num_vertices) {
        return;
    }

    let v_i = vertices[index];
    let is_pinned_i = (v_i.inv_mass <= 0.0);
    let p_i = v_i.prev_pos;
    let local_len_i = local_edge_lengths[index];

    let cell_size = params.cell_size;
    let table_size = params.table_size;
    let base_cell = vec3<i32>(floor(p_i / cell_size));

    // 頂点ローカル Top-K 保持域 (距離昇順)。重複排除キー付き。
    var vt_n = 0u;
    var vt_dists: array<f32, 8>;
    var vt_data: array<GpuVtPair, 8>;
    var vt_eval = 0u;
    var vt_drop = 0u;
    var ee_n = 0u;
    var ee_dists: array<f32, 8>;
    var ee_data: array<GpuEePair, 8>;
    var ee_eval = 0u;
    var ee_drop = 0u;
    let quota_vt = clamp(params.quota_vt, 1u, 8u);
    let quota_ee = clamp(params.quota_ee, 1u, 8u);

    let margin = params.safety_margin;
    // 速度スイープホライゾン (AutoVelocityモード時のみ有効、Fixed時は0)
    // フレーム内に接近するペアを取りこぼさないための予測拡張。頂点ローカルで完結しCPU readback不要。
    let horizon_i_raw = length(v_i.velocity) * params.dt_frame * params.velocity_horizon_scale;
    let horizon_i = min(horizon_i_raw, params.max_horizon);

    // 27近傍セル走査
    for (var gx = -1; gx <= 1; gx = gx + 1) {
        for (var gy = -1; gy <= 1; gy = gy + 1) {
            for (var gz = -1; gz <= 1; gz = gz + 1) {
                let cell_coord = base_cell + vec3<i32>(gx, gy, gz);
                let h = hash_coords(cell_coord, table_size);

                let start = cell_starts[h];
                let end = cell_starts[h + 1u];
                if (start >= end) {
                    continue;
                }

                for (var k = start; k < end; k = k + 1u) {
                    let j = sorted_indices[k];
                    if (j >= params.num_vertices || j == index) {
                        continue;
                    }

                    let v_j = vertices[j];
                    let p_j = v_j.prev_pos;

                    // 枝切り: ハッシュ値が同じでも実際の空間セルが異なる場合はスキップ（ハッシュ衝突排除）
                    let j_cell = vec3<i32>(floor(p_j / cell_size));
                    if (!all(j_cell == cell_coord)) {
                        continue;
                    }

                    let is_pinned_j = (v_j.inv_mass <= 0.0);
                    if (!(is_pinned_i && is_pinned_j)) {
                        // 相手頂点の速度ホライゾン (Fixedモードでは0に潰す)
                        let horizon_j_raw = length(v_j.velocity) * params.dt_frame * params.velocity_horizon_scale;
                        let horizon_j_full = min(horizon_j_raw, params.max_horizon);
                        let mode_f = f32(params.margin_mode);
                        let horizon_j = horizon_j_full * mode_f;
                        let horizon_i_eff = horizon_i * mode_f;

                        let diff_vv = p_i - p_j;
                        let dist_sq_vv = dot(diff_vv, diff_vv);

                        let effective_thick = v_i.thickness + v_j.thickness;
                        let local_len_j = local_edge_lengths[j];

                        // 広域外接球による早期枝切り (マージン+速度ホライゾン付き)
                        let broad_bound = effective_thick + (local_len_i + local_len_j) * 1.3 + margin + horizon_i_eff + horizon_j;
                        if (dist_sq_vv <= broad_bound * broad_bound) {
                            let same_island = (island_ids[index] == island_ids[j]);
                            let dist_vv = sqrt(max(dist_sq_vv, 0.0));
                            let is_near_topology = same_island && (params.exclude_neighbors != 0u) && is_topologically_near_folded(index, j, dist_vv);

                            if (!is_near_topology) {
                                // ==============================================
                                // 1. V-T (頂点 対 面) ペア収集
                                // ==============================================
                                let vt_bound = effective_thick + local_len_j * 1.3 + margin + horizon_i_eff + horizon_j;
                                if (dist_sq_vv <= vt_bound * vt_bound) {
                                    let star_start = star_offsets[j];
                                    let star_end = star_offsets[j + 1u];
                                    if (star_start < star_end) {
                                        let num_stars = min(star_end - star_start, 8u);

                                        for (var s = 0u; s < num_stars; s = s + 1u) {
                                            let pair = star_indices[star_start + s];
                                            let v0 = pair.v0;
                                            let v1 = pair.v1;

                                            // 自頂点が三角形を構成する頂点でなければ
                                            if (index != v0 && index != v1 && v0 < params.num_vertices && v1 < params.num_vertices) {
                                                let p_v0 = vertices[v0].prev_pos;
                                                let p_v1 = vertices[v1].prev_pos;

                                                let res = closest_point_on_triangle_bary(p_i, p_j, p_v0, p_v1);
                                                let diff_vt = p_i - res.pt;
                                                let dist_sq_vt = dot(diff_vt, diff_vt);

                                                let check_thick = effective_thick + margin + horizon_i_eff + horizon_j;
                                                if (dist_sq_vt < check_thick * check_thick) {
                                                    // V-T候補を受理: 面トリプルで重複排除し距離上位quota件を保持
                                                    // (同一面は中心違いで最大3回評価されるため)
                                                    let t0 = min(min(j, v0), v1);
                                                    let t2 = max(max(j, v0), v1);
                                                    let t1 = j + v0 + v1 - t0 - t2;
                                                    var dup = false;
                                                    for (var qk = 0u; qk < 8u; qk = qk + 1u) {
                                                        if (qk >= vt_n) {
                                                            break;
                                                        }
                                                        let e = vt_data[qk];
                                                        let s0 = min(min(e.vert_j, e.v0), e.v1);
                                                        let s2 = max(max(e.vert_j, e.v0), e.v1);
                                                        let s1 = e.vert_j + e.v0 + e.v1 - s0 - s2;
                                                        if (s0 == t0 && s1 == t1 && s2 == t2) {
                                                            dup = true;
                                                            break;
                                                        }
                                                    }
                                                    if (!dup) {
                                                        vt_eval = vt_eval + 1u;
                                                        var pos = vt_n;
                                                        for (var qk = 0u; qk < 8u; qk = qk + 1u) {
                                                            if (qk < vt_n && dist_sq_vt < vt_dists[qk]) {
                                                                pos = qk;
                                                                break;
                                                            }
                                                        }
                                                        if (pos < quota_vt) {
                                                            var up = min(vt_n, quota_vt - 1u);
                                                            var qk = up;
                                                            loop {
                                                                if (qk <= pos) {
                                                                    break;
                                                                }
                                                                vt_dists[qk] = vt_dists[qk - 1u];
                                                                vt_data[qk] = vt_data[qk - 1u];
                                                                qk = qk - 1u;
                                                            }
                                                            vt_dists[pos] = dist_sq_vt;
                                                            vt_data[pos] = GpuVtPair(index, j, v0, v1);
                                                            if (vt_n < quota_vt) {
                                                                vt_n = vt_n + 1u;
                                                            }
                                                        } else {
                                                            vt_drop = vt_drop + 1u;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                // ==============================================
                                // 2. E-E (辺 対 辺) ペア収集
                                // ==============================================
                                let ee_bound = effective_thick + (local_len_i + local_len_j) * 1.3 + margin + horizon_i_eff + horizon_j;
                                if (dist_sq_vv <= ee_bound * ee_bound) {
                                    let adj_i_start = adj_offsets[index];
                                    let adj_i_end = adj_offsets[index + 1u];
                                    let adj_j_start = adj_offsets[j];
                                    let adj_j_end = adj_offsets[j + 1u];

                                    if (adj_i_start < adj_i_end && adj_j_start < adj_j_end) {
                                        let num_edges_i = min(adj_i_end - adj_i_start, 8u);
                                        let num_edges_j = min(adj_j_end - adj_j_start, 8u);

                                        for (var ei = 0u; ei < num_edges_i; ei = ei + 1u) {
                                            let ui = adj_indices[adj_i_start + ei];
                                            if (ui >= params.num_vertices) {
                                                continue;
                                            }
                                            let v_ui = vertices[ui];
                                            let p_ui = v_ui.prev_pos;

                                            for (var ej = 0u; ej < num_edges_j; ej = ej + 1u) {
                                                let vj = adj_indices[adj_j_start + ej];
                                                if (vj >= params.num_vertices) {
                                                    continue;
                                                }
                                                if (ui != j && ui != vj && index != vj) {
                                                    let v_vj = vertices[vj];
                                                    let edge_j_has_pin = (v_j.inv_mass <= 0.0) || (v_vj.inv_mass <= 0.0);

                                                    // 重複排除: 相手がピンでない動的エッジ同士なら index < j
                                                    if (!is_pinned_i && (edge_j_has_pin || index < j)) {
                                                        let p_vj = v_vj.prev_pos;
                                                        let st = closest_points_segments(p_i, p_ui, p_j, p_vj);
                                                        let pt1 = p_i + st.x * (p_ui - p_i);
                                                        let pt2 = p_j + st.y * (p_vj - p_j);

                                                        let diff_ee = pt1 - pt2;
                                                        let dist_sq_ee = dot(diff_ee, diff_ee);
                                                        let check_thick = effective_thick + margin + horizon_i_eff + horizon_j;

                                                        if (dist_sq_ee < check_thick * check_thick && dist_sq_ee > EPSILON) {
                                                            // E-E候補を受理: 距離上位quota件を保持
                                                            // (index<j重複排除済みのためスレッド内重複なし)
                                                            ee_eval = ee_eval + 1u;
                                                            var pos = ee_n;
                                                            for (var qk = 0u; qk < 8u; qk = qk + 1u) {
                                                                if (qk < ee_n && dist_sq_ee < ee_dists[qk]) {
                                                                    pos = qk;
                                                                    break;
                                                                }
                                                            }
                                                            if (pos < quota_ee) {
                                                                var up = min(ee_n, quota_ee - 1u);
                                                                var qk = up;
                                                                loop {
                                                                    if (qk <= pos) {
                                                                        break;
                                                                    }
                                                                    ee_dists[qk] = ee_dists[qk - 1u];
                                                                    ee_data[qk] = ee_data[qk - 1u];
                                                                    qk = qk - 1u;
                                                                }
                                                                ee_dists[pos] = dist_sq_ee;
                                                                ee_data[pos] = GpuEePair(index, ui, j, vj);
                                                                if (ee_n < quota_ee) {
                                                                    ee_n = ee_n + 1u;
                                                                }
                                                            } else {
                                                                ee_drop = ee_drop + 1u;
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
                    }
                }
            }
        }
    }

    // 頂点メジャー固定スロットへ書き戻し (未使用枠はセンチネル)。
    // 物理配置は index*quota+k のコンパクト配置 (quotaは論理枠数)。
    // 収集は毎フレーム全論理範囲を書き直すため quota 変更時も整合する。
    let base_vt = index * quota_vt;
    for (var k = 0u; k < 8u; k = k + 1u) {
        if (k >= quota_vt) {
            break;
        }
        if (k < vt_n) {
            active_vt_pairs[base_vt + k] = vt_data[k];
        } else {
            active_vt_pairs[base_vt + k] = GpuVtPair(0xFFFFFFFFu, 0u, 0u, 0u);
        }
    }

    let base_ee = index * quota_ee;
    for (var k = 0u; k < 8u; k = k + 1u) {
        if (k >= quota_ee) {
            break;
        }
        if (k < ee_n) {
            active_ee_pairs[base_ee + k] = ee_data[k];
        } else {
            active_ee_pairs[base_ee + k] = GpuEePair(0xFFFFFFFFu, 0u, 0u, 0u);
        }
    }

    // 統計のみアトミック加算 (格納自体は競合なし)
    if (vt_eval > 0u) {
        atomicAdd(&counters.vt_count, vt_eval);
    }
    if (vt_drop > 0u) {
        atomicAdd(&counters.vt_dropped, vt_drop);
    }
    if (ee_eval > 0u) {
        atomicAdd(&counters.ee_count, ee_eval);
    }
    if (ee_drop > 0u) {
        atomicAdd(&counters.ee_dropped, ee_drop);
    }
}
