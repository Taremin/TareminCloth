// 純粋球対球（Vertex-Vertex）自己衝突解決シェーダー
// 仮想コライダー粒子（スレーブ粒子）を含む全粒子に対してV-V判定を行い、
// 衝突反発変位を重心座標重みで親頂点へ固定小数点アトミック加算する。
//
// 1. 同一面（同一三角形）内の粒子ペアは自爆防止のため無条件除外。
// 2. 親頂点を共有するトポロジー近傍要素（エッジ共有・頂点共有・同一エッジ）に対しては、
//    レスト距離スケーリング（Rest-Distance Folded Thresholding / Bridson 2002）を適用。
//    平坦時の自爆・座屈ノイズを100%防止しつつ、シワ・折り畳み接触（初期距離の50%未満に圧縮された急峻な折り目）を確実に検知・遮断する。
// 3. 親頂点を共有しない独立した面同士は、常に通常の厚み min_dist で反発する。

struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct GpuVirtualVertexDef {
    parent_indices: vec4<u32>, // x, y, z: 親三角形の3頂点, w: 親face_id
    bary_weights: vec4<f32>,   // x, y, z: 重心座標 (u, v, w), w: thickness
};

struct SelfCollisionVvParams {
    cell_size: f32,
    table_size: u32,
    num_real_vertices: u32,
    num_total_particles: u32,
    relief_factor: f32,
    max_displacement_ratio: f32,
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

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(2) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(3) var<uniform> params: SelfCollisionVvParams;
@group(0) @binding(4) var<storage, read_write> accum: array<SelfCollisionAccum>;
@group(0) @binding(5) var<storage, read> virt_defs: array<GpuVirtualVertexDef>;
@group(0) @binding(6) var<storage, read> rest_positions: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> two_hop_offsets: array<u32>;
@group(0) @binding(8) var<storage, read> two_hop_indices: array<u32>;
@group(0) @binding(9) var<storage, read> two_hop_rest_lengths: array<f32>;

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

fn hash_coords(coord: vec3<i32>, table_size: u32) -> u32 {
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    let n = (u32(coord.x) * p1) ^ (u32(coord.y) * p2) ^ (u32(coord.z) * p3);
    return n % table_size;
}

// 粒子idxから親頂点インデックス3つ、重心座標、親face_id、初期レスト座標を取得する
struct ParticleInfo {
    parents: vec3<u32>,
    bary: vec3<f32>,
    parent_face_id: u32,
    rest_pos: vec3<f32>,
    is_virtual: bool,
};

fn get_particle_info(idx: u32) -> ParticleInfo {
    if (idx < params.num_real_vertices) {
        let r_pos = rest_positions[idx].xyz;
        return ParticleInfo(
            vec3<u32>(idx, idx, idx),
            vec3<f32>(1.0, 0.0, 0.0),
            0xFFFFFFFFu,
            r_pos,
            false
        );
    } else {
        let virt_idx = idx - params.num_real_vertices;
        let def = virt_defs[virt_idx];
        let p0 = rest_positions[def.parent_indices.x].xyz;
        let p1 = rest_positions[def.parent_indices.y].xyz;
        let p2 = rest_positions[def.parent_indices.z].xyz;
        let r_pos = p0 * def.bary_weights.x + p1 * def.bary_weights.y + p2 * def.bary_weights.z;
        return ParticleInfo(
            def.parent_indices.xyz,
            def.bary_weights.xyz,
            def.parent_indices.w,
            r_pos,
            true
        );
    }
}

// 2つの粒子が親頂点を1つでも共有しているかを判定（トポロジー隣接判定）
fn share_any_parent(info_a: ParticleInfo, info_b: ParticleInfo) -> bool {
    let a0 = info_a.parents.x;
    let a1 = info_a.parents.y;
    let a2 = info_a.parents.z;

    let b0 = info_b.parents.x;
    let b1 = info_b.parents.y;
    let b2 = info_b.parents.z;

    if (!info_a.is_virtual && !info_b.is_virtual) {
        return a0 == b0;
    }

    if (!info_a.is_virtual) {
        return (a0 == b0) || (a0 == b1) || (a0 == b2);
    }

    if (!info_b.is_virtual) {
        return (b0 == a0) || (b0 == a1) || (b0 == a2);
    }

    return (a0 == b0) || (a0 == b1) || (a0 == b2)
        || (a1 == b0) || (a1 == b1) || (a1 == b2)
        || (a2 == b0) || (a2 == b1) || (a2 == b2);
}

// トポロジー2ホップ近傍判定および初期レスト長取得（Bridson 2002: 動的折り畳み検出用）
struct TwoHopResult {
    is_two_hop: bool,
    rest_length: f32,
};

fn query_two_hop(vert_a: u32, vert_b: u32) -> TwoHopResult {
    if (vert_a == vert_b) {
        return TwoHopResult(true, 0.0);
    }
    let start = two_hop_offsets[vert_a];
    let end = two_hop_offsets[vert_a + 1u];

    var low = start;
    var high = end;
    while (low < high) {
        let mid = low + (high - low) / 2u;
        let val = two_hop_indices[mid];
        if (val == vert_b) {
            return TwoHopResult(true, two_hop_rest_lengths[mid]);
        } else if (val < vert_b) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }
    return TwoHopResult(false, 0.0);
}

// 変位を親頂点へ分配する
fn distribute_disp(info: ParticleInfo, disp: vec3<f32>) {
    if (!info.is_virtual) {
        add_disp_atomic(info.parents.x, disp);
    } else {
        if (info.bary.x > 0.001) {
            add_disp_atomic(info.parents.x, disp * info.bary.x);
        }
        if (info.bary.y > 0.001) {
            add_disp_atomic(info.parents.y, disp * info.bary.y);
        }
        if (info.bary.z > 0.001) {
            add_disp_atomic(info.parents.z, disp * info.bary.z);
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_total_particles) {
        return;
    }

    let v_i = vertices[index];
    let p_i = v_i.prev_pos;
    let thick_i = v_i.thickness;

    let info_i = get_particle_info(index);

    let cell = vec3<i32>(floor(p_i / params.cell_size));
    let min_cell = cell - vec3<i32>(1);
    let max_cell = cell + vec3<i32>(1);

    // 27近傍セルを走査
    for (var cx = min_cell.x; cx <= max_cell.x; cx = cx + 1) {
        for (var cy = min_cell.y; cy <= max_cell.y; cy = cy + 1) {
            for (var cz = min_cell.z; cz <= max_cell.z; cz = cz + 1) {
                let neighbor_cell = vec3<i32>(cx, cy, cz);
                let h = hash_coords(neighbor_cell, params.table_size);

                let cell_start = cell_starts[h];
                let cell_end = cell_starts[h + 1u];

                for (var k = cell_start; k < cell_end; k = k + 1u) {
                    let j = sorted_indices[k];
                    // 対称性: index < j の側でのみ1回だけ評価
                    if (index >= j) {
                        continue;
                    }

                    let v_j = vertices[j];
                    let p_j = v_j.prev_pos;

                    // ハッシュ衝突の枝切り
                    let j_cell = vec3<i32>(floor(p_j / params.cell_size));
                    if (!all(j_cell == neighbor_cell)) {
                        continue;
                    }

                    let info_j = get_particle_info(j);

                    let min_dist = thick_i + v_j.thickness;

                    // 幾何学的に min_dist 未満に近づいていないペアは即座に早期枝切り
                    let delta = p_i - p_j;
                    let dist_sq = dot(delta, delta);
                    if (dist_sq >= min_dist * min_dist || dist_sq <= EPSILON) {
                        continue;
                    }
                    let dist_val = sqrt(dist_sq);

                    var is_topological_neighbor = false;

                    if (!info_i.is_virtual && !info_j.is_virtual) {
                        // 【ケース1】元頂点同士 (Real - Real)
                        if (index == j) {
                            continue;
                        }
                        let q = query_two_hop(index, j);
                        is_topological_neighbor = q.is_two_hop;
                    } else if (!info_i.is_virtual && info_j.is_virtual) {
                        // 【ケース2】元頂点と仮想頂点 (Real - Virt)
                        let v_idx = info_i.parents.x;
                        let p0 = info_j.parents.x;
                        let p1 = info_j.parents.y;
                        let p2 = info_j.parents.z;
                        if (v_idx == p0 || v_idx == p1 || v_idx == p2) {
                            continue;
                        }
                        let q0 = query_two_hop(v_idx, p0);
                        let q1 = query_two_hop(v_idx, p1);
                        let q2 = query_two_hop(v_idx, p2);
                        is_topological_neighbor = q0.is_two_hop || q1.is_two_hop || q2.is_two_hop;
                    } else if (info_i.is_virtual && !info_j.is_virtual) {
                        // 【ケース3】仮想頂点と元頂点 (Virt - Real)
                        let v_idx = info_j.parents.x;
                        let p0 = info_i.parents.x;
                        let p1 = info_i.parents.y;
                        let p2 = info_i.parents.z;
                        if (v_idx == p0 || v_idx == p1 || v_idx == p2) {
                            continue;
                        }
                        let q0 = query_two_hop(v_idx, p0);
                        let q1 = query_two_hop(v_idx, p1);
                        let q2 = query_two_hop(v_idx, p2);
                        is_topological_neighbor = q0.is_two_hop || q1.is_two_hop || q2.is_two_hop;
                    } else {
                        // 【ケース4】仮想頂点同士 (Virt - Virt)
                        if (info_i.parent_face_id == info_j.parent_face_id) {
                            continue;
                        }
                        if (share_any_parent(info_i, info_j)) {
                            is_topological_neighbor = true;
                        } else {
                            let q = query_two_hop(info_i.parents.x, info_j.parents.x);
                            is_topological_neighbor = q.is_two_hop;
                        }
                    }

                    // 2. トポロジー近傍要素に対するレスト距離動的折り畳み検出 (Bridson 2002)
                    var threshold = min_dist;
                    if (is_topological_neighbor) {
                        let delta_rest = info_i.rest_pos - info_j.rest_pos;
                        let l0 = length(delta_rest);
                        // 平坦時 (dist_val >= l0 * 0.5) は自爆防止のため完全除外
                        if (dist_val >= l0 * 0.5) {
                            continue;
                        }
                        // レスト長の 50% 未満に圧縮された場合のみシワ・折り畳みとして反発を許可
                        threshold = min(min_dist, l0 * 0.5);
                    }

                    // 3. 衝突反発変位の計算・分配（作用・反作用）
                    if (dist_val < threshold) {
                        let pen = threshold - dist_val;
                        let normal = delta / dist_val;
                        let disp = normal * (pen * 0.5);

                        distribute_disp(info_i, disp);
                        distribute_disp(info_j, -disp);
                    }
                }
            }
        }
    }
}
