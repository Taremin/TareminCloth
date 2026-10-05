// 純粋球対球（Vertex-Vertex）自己衝突解決シェーダー
// 仮想コライダー粒子（スレーブ粒子）を含む全粒子に対してV-V判定を行い、
// 衝突反発変位を重心座標重みで親頂点へ固定小数点アトミック加算する。

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

// 粒子idxから親頂点インデックス3つと重心座標を取得する
struct ParticleInfo {
    parents: vec3<u32>,
    bary: vec3<f32>,
    is_virtual: bool,
};

fn get_particle_info(idx: u32) -> ParticleInfo {
    if (idx < params.num_real_vertices) {
        return ParticleInfo(
            vec3<u32>(idx, idx, idx),
            vec3<f32>(1.0, 0.0, 0.0),
            false
        );
    } else {
        let virt_idx = idx - params.num_real_vertices;
        let def = virt_defs[virt_idx];
        return ParticleInfo(
            def.parent_indices.xyz,
            def.bary_weights.xyz,
            true
        );
    }
}

// 2つの粒子が親頂点を1つでも共有しているかを判定（同一面・エッジ隣接面・頂点共有面の自爆除外）
fn share_any_parent(info_a: ParticleInfo, info_b: ParticleInfo) -> bool {
    let a0 = info_a.parents.x;
    let a1 = info_a.parents.y;
    let a2 = info_a.parents.z;

    let b0 = info_b.parents.x;
    let b1 = info_b.parents.y;
    let b2 = info_b.parents.z;

    if (!info_a.is_virtual && !info_b.is_virtual) {
        // 両方が元頂点の場合、同一頂点のみ除外 (隣接エッジ判定は距離で自然解決)
        return a0 == b0;
    }

    if (!info_a.is_virtual) {
        // a が元頂点、b が仮想頂点の場合: b の親に a0 が含まれていれば除外
        return (a0 == b0) || (a0 == b1) || (a0 == b2);
    }

    if (!info_b.is_virtual) {
        // b が元頂点、a が仮想頂点の場合: a の親に b0 が含まれていれば除外
        return (b0 == a0) || (b0 == a1) || (b0 == a2);
    }

    // 両方が仮想頂点の場合: 3x3 比較で親頂点を1つでも共有していれば除外
    return (a0 == b0) || (a0 == b1) || (a0 == b2)
        || (a1 == b0) || (a1 == b1) || (a1 == b2)
        || (a2 == b0) || (a2 == b1) || (a2 == b2);
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

                    // トポロジー除外（親頂点共有判定）
                    if (share_any_parent(info_i, info_j)) {
                        continue;
                    }

                    let delta = p_i - p_j;
                    let dist_sq = dot(delta, delta);
                    let min_dist = thick_i + v_j.thickness;

                    if (dist_sq < min_dist * min_dist && dist_sq > EPSILON) {
                        let dist_val = sqrt(dist_sq);
                        let pen = min_dist - dist_val;
                        let normal = delta / dist_val;

                        // 衝突反発変位（作用・反作用）
                        let disp = normal * (pen * 0.5);

                        distribute_disp(info_i, disp);
                        distribute_disp(info_j, -disp);
                    }
                }
            }
        }
    }
}
