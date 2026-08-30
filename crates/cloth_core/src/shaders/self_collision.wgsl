struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
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
    _pad1: u32,
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

const EPSILON: f32 = 1e-7;

fn hash_coords(coord: vec3<i32>, table_size: u32) -> u32 {
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    let n = (u32(coord.x) * p1) ^ (u32(coord.y) * p2) ^ (u32(coord.z) * p3);
    return n % table_size;
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

    var p_i = v_i.prev_pos;
    let thick_i = v_i.thickness;
    let layer_i = v_i.layer_id;

    let cell = vec3<i32>(floor(p_i / params.cell_size));

    // 近傍 27 セルを走査
    var total_disp = vec3<f32>(0.0);
    var contact_count = 0.0;
    var has_untangle_contact = false;

    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dz = -1; dz <= 1; dz = dz + 1) {
                let neighbor_cell = cell + vec3<i32>(dx, dy, dz);
                let h = hash_coords(neighbor_cell, params.table_size);

                var other_idx = atomicLoad(&cell_heads[h]);
                var iter_count = 0u;

                while (other_idx >= 0 && iter_count < 128u) {
                    let j = u32(other_idx);
                    if (index != j) {
                        // 1. トポロジカル隣接頂点除外フィルタ
                        var skip = false;
                        if (params.exclude_neighbors != 0u) {
                            let adj_start = adj_offsets[index];
                            let adj_end = adj_offsets[index + 1u];
                            for (var k = adj_start; k < adj_end; k = k + 1u) {
                                if (adj_indices[k] == j) {
                                    skip = true;
                                    break;
                                }
                            }
                        }

                        if (!skip) {
                            let v_j = vertices[j];
                            let p_j = v_j.prev_pos;
                            let thick_j = v_j.thickness;
                            let layer_j = v_j.layer_id;

                            let island_i = island_ids[index];
                            let island_j = island_ids[j];
                            let is_different_island = (island_i != island_j);

                            let delta = p_i - p_j;
                            let dist = length(delta);
                            let min_dist = thick_i + thick_j;

                            let local_len_i = local_edge_lengths[index];
                            let local_len_j = local_edge_lengths[j];
                            // 粗いメッシュでも隙間から素通しにならないよう、局所エッジ長を考慮した実効衝突マージン
                            let edge_margin = min(local_len_i, local_len_j) * 0.04;
                            let effective_thick = min(max(min_dist, edge_margin), min_dist * 3.0);

                            var collided = false;
                            var n = vec3<f32>(0.0);
                            var penetration = 0.0;
                            var is_untangling = false;

                            // 相手頂点 j の法線の取得 (多層布Untangling用)
                            let n_j = normals[j].xyz;
                            let n_j_len = length(n_j);
                            let has_valid_normal_j = (n_j_len > 0.5);
                            var n_j_unit = vec3<f32>(0.0, 0.0, 1.0);
                            if (has_valid_normal_j) {
                                n_j_unit = n_j / n_j_len;
                            }

                            let h_j = dot(delta, n_j_unit); // 相手の法線方向から見た自分の高さ (< 0 なら裏側)

                            // 1. 通常接触: 従来の球対球 (Vertex-Vertex) PBD反発
                            // 距離が衝突厚み未満なら、いかなる場合も互いを押し離す基本ベクトルを適用
                            if (dist < effective_thick && dist > EPSILON) {
                                collided = true;
                                n = delta / dist;
                                penetration = effective_thick - dist;
                            }

                            // 2. 多層布Untangling (外側レイヤー i が内側レイヤー j の裏側に潜り込んだ場合のみ法線脱出)
                            if (params.enable_normal_untangling != 0u && has_valid_normal_j) {
                                let can_untangle = (layer_i > layer_j);

                                if (can_untangle && h_j < 0.0) {
                                    let d_tangent = delta - h_j * n_j_unit;
                                    let r_sq = dot(d_tangent, d_tangent);

                                    // 広大なエッジ長ではなく、実効厚みの至近距離に限定
                                    let r_max = effective_thick * 2.0;
                                    let r_max_sq = r_max * r_max;
                                    let h_max = effective_thick * 2.0;

                                    if (r_sq < r_max_sq && h_j > -h_max) {
                                        collided = true;
                                        is_untangling = true;
                                        n = n_j_unit; // 相手の表側法線方向へ押し戻す
                                        penetration = min(effective_thick - h_j, effective_thick * 2.0);
                                    }
                                }
                            }

                            if (collided && penetration > 0.0) {
                                var disp = vec3<f32>(0.0);
                                if (is_untangling) {
                                    has_untangle_contact = true;
                                    // Untangling (多層布の法線脱出: 外側レイヤー i のみを動かす)
                                    disp = n * penetration;
                                } else {
                                    // 通常接触
                                    if (layer_i == layer_j) {
                                        let w_sum = v_i.inv_mass + v_j.inv_mass;
                                        if (w_sum > EPSILON) {
                                            let ratio = v_i.inv_mass / w_sum;
                                            disp = n * (penetration * ratio * 0.5);
                                        }
                                    } else if (layer_i > layer_j) {
                                        disp = n * penetration;
                                    }
                                }
                                total_disp = total_disp + disp;
                                contact_count = contact_count + 1.0;
                            }
                        }
                    }
                    other_idx = vert_next[j];
                    iter_count = iter_count + 1u;
                }
            }
        }
    }

    // 近傍接触ペアからの変位を平均化し、1サブステップあたりの最大移動量を適正にクランプ
    if (contact_count > 0.0) {
        var avg_disp = total_disp / contact_count;
        // 通常衝突はソフトリリーフで穏やかに押し合い、Untangling(脱出)は確実に押し出す
        if (params.enable_relief != 0u && !has_untangle_contact) {
            avg_disp = avg_disp * params.relief_factor;
        }
        let disp_len = length(avg_disp);
        let ratio = select(0.10, params.max_displacement_ratio, params.max_displacement_ratio > 0.0);
        let max_step_disp = min(local_edge_lengths[index] * ratio, 0.020); // 縫合力に負けずバネを壊さない安全値 (最大20mm)
        if (disp_len > max_step_disp && disp_len > EPSILON) {
            avg_disp = (avg_disp / disp_len) * max_step_disp;
        }
        p_i = p_i + avg_disp;
    }

    v_i.prev_pos = p_i;
    vertices[index] = v_i;
}
