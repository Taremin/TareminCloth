// Edge-to-Edge Collision Pass (Cloth Edge vs Collider Edge)
// 布のエッジとコライダー稜線間の厳密な線分間最短距離（E-E）接触解決パス。
// 空間ハッシュ（近傍27セル走査）および固定小数点アトミック変位蓄積により、
// 尖った角や稜線同士のすり抜けを完全阻止しつつ、O(Ne * K)の定数時間で高速実行する。

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

struct GpuColliderEdge {
    p0: vec3<f32>,
    thickness: f32,
    p1: vec3<f32>,
    friction: f32,
};

struct EdgeCollisionParams {
    cell_size: f32,
    table_size: u32,
    num_cloth_edges: u32,
    num_collider_edges: u32,
    edge_margin_scale: f32,
    edge_margin_offset: f32,
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

@group(0) @binding(0) var<storage, read> cloth_edges: array<GpuEdge>;
@group(0) @binding(1) var<storage, read> vertices: array<GpuVertex>;
@group(0) @binding(2) var<storage, read> collider_edges: array<GpuColliderEdge>;
@group(0) @binding(3) var<storage, read> collider_cell_starts: array<u32>;
@group(0) @binding(4) var<storage, read> collider_sorted_indices: array<u32>;
@group(0) @binding(5) var<uniform> params: EdgeCollisionParams;
@group(0) @binding(6) var<storage, read_write> accum: array<SelfCollisionAccum>;

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

struct SegmentDistanceResult {
    s: f32,
    t: f32,
    dist: f32,
};

// 2線分 p0->p1 と q0->q1 間の最短距離および線分パラメータ s, t in [0, 1]
fn closest_points_segments(
    p0: vec3<f32>, p1: vec3<f32>,
    q0: vec3<f32>, q1: vec3<f32>
) -> SegmentDistanceResult {
    let d1 = p1 - p0;
    let d2 = q1 - q0;
    let r = p0 - q0;

    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);

    var s = 0.0;
    var t = 0.0;

    if (a <= EPSILON && e <= EPSILON) {
        let diff = p0 - q0;
        return SegmentDistanceResult(0.0, 0.0, length(diff));
    }
    if (a <= EPSILON) {
        s = 0.0;
        t = clamp(f / e, 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (e <= EPSILON) {
            t = 0.0;
            s = clamp(-c / a, 0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;

            if (abs(denom) > EPSILON) {
                s = clamp((b * f - c * e) / denom, 0.0, 1.0);
            } else {
                s = 0.0;
            }

            t = (b * s + f) / e;

            if (t < 0.0) {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }

    let closest_p = p0 + d1 * s;
    let closest_q = q0 + d2 * t;
    let diff = closest_p - closest_q;
    return SegmentDistanceResult(s, t, length(diff));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let edge_idx = global_id.x;
    if (edge_idx >= params.num_cloth_edges || params.num_collider_edges == 0u) {
        return;
    }

    let e = cloth_edges[edge_idx];
    let u0 = e.v0;
    let u1 = e.v1;

    let v0 = vertices[u0];
    let v1 = vertices[u1];

    let w0 = v0.inv_mass;
    let w1 = v1.inv_mass;
    let w_sum = w0 + w1;
    if (w_sum <= EPSILON) {
        return;
    }

    let p0 = v0.prev_pos;
    let p1 = v1.prev_pos;

    let p_mid = (p0 + p1) * 0.5;
    let cloth_thickness = max(v0.thickness, v1.thickness);

    let cell_coord = vec3<i32>(floor(p_mid / params.cell_size));

    let cloth_aabb_min = min(p0, p1);
    let cloth_aabb_max = max(p0, p1);

    // 27近傍セル走査
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let neighbor = cell_coord + vec3<i32>(dx, dy, dz);
                let h = hash_coords(neighbor, params.table_size);

                let start = collider_cell_starts[h];
                let end = collider_cell_starts[h + 1u];

                for (var k = start; k < end; k = k + 1u) {
                    let col_idx = collider_sorted_indices[k];
                    let col_edge = collider_edges[col_idx];

                    let q0 = col_edge.p0;
                    let q1 = col_edge.p1;

                    let target_dist = (cloth_thickness + col_edge.thickness) * params.edge_margin_scale + params.edge_margin_offset;

                    // 1. AABB 先行枝切り
                    let col_aabb_min = min(q0, q1);
                    let col_aabb_max = max(q0, q1);

                    if (cloth_aabb_max.x + target_dist < col_aabb_min.x ||
                        cloth_aabb_min.x - target_dist > col_aabb_max.x ||
                        cloth_aabb_max.y + target_dist < col_aabb_min.y ||
                        cloth_aabb_min.y - target_dist > col_aabb_max.y ||
                        cloth_aabb_max.z + target_dist < col_aabb_min.z ||
                        cloth_aabb_min.z - target_dist > col_aabb_max.z) {
                        continue;
                    }

                    // 2. 厳密な線分間最短距離 (E-E) 判定
                    let res = closest_points_segments(p0, p1, q0, q1);
                    if (res.dist < target_dist) {
                        let pt_cloth = p0 + (p1 - p0) * res.s;
                        let pt_col = q0 + (q1 - q0) * res.t;
                        let diff = pt_cloth - pt_col;

                        var normal = vec3<f32>(0.0, 0.0, 1.0);
                        if (res.dist > EPSILON) {
                            normal = diff / res.dist;
                        } else {
                            // ほぼ交差している場合、線分外積から法線を推定
                            let d_cloth = p1 - p0;
                            let d_col = q1 - q0;
                            let c = cross(d_cloth, d_col);
                            let c_len = length(c);
                            if (c_len > EPSILON) {
                                normal = c / c_len;
                            }
                        }

                        let penetration = target_dist - res.dist;
                        let push = normal * penetration;

                        // コライダーは無限質量 (動かない) ため、100% 布端点にのみ分配
                        let a = 1.0 - res.s;
                        let b = res.s;
                        let denom = a * a * w0 + b * b * w1;
                        if (denom > EPSILON) {
                            let factor0 = (a * w0) / denom;
                            let factor1 = (b * w1) / denom;
                            add_disp_atomic(u0, push * factor0);
                            add_disp_atomic(u1, push * factor1);
                        }
                    }
                }
            }
        }
    }
}
