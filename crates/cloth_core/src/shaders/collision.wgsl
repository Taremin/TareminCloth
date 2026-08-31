struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct GpuCollider {
    collider_type: u32, // 0: Sphere, 1: Capsule, 2: Plane
    friction: f32,
    restitution: f32,
    _pad0: f32,
    point_a: vec3<f32>,
    radius: f32,
    point_b: vec3<f32>,
    _pad1: f32,
};

struct GpuMeshTriangle {
    p0: vec3<f32>,
    friction: f32,
    p1: vec3<f32>,
    thickness: f32,
    p2: vec3<f32>,
    restitution: f32,
    flags: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct CollisionParams {
    num_vertices: u32,
    num_colliders: u32,
    num_mesh_triangles: u32,
    dt: f32,
    edge_margin_scale: f32,
    edge_margin_offset: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> colliders: array<GpuCollider>;
@group(0) @binding(2) var<storage, read> mesh_triangles: array<GpuMeshTriangle>;
@group(0) @binding(3) var<uniform> params: CollisionParams;
@group(0) @binding(4) var<storage, read> mesh_bounds: array<vec4<f32>>;

const EPSILON: f32 = 1e-7;

// 接触応答（クーロン摩擦と非弾性反発係数の適用）
// PBD/XPBDの接触定式化 (Velocity Projection via Normal Position Adjustment):
// PBDでは速度が v = (p - x_n) / dt で暗黙更新されるため、
// 位置 p をコライダー表面に押し出しても、x_n が衝突前の上空位置のままだと
// 下向きの突入速度が速度として引き継がれ、次ステップでコライダーを貫通してしまう。
// したがって、法線方向の相対速度を希望の反発速度 desired_v_out_n (-e * v_in) に
// 調停するため、x_n の法線成分を target_pos - normal * (desired_v_out_n * dt) に投影する。
fn apply_contact_response(
    target_pos: vec3<f32>,
    normal: vec3<f32>,
    friction: f32,
    restitution: f32,
    dt: f32,
    p_initial: vec3<f32>,
    x_n: ptr<function, vec3<f32>>,
    p: ptr<function, vec3<f32>>
) {
    // 1. 接線方向すべり変位の抽出とクーロン摩擦の適用
    let total_disp = target_pos - *x_n;
    let tangent_disp = total_disp - dot(total_disp, normal) * normal;
    *p = target_pos - tangent_disp * clamp(friction, 0.0, 1.0);

    // 2. 法線突入速度の吸収と反発 (Velocity Projection)
    let effective_dt = max(dt, 1e-6);
    let v_in_n = dot((p_initial - *x_n) / effective_dt, normal);
    var desired_v_out_n = 0.0;
    if (v_in_n < 0.0) {
        desired_v_out_n = -clamp(restitution, 0.0, 1.0) * v_in_n;
    }
    let delta_x_n = dot(target_pos - *x_n, normal) - desired_v_out_n * effective_dt;
    *x_n = *x_n + normal * delta_x_n;

    // 3. 接触押し出しに伴う異常初速（局所スパイク）の物理クランプ (Velocity Clamping)
    // コライダー裏面リカバリー等の急激な位置補正が時速数百km/hの暴走速度に変換されるのを防ぐ
    let max_safe_vel = 25.0; // 最大許容速度 25 m/s (時速 90km/h 相当)
    let v_post = (*p - *x_n) / effective_dt;
    let v_speed = length(v_post);
    if (v_speed > max_safe_vel) {
        *x_n = *p - (v_post / v_speed) * (max_safe_vel * effective_dt);
    }
}

// 点から三角形への最近傍点
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_vertices) {
        return;
    }

    var v = vertices[index];
    if (v.inv_mass <= 0.0) {
        return;
    }

    var p = v.prev_pos;
    let p_initial = p;
    var x_n = v.position;
    let thickness = v.thickness;
    let dt = params.dt;

    // 1. プリミティブコライダー (Sphere, Capsule, Plane)
    for (var i = 0u; i < params.num_colliders; i = i + 1u) {
        let col = colliders[i];

        if (col.collider_type == 0u) {
            let center = col.point_a;
            let target_r = col.radius + thickness;
            let delta = p - center;
            let dist = length(delta);

            if (dist < target_r) {
                var normal = vec3<f32>(0.0, 0.0, 1.0);
                if (dist > EPSILON) {
                    normal = delta / dist;
                }
                let target_pos = center + normal * target_r;
                apply_contact_response(target_pos, normal, col.friction, col.restitution, dt, p_initial, &x_n, &p);
            }
        } else if (col.collider_type == 1u) {
            let a = col.point_a;
            let b = col.point_b;
            let target_r = col.radius + thickness;

            let ab = b - a;
            let l2 = dot(ab, ab);
            var t = 0.0;
            if (l2 > EPSILON) {
                t = clamp(dot(p - a, ab) / l2, 0.0, 1.0);
            }
            let closest = a + ab * t;
            let delta = p - closest;
            let dist = length(delta);

            if (dist < target_r) {
                var normal = vec3<f32>(0.0, 0.0, 1.0);
                if (dist > EPSILON) {
                    normal = delta / dist;
                }
                let target_pos = closest + normal * target_r;
                apply_contact_response(target_pos, normal, col.friction, col.restitution, dt, p_initial, &x_n, &p);
            }
        } else if (col.collider_type == 2u) {
            let plane_pt = col.point_a;
            let plane_n = normalize(col.point_b);
            let dist = dot(p - plane_pt, plane_n);

            if (dist < thickness) {
                let target_pos = p + plane_n * (thickness - dist);
                apply_contact_response(target_pos, plane_n, col.friction, col.restitution, dt, p_initial, &x_n, &p);
            }
        }
    }

    // 2. メッシュコライダー (Triangle Mesh: 2段階フェッチによるVRAM帯域最適化)
    for (var i = 0u; i < params.num_mesh_triangles; i = i + 1u) {
        // 第1段階: 16バイトの軽量境界球 (vec4: xyz=中心, w=外接半径+厚み) のみをフェッチして判定
        let b = mesh_bounds[i];
        let diff = p - b.xyz;
        let r = b.w * 1.5 + thickness;
        if (dot(diff, diff) > r * r) {
            continue; // 近傍外の三角形はここで即座にスキップ
        }

        // 第2段階: 境界球と交差した三角形のみ、48バイトの詳細頂点データをフェッチ
        let tri = mesh_triangles[i];
        let target_dist = thickness + tri.thickness;

        let q = closest_point_on_triangle(p, tri.p0, tri.p1, tri.p2);
        let delta = p - q;
        let dist = length(delta);

        let cross_prod = cross(tri.p1 - tri.p0, tri.p2 - tri.p0);
        let cross_len = length(cross_prod);
        if (cross_len < EPSILON) {
            continue;
        }
        let face_normal = cross_prod / cross_len;
        let signed_dist = dot(delta, face_normal);

        let is_single_sided = (tri.flags & 1u) != 0u;
        if (is_single_sided) {
            if (signed_dist >= 0.0) {
                // 表側: 通常の厚み判定 (法線方向のみ押し出し)
                if (dist < target_dist) {
                    var normal = face_normal;
                    if (dist > EPSILON) {
                        normal = delta / dist;
                    }
                    let target_pos = p + normal * (target_dist - dist);
                    apply_contact_response(target_pos, normal, tri.friction, tri.restitution, dt, p_initial, &x_n, &p);
                }
            } else {
                // 裏側 (メッシュ内部侵入): 境界球範囲内のめり込みを表側へ押し戻す貫通リカバリー
                let max_recovery_depth = b.w + target_dist;
                if (dist < max_recovery_depth) {
                    let normal = face_normal;
                    let target_pos = p + normal * (target_dist - signed_dist);
                    apply_contact_response(target_pos, normal, tri.friction, tri.restitution, dt, p_initial, &x_n, &p);
                }
            }
        } else {
            // 両面コライダー: 厚み target_dist の両面薄皮判定
            if (dist < target_dist && signed_dist >= -target_dist) {
                var normal = face_normal;
                if (dist > EPSILON) {
                    if (signed_dist > 0.0) {
                        normal = delta / dist;
                    }
                }
                let target_pos = p + normal * (target_dist - dist);
                apply_contact_response(target_pos, normal, tri.friction, tri.restitution, dt, p_initial, &x_n, &p);
            }
        }
    }

    v.prev_pos = p;
    v.position = x_n; // 法線突入速度を吸収した基準位置
    vertices[index] = v;
}
