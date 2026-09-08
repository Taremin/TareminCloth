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
    num_clusters: u32,
    dt: f32,
    edge_margin_scale: f32,
    edge_margin_offset: f32,
    enable_cluster_culling: u32,
    enable_single_sided_recovery: u32,
    sweep_margin_offset: f32,
    num_bones: u32,
    enable_bone_sdf: u32,
};

struct GpuBoneInfo {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    uvw_scale: vec4<f32>,
    uvw_offset: vec4<f32>,
    params: vec4<f32>, // x: friction, y: thickness, z: restitution, w: blend_k
};

struct GpuBoneTransform {
    world_matrix: mat4x4<f32>,
    inv_world_matrix: mat4x4<f32>,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> colliders: array<GpuCollider>;
@group(0) @binding(2) var<storage, read> mesh_triangles: array<GpuMeshTriangle>;
@group(0) @binding(3) var<uniform> params: CollisionParams;
@group(0) @binding(4) var<storage, read> mesh_bounds: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> collider_group_bounds: array<vec4<f32>>;
@group(0) @binding(6) var bone_sdf_texture: texture_3d<f32>;
@group(0) @binding(7) var bone_sdf_sampler: sampler;
@group(0) @binding(8) var<storage, read> bone_infos: array<GpuBoneInfo>;
@group(0) @binding(9) var<storage, read> bone_transforms: array<GpuBoneTransform>;


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
}

fn smin_poly(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / max(k, 1e-6);
    return min(a, b) - h * h * k * 0.25;
}

// ボーンSDFコライダーの衝突判定および押し戻し
fn solve_bone_sdf_collision(
    p: ptr<function, vec3<f32>>,
    x_n: ptr<function, vec3<f32>>,
    dt: f32,
    p_initial: vec3<f32>,
    vertex_thickness: f32
) {
    if (params.enable_bone_sdf == 0u || params.num_bones == 0u) {
        return;
    }

    var best_dist = 1e6;
    var best_normal = vec3<f32>(0.0, 0.0, 0.0);
    var best_friction = 0.0;
    var best_restitution = 0.0;
    var best_target_dist = 0.0;
    var hit_count = 0u;

    for (var b = 0u; b < params.num_bones; b = b + 1u) {
        let info = bone_infos[b];
        let transform = bone_transforms[b];

        // 1. 布の現在位置 p をボーンローカル座標に射影
        let p_local = (transform.inv_world_matrix * vec4<f32>(*p, 1.0)).xyz;

        // 2. ローカルAABBによる早期カリング (マージン込み)
        let target_dist = vertex_thickness + info.params.y;
        let margin = target_dist + 0.05;
        if (p_local.x < info.aabb_min.x - margin || p_local.x > info.aabb_max.x + margin ||
            p_local.y < info.aabb_min.y - margin || p_local.y > info.aabb_max.y + margin ||
            p_local.z < info.aabb_min.z - margin || p_local.z > info.aabb_max.z + margin) {
            continue;
        }

        // 3. テクスチャアトラスのUVW座標へ変換
        let uvw = p_local * info.uvw_scale.xyz + info.uvw_offset.xyz;

        // 各ボーンに割り当てられた3Dタイルスロット範囲 [uvw_min, uvw_max]
        let uvw_min = info.aabb_min.xyz * info.uvw_scale.xyz + info.uvw_offset.xyz;
        let uvw_max = info.aabb_max.xyz * info.uvw_scale.xyz + info.uvw_offset.xyz;

        // 隣接ボーンのスライスへの侵入を完全に防止
        if (uvw.x < uvw_min.x || uvw.x > uvw_max.x ||
            uvw.y < uvw_min.y || uvw.y > uvw_max.y ||
            uvw.z < uvw_min.z || uvw.z > uvw_max.z) {
            continue;
        }

        // トリリニア補間での隣接スライスにじみを防ぐ安全クランプ
        // 単一メッシュSDF (num_bones == 1u) の場合は隣接スライスが存在しないため、境界クランプを極小化 (0.002)
        var eps_slice = (uvw_max - uvw_min) * 0.015;
        if (params.num_bones == 1u) {
            eps_slice = (uvw_max - uvw_min) * 0.002;
        }
        let uvw_safe = clamp(uvw, uvw_min + eps_slice, uvw_max - eps_slice);

        // 4. トリリニアサンプリング (Distance, Alpha/Weight)
        let sample_val = textureSampleLevel(bone_sdf_texture, bone_sdf_sampler, uvw_safe, 0.0);
        let raw_dist = sample_val.r;
        let alpha = sample_val.g;

        // Alphaマスク: weight_threshold以下の領域はカリングして安全圏(+10.0)へ逃がす
        let threshold = max(info.params.w, 0.01);
        var retract_offset = 0.0;
        // ハイブリッドモード（関節メッシュコライダーが存在する場合）のみ、関節境界で剛体角をリトラクト
        // メッシュコライダーが存在しないSDF単体時は、コライダー壁としての貫通防止機能を優先して角を維持
        if (params.num_mesh_triangles > 0u) {
            let blend_mask = smoothstep(threshold, 0.7, alpha);
            retract_offset = (1.0 - blend_mask) * 0.06; // 最大6cm内側へリトラクト
        }
        let effective_dist = select(10.0, raw_dist + retract_offset, alpha > threshold);

        if (effective_dist < target_dist) {
            // 衝突・厚み内侵入を検知した場合のみ法線中心差分を計算
            let eps = 0.005; // 5mm
            let eps_uv = eps * info.uvw_scale.xyz;

            let u_min_safe = uvw_min.x + eps_slice.x;
            let u_max_safe = uvw_max.x - eps_slice.x;
            let v_min_safe = uvw_min.y + eps_slice.y;
            let v_max_safe = uvw_max.y - eps_slice.y;
            let w_min_safe = uvw_min.z + eps_slice.z;
            let w_max_safe = uvw_max.z - eps_slice.z;

            let d_x_pos = textureSampleLevel(bone_sdf_texture, bone_sdf_sampler, vec3<f32>(clamp(uvw.x + eps_uv.x, u_min_safe, u_max_safe), uvw_safe.y, uvw_safe.z), 0.0).r;
            let d_x_neg = textureSampleLevel(bone_sdf_texture, bone_sdf_sampler, vec3<f32>(clamp(uvw.x - eps_uv.x, u_min_safe, u_max_safe), uvw_safe.y, uvw_safe.z), 0.0).r;
            let d_y_pos = textureSampleLevel(bone_sdf_texture, bone_sdf_sampler, vec3<f32>(uvw_safe.x, clamp(uvw.y + eps_uv.y, v_min_safe, v_max_safe), uvw_safe.z), 0.0).r;
            let d_y_neg = textureSampleLevel(bone_sdf_texture, bone_sdf_sampler, vec3<f32>(uvw_safe.x, clamp(uvw.y - eps_uv.y, v_min_safe, v_max_safe), uvw_safe.z), 0.0).r;
            let d_z_pos = textureSampleLevel(bone_sdf_texture, bone_sdf_sampler, vec3<f32>(uvw_safe.x, uvw_safe.y, clamp(uvw.z + eps_uv.z, w_min_safe, w_max_safe)), 0.0).r;
            let d_z_neg = textureSampleLevel(bone_sdf_texture, bone_sdf_sampler, vec3<f32>(uvw_safe.x, uvw_safe.y, clamp(uvw.z - eps_uv.z, w_min_safe, w_max_safe)), 0.0).r;

            var dx = d_x_pos - d_x_neg;
            if (abs(dx) < 1e-6) {
                dx = select(raw_dist - d_x_neg, d_x_pos - raw_dist, uvw.x < (u_min_safe + u_max_safe) * 0.5);
            }
            var dy = d_y_pos - d_y_neg;
            if (abs(dy) < 1e-6) {
                dy = select(raw_dist - d_y_neg, d_y_pos - raw_dist, uvw.y < (v_min_safe + v_max_safe) * 0.5);
            }
            var dz = d_z_pos - d_z_neg;
            if (abs(dz) < 1e-6) {
                dz = select(raw_dist - d_z_neg, d_z_pos - raw_dist, uvw.z < (w_min_safe + w_max_safe) * 0.5);
            }

            var grad_local = vec3<f32>(dx, dy, dz);
            let grad_len = length(grad_local);
            var normal_local = vec3<f32>(0.0, 0.0, 1.0);
            if (grad_len > 1e-6) {
                normal_local = grad_local / grad_len;
            } else {
                let center_local = (info.aabb_min.xyz + info.aabb_max.xyz) * 0.5;
                let diff_c = p_local - center_local;
                let len_c = length(diff_c);
                if (len_c > 1e-6) {
                    normal_local = diff_c / len_c;
                }
            }

            // ワールド法線へ変換
            let normal_world = normalize((transform.world_matrix * vec4<f32>(normal_local, 0.0)).xyz);

            if (hit_count == 0u || effective_dist < best_dist) {
                best_dist = effective_dist;
                best_normal = normal_world;
                best_friction = info.params.x;
                best_restitution = info.params.z;
                best_target_dist = target_dist;
            }
            hit_count = hit_count + 1u;
        }
    }

    if (hit_count > 0u && best_dist < best_target_dist) {
        let penetration = best_target_dist - best_dist;
        let target_pos = *p + best_normal * penetration;
        apply_contact_response(target_pos, best_normal, best_friction, best_restitution, dt, p_initial, x_n, p);
    }
}

struct ClosestResult {
    point: vec3<f32>,
    is_face: bool,
};

// 空間上の点 p から三角形 (a, b, c) への最近傍点および面内部フラグ
fn closest_point_on_triangle_ext(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> ClosestResult {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return ClosestResult(a, false);
    }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return ClosestResult(b, false);
    }

    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return ClosestResult(a + v * ab, false);
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return ClosestResult(c, false);
    }

    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return ClosestResult(a + w * ac, false);
    }

    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return ClosestResult(b + w * (c - b), false);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return ClosestResult(a + ab * v + ac * w, true);
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

    // 2. メッシュコライダー (Triangle Mesh: 2段階フェッチ + CCD & 表側優先・裏面抑制リカバリー)
    var has_front_contact = false;
    var best_recovery_depth = 999999.0;
    var best_recovery_pos = p;
    var best_recovery_normal = vec3<f32>(0.0);
    var best_recovery_fric = 0.0;
    var best_recovery_rest = 0.0;
    var has_valid_recovery = false;

    let total_tris = params.num_mesh_triangles;
    let num_groups = params.num_clusters;
    let use_cluster = (params.enable_cluster_culling != 0u) && (total_tris >= 32u);

    if (!use_cluster) {
        // [直接全走査パス - デフォルト OFF]
        for (var i = 0u; i < total_tris; i = i + 1u) {
            let b = mesh_bounds[i];
            let diff = p - b.xyz;
            let diff_old = v.position - b.xyz;
            let r = b.w * 1.5 + thickness;
            if (dot(diff, diff) > r * r && dot(diff_old, diff_old) > r * r) {
                continue;
            }

            let tri = mesh_triangles[i];
            let target_dist = thickness + tri.thickness;

            let cross_prod = cross(tri.p1 - tri.p0, tri.p2 - tri.p0);
            let cross_len = length(cross_prod);
            if (cross_len < EPSILON) {
                continue;
            }
            let face_normal = cross_prod / cross_len;
            let is_single_sided = (tri.flags & 1u) != 0u;

            // (A) 連続衝突判定 (CCD)
            let isect = intersect_segment_triangle(v.position, p, tri.p0, tri.p1, tri.p2);
            if (isect.x > 0.5) {
                let t_hit = isect.y;
                let hit_pt = v.position + t_hit * (p - v.position);
                var normal = face_normal;
                let move_dir = p - v.position;
                if (dot(move_dir, face_normal) > 0.0 && !is_single_sided) {
                    normal = -face_normal;
                }
                let target_pos = hit_pt + normal * target_dist;
                apply_contact_response(target_pos, normal, tri.friction, tri.restitution, dt, p_initial, &x_n, &p);
                has_front_contact = true;
            }

            // (B) 近傍厚み接触 & 静的貫通リカバリー
            let res = closest_point_on_triangle_ext(p, tri.p0, tri.p1, tri.p2);
            let q = res.point;
            let delta = p - q;
            let dist = length(delta);
            let signed_dist = dot(delta, face_normal);

            if (is_single_sided) {
                if (signed_dist >= 0.0) {
                    if (dist < target_dist) {
                        var normal = face_normal;
                        if (dist > EPSILON) {
                            normal = delta / dist;
                        }
                        let target_pos = p + normal * (target_dist - dist);
                        apply_contact_response(target_pos, normal, tri.friction, tri.restitution, dt, p_initial, &x_n, &p);
                        has_front_contact = true;
                    }
                } else {
                    if (params.enable_single_sided_recovery != 0u) {
                        let max_recovery_depth = target_dist * 3.0;
                        if (res.is_face && dist < max_recovery_depth && dist <= abs(signed_dist) * 1.1 + EPSILON) {
                            if (dist < best_recovery_depth) {
                                best_recovery_depth = dist;
                                best_recovery_normal = face_normal;
                                best_recovery_pos = p + face_normal * (target_dist - signed_dist);
                                best_recovery_fric = tri.friction;
                                best_recovery_rest = tri.restitution;
                                has_valid_recovery = true;
                            }
                        }
                    }
                }
            } else {
                if (dist < target_dist && signed_dist >= -target_dist) {
                    var normal = face_normal;
                    if (dist > EPSILON) {
                        normal = delta / dist;
                    }
                    let target_pos = p + normal * (target_dist - dist);
                    apply_contact_response(target_pos, normal, tri.friction, tri.restitution, dt, p_initial, &x_n, &p);
                    has_front_contact = true;
                }
            }
        }
    } else {
        // [16面クラスタ境界球カリングパス - オプション ON]
        let sweep_offset = max(params.sweep_margin_offset, 0.0);

        for (var g = 0u; g < num_groups; g = g + 1u) {
            let gb = collider_group_bounds[g];
            let diff_g = p - gb.xyz;
            let diff_old_g = v.position - gb.xyz;
            let rg = gb.w + thickness * 4.0 + sweep_offset;
            if (dot(diff_g, diff_g) > rg * rg && dot(diff_old_g, diff_old_g) > rg * rg) {
                continue;
            }

            let start_i = g * 16u;
            let end_i = min(start_i + 16u, total_tris);

            for (var i = start_i; i < end_i; i = i + 1u) {
                let b = mesh_bounds[i];
                let diff = p - b.xyz;
                let diff_old = v.position - b.xyz;
                let r = b.w + thickness * 4.0;
                if (dot(diff, diff) > r * r && dot(diff_old, diff_old) > r * r) {
                    continue;
                }

                let tri = mesh_triangles[i];
                let target_dist = thickness + tri.thickness;

                let cross_prod = cross(tri.p1 - tri.p0, tri.p2 - tri.p0);
                let cross_len = length(cross_prod);
                if (cross_len < EPSILON) {
                    continue;
                }
                let face_normal = cross_prod / cross_len;
                let is_single_sided = (tri.flags & 1u) != 0u;

                // (A) 連続衝突判定 (CCD)
                let isect = intersect_segment_triangle(v.position, p, tri.p0, tri.p1, tri.p2);
                if (isect.x > 0.5) {
                    let t_hit = isect.y;
                    let hit_pt = v.position + t_hit * (p - v.position);
                    var normal = face_normal;
                    let move_dir = p - v.position;
                    if (dot(move_dir, face_normal) > 0.0 && !is_single_sided) {
                        normal = -face_normal;
                    }
                    let target_pos = hit_pt + normal * target_dist;
                    apply_contact_response(target_pos, normal, tri.friction, tri.restitution, dt, p_initial, &x_n, &p);
                    has_front_contact = true;
                }

                // (B) 近傍厚み接触 & 静的貫通リカバリー
                let res = closest_point_on_triangle_ext(p, tri.p0, tri.p1, tri.p2);
                let q = res.point;
                let delta = p - q;
                let dist = length(delta);
                let signed_dist = dot(delta, face_normal);

                if (is_single_sided) {
                    if (signed_dist >= 0.0) {
                        if (dist < target_dist) {
                            var normal = face_normal;
                            if (dist > EPSILON) {
                                normal = delta / dist;
                            }
                            let target_pos = p + normal * (target_dist - dist);
                            apply_contact_response(target_pos, normal, tri.friction, tri.restitution, dt, p_initial, &x_n, &p);
                            has_front_contact = true;
                        }
                    } else {
                        if (params.enable_single_sided_recovery != 0u) {
                            let max_recovery_depth = target_dist * 3.0;
                            if (res.is_face && dist < max_recovery_depth && dist <= abs(signed_dist) * 1.1 + EPSILON) {
                                if (dist < best_recovery_depth) {
                                    best_recovery_depth = dist;
                                    best_recovery_normal = face_normal;
                                    best_recovery_pos = p + face_normal * (target_dist - signed_dist);
                                    best_recovery_fric = tri.friction;
                                    best_recovery_rest = tri.restitution;
                                    has_valid_recovery = true;
                                }
                            }
                        }
                    }
                } else {
                    if (dist < target_dist && signed_dist >= -target_dist) {
                        var normal = face_normal;
                        if (dist > EPSILON) {
                            normal = delta / dist;
                        }
                        let target_pos = p + normal * (target_dist - dist);
                        apply_contact_response(target_pos, normal, tri.friction, tri.restitution, dt, p_initial, &x_n, &p);
                        has_front_contact = true;
                    }
                }
            }
        }
    }

    // フェーズ2: 表側接触が一切ない完全なめり込み頂点のみ、最適な最近傍1面から安全に脱出リカバリー
    if (!has_front_contact && has_valid_recovery) {
        apply_contact_response(best_recovery_pos, best_recovery_normal, best_recovery_fric, best_recovery_rest, dt, p_initial, &x_n, &p);
    }

    // フェーズ3: ボーン局所SDFコライダーの判定・押し戻し
    // メッシュコライダーで既に接触・押し戻しが行われた頂点は、二重適用による干渉・摩擦引き攣れを防ぐためスキップ
    if (!has_front_contact && !has_valid_recovery) {
        solve_bone_sdf_collision(&p, &x_n, dt, p_initial, thickness);
    }

    v.prev_pos = p;
    v.position = x_n; // 法線突入速度を吸収した基準位置
    vertices[index] = v;
}

