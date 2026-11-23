use std::collections::{HashMap, HashSet};
use std::mem::{offset_of, size_of};

use cloth_core::mesh::{
    EdgeCollisionParams, GpuBendingConstraint, GpuCollider, GpuColliderEdge, GpuDistanceConstraint,
    GpuEdge, GpuEePair, GpuMeshTriangle, GpuPinConstraint, GpuSewingConstraint, GpuStarPair,
    GpuVertex, GpuVtPair, PairCollectParams, PairCounters, PairSolveParams, SelfCollisionParams,
    SimParams,
};
use cloth_core::sdf_baker::{
    GpuBakeMeshParams, GpuBakeMeshTriangle, GpuBakeParams, GpuBakeTriangle, GpuHierFineParams,
};
use cloth_core::simulation::types::{
    CollisionParams, DispatchInfo, GpuAtomicAccum, GpuBoneInfo, GpuBoneTransform,
    GpuBoneTriangleSource, GpuSelfCollisionAccum, GpuSkinningVertex, NormalParams, PinParams,
    WrinkleFieldParams,
};
use cloth_core::spatial_hash::SpatialHashParams;
use cloth_core::virtual_mesh::{GpuVirtualVertexDef, SelfCollisionVvParams, VirtualForwardParams};

use naga::front::wgsl;
use naga::proc::Layouter;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedMember {
    name: String,
    offset: u32,
    size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedStruct {
    name: String,
    size: u32,
    members: Vec<ParsedMember>,
}

fn parse_wgsl_structs(filename: &str, source: &str) -> HashMap<String, ParsedStruct> {
    let module = wgsl::parse_str(source).unwrap_or_else(|e| {
        panic!("Failed to parse WGSL in {}: {:?}", filename, e);
    });
    let mut layouter = Layouter::default();
    let ctx = module.to_ctx();
    layouter.update(ctx).unwrap_or_else(|e| {
        panic!("Failed to compute layout for {}: {:?}", filename, e);
    });

    let mut map = HashMap::new();
    for (_handle, ty) in module.types.iter() {
        if let naga::TypeInner::Struct { ref members, span } = ty.inner {
            if let Some(ref name) = ty.name {
                let parsed_members = members
                    .iter()
                    .map(|m| ParsedMember {
                        name: m.name.clone().unwrap_or_default(),
                        offset: m.offset,
                        size: layouter[m.ty].size,
                    })
                    .collect();
                map.insert(
                    name.clone(),
                    ParsedStruct {
                        name: name.clone(),
                        size: span,
                        members: parsed_members,
                    },
                );
            }
        }
    }
    map
}

const ALL_SHADERS: &[(&str, &str)] = &[
    ("collision.wgsl", include_str!("../src/shaders/collision.wgsl")),
    ("edge_collision.wgsl", include_str!("../src/shaders/edge_collision.wgsl")),
    ("self_collision.wgsl", include_str!("../src/shaders/self_collision.wgsl")),
    ("self_collision_untangle.wgsl", include_str!("../src/shaders/self_collision_untangle.wgsl")),
    ("predict.wgsl", include_str!("../src/shaders/predict.wgsl")),
    ("update_vel.wgsl", include_str!("../src/shaders/update_vel.wgsl")),
    ("distance.wgsl", include_str!("../src/shaders/distance.wgsl")),
    ("distance_atomic.wgsl", include_str!("../src/shaders/distance_atomic.wgsl")),
    ("bending.wgsl", include_str!("../src/shaders/bending.wgsl")),
    ("sewing.wgsl", include_str!("../src/shaders/sewing.wgsl")),
    ("sewing_shrink.wgsl", include_str!("../src/shaders/sewing_shrink.wgsl")),
    ("pin.wgsl", include_str!("../src/shaders/pin.wgsl")),
    ("spatial_grid_sort.wgsl", include_str!("../src/shaders/spatial_grid_sort.wgsl")),
    ("compute_normals.wgsl", include_str!("../src/shaders/compute_normals.wgsl")),
    ("extract_positions.wgsl", include_str!("../src/shaders/extract_positions.wgsl")),
    ("bake_sdf.wgsl", include_str!("../src/shaders/bake_sdf.wgsl")),
    ("bake_mesh_sdf.wgsl", include_str!("../src/shaders/bake_mesh_sdf.wgsl")),
    ("bake_mesh_sdf_hier_common.wgsl", include_str!("../src/shaders/bake_mesh_sdf_hier_common.wgsl")),
    (
        "bake_mesh_sdf_hier_coarse_main.wgsl",
        concat!(
            include_str!("../src/shaders/bake_mesh_sdf_hier_common.wgsl"),
            "\n",
            include_str!("../src/shaders/bake_mesh_sdf_hier_coarse_main.wgsl")
        ),
    ),
    (
        "bake_mesh_sdf_hier_fine_main.wgsl",
        concat!(
            include_str!("../src/shaders/bake_mesh_sdf_hier_common.wgsl"),
            "\n",
            include_str!("../src/shaders/bake_mesh_sdf_hier_fine_main.wgsl")
        ),
    ),
    ("skinning.wgsl", include_str!("../src/shaders/skinning.wgsl")),
    ("prep_bone_triangles.wgsl", include_str!("../src/shaders/prep_bone_triangles.wgsl")),
    ("self_collision_apply.wgsl", include_str!("../src/shaders/self_collision_apply.wgsl")),
    ("self_collision_collect_pairs.wgsl", include_str!("../src/shaders/self_collision_collect_pairs.wgsl")),
    ("self_collision_solve_vt.wgsl", include_str!("../src/shaders/self_collision_solve_vt.wgsl")),
    ("self_collision_solve_ee.wgsl", include_str!("../src/shaders/self_collision_solve_ee.wgsl")),
    ("self_collision_ee_direct.wgsl", include_str!("../src/shaders/self_collision_ee_direct.wgsl")),
    ("self_collision_vv.wgsl", include_str!("../src/shaders/self_collision_vv.wgsl")),
    ("virtual_vertices_forward.wgsl", include_str!("../src/shaders/virtual_vertices_forward.wgsl")),
    ("spatial_grid_sort_edges.wgsl", include_str!("../src/shaders/spatial_grid_sort_edges.wgsl")),
    ("wrinkle_field.wgsl", include_str!("../src/shaders/wrinkle_field.wgsl")),
];

macro_rules! check_member {
    ($file:expr, $wgsl_struct:expr, $rust_type:ty, $field:ident) => {
        let rust_offset = offset_of!($rust_type, $field) as u32;
        let member = $wgsl_struct.members.iter().find(|m| m.name == stringify!($field));
        match member {
            Some(m) => {
                assert_eq!(
                    m.offset, rust_offset,
                    "[{}] Struct '{}' field '{}' offset mismatch! WGSL offset={}, Rust offset={}",
                    $file, $wgsl_struct.name, stringify!($field), m.offset, rust_offset
                );
            }
            None => {
                panic!(
                    "[{}] Struct '{}' missing expected field '{}'!",
                    $file, $wgsl_struct.name, stringify!($field)
                );
            }
        }
    };
}

macro_rules! check_struct_size {
    ($file:expr, $wgsl_struct:expr, $rust_type:ty) => {
        let rust_size = size_of::<$rust_type>() as u32;
        assert_eq!(
            $wgsl_struct.size, rust_size,
            "[{}] Struct '{}' size mismatch! WGSL size={}, Rust size={}",
            $file, $wgsl_struct.name, $wgsl_struct.size, rust_size
        );
    };
}

#[test]
fn test_all_wgsl_files_in_directory_are_registered() {
    let shaders_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join("shaders");
    let registered: std::collections::HashSet<&str> =
        ALL_SHADERS.iter().map(|(name, _)| *name).collect();
    let mut unregistered = Vec::new();
    for entry in std::fs::read_dir(&shaders_dir).expect("Failed to read src/shaders directory") {
        let entry = entry.expect("Invalid dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("wgsl") {
            if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
                if !registered.contains(fname) {
                    unregistered.push(fname.to_string());
                }
            }
        }
    }
    assert!(
        unregistered.is_empty(),
        "src/shaders/ 内の以下のWGSLファイルが ALL_SHADERS に未登録です: {:?}",
        unregistered
    );
}

#[test]
fn test_all_shaders_parse_successfully() {
    for (name, src) in ALL_SHADERS {
        let structs = parse_wgsl_structs(name, src);
        assert!(!structs.is_empty(), "Shader {} should contain structs", name);
    }
}

#[test]
fn test_cross_shader_struct_consistency() {
    let mut canonical_structs: HashMap<String, (&str, ParsedStruct)> = HashMap::new();

    for &(filename, src) in ALL_SHADERS {
        let structs = parse_wgsl_structs(filename, src);
        for (struct_name, s) in structs {
            if let Some((first_file, ref canonical)) = canonical_structs.get(&struct_name) {
                assert_eq!(
                    s.size, canonical.size,
                    "Struct '{}' size inconsistent between '{}' ({}B) and '{}' ({}B)",
                    struct_name, first_file, canonical.size, filename, s.size
                );
                assert_eq!(
                    s.members, canonical.members,
                    "Struct '{}' member layout inconsistent between '{}' and '{}'!\nLeft ({:?}):\n{:#?}\nRight ({:?}):\n{:#?}",
                    struct_name, first_file, filename, first_file, canonical.members, filename, s.members
                );
            } else {
                canonical_structs.insert(struct_name, (filename, s));
            }
        }
    }
}

macro_rules! verify_wgsl_struct {
    ($file:expr, $structs:expr, $hits:expr, $wgsl_name:literal => $rust_type:ty { $($field:ident),+ $(,)? }) => {
        if let Some(s) = $structs.get($wgsl_name) {
            *$hits.entry($wgsl_name).or_insert(0) += 1;
            check_struct_size!($file, s, $rust_type);
            $(
                check_member!($file, s, $rust_type, $field);
            )+
        }
    };
}

#[test]
fn test_rust_and_wgsl_struct_alignment() {
    const REQUIRED_BUFFER_STRUCTS: &[&str] = &[
        "GpuVertex",
        "SimParams",
        "CollisionParams",
        "GpuBoneInfo",
        "GpuBoneTransform",
        "GpuEdge",
        "SelfCollisionParams",
        "SpatialHashParams",
        "PinParams",
        "NormalParams",
        "DispatchInfo",
        "GpuCollider",
        "GpuMeshTriangle",
        "GpuDistanceConstraint",
        "GpuBendingConstraint",
        "GpuPinConstraint",
        "GpuSewingConstraint",
        "GpuStarPair",
        "PairCollectParams",
        "PairCounters",
        "SolveParams",
        "RestVertex",
        "BoneTriangleSource",
        "BakeParams",
        "BakeTriangle",
        "BakeMeshParams",
        "BakeMeshTriangle",
        "GpuColliderEdge",
        "EdgeCollisionParams",
        "WrinkleFieldParams",
        "HierFineParams",
        "GpuVirtualVertexDef",
        "VirtualForwardParams",
        "SelfCollisionVvParams",
        "SelfCollisionAccum",
        "GpuVtPair",
        "GpuEePair",
        "AtomicAccum",
    ];
    // WGSL 関数内部のローカル変数・戻り値専用構造体（GPU バッファ境界に載らないもの）
    const LOCAL_HELPER_STRUCTS: &[&str] = &[
        "ClosestResult",
        "SegmentDistanceResult",
        "ClosestBaryResult",
        "SegmentEval",
        "TwoHopResult",
        "ParticleInfo",
    ];

    let mut hit_counts: HashMap<&'static str, usize> = HashMap::new();
    let mut seen_wgsl_structs: HashSet<String> = HashSet::new();

    for &(filename, src) in ALL_SHADERS {
        let structs = parse_wgsl_structs(filename, src);
        for name in structs.keys() {
            seen_wgsl_structs.insert(name.clone());
        }

        verify_wgsl_struct!(filename, structs, hit_counts, "GpuVertex" => GpuVertex {
            position, inv_mass, prev_pos, layer_id, velocity, thickness
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "SimParams" => SimParams {
            gravity, damping, substeps, num_vertices, num_distance_constraints,
            num_bending_constraints, num_sewing_constraints, sewing_compliance,
            enable_sewing_lock, sewing_lock_distance, tension_damping,
            compression_damping, shear_damping, bending_damping
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "CollisionParams" => CollisionParams {
            num_vertices, num_colliders, num_mesh_triangles, num_clusters, dt,
            edge_margin_scale, edge_margin_offset, enable_cluster_culling,
            enable_single_sided_recovery, sweep_margin_offset, num_bones, enable_bone_sdf
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "GpuBoneInfo" => GpuBoneInfo {
            aabb_min, aabb_max, uvw_scale, uvw_offset, params
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "GpuBoneTransform" => GpuBoneTransform {
            world_matrix, inv_world_matrix
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "GpuEdge" => GpuEdge {
            v0, v1
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "SelfCollisionParams" => SelfCollisionParams {
            cell_size, table_size, num_vertices, num_edges, relief_factor,
            max_displacement_ratio, enable_relief, enable_normal_untangling,
            exclude_neighbors, max_search_iterations, enable_ee, _pad3
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "SpatialHashParams" => SpatialHashParams {
            cell_size, table_size, num_vertices, num_blocks
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "PinParams" => PinParams {
            num_pins
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "NormalParams" => NormalParams {
            num_vertices
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "DispatchInfo" => DispatchInfo {
            color_offset, color_count
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "GpuCollider" => GpuCollider {
            collider_type, friction, restitution, point_a, radius, point_b
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "GpuMeshTriangle" => GpuMeshTriangle {
            p0, friction, p1, thickness, p2, restitution, flags
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "GpuDistanceConstraint" => GpuDistanceConstraint {
            v0, v1, rest_length, tension_compliance, compression_compliance, constraint_type
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "GpuBendingConstraint" => GpuBendingConstraint {
            v0, v1, v2, v3, rest_length, compliance
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "GpuPinConstraint" => GpuPinConstraint {
            vertex_idx, weight, target_pos
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "GpuSewingConstraint" => GpuSewingConstraint {
            v0, v1, current_rest_len, target_rest_len, shrink_speed, compliance, lock_on_close
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "GpuStarPair" => GpuStarPair {
            v0, v1
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "PairCollectParams" => PairCollectParams {
            cell_size, table_size, num_vertices, max_vt_pairs, max_ee_pairs,
            safety_margin, exclude_neighbors, margin_mode, dt_frame,
            velocity_horizon_scale, max_horizon, quota_vt, quota_ee
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "PairCounters" => PairCounters {
            vt_count, ee_count, vt_dropped, ee_dropped
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "SolveParams" => PairSolveParams {
            num_vertices, max_vt_pairs, max_ee_pairs, enable_normal_untangling
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "RestVertex" => GpuSkinningVertex {
            pos, _pad0, normal, _pad1, bone_indices, bone_weights
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "BoneTriangleSource" => GpuBoneTriangleSource {
            i0, i1, i2, bone_idx, w0, w1, w2, _pad
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "BakeParams" => GpuBakeParams {
            local_min, tri_start, local_max, tri_count, tile_col, tile_row,
            tile_layer, res, total_width, total_height, total_depth, row_pitch
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "BakeTriangle" => GpuBakeTriangle {
            p0, w0, p1, w1, p2, w2, normal, _pad
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "BakeMeshParams" => GpuBakeMeshParams {
            local_min, tri_count, local_max, row_pitch, width, height, depth, _pad0
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "BakeMeshTriangle" => GpuBakeMeshTriangle {
            p0, _pad0, p1, _pad1, p2, _pad2, normal, _pad3
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "GpuColliderEdge" => GpuColliderEdge {
            p0, thickness, p1, friction, normal, flags
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "EdgeCollisionParams" => EdgeCollisionParams {
            cell_size, table_size, num_cloth_edges, num_collider_edges,
            edge_margin_scale, edge_margin_offset
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "WrinkleFieldParams" => WrinkleFieldParams {
            bone_origin, bone_axis, bone_normal, bone_binormal, z_range, r_range,
            enabled, use_texture, valley_window, crest_window
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "HierFineParams" => GpuHierFineParams {
            coarse_w, coarse_h, coarse_d, band, z_offset, _pad0, _pad1, _pad2
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "GpuVirtualVertexDef" => GpuVirtualVertexDef {
            parent_indices, bary_weights
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "VirtualForwardParams" => VirtualForwardParams {
            num_real_vertices, num_virtual_vertices, _pad0, _pad1
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "SelfCollisionVvParams" => SelfCollisionVvParams {
            cell_size, table_size, num_real_vertices, num_total_particles,
            relief_factor, max_displacement_ratio, _pad0, _pad1
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "SelfCollisionAccum" => GpuSelfCollisionAccum {
            dx, dy, dz, count, pin_dx, pin_dy, pin_dz, pin_count,
            ccd_pos_x, ccd_pos_y, ccd_pos_z, ccd_neg_x, ccd_neg_y, ccd_neg_z,
            untangle_cand_v, ee_ccd_count
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "GpuVtPair" => GpuVtPair {
            vert_i, vert_j, v0, v1
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "GpuEePair" => GpuEePair {
            v_i, v_ui, v_j, v_vj
        });
        verify_wgsl_struct!(filename, structs, hit_counts, "AtomicAccum" => GpuAtomicAccum {
            dx, dy, dz, count
        });
    }

    // 空振り (Vacuous Truth) 防止アサーション: 全38バッファ構造体が最低1回はWGSLと突合されること
    for required in REQUIRED_BUFFER_STRUCTS {
        assert!(
            hit_counts.get(required).copied().unwrap_or(0) > 0,
            "Struct '{}' was never checked against any WGSL shader (vacuous assertion)!",
            required
        );
    }
    assert_eq!(
        hit_counts.get("SelfCollisionAccum").copied().unwrap_or(0),
        8,
        "SelfCollisionAccum は全8つの自己衝突/エッジ衝突シェーダーで一致検証されるべき"
    );

    // 未知 WGSL 構造体の追加漏れ防止アサーション
    let allowed: HashSet<&str> = REQUIRED_BUFFER_STRUCTS
        .iter()
        .chain(LOCAL_HELPER_STRUCTS.iter())
        .copied()
        .collect();
    let unknown: Vec<String> = seen_wgsl_structs
        .into_iter()
        .filter(|name| !allowed.contains(name.as_str()))
        .collect();
    assert!(
        unknown.is_empty(),
        "test_shader_alignment.rs で未網羅の WGSL 構造体が検出されました: {:?}",
        unknown
    );
}

