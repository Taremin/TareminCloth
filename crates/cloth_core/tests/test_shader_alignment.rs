use std::collections::HashMap;
use std::mem::{offset_of, size_of};

use cloth_core::mesh::{
    GpuBendingConstraint, GpuCollider, GpuDistanceConstraint, GpuMeshTriangle, GpuPinConstraint,
    GpuSewingConstraint, GpuStarPair, GpuVertex, SelfCollisionParams, SimParams,
};
use cloth_core::simulation::types::{CollisionParams, DispatchInfo, NormalParams, PinParams};
use cloth_core::spatial_hash::SpatialHashParams;

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
    ("predict.wgsl", include_str!("../src/shaders/predict.wgsl")),
    ("update_vel.wgsl", include_str!("../src/shaders/update_vel.wgsl")),
    ("distance.wgsl", include_str!("../src/shaders/distance.wgsl")),
    ("distance_atomic.wgsl", include_str!("../src/shaders/distance_atomic.wgsl")),
    ("bending.wgsl", include_str!("../src/shaders/bending.wgsl")),
    ("sewing.wgsl", include_str!("../src/shaders/sewing.wgsl")),
    ("pin.wgsl", include_str!("../src/shaders/pin.wgsl")),
    ("spatial_hash_build.wgsl", include_str!("../src/shaders/spatial_hash_build.wgsl")),
    ("compute_normals.wgsl", include_str!("../src/shaders/compute_normals.wgsl")),
    ("extract_positions.wgsl", include_str!("../src/shaders/extract_positions.wgsl")),
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

#[test]
fn test_rust_and_wgsl_struct_alignment() {
    for &(filename, src) in ALL_SHADERS {
        let structs = parse_wgsl_structs(filename, src);

        if let Some(s) = structs.get("GpuVertex") {
            check_struct_size!(filename, s, GpuVertex);
            check_member!(filename, s, GpuVertex, position);
            check_member!(filename, s, GpuVertex, inv_mass);
            check_member!(filename, s, GpuVertex, prev_pos);
            check_member!(filename, s, GpuVertex, layer_id);
            check_member!(filename, s, GpuVertex, velocity);
            check_member!(filename, s, GpuVertex, thickness);
        }

        if let Some(s) = structs.get("SimParams") {
            check_struct_size!(filename, s, SimParams);
            check_member!(filename, s, SimParams, gravity);
            check_member!(filename, s, SimParams, damping);
            check_member!(filename, s, SimParams, substeps);
            check_member!(filename, s, SimParams, num_vertices);
            check_member!(filename, s, SimParams, num_distance_constraints);
            check_member!(filename, s, SimParams, num_bending_constraints);
            check_member!(filename, s, SimParams, num_sewing_constraints);
        }

        if let Some(s) = structs.get("CollisionParams") {
            check_struct_size!(filename, s, CollisionParams);
            check_member!(filename, s, CollisionParams, num_vertices);
            check_member!(filename, s, CollisionParams, num_colliders);
            check_member!(filename, s, CollisionParams, num_mesh_triangles);
            check_member!(filename, s, CollisionParams, num_clusters);
            check_member!(filename, s, CollisionParams, dt);
            check_member!(filename, s, CollisionParams, edge_margin_scale);
            check_member!(filename, s, CollisionParams, edge_margin_offset);
            check_member!(filename, s, CollisionParams, enable_cluster_culling);
            check_member!(filename, s, CollisionParams, enable_single_sided_recovery);
            check_member!(filename, s, CollisionParams, sweep_margin_offset);
        }

        if let Some(s) = structs.get("SelfCollisionParams") {
            check_struct_size!(filename, s, SelfCollisionParams);
            check_member!(filename, s, SelfCollisionParams, cell_size);
            check_member!(filename, s, SelfCollisionParams, table_size);
            check_member!(filename, s, SelfCollisionParams, num_vertices);
            check_member!(filename, s, SelfCollisionParams, relief_factor);
            check_member!(filename, s, SelfCollisionParams, max_displacement_ratio);
            check_member!(filename, s, SelfCollisionParams, enable_relief);
            check_member!(filename, s, SelfCollisionParams, enable_normal_untangling);
            check_member!(filename, s, SelfCollisionParams, exclude_neighbors);
            check_member!(filename, s, SelfCollisionParams, max_search_iterations);
        }

        if let Some(s) = structs.get("SpatialHashParams") {
            check_struct_size!(filename, s, SpatialHashParams);
            check_member!(filename, s, SpatialHashParams, cell_size);
            check_member!(filename, s, SpatialHashParams, table_size);
            check_member!(filename, s, SpatialHashParams, num_vertices);
        }

        if let Some(s) = structs.get("PinParams") {
            check_struct_size!(filename, s, PinParams);
            check_member!(filename, s, PinParams, num_pins);
        }

        if let Some(s) = structs.get("NormalParams") {
            check_struct_size!(filename, s, NormalParams);
            check_member!(filename, s, NormalParams, num_vertices);
        }

        if let Some(s) = structs.get("DispatchInfo") {
            check_struct_size!(filename, s, DispatchInfo);
            check_member!(filename, s, DispatchInfo, color_offset);
            check_member!(filename, s, DispatchInfo, color_count);
        }

        if let Some(s) = structs.get("GpuCollider") {
            check_struct_size!(filename, s, GpuCollider);
            check_member!(filename, s, GpuCollider, collider_type);
            check_member!(filename, s, GpuCollider, friction);
            check_member!(filename, s, GpuCollider, restitution);
            check_member!(filename, s, GpuCollider, point_a);
            check_member!(filename, s, GpuCollider, radius);
            check_member!(filename, s, GpuCollider, point_b);
        }

        if let Some(s) = structs.get("GpuMeshTriangle") {
            check_struct_size!(filename, s, GpuMeshTriangle);
            check_member!(filename, s, GpuMeshTriangle, p0);
            check_member!(filename, s, GpuMeshTriangle, friction);
            check_member!(filename, s, GpuMeshTriangle, p1);
            check_member!(filename, s, GpuMeshTriangle, thickness);
            check_member!(filename, s, GpuMeshTriangle, p2);
            check_member!(filename, s, GpuMeshTriangle, restitution);
            check_member!(filename, s, GpuMeshTriangle, flags);
        }

        if let Some(s) = structs.get("GpuDistanceConstraint") {
            check_struct_size!(filename, s, GpuDistanceConstraint);
            check_member!(filename, s, GpuDistanceConstraint, v0);
            check_member!(filename, s, GpuDistanceConstraint, v1);
            check_member!(filename, s, GpuDistanceConstraint, rest_length);
            check_member!(filename, s, GpuDistanceConstraint, tension_compliance);
            check_member!(filename, s, GpuDistanceConstraint, compression_compliance);
            check_member!(filename, s, GpuDistanceConstraint, constraint_type);
        }

        if let Some(s) = structs.get("GpuBendingConstraint") {
            check_struct_size!(filename, s, GpuBendingConstraint);
            check_member!(filename, s, GpuBendingConstraint, v0);
            check_member!(filename, s, GpuBendingConstraint, v1);
            check_member!(filename, s, GpuBendingConstraint, v2);
            check_member!(filename, s, GpuBendingConstraint, v3);
            check_member!(filename, s, GpuBendingConstraint, rest_length);
            check_member!(filename, s, GpuBendingConstraint, compliance);
        }

        if let Some(s) = structs.get("GpuPinConstraint") {
            check_struct_size!(filename, s, GpuPinConstraint);
            check_member!(filename, s, GpuPinConstraint, vertex_idx);
            check_member!(filename, s, GpuPinConstraint, weight);
            check_member!(filename, s, GpuPinConstraint, target_pos);
        }

        if let Some(s) = structs.get("GpuSewingConstraint") {
            check_struct_size!(filename, s, GpuSewingConstraint);
            check_member!(filename, s, GpuSewingConstraint, v0);
            check_member!(filename, s, GpuSewingConstraint, v1);
            check_member!(filename, s, GpuSewingConstraint, current_rest_len);
            check_member!(filename, s, GpuSewingConstraint, target_rest_len);
            check_member!(filename, s, GpuSewingConstraint, shrink_speed);
            check_member!(filename, s, GpuSewingConstraint, compliance);
        }

        if let Some(s) = structs.get("GpuStarPair") {
            check_struct_size!(filename, s, GpuStarPair);
            check_member!(filename, s, GpuStarPair, v0);
            check_member!(filename, s, GpuStarPair, v1);
        }
    }
}
