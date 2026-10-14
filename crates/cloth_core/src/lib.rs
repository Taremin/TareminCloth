pub mod context;
pub mod config;
pub mod mesh;
pub mod coloring;
pub use coloring::color_edge_pairs;
pub mod geometry;
pub use geometry::{find_proximity_violations, find_triangle_intersections, tri_tri_intersect};
pub mod pair_audit;
pub use pair_audit::{audit_pairs, AuditResult};
pub mod smooth;
pub mod spatial_hash;
pub mod simulation;
pub mod virtual_mesh;
pub mod debug_recorder;
pub mod renderer;
pub mod sdf_baker;
pub mod pattern_remesh;
pub mod quad_remesh;

pub use pattern_remesh::{polygon_signed_area_2d, remesh_pattern_2d_core, triangle_signed_area_2d};
pub use quad_remesh::{remesh_cloth_to_quad_3d_core, QuadRemeshOutput};

pub use context::{GpuBufferLimits, GpuContext, GpuContextError, GpuDeviceInfo};
pub use config::SimConfig;
pub use mesh::{ClothMesh, GpuBendingConstraint, GpuCollider, GpuDistanceConstraint, GpuMeshTriangle, GpuPinConstraint, GpuSewingConstraint, GpuVertex, SimParams, areal_inv_masses};
pub use smooth::{
    TENSION_EXPAND, GRAB_PIN_THRESHOLD,
    FALLOFF_SMOOTH, FALLOFF_SPHERE, FALLOFF_SHARP, FALLOFF_LINEAR, FALLOFF_CONSTANT,
    build_adjacency_csr, laplacian_smooth_targets, radial_expand_targets,
    falloff_weights, verts_in_brush, depth_keep_mask, select_grab_pins,
    radial_adjust, radial_adjust_with_scale, grab_drag_targets, ray_plane_hit,
};
pub use simulation::{DynamicBoneSdfSetup, GpuClothSimulator};
pub use simulation::types::{GpuBoneInfo, GpuBoneTransform, GpuBoneTriangleSource, GpuSkinningVertex};
pub use debug_recorder::{BoneFrameState, BoneSdfRecord, ColliderRecord, DebugRecordOptions, ElasticScaleRecord, FrameRecord, FrameStats, JsonlRecord, PinRecord, SimulationDebugRecorder, SimulationMetadata};
pub use renderer::{
    render_mesh_to_buffer, render_mesh_to_png_file, render_scene_to_buffer,
    render_scene_to_png_file, RenderOptions, RenderResult, SceneData,
};
pub use sdf_baker::{
    bake_bone_sdf_gpu, bake_mesh_sdf_coarse, bake_mesh_sdf_gpu, bake_mesh_sdf_hierarchical,
    BoneInput, GpuBakeMeshParams, GpuBakeMeshTriangle, GpuHierFineParams, GpuMeshSdfBakeResult,
    GpuMeshSdfCoarseResult, GpuBakeParams, GpuBakeTriangle, GpuBoneSdfBakeResult,
    MESH_SDF_HIER_RATIO, MESH_SDF_HIER_SLAB_LAYERS,
};

