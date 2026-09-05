pub mod context;
pub mod mesh;
pub mod coloring;
pub mod spatial_hash;
pub mod simulation;
pub mod debug_recorder;
pub mod renderer;
pub mod sdf_baker;

pub use context::{GpuContext, GpuContextError, GpuDeviceInfo};
pub use mesh::{ClothMesh, GpuBendingConstraint, GpuCollider, GpuDistanceConstraint, GpuMeshTriangle, GpuPinConstraint, GpuSewingConstraint, GpuVertex, SimParams};
pub use simulation::{DynamicBoneSdfSetup, GpuClothSimulator};
pub use simulation::types::{GpuBoneInfo, GpuBoneTransform, GpuBoneTriangleSource, GpuSkinningVertex};
pub use debug_recorder::{ColliderRecord, FrameRecord, FrameStats, JsonlRecord, PinRecord, SimulationDebugRecorder, SimulationMetadata};
pub use renderer::{
    render_mesh_to_buffer, render_mesh_to_png_file, render_scene_to_buffer,
    render_scene_to_png_file, RenderOptions, RenderResult, SceneData,
};
pub use sdf_baker::{bake_bone_sdf_gpu, BoneInput, GpuBakeParams, GpuBakeTriangle, GpuBoneSdfBakeResult};

