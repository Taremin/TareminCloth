pub mod context;
pub mod mesh;
pub mod coloring;
pub mod spatial_hash;
pub mod simulation;
pub mod debug_recorder;
pub mod renderer;

pub use context::{GpuContext, GpuContextError, GpuDeviceInfo};
pub use mesh::{ClothMesh, GpuBendingConstraint, GpuCollider, GpuDistanceConstraint, GpuMeshTriangle, GpuPinConstraint, GpuSewingConstraint, GpuVertex, SimParams};
pub use simulation::GpuClothSimulator;
pub use debug_recorder::{ColliderRecord, FrameRecord, FrameStats, JsonlRecord, PinRecord, SimulationDebugRecorder, SimulationMetadata};
pub use renderer::{render_mesh_to_buffer, render_mesh_to_png_file, RenderOptions, RenderResult};

