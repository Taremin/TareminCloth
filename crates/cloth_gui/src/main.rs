use std::env;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use winit::event_loop::EventLoop;

mod protocol;
mod server;
mod camera;
mod mesh_render;
mod demo_scenes;
mod app;
mod coupled_mode;

use cloth_core::mesh::ClothMesh;
use cloth_core::simulation::GpuClothSimulator;
use cloth_core::GpuContext;
use protocol::GuiCommand;
use server::TcpServerHandle;
use app::GuiApp;
use demo_scenes::DemoSceneType;

#[cfg(target_os = "windows")]
#[link(name = "winmm")]
extern "system" {
    fn timeBeginPeriod(period: u32) -> u32;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "windows")]
    unsafe {
        timeBeginPeriod(1);
    }

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,taremin_cloth_gui=info")
    ).init();
    log::info!("=== Starting Taremin Cloth GUI ===");

    let mut port = 9055u16;
    let mut headless = false;
    let mut initial_demo = DemoSceneType::SphereDraping;

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" => {
                if i + 1 < args.len() {
                    if let Ok(p) = args[i + 1].parse::<u16>() {
                        port = p;
                        i += 1;
                    }
                }
            }
            "--headless" => {
                headless = true;
            }
            "--demo" | "-d" => {
                if i + 1 < args.len() {
                    let d = args[i + 1].to_lowercase();
                    initial_demo = match d.as_str() {
                        "none" => DemoSceneType::None,
                        "sphere" | "sphere_draping" | "1" => DemoSceneType::SphereDraping,
                        "curtain" | "two_point_curtain" | "2" => DemoSceneType::TwoPointCurtain,
                        "folding" | "ground_folding" | "accordion" | "3" => DemoSceneType::GroundFolding,
                        "layer" | "multi_layer" | "4" => DemoSceneType::MultiLayerCloth,
                        "twisting" | "twist" | "5" => DemoSceneType::ClothTwisting,
                        "funnel" | "funnel_pass" | "6" => DemoSceneType::FunnelPass,
                        "sewing" | "garment_sewing" | "7" => DemoSceneType::GarmentSewing,
                        _ => {
                            log::warn!("Unknown demo name: {}, falling back to SphereDraping", d);
                            DemoSceneType::SphereDraping
                        }
                    };
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let bind_addr = format!("127.0.0.1:{}", port);
    let server_handle = TcpServerHandle::start(&bind_addr)?;

    if headless {
        log::info!("[Headless] Running in headless server mode on {}", bind_addr);
        run_headless_loop(server_handle)?;
    } else {
        log::info!("[GUI] Launching GUI window with initial demo: {:?}", initial_demo);
        let event_loop = EventLoop::new()?;
        let mut app = GuiApp::new(server_handle, initial_demo);
        event_loop.run_app(&mut app)?;
        app.cleanup();
    }

    // 共有パイプラインキャッシュとグローバルGPUコンテキストを明示的に解放
    cloth_core::simulation::pipeline_cache::clear_cache();
    cloth_core::GpuContext::clear_global_context();

    log::info!("=== Taremin Cloth GUI Exited Normally ===");
    Ok(())
}

/// ヘッドレスサーバーモード（ウィンドウなし・テストおよびバッチ処理用）
fn run_headless_loop(server: TcpServerHandle) -> Result<(), Box<dyn std::error::Error>> {
    let gpu_context = GpuContext::get_or_init()?;
    let mut simulator: Option<GpuClothSimulator> = None;
    let mut is_running = false;
    let mut frame_seq = 0u64;

    loop {
        // コマンド/タスク処理
        while let Ok(task) = server.task_rx.try_recv() {
            match task {
                server::MainThreadTask::InitScene { data, respond_to } => {
                    log::info!("[Headless] InitScene starting: obj={}, verts={}", data.object_name, data.positions.len());
                    let faces_opt = if !data.faces.is_empty() { Some(data.faces.as_slice()) } else { None };
                    let sew_opt = data.sewing_springs.as_deref();

                    let mesh = ClothMesh::from_raw_with_opts(
                        &data.positions,
                        &data.edges,
                        faces_opt,
                        Some(&data.inv_masses),
                        sew_opt,
                        data.layer_ids.as_deref(),
                        None,
                        data.layer_id,
                        data.thickness,
                        data.stiffness,
                        data.compression_stiffness,
                        data.shear_stiffness,
                        data.bending_stiffness,
                        data.sewing_shrink_speed,
                        Some(data.sewing_stiffness),
                        Some(data.enable_sewing_lock),
                        data.enable_coarse_constraints,
                    );


                    let mut sim = GpuClothSimulator::with_options(
                        Arc::clone(&gpu_context),
                        mesh,
                        data.workgroup_size,
                        data.solver_mode,
                    );

                    // 全シミュレーションパラメータ、コライダー、ボーンSDFの登録
                    apply_simulation_parameters(&mut sim, &data);

                    simulator = Some(sim);
                    is_running = true;
                    frame_seq = 0;

                    if let Ok(mut state) = server.shared_state.write() {
                        state.num_vertices = data.positions.len() as u32;
                        state.latest_coords = data.positions;
                        state.is_running = true;
                        state.frame_seq = 0;
                    }
                    log::info!("[Headless] InitScene completed successfully (colliders: {}, mesh_tris: {}, bone_sdf: {})", data.colliders.len(), data.mesh_triangles.len(), data.bone_sdf.is_some());
                    let _ = respond_to.send(protocol::GuiResponse::Ack { message: "OK".to_string() });
                }
                server::MainThreadTask::Step { dt, substeps, solver_iterations, respond_to } => {
                    if let Some(ref mut sim) = simulator {
                        sim.solver_iterations = solver_iterations;
                        sim.step(dt, substeps);
                        frame_seq += 1;
                        if let Ok(mut state) = server.shared_state.write() {
                            state.frame_seq = frame_seq;
                            let mut coords = vec![0.0f32; sim.num_vertices as usize * 3];
                            sim.get_positions_flat(&mut coords);
                            state.latest_coords = coords.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
                        }
                    }
                    let _ = respond_to.send(protocol::GuiResponse::Ack { message: "OK".to_string() });
                }
                server::MainThreadTask::Command(cmd) => match cmd {
                    GuiCommand::Play => is_running = true,
                    GuiCommand::Pause => is_running = false,
                    GuiCommand::Reset => {
                        if let Some(ref mut sim) = simulator {
                            sim.reset();
                            frame_seq = 0;
                        }
                    }
                    GuiCommand::UpdateColliders { colliders, mesh_triangles } => {
                        if let Some(ref mut sim) = simulator {
                            register_colliders(sim, &colliders, &mesh_triangles);
                        }
                    }
                    GuiCommand::UpdateBoneTransforms { transforms } => {
                        if let Some(ref mut sim) = simulator {
                            let gpu_transforms: Vec<cloth_core::simulation::GpuBoneTransform> = transforms.iter().map(|mat| {
                                let m = glam::Mat4::from_cols_array_2d(mat);
                                let inv_m = m.inverse();
                                cloth_core::simulation::GpuBoneTransform {
                                    world_matrix: *mat,
                                    inv_world_matrix: inv_m.to_cols_array_2d(),
                                }
                            }).collect();
                            sim.update_bone_transforms(&gpu_transforms);
                        }
                    }
                    GuiCommand::UpdateElasticScales { edge_indices, scales } => {
                        if let Some(ref mut sim) = simulator {
                            sim.set_edge_rest_length_scales(&edge_indices, &scales);
                        }
                    }
                    GuiCommand::UpdatePins { vertex_indices, positions, weights } => {
                        if let Some(ref mut sim) = simulator {
                            for i in 0..vertex_indices.len() {
                                sim.set_pin_target(vertex_indices[i], positions[i], weights[i]);
                            }
                        }
                    }
                    GuiCommand::SetParams(p) => {
                        if let Some(ref mut sim) = simulator {
                            let mut cfg = sim.export_config();
                            p.apply_to_sim_config(&mut cfg);
                            sim.apply_config(&cfg);
                        }
                    }
                    GuiCommand::Quit => {
                        log::info!("[Headless] Quit requested");
                        return Ok(());
                    }
                    _ => {}
                },
            }
        }

        // 継続物理ステップ
        if is_running {
            if let Some(ref mut sim) = simulator {
                let t0 = std::time::Instant::now();
                sim.step(1.0 / 60.0, 15);
                let elapsed = t0.elapsed().as_secs_f32();
                frame_seq += 1;

                if let Ok(mut state) = server.shared_state.write() {
                    state.frame_seq = frame_seq;
                    state.step_time_ms = elapsed * 1000.0;
                    state.physics_fps = if elapsed > 0.0 { 1.0 / elapsed } else { 0.0 };

                    if state.is_connected {
                        let mut coords = vec![0.0f32; sim.num_vertices as usize * 3];
                        sim.get_positions_flat(&mut coords);
                        state.latest_coords = coords.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
                    }
                }
            }
        }

        thread::sleep(Duration::from_millis(2));
    }
}

fn apply_simulation_parameters(sim: &mut GpuClothSimulator, data: &protocol::SceneInitData) {
    let cfg = data.to_sim_config();
    sim.apply_config(&cfg);

    // 伸縮グループ
    if let Some(ref eb) = data.elastic_bands {
        if !eb.edge_indices.is_empty() {
            sim.set_edge_rest_length_scales(&eb.edge_indices, &eb.scales);
        }
    }

    // コライダー登録
    register_colliders(sim, &data.colliders, &data.mesh_triangles);

    // ボーンSDF登録
    if let Some(ref bsdf) = data.bone_sdf {
        register_bone_sdf(sim, bsdf);
    }
}

fn register_bone_sdf(sim: &mut GpuClothSimulator, bone_sdf: &protocol::GuiBoneSdfData) {
    use base64::Engine;
    if let Ok(tex_bytes) = base64::engine::general_purpose::STANDARD.decode(&bone_sdf.texture_base64) {
        let bone_infos: Vec<cloth_core::simulation::GpuBoneInfo> = bone_sdf.bone_infos.iter().map(|raw| {
            cloth_core::simulation::GpuBoneInfo {
                aabb_min: [raw[0], raw[1], raw[2], raw[3]],
                aabb_max: [raw[4], raw[5], raw[6], raw[7]],
                uvw_scale: [raw[8], raw[9], raw[10], raw[11]],
                uvw_offset: [raw[12], raw[13], raw[14], raw[15]],
                params: [raw[16], raw[17], raw[18], raw[19]],
            }
        }).collect();
        sim.set_bone_sdf_colliders(
            bone_sdf.width,
            bone_sdf.height,
            bone_sdf.depth,
            &tex_bytes,
            &bone_infos,
        );
        if !bone_sdf.bone_transforms.is_empty() {
            let gpu_transforms: Vec<cloth_core::simulation::GpuBoneTransform> = bone_sdf.bone_transforms.iter().map(|mat| {
                let m = glam::Mat4::from_cols_array_2d(mat);
                let inv_m = m.inverse();
                cloth_core::simulation::GpuBoneTransform {
                    world_matrix: *mat,
                    inv_world_matrix: inv_m.to_cols_array_2d(),
                }
            }).collect();
            sim.update_bone_transforms(&gpu_transforms);
            log::info!("[Headless] Applied {} initial bone transforms for Bone SDF", gpu_transforms.len());
        }
        log::info!("[Headless] Registered Bone SDF 3D Texture: {}x{}x{} ({} bones)", bone_sdf.width, bone_sdf.height, bone_sdf.depth, bone_infos.len());
    } else {
        log::error!("[Headless] Failed to decode Bone SDF texture base64 data");
    }
}

fn register_colliders(
    sim: &mut GpuClothSimulator,
    colliders: &[protocol::GuiColliderData],
    mesh_triangles: &[protocol::GuiMeshTriangleData],
) {
    sim.clear_colliders();
    for col in colliders {
        match col.collider_type {
            0 => sim.add_sphere_collider(col.point_a, col.radius, col.friction, col.restitution),
            1 => sim.add_capsule_collider(col.point_a, col.point_b, col.radius, col.friction, col.restitution),
            2 => sim.add_plane_collider(col.point_a, col.point_b, col.friction, col.restitution),
            _ => {}
        }
    }
    if !mesh_triangles.is_empty() {
        let gpu_tris: Vec<cloth_core::mesh::GpuMeshTriangle> = mesh_triangles.iter().map(|t| {
            cloth_core::mesh::GpuMeshTriangle {
                p0: t.p0,
                friction: t.friction,
                p1: t.p1,
                thickness: t.thickness,
                p2: t.p2,
                restitution: t.restitution,
                flags: t.flags,
                _pad: [0.0; 3],
            }
        }).collect();
        sim.set_mesh_triangles(&gpu_tris);
    }
}

