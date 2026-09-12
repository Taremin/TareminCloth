use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use cloth_core::mesh::ClothMesh;
use cloth_core::simulation::GpuClothSimulator;
use cloth_core::GpuContext;

use crate::camera::OrbitCamera;
use crate::mesh_render::MeshRenderer;
use crate::protocol::{GuiCommand, GuiResponse, SceneInitData};
use crate::server::{MainThreadTask, TcpServerHandle};

pub struct GuiApp {
    server: TcpServerHandle,
    window: Option<Arc<Window>>,
    gpu_context: Option<Arc<GpuContext>>,
    surface: Option<wgpu::Surface<'static>>,
    surface_config: Option<wgpu::SurfaceConfiguration>,
    depth_texture: Option<wgpu::TextureView>,
    renderer: Option<MeshRenderer>,
    egui_renderer: Option<egui_wgpu::Renderer>,
    egui_state: Option<egui_winit::State>,
    egui_ctx: egui::Context,
    camera: OrbitCamera,

    // 物理シミュレーター
    simulator: Option<GpuClothSimulator>,
    is_running: bool,
    frame_seq: u64,
    last_sim_time: Instant,
    time_accumulator: f32,
    physics_fps: f32,
    physics_frame_ms: f32,
    physics_max_fps: f32,
    sim_step_count: u32,
    last_fps_calc_time: Instant,
    render_fps: f32,
    render_frame_ms: f32,
    render_max_fps: f32,
    render_count: u32,

    // UI設定
    draw_cloth: bool,
    draw_wireframe: bool,
    draw_colliders: bool,
    draw_voxels: bool,
    draw_bbox: bool,
    highlight_backface: bool,
    base_gravity: [f32; 3],
    gravity_scale: f32,
    gravity: [f32; 3],
    stiffness: f32,
    tension_stiffness: f32,
    compression_stiffness: f32,
    shear_stiffness: f32,
    bending_stiffness: f32,
    air_damping: f32,
    tension_damping: f32,
    compression_damping: f32,
    shear_damping: f32,
    bending_damping: f32,
    substeps: u32,
    solver_iterations: u32,
    target_fps: f32,
    scene_fps: f32,
    next_frame_time: Instant,

    // マウス入力 & インタラクティブ操作 (Grab / Pin)
    is_orbiting: bool,
    is_panning: bool,
    last_mouse_pos: (f32, f32),
    grabbed_vert: Option<u32>,
    grab_initial_pos: glam::Vec3,
    grab_plane_point: glam::Vec3,
    grab_plane_normal: glam::Vec3,
    grab_initial_hit: glam::Vec3,
    grab_target_pos: Option<glam::Vec3>,
    pinned_verts: HashMap<u32, [f32; 3]>,
    cached_positions: Vec<[f32; 3]>,
}


impl GuiApp {
    pub fn new(server: TcpServerHandle) -> Self {
        let egui_ctx = egui::Context::default();
        Self {
            server,
            window: None,
            gpu_context: None,
            surface: None,
            surface_config: None,
            depth_texture: None,
            renderer: None,
            egui_renderer: None,
            egui_state: None,
            egui_ctx,
            camera: OrbitCamera::default(),
            simulator: None,
            is_running: false,
            frame_seq: 0,
            last_sim_time: Instant::now(),
            time_accumulator: 0.0,
            physics_fps: 0.0,
            physics_frame_ms: 0.0,
            physics_max_fps: 0.0,
            sim_step_count: 0,
            last_fps_calc_time: Instant::now(),
            render_fps: 0.0,
            render_frame_ms: 0.0,
            render_max_fps: 0.0,
            render_count: 0,
            draw_cloth: true,
            draw_wireframe: false,
            draw_colliders: true,
            draw_voxels: true,
            draw_bbox: true,
            highlight_backface: true,
            base_gravity: [0.0, 0.0, -9.81],
            gravity_scale: 1.0,
            gravity: [0.0, 0.0, -9.81],
            stiffness: 1000.0,
            tension_stiffness: 1000.0,
            compression_stiffness: 100.0,
            shear_stiffness: 100.0,
            bending_stiffness: 10.0,
            air_damping: 1.0,
            tension_damping: 5.0,
            compression_damping: 5.0,
            shear_damping: 5.0,
            bending_damping: 0.5,
            substeps: 20,
            solver_iterations: 2,
            target_fps: 60.0,
            scene_fps: 60.0,
            next_frame_time: Instant::now(),
            is_orbiting: false,
            is_panning: false,
            last_mouse_pos: (0.0, 0.0),
            grabbed_vert: None,
            grab_initial_pos: glam::Vec3::ZERO,
            grab_plane_point: glam::Vec3::ZERO,
            grab_plane_normal: glam::Vec3::Z,
            grab_initial_hit: glam::Vec3::ZERO,
            grab_target_pos: None,
            pinned_verts: HashMap::new(),
            cached_positions: Vec::new(),
        }
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        // GpuContext の取得 (cloth_core と共通)
        let gpu_context = GpuContext::get_or_init().expect("Failed to get or init GpuContext");

        // Surface の生成
        let surface = gpu_context.instance.create_surface(window.clone()).expect("Failed to create surface");

        let caps = surface.get_capabilities(&gpu_context.adapter);
        let surface_format = caps.formats.iter().copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        // 自前タイマーで正確にフレーム制御するため、VSync待機による競合・遅延を避けて AutoNoVsync/Mailbox/Immediate を優先
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::AutoNoVsync) {
            wgpu::PresentMode::AutoNoVsync
        } else if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::AutoVsync
        };
        log::info!("[GPU] Available present modes: {:?}, selected: {:?}", caps.present_modes, present_mode);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&gpu_context.device, &config);

        let depth_view = create_depth_texture(&gpu_context.device, width, height);
        let renderer = MeshRenderer::new(&gpu_context.device, surface_format);

        let egui_renderer = egui_wgpu::Renderer::new(
            &gpu_context.device,
            surface_format,
            None,
            1,
            false,
        );

        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        self.window = Some(window);
        self.gpu_context = Some(gpu_context);
        self.surface = Some(surface);
        self.surface_config = Some(config);
        self.depth_texture = Some(depth_view);
        self.renderer = Some(renderer);
        self.egui_renderer = Some(egui_renderer);
        self.egui_state = Some(egui_state);
    }

    fn pick_vertex(&mut self, screen_pos: (f32, f32), screen_size: (f32, f32)) -> Option<u32> {
        let Some(ref mut sim) = self.simulator else { return None };
        let n_verts = sim.num_vertices as usize;
        if n_verts == 0 { return None; }

        // 最新の座標をGPUからリードバック
        if self.cached_positions.len() != n_verts {
            self.cached_positions.resize(n_verts, [0.0; 3]);
        }
        let mut flat = vec![0.0f32; n_verts * 3];
        sim.get_positions_flat(&mut flat);
        for (i, chunk) in flat.chunks_exact(3).enumerate() {
            self.cached_positions[i] = [chunk[0], chunk[1], chunk[2]];
        }

        let mut best_dist_sq = 45.0 * 45.0; // 半径45px以内
        let mut best_idx = None;

        for (i, pos) in self.cached_positions.iter().enumerate() {
            let world_p = glam::Vec3::from_array(*pos);
            if let Some((sx, sy)) = self.camera.world_to_screen(world_p, screen_size) {
                let dx = sx - screen_pos.0;
                let dy = sy - screen_pos.1;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    best_idx = Some(i as u32);
                }
            }
        }
        best_idx
    }

    fn init_scene_from_data(&mut self, data: SceneInitData) {
        log::info!("[App] Initializing scene for object: {} (verts: {})", data.object_name, data.positions.len());
        let Some(ref context) = self.gpu_context else { return };

        let faces_opt = if !data.faces.is_empty() { Some(data.faces.as_slice()) } else { None };
        let sew_opt = data.sewing_springs.as_deref();

        let mesh = ClothMesh::from_raw(
            &data.positions,
            &data.edges,
            faces_opt,
            Some(&data.inv_masses),
            sew_opt,
            None,
            None,
            data.layer_id,
            data.thickness,
            data.stiffness,
            data.compression_stiffness,
            data.shear_stiffness,
            data.bending_stiffness,
            data.sewing_shrink_speed,
        );

        let mut sim = GpuClothSimulator::with_options(
            Arc::clone(context),
            mesh,
            data.workgroup_size,
            data.solver_mode,
        );

        // 剛性4種
        sim.set_stiffness_all(
            data.stiffness,
            data.compression_stiffness,
            data.shear_stiffness,
            data.bending_stiffness,
        );
        self.stiffness = data.stiffness;
        self.tension_stiffness = data.stiffness;
        self.compression_stiffness = data.compression_stiffness;
        self.shear_stiffness = data.shear_stiffness;
        self.bending_stiffness = data.bending_stiffness;

        // 減衰
        sim.set_damping(data.air_damping);
        sim.set_damping_all(
            data.tension_damping,
            data.compression_damping,
            data.shear_damping,
            data.bending_damping,
        );
        self.air_damping = data.air_damping;
        self.tension_damping = data.tension_damping;
        self.compression_damping = data.compression_damping;
        self.shear_damping = data.shear_damping;
        self.bending_damping = data.bending_damping;

        // 重力
        self.gravity_scale = data.gravity_scale;
        if data.gravity_scale > 1e-4 {
            self.base_gravity = [
                data.gravity[0] / data.gravity_scale,
                data.gravity[1] / data.gravity_scale,
                data.gravity[2] / data.gravity_scale,
            ];
        } else {
            self.base_gravity = [0.0, 0.0, -9.81];
        }
        self.gravity = data.gravity;
        sim.set_gravity(data.gravity[0], data.gravity[1], data.gravity[2]);

        // ソルバー設定
        sim.solver_iterations = data.solver_iterations;
        self.solver_iterations = data.solver_iterations;
        self.substeps = data.substeps;
        if let Some(fps) = data.fps {
            if fps > 0.0 {
                self.scene_fps = fps;
                self.target_fps = fps;
                log::info!("[App] Synchronized simulation target FPS with Blender scene: {:.1} FPS", fps);
            }
        }

        // 自己衝突オプション
        if let Some(ref sc) = data.self_collision {
            sim.set_enable_self_collision(sc.enabled);
            sim.set_self_collision_options(
                sc.relief_factor,
                sc.max_displacement_ratio,
                sc.exclude_neighbors,
                sc.enable_normal_untangling,
                256,
            );
            sim.set_coupled_self_collision_options(sc.coupled_mode, sc.post_relaxation_iters);
            sim.set_enable_edge_collision(sc.enable_edge_collision);
            sim.set_edge_margin_scale(sc.edge_margin_scale);
            sim.set_edge_margin_offset(sc.edge_margin_offset);
        }

        // 伸縮グループ
        if let Some(ref eb) = data.elastic_bands {
            if !eb.edge_indices.is_empty() {
                sim.set_edge_rest_length_scales(&eb.edge_indices, &eb.scales);
            }
        }

        // コライダーの登録
        Self::register_colliders_to_sim(&mut sim, &data.colliders, &data.mesh_triangles);

        // ボーンSDFの登録
        if let Some(ref bsdf) = data.bone_sdf {
            Self::register_bone_sdf_to_sim(&mut sim, bsdf);
        }

        // レンダラーのトポロジー更新 (布、コライダー、SDF枠線ライン、SDF表面ボクセル)
        if let Some(ref mut r) = self.renderer {
            r.update_mesh_topology(&context.device, &data.faces, &data.edges);
            r.update_collider_topology(&context.device, &data.colliders, &data.mesh_triangles);
            r.update_lines(&context.device, &data.extra_lines);
            r.update_voxels(&context.device, &data.voxels);
        }

        // カメラターゲットをメッシュ中心に自動フィット
        if !data.positions.is_empty() {
            let mut center = glam::Vec3::ZERO;
            for p in &data.positions {
                center += glam::Vec3::from_array(*p);
            }
            center /= data.positions.len() as f32;
            self.camera.target = center;
        }

        self.simulator = Some(sim);
        self.frame_seq = 0;
        self.is_running = true;
        self.cached_positions = data.positions.clone();
        self.grabbed_vert = None;
        self.grab_target_pos = None;
        self.pinned_verts.clear();

        // サーバー状態を更新
        if let Ok(mut state) = self.server.shared_state.write() {
            state.num_vertices = data.positions.len() as u32;
            state.latest_coords = data.positions;
            state.is_running = true;
            state.frame_seq = 0;
        }
    }

    fn register_bone_sdf_to_sim(
        sim: &mut GpuClothSimulator,
        bone_sdf: &crate::protocol::GuiBoneSdfData,
    ) {
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
                log::info!("[App] Applied {} initial bone transforms for Bone SDF", gpu_transforms.len());
            }
            log::info!("[App] Registered Bone SDF 3D Texture ({}x{}x{}, {} bones)", bone_sdf.width, bone_sdf.height, bone_sdf.depth, bone_infos.len());
        } else {
            log::error!("[App] Failed to decode Bone SDF texture base64 data");
        }
    }

    fn register_colliders_to_sim(
        sim: &mut GpuClothSimulator,
        colliders: &[crate::protocol::GuiColliderData],
        mesh_triangles: &[crate::protocol::GuiMeshTriangleData],
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

    fn step_physics(&mut self) {
        let Some(ref mut sim) = self.simulator else { return };
        if !self.is_running {
            self.last_sim_time = Instant::now();
            self.time_accumulator = 0.0;
            return;
        }

        let now = Instant::now();
        let elapsed = now.duration_since(self.last_sim_time).as_secs_f32();
        self.last_sim_time = now;

        // 目標FPSに合わせた固定時間刻みで実時間同期 (Maxモード時は60FPS基準dtで計算)
        let sim_fps = if self.target_fps >= 1000.0 { 60.0 } else { self.target_fps };
        let fixed_dt = 1.0 / sim_fps.max(1.0);

        // ウィンドウドラッグや遅延による過剰なアキュムレータ蓄積（死のスパイラル）を防止
        // 最大でも fixed_dt * 1.5 までしか持ち越さない
        self.time_accumulator = (self.time_accumulator + elapsed).min(fixed_dt * 1.5);

        // インタラクティブ表示時のフレーム内最大ステップ数:
        // 60FPS以上なら1フレーム1ステップに制限して描画の滑らかさを死守
        // 30FPS以下なら最大2ステップまで許容
        let max_steps = if self.target_fps >= 50.0 { 1 } else { 2 };
        let mut steps_done = 0;
        let t0 = Instant::now();

        // タイマーの微小な揺らぎ（16.5ms等）でステップがスキップされないよう2msの許容余裕を持たせる
        let step_threshold = (fixed_dt - 0.002).max(0.001);
        while self.time_accumulator >= step_threshold && steps_done < max_steps {
            sim.solver_iterations = self.solver_iterations;
            sim.step(fixed_dt, self.substeps);
            self.time_accumulator = (self.time_accumulator - fixed_dt).max(0.0);
            self.frame_seq += 1;
            steps_done += 1;
        }

        // Maxモード（ベンチマーク）時はアキュムレータに関わらず毎フレーム1ステップ進める
        if self.target_fps >= 1000.0 && steps_done == 0 {
            sim.solver_iterations = self.solver_iterations;
            sim.step(fixed_dt, self.substeps);
            self.frame_seq += 1;
            steps_done = 1;
        }

        if steps_done > 0 {
            let total_elapsed = t0.elapsed().as_secs_f32();
            let step_ms = (total_elapsed / steps_done as f32) * 1000.0;
            self.physics_frame_ms = if self.physics_frame_ms <= 0.0 { step_ms } else { 0.9 * self.physics_frame_ms + 0.1 * step_ms };
            self.physics_max_fps = if self.physics_frame_ms > 0.0 { 1000.0 / self.physics_frame_ms } else { 0.0 };
            self.sim_step_count += steps_done as u32;

            // サーバー共有状態の更新
            if let Ok(mut state) = self.server.shared_state.write() {
                state.frame_seq = self.frame_seq;
                state.physics_fps = self.physics_fps;
                state.render_fps = self.render_fps;
                state.step_time_ms = self.physics_frame_ms;
                state.is_running = self.is_running;

                // Blenderから明示的に要求された時（またはユーザーがドラッグ・ピックした時）のみ座標をリードバック
                // 不要な33msごとのGPU同期ブロッキング（ストール）を完全に排除して60FPSを死守
                if state.needs_coords {
                    state.needs_coords = false;
                    let n_verts = sim.num_vertices as usize;
                    if self.cached_positions.len() != n_verts {
                        self.cached_positions.resize(n_verts, [0.0; 3]);
                    }
                    let mut flat = vec![0.0f32; n_verts * 3];
                    sim.get_positions_flat(&mut flat);
                    for (i, chunk) in flat.chunks_exact(3).enumerate() {
                        self.cached_positions[i] = [chunk[0], chunk[1], chunk[2]];
                    }
                    state.latest_coords = self.cached_positions.clone();
                }
            }
        }
    }

    fn render_frame(&mut self) {
        let render_t0 = Instant::now();

        let Some(ref window) = self.window else { return };
        let Some(ref surface) = self.surface else { return };
        let Some(ref gpu_context) = self.gpu_context else { return };
        let Some(ref config) = self.surface_config else { return };
        let Some(ref depth_view) = self.depth_texture else { return };
        let Some(ref renderer) = self.renderer else { return };
        let Some(ref mut egui_renderer) = self.egui_renderer else { return };
        let Some(ref mut egui_state) = self.egui_state else { return };

        let output = match surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => {
                surface.configure(&gpu_context.device, config);
                return;
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                log::error!("[App] Surface OutOfMemory");
                return;
            }
            Err(e) => {
                log::warn!("[App] Surface texture error: {}", e);
                return;
            }
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let aspect = config.width as f32 / config.height as f32;
        let (view_proj, eye) = self.camera.view_projection(aspect);

        renderer.update_uniforms(&gpu_context.queue, view_proj, eye, self.highlight_backface);

        // egui UI の構築
        let mut stiffness_changed = false;
        let mut air_damping_changed = false;
        let mut damping_changed = false;
        let mut gravity_changed = false;
        let mut iterations_changed = false;

        let raw_input = egui_state.take_egui_input(window);
        self.egui_ctx.begin_pass(raw_input);

        egui::Window::new("Taremin Cloth GUI")
            .default_width(280.0)
            .show(&self.egui_ctx, |ui| {
                ui.heading("Performance");
                ui.horizontal(|ui| {
                    ui.label("Target:");
                    if self.target_fps >= 1000.0 {
                        ui.colored_label(egui::Color32::from_rgb(255, 200, 80), "Max (Unlimited)");
                    } else {
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 200, 80),
                            format!("{:.0} FPS", self.target_fps),
                        );
                    }
                    if (self.target_fps - self.scene_fps).abs() < 0.1 {
                        ui.label(
                            egui::RichText::new("(Blender Scene Sync)")
                                .small()
                                .color(egui::Color32::from_rgb(160, 160, 160)),
                        );
                    } else if self.target_fps >= 1000.0 {
                        ui.label(
                            egui::RichText::new("(Benchmark)")
                                .small()
                                .color(egui::Color32::from_rgb(160, 160, 160)),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new("(Manual)")
                                .small()
                                .color(egui::Color32::from_rgb(160, 160, 160)),
                        );
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Physics:");
                    ui.colored_label(
                        egui::Color32::from_rgb(50, 220, 100),
                        format!("{:.1} FPS ({:.2} ms)", self.physics_fps, self.physics_frame_ms),
                    );
                    ui.label(
                        egui::RichText::new(format!("Max: {:.0} FPS", self.physics_max_fps))
                            .small()
                            .color(egui::Color32::from_rgb(150, 150, 150)),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Render:");
                    ui.colored_label(
                        egui::Color32::from_rgb(100, 180, 255),
                        format!("{:.1} FPS ({:.2} ms)", self.render_fps, self.render_frame_ms),
                    );
                    ui.label(
                        egui::RichText::new(format!("Max: {:.0} FPS", self.render_max_fps))
                            .small()
                            .color(egui::Color32::from_rgb(150, 150, 150)),
                    );
                });

                let is_connected = self.server.shared_state.read().map(|s| s.is_connected).unwrap_or(false);
                ui.horizontal(|ui| {
                    ui.label("Blender:");
                    if is_connected {
                        ui.colored_label(egui::Color32::GREEN, "● Connected");
                    } else {
                        ui.colored_label(egui::Color32::GRAY, "○ Disconnected");
                    }
                });

                ui.separator();
                ui.heading("Control");
                ui.horizontal(|ui| {
                    if ui.button(if self.is_running { "⏸ Pause" } else { "▶ Play" }).clicked() {
                        self.is_running = !self.is_running;
                    }
                    if ui.button("⏭ Step").clicked() {
                        if let Some(ref mut sim) = self.simulator {
                            sim.solver_iterations = self.solver_iterations;
                            sim.step(1.0 / 60.0, self.substeps);
                        }
                    }
                    if ui.button("🔄 Reset").clicked() {
                        if let Some(ref mut sim) = self.simulator {
                            sim.reset();
                            self.frame_seq = 0;
                        }
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("📌 Clear Pins").clicked() {
                        self.pinned_verts.clear();
                        if let Some(ref mut sim) = self.simulator {
                            sim.clear_dynamic_pins();
                        }
                    }
                });
                ui.label(
                    egui::RichText::new("🖱 [Left Drag] Grab  |  [P] Toggle Pin")
                        .small()
                        .color(egui::Color32::from_rgb(180, 180, 180)),
                );

                ui.separator();
                egui::CollapsingHeader::new("Display").default_open(true).show(ui, |ui| {
                    ui.checkbox(&mut self.draw_cloth, "Cloth Mesh");
                    ui.checkbox(&mut self.draw_wireframe, "Wireframe");
                    ui.checkbox(&mut self.draw_colliders, "Colliders");
                    ui.checkbox(&mut self.draw_voxels, "SDF Surface Voxels");
                    ui.checkbox(&mut self.draw_bbox, "SDF AABB Bounds");
                    ui.checkbox(&mut self.highlight_backface, "Highlight Backface (Red)");
                });

                ui.separator();
                egui::CollapsingHeader::new("Stiffness").default_open(true).show(ui, |ui| {
                    if ui.add(egui::Slider::new(&mut self.tension_stiffness, 0.1..=10000.0).logarithmic(true).text("Tension")).changed() {
                        self.stiffness = self.tension_stiffness;
                        stiffness_changed = true;
                    }
                    if ui.add(egui::Slider::new(&mut self.compression_stiffness, 0.1..=10000.0).logarithmic(true).text("Compression")).changed() {
                        stiffness_changed = true;
                    }
                    if ui.add(egui::Slider::new(&mut self.shear_stiffness, 0.1..=10000.0).logarithmic(true).text("Shear")).changed() {
                        stiffness_changed = true;
                    }
                    if ui.add(egui::Slider::new(&mut self.bending_stiffness, 0.0..=1000.0).text("Bending")).changed() {
                        stiffness_changed = true;
                    }
                });

                ui.separator();
                egui::CollapsingHeader::new("Damping").default_open(false).show(ui, |ui| {
                    if ui.add(egui::Slider::new(&mut self.air_damping, 0.0..=20.0).text("Air Damping")).changed() {
                        air_damping_changed = true;
                    }
                    if ui.add(egui::Slider::new(&mut self.tension_damping, 0.0..=50.0).text("Tension")).changed() {
                        damping_changed = true;
                    }
                    if ui.add(egui::Slider::new(&mut self.compression_damping, 0.0..=50.0).text("Compression")).changed() {
                        damping_changed = true;
                    }
                    if ui.add(egui::Slider::new(&mut self.shear_damping, 0.0..=50.0).text("Shear")).changed() {
                        damping_changed = true;
                    }
                    if ui.add(egui::Slider::new(&mut self.bending_damping, 0.0..=20.0).text("Bending")).changed() {
                        damping_changed = true;
                    }
                });

                ui.separator();
                egui::CollapsingHeader::new("Simulation").default_open(true).show(ui, |ui| {
                    let grav_label = format!("Gravity ({:.2}x)", self.gravity_scale);
                    let grav_slider = ui.add(egui::Slider::new(&mut self.gravity_scale, 0.0..=5.0).text(grav_label));
                    if grav_slider.changed() {
                        self.gravity = [
                            self.base_gravity[0] * self.gravity_scale,
                            self.base_gravity[1] * self.gravity_scale,
                            self.base_gravity[2] * self.gravity_scale,
                        ];
                        gravity_changed = true;
                    }
                    grav_slider.on_hover_text(format!("Effective Gravity: [{:.2}, {:.2}, {:.2}] m/s²", self.gravity[0], self.gravity[1], self.gravity[2]));

                    ui.add(egui::Slider::new(&mut self.substeps, 1..=60).text("Substeps"));
                    if ui.add(egui::Slider::new(&mut self.solver_iterations, 1..=10).text("Iterations")).changed() {
                        iterations_changed = true;
                    }

                    ui.add_space(4.0);
                    ui.label("Target FPS:");
                    ui.horizontal_wrapped(|ui| {
                        let is_scene = (self.target_fps - self.scene_fps).abs() < 0.1;
                        if ui.selectable_label(is_scene, format!("Scene ({:.0})", self.scene_fps)).clicked() {
                            self.target_fps = self.scene_fps;
                        }
                        if (self.scene_fps - 30.0).abs() >= 0.1 && ui.selectable_label((self.target_fps - 30.0).abs() < 0.1, "30").clicked() {
                            self.target_fps = 30.0;
                        }
                        if (self.scene_fps - 60.0).abs() >= 0.1 && ui.selectable_label((self.target_fps - 60.0).abs() < 0.1, "60").clicked() {
                            self.target_fps = 60.0;
                        }
                        if (self.scene_fps - 120.0).abs() >= 0.1 && ui.selectable_label((self.target_fps - 120.0).abs() < 0.1, "120").clicked() {
                            self.target_fps = 120.0;
                        }
                        if ui.selectable_label(self.target_fps >= 1000.0, "Max").clicked() {
                            self.target_fps = 1000.0;
                        }
                    });
                    let mut custom_fps = self.target_fps.min(240.0);
                    if ui.add(egui::Slider::new(&mut custom_fps, 10.0..=240.0).text("Custom")).changed() {
                        self.target_fps = custom_fps;
                    }
                });
            });

        // UI操作による物理パラメータ変更の即時反映
        if let Some(ref mut sim) = self.simulator {
            if gravity_changed {
                sim.set_gravity(self.gravity[0], self.gravity[1], self.gravity[2]);
            }
            if stiffness_changed {
                sim.set_stiffness_all(
                    self.tension_stiffness,
                    self.compression_stiffness,
                    self.shear_stiffness,
                    self.bending_stiffness,
                );
            }
            if air_damping_changed {
                sim.set_damping(self.air_damping);
            }
            if damping_changed {
                sim.set_damping_all(
                    self.tension_damping,
                    self.compression_damping,
                    self.shear_damping,
                    self.bending_damping,
                );
            }
            if iterations_changed {
                sim.solver_iterations = self.solver_iterations;
            }
        }

        // 2D オーバーレイ描画 (Pin / Grab マーカー)
        let painter = self.egui_ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("overlay")));
        let screen_size = (config.width as f32, config.height as f32);
        let ppp = window.scale_factor() as f32;

        // ピン留め頂点の描画（赤丸）
        for (&_v_idx, &p_pos) in &self.pinned_verts {
            let world_p = glam::Vec3::from_array(p_pos);
            if let Some((sx, sy)) = self.camera.world_to_screen(world_p, screen_size) {
                let center = egui::pos2(sx / ppp, sy / ppp);
                painter.circle_filled(center, 5.0, egui::Color32::from_rgb(255, 60, 60));
                painter.circle_stroke(center, 5.0, egui::Stroke::new(1.5_f32, egui::Color32::WHITE));
            }
        }

        // ドラッグ中頂点の描画（黄丸 + 接続ライン）
        if let (Some(_v_idx), Some(target_pos)) = (self.grabbed_vert, self.grab_target_pos) {
            let orig_pos = self.grab_initial_pos;
            let orig_screen = self.camera.world_to_screen(orig_pos, screen_size);
            let target_screen = self.camera.world_to_screen(target_pos, screen_size);

            if let (Some((ox, oy)), Some((tx, ty))) = (orig_screen, target_screen) {
                let p_orig = egui::pos2(ox / ppp, oy / ppp);
                let p_target = egui::pos2(tx / ppp, ty / ppp);

                painter.line_segment([p_orig, p_target], egui::Stroke::new(2.0_f32, egui::Color32::YELLOW));
                painter.circle_filled(p_orig, 3.0, egui::Color32::WHITE);
                painter.circle_filled(p_target, 6.0, egui::Color32::from_rgb(255, 220, 0));
                painter.circle_stroke(p_target, 6.0, egui::Stroke::new(1.5_f32, egui::Color32::BLACK));
            }
        }

        let full_output = self.egui_ctx.end_pass();
        egui_state.handle_platform_output(window, full_output.platform_output);

        let tris = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, delta) in &full_output.textures_delta.set {
            egui_renderer.update_texture(&gpu_context.device, &gpu_context.queue, *id, delta);
        }

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [config.width, config.height],
            pixels_per_point: window.scale_factor() as f32,
        };

        let mut encoder = gpu_context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Frame Encoder"),
        });

        egui_renderer.update_buffers(
            &gpu_context.device,
            &gpu_context.queue,
            &mut encoder,
            &tris,
            &screen_descriptor,
        );

        // 3Dシーンレンダーパス
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3D Scene Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.12, g: 0.14, b: 0.17, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if let Some(ref sim) = self.simulator {
                if self.draw_cloth {
                    renderer.render(&mut pass, sim.vertex_buffer(), true, self.draw_wireframe);
                }
            }
            // コライダーの描画
            if self.draw_colliders {
                renderer.render_colliders(&mut pass);
            }
            // SDF表面ボクセルの描画
            if self.draw_voxels {
                renderer.render_voxels(&mut pass);
            }
            // SDF AABB枠線（3Dライン）の描画
            if self.draw_bbox {
                renderer.render_lines(&mut pass);
            }
        }

        // egui UIレンダーパス
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            }).forget_lifetime();

            egui_renderer.render(&mut pass, &tris, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            egui_renderer.free_texture(id);
        }

        gpu_context.queue.submit(std::iter::once(encoder.finish()));
        let t_pres = Instant::now();
        output.present();
        let pres_ms = t_pres.elapsed().as_secs_f32() * 1000.0;

        let render_ms = render_t0.elapsed().as_secs_f32() * 1000.0;
        self.render_frame_ms = if self.render_frame_ms <= 0.0 { render_ms } else { 0.9 * self.render_frame_ms + 0.1 * render_ms };
        self.render_max_fps = if self.render_frame_ms > 0.0 { 1000.0 / self.render_frame_ms } else { 0.0 };
        self.render_count += 1;
        if pres_ms > 5.0 {
            log::debug!("[GPU] output.present took {:.2} ms", pres_ms);
        }
    }

    fn process_tasks(&mut self, event_loop: &ActiveEventLoop) -> bool {
        let mut had_tasks = false;
        while let Ok(task) = self.server.task_rx.try_recv() {
            had_tasks = true;
            match task {
                MainThreadTask::InitScene { data, respond_to } => {
                    self.init_scene_from_data(data);
                    let _ = respond_to.send(GuiResponse::Ack { message: "OK".to_string() });
                }
                MainThreadTask::Step { dt, substeps, solver_iterations, respond_to } => {
                    if let Some(ref mut sim) = self.simulator {
                        sim.solver_iterations = solver_iterations;
                        sim.step(dt, substeps);
                        self.frame_seq += 1;
                    }
                    let _ = respond_to.send(GuiResponse::Ack { message: "OK".to_string() });
                }
                MainThreadTask::Command(cmd) => match cmd {
                    GuiCommand::Play => {
                        self.is_running = true;
                    }
                    GuiCommand::Pause => {
                        self.is_running = false;
                    }
                    GuiCommand::Reset => {
                        if let Some(ref mut sim) = self.simulator {
                            sim.reset();
                            self.frame_seq = 0;
                        }
                    }
                    GuiCommand::SetParams(p) => {
                        if let Some(scale) = p.gravity_scale {
                            self.gravity_scale = scale;
                            self.gravity = [
                                self.base_gravity[0] * scale,
                                self.base_gravity[1] * scale,
                                self.base_gravity[2] * scale,
                            ];
                            if let Some(ref mut sim) = self.simulator {
                                sim.set_gravity(self.gravity[0], self.gravity[1], self.gravity[2]);
                            }
                        } else if let Some(g) = p.gravity {
                            self.gravity = g;
                            if self.gravity_scale > 1e-4 {
                                self.base_gravity = [g[0] / self.gravity_scale, g[1] / self.gravity_scale, g[2] / self.gravity_scale];
                            }
                            if let Some(ref mut sim) = self.simulator {
                                sim.set_gravity(g[0], g[1], g[2]);
                            }
                        }
                        if let Some(air) = p.air_damping {
                            self.air_damping = air;
                            if let Some(ref mut sim) = self.simulator {
                                sim.set_damping(air);
                            }
                        }
                        if p.tension_damping.is_some() || p.compression_damping.is_some() || p.shear_damping.is_some() || p.bending_damping.is_some() {
                            if let Some(v) = p.tension_damping { self.tension_damping = v; }
                            if let Some(v) = p.compression_damping { self.compression_damping = v; }
                            if let Some(v) = p.shear_damping { self.shear_damping = v; }
                            if let Some(v) = p.bending_damping { self.bending_damping = v; }
                            if let Some(ref mut sim) = self.simulator {
                                sim.set_damping_all(
                                    self.tension_damping,
                                    self.compression_damping,
                                    self.shear_damping,
                                    self.bending_damping,
                                );
                            }
                        }
                        if p.stiffness.is_some() || p.tension_stiffness.is_some() || p.compression_stiffness.is_some() || p.shear_stiffness.is_some() || p.bending_stiffness.is_some() {
                            if let Some(v) = p.tension_stiffness.or(p.stiffness) { self.tension_stiffness = v; self.stiffness = v; }
                            if let Some(v) = p.compression_stiffness { self.compression_stiffness = v; }
                            if let Some(v) = p.shear_stiffness { self.shear_stiffness = v; }
                            if let Some(v) = p.bending_stiffness { self.bending_stiffness = v; }
                            if let Some(ref mut sim) = self.simulator {
                                sim.set_stiffness_all(
                                    self.tension_stiffness,
                                    self.compression_stiffness,
                                    self.shear_stiffness,
                                    self.bending_stiffness,
                                );
                            }
                        }
                        if let Some(sub) = p.substeps {
                            self.substeps = sub;
                        }
                        if let Some(iters) = p.solver_iterations {
                            self.solver_iterations = iters;
                            if let Some(ref mut sim) = self.simulator {
                                sim.solver_iterations = iters;
                            }
                        }
                        if let Some(fps) = p.target_fps {
                            self.target_fps = fps;
                        }
                    }
                    GuiCommand::UpdateColliders { colliders, mesh_triangles } => {
                        if let Some(ref mut sim) = self.simulator {
                            Self::register_colliders_to_sim(sim, &colliders, &mesh_triangles);
                        }
                        if let (Some(ref mut r), Some(ref gpu)) = (&mut self.renderer, &self.gpu_context) {
                            r.update_collider_topology(&gpu.device, &colliders, &mesh_triangles);
                        }
                    }
                    GuiCommand::UpdateBoneTransforms { transforms } => {
                        if let Some(ref mut sim) = self.simulator {
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
                        if let Some(ref mut sim) = self.simulator {
                            sim.set_edge_rest_length_scales(&edge_indices, &scales);
                        }
                    }
                    GuiCommand::UpdatePins { vertex_indices, positions, weights } => {
                        if let Some(ref mut sim) = self.simulator {
                            for i in 0..vertex_indices.len() {
                                sim.set_pin_target(vertex_indices[i], positions[i], weights[i]);
                            }
                        }
                    }
                    GuiCommand::Quit => {
                        event_loop.exit();
                        std::process::exit(0);
                    }
                    _ => {}
                },
            }
        }
        had_tasks
    }
}

impl ApplicationHandler for GuiApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attrs = Window::default_attributes()
                .with_title("Taremin Cloth GUI - High Speed XPBD Engine")
                .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

            let window = Arc::new(event_loop.create_window(window_attrs).expect("Failed to create window"));
            self.init_gpu(window);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        // egui へイベントを中継
        if let Some(ref mut egui_state) = self.egui_state {
            if let Some(ref window) = self.window {
                let response = egui_state.on_window_event(window, &event);
                if response.repaint {
                    window.request_redraw();
                }
                if response.consumed {
                    return;
                }
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let (Some(ref mut config), Some(ref surface), Some(ref gpu_context)) =
                    (&mut self.surface_config, &self.surface, &self.gpu_context)
                {
                    config.width = new_size.width.max(1);
                    config.height = new_size.height.max(1);
                    surface.configure(&gpu_context.device, config);
                    self.depth_texture = Some(create_depth_texture(&gpu_context.device, config.width, config.height));
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                match button {
                    MouseButton::Left => {
                        if state == ElementState::Pressed {
                            if !self.egui_ctx.is_pointer_over_area() {
                                let screen_size = if let Some(ref config) = self.surface_config {
                                    (config.width as f32, config.height as f32)
                                } else {
                                    (1280.0, 720.0)
                                };
                                if let Some(v_idx) = self.pick_vertex(self.last_mouse_pos, screen_size) {
                                    let v_pos = if (v_idx as usize) < self.cached_positions.len() {
                                        glam::Vec3::from_array(self.cached_positions[v_idx as usize])
                                    } else {
                                        glam::Vec3::ZERO
                                    };
                                    self.grabbed_vert = Some(v_idx);
                                    self.grab_initial_pos = v_pos;
                                    self.grab_target_pos = Some(v_pos);
                                    self.grab_plane_point = v_pos;
                                    let view_dir = (self.camera.eye_position() - v_pos).normalize_or_zero();
                                    self.grab_plane_normal = if view_dir.length_squared() > 1e-4 {
                                        view_dir
                                    } else {
                                        glam::Vec3::Z
                                    };

                                    let (ray_origin, ray_dir) = self.camera.screen_to_ray(self.last_mouse_pos, screen_size);
                                    let denom = ray_dir.dot(self.grab_plane_normal);
                                    if denom.abs() > 1e-6 {
                                        let t = (self.grab_plane_point - ray_origin).dot(self.grab_plane_normal) / denom;
                                        self.grab_initial_hit = ray_origin + ray_dir * t;
                                    } else {
                                        self.grab_initial_hit = v_pos;
                                    }

                                    if let Some(ref mut sim) = self.simulator {
                                        sim.set_pin_target(v_idx, [v_pos.x, v_pos.y, v_pos.z], 1.0);
                                    }
                                }
                            }
                        } else if state == ElementState::Released {
                            if let Some(v_idx) = self.grabbed_vert.take() {
                                if !self.pinned_verts.contains_key(&v_idx) {
                                    if let Some(ref mut sim) = self.simulator {
                                        sim.release_pin(v_idx);
                                    }
                                }
                                self.grab_target_pos = None;
                            }
                        }
                    }
                    MouseButton::Right | MouseButton::Middle => {
                        self.is_orbiting = state == ElementState::Pressed;
                    }
                    _ => {}
                }
                if let Some(ref window) = self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
                let (dx, dy) = (x - self.last_mouse_pos.0, y - self.last_mouse_pos.1);
                self.last_mouse_pos = (x, y);

                let mut need_redraw = false;
                if let Some(v_idx) = self.grabbed_vert {
                    if let Some(ref config) = self.surface_config {
                        let screen_size = (config.width as f32, config.height as f32);
                        let (ray_origin, ray_dir) = self.camera.screen_to_ray(self.last_mouse_pos, screen_size);
                        let denom = ray_dir.dot(self.grab_plane_normal);
                        if denom.abs() > 1e-6 {
                            let t = (self.grab_plane_point - ray_origin).dot(self.grab_plane_normal) / denom;
                            let current_hit = ray_origin + ray_dir * t;
                            let delta = current_hit - self.grab_initial_hit;
                            let target = self.grab_initial_pos + delta;
                            self.grab_target_pos = Some(target);
                            if let Some(ref mut sim) = self.simulator {
                                sim.set_pin_target(v_idx, [target.x, target.y, target.z], 1.0);
                            }
                            if self.pinned_verts.contains_key(&v_idx) {
                                self.pinned_verts.insert(v_idx, target.to_array());
                            }
                            need_redraw = true;
                        }
                    }
                } else if self.is_orbiting {
                    self.camera.rotate(dx, dy);
                    need_redraw = true;
                } else if self.is_panning {
                    self.camera.pan(dx, dy);
                    need_redraw = true;
                }

                if need_redraw {
                    if let Some(ref window) = self.window {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::KeyP),
                    ..
                },
                ..
            } => {
                let target_vert = self.grabbed_vert.or_else(|| {
                    if let Some(ref config) = self.surface_config {
                        let screen_size = (config.width as f32, config.height as f32);
                        self.pick_vertex(self.last_mouse_pos, screen_size)
                    } else {
                        None
                    }
                });

                if let Some(v_idx) = target_vert {
                    if self.pinned_verts.contains_key(&v_idx) {
                        self.pinned_verts.remove(&v_idx);
                        if self.grabbed_vert != Some(v_idx) {
                            if let Some(ref mut sim) = self.simulator {
                                sim.release_pin(v_idx);
                            }
                        }
                        log::info!("[App] Unpinned vertex {}", v_idx);
                    } else {
                        let pos = if let Some(target) = self.grab_target_pos.filter(|_| self.grabbed_vert == Some(v_idx)) {
                            target
                        } else if (v_idx as usize) < self.cached_positions.len() {
                            glam::Vec3::from_array(self.cached_positions[v_idx as usize])
                        } else {
                            glam::Vec3::ZERO
                        };
                        self.pinned_verts.insert(v_idx, pos.to_array());
                        if let Some(ref mut sim) = self.simulator {
                            sim.set_pin_target(v_idx, [pos.x, pos.y, pos.z], 1.0);
                        }
                        log::info!("[App] Pinned vertex {} at {:?}", v_idx, pos);
                    }
                    if let Some(ref window) = self.window {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.05,
                };
                self.camera.zoom(scroll);
                if let Some(ref window) = self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                let t_start = Instant::now();
                let mut t_waited = 0.0f32;

                // 目標FPSに応じた高精度絶対時刻フレームペーシング
                if self.is_running && self.target_fps < 1000.0 {
                    let now = Instant::now();
                    if now < self.next_frame_time {
                        let remaining = self.next_frame_time - now;
                        // 2ms以上余裕があれば、1.5ms手前まで高精度スリープ (timeBeginPeriod(1)が有効なため1ms精度で動作)
                        if remaining > std::time::Duration::from_millis(2) {
                            std::thread::sleep(remaining - std::time::Duration::from_micros(1500));
                        }
                        // 最後の1.5ms未満は超高精度スピン待機でピッタリ目標時刻に合わせる
                        while Instant::now() < self.next_frame_time {
                            std::hint::spin_loop();
                        }
                    }
                    t_waited = t_start.elapsed().as_secs_f32() * 1000.0;
                    let current_now = Instant::now();
                    let target_dt = std::time::Duration::from_secs_f32(1.0 / self.target_fps.max(1.0));
                    if current_now > self.next_frame_time + target_dt {
                        self.next_frame_time = current_now + target_dt;
                    } else {
                        self.next_frame_time += target_dt;
                    }
                }

                // コマンド/タスクキューを処理
                let t_tasks_start = Instant::now();
                self.process_tasks(event_loop);
                let t_tasks = t_tasks_start.elapsed().as_secs_f32() * 1000.0;

                // 物理ステップ実行
                let t_phys_start = Instant::now();
                self.step_physics();
                let t_phys = t_phys_start.elapsed().as_secs_f32() * 1000.0;

                // フレーム描画
                let t_render_start = Instant::now();
                self.render_frame();
                let t_render = t_render_start.elapsed().as_secs_f32() * 1000.0;

                // FPS集計 (0.5秒間隔)
                let fps_elapsed = self.last_fps_calc_time.elapsed().as_secs_f32();
                if fps_elapsed >= 0.5 {
                    self.physics_fps = (self.sim_step_count as f32) / fps_elapsed;
                    self.render_fps = (self.render_count as f32) / fps_elapsed;
                    log::debug!("[Profile] phys_fps={:.1}, render_fps={:.1} | wait={:.2}ms, tasks={:.2}ms, phys={:.2}ms, render={:.2}ms",
                        self.physics_fps, self.render_fps, t_waited, t_tasks, t_phys, t_render);
                    self.sim_step_count = 0;
                    self.render_count = 0;
                    self.last_fps_calc_time = Instant::now();
                }

                let is_interacting = self.is_orbiting || self.is_panning || self.grabbed_vert.is_some();

                if self.is_running {
                    // 次フレームの再描画を即座に要求 (イベントループのディスパッチ遅延は次回フレームの先頭待機で自然に吸収される)
                    if let Some(ref window) = self.window {
                        window.request_redraw();
                    }
                } else if is_interacting {
                    // 一時停止中でもカメラ・ドラッグ操作中は滑らかに描画 (最大120FPS)
                    let target_dt = std::time::Duration::from_secs_f32(1.0 / 120.0);
                    let elapsed = self.last_sim_time.elapsed();
                    if elapsed < target_dt {
                        std::thread::sleep(target_dt - elapsed);
                    }
                    if let Some(ref window) = self.window {
                        window.request_redraw();
                    }
                } else {
                    // 完全静止時: 再描画リクエストを出さずに待機 (GPU使用率0%)
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let had_tasks = self.process_tasks(event_loop);
        if had_tasks {
            if let Some(ref window) = self.window {
                window.request_redraw();
            }
        }

        if self.is_running {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
            if let Some(ref window) = self.window {
                window.request_redraw();
            }
        } else {
            // 一時停止中: 20ms後に再度起きてタスクをチェック (CPU使用率はほぼ0%)
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
                Instant::now() + std::time::Duration::from_millis(20),
            ));
        }
    }
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
