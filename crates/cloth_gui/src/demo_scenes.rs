//! 独立GUI用プロシージャルデモシーン生成モジュール
//!
//! 定番の布シミュレーションデモ（球体ドレープ、2点吊りカーテン、長い布の地面折り畳み等）を
//! Blenderなしで即座に初期化・テストするためのシーン定義を提供します。

use cloth_core::mesh::areal_inv_masses;
use crate::protocol::{
    GuiColliderData, GuiSelfCollisionData, SceneInitData,
};

/// デモシーンの種類
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoSceneType {
    None,
    SphereDraping,
    TwoPointCurtain,
    GroundFolding,
    MultiLayerCloth,
    ClothTwisting,
    FunnelPass,
    GarmentSewing,
}

impl DemoSceneType {
    pub const ALL: &'static [DemoSceneType] = &[
        DemoSceneType::None,
        DemoSceneType::SphereDraping,
        DemoSceneType::TwoPointCurtain,
        DemoSceneType::GroundFolding,
        DemoSceneType::MultiLayerCloth,
        DemoSceneType::ClothTwisting,
        DemoSceneType::FunnelPass,
        DemoSceneType::GarmentSewing,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            DemoSceneType::None => "None (Waiting for Blender)",
            DemoSceneType::SphereDraping => "1. Sphere Draping",
            DemoSceneType::TwoPointCurtain => "2. Two-Point Curtain",
            DemoSceneType::GroundFolding => "3. Ground Folding (Accordion)",
            DemoSceneType::MultiLayerCloth => "4. Multi-Layer Cloth (Preview)",
            DemoSceneType::ClothTwisting => "5. Cloth Twisting (Preview)",
            DemoSceneType::FunnelPass => "6. Funnel Pass (Preview)",
            DemoSceneType::GarmentSewing => "7. Garment Sewing (Preview)",
        }
    }

    pub fn create_scene(&self) -> Option<SceneInitData> {
        match self {
            DemoSceneType::None => None,
            DemoSceneType::SphereDraping => Some(create_sphere_draping()),
            DemoSceneType::TwoPointCurtain => Some(create_two_point_curtain()),
            DemoSceneType::GroundFolding => Some(create_ground_folding()),
            DemoSceneType::MultiLayerCloth => Some(create_multi_layer_cloth()),
            DemoSceneType::ClothTwisting => Some(create_cloth_twisting()),
            DemoSceneType::FunnelPass => Some(create_funnel_pass()),
            DemoSceneType::GarmentSewing => Some(create_garment_sewing()),
        }
    }
}

/// プロシージャル生成されたグリッドメッシュ
pub struct GridMesh {
    pub positions: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub edges: Vec<[u32; 2]>,
    pub inv_masses: Vec<f32>,
}

/// 平面グリッド布メッシュの生成
/// - nx, ny: 格子頂点数 (X, Y)
/// - width, height: 布の物理幅と高さ (m)
/// - center: グリッド中心位置 [x, y, z]
/// - plane: 0: 水平 (XY平面, 法線+Z), 1: 垂直 (XZ平面, 法線+Y)
/// - areal_density: 面密度 (kg/m^2)
pub fn generate_grid_mesh(
    nx: u32,
    ny: u32,
    width: f32,
    height: f32,
    center: [f32; 3],
    plane: u32,
    areal_density: f32,
) -> GridMesh {
    assert!(nx >= 2 && ny >= 2, "Grid dimensions must be at least 2x2");
    let num_verts = (nx * ny) as usize;
    let mut positions = Vec::with_capacity(num_verts);

    let dx = width / (nx - 1) as f32;
    let dy = height / (ny - 1) as f32;
    let x_start = center[0] - width * 0.5;
    let y_start = center[1] - height * 0.5;
    let z_start = center[2] - height * 0.5;

    for j in 0..ny {
        for i in 0..nx {
            let u = x_start + i as f32 * dx;
            let pos = match plane {
                0 => [u, y_start + j as f32 * dy, center[2]], // 水平 (XY)
                1 => [u, center[1], z_start + j as f32 * dy], // 垂直 (XZ)
                _ => [u, y_start + j as f32 * dy, center[2]],
            };
            positions.push(pos);
        }
    }

    let mut faces = Vec::with_capacity(((nx - 1) * (ny - 1) * 2) as usize);
    let mut edge_set = std::collections::HashSet::new();

    for j in 0..(ny - 1) {
        for i in 0..(nx - 1) {
            let v00 = j * nx + i;
            let v10 = j * nx + (i + 1);
            let v01 = (j + 1) * nx + i;
            let v11 = (j + 1) * nx + (i + 1);

            // 対角線分割で2つの三角形を生成
            faces.push([v00, v10, v11]);
            faces.push([v00, v11, v01]);

            let mut add_edge = |a: u32, b: u32| {
                edge_set.insert((a.min(b), a.max(b)));
            };
            add_edge(v00, v10);
            add_edge(v10, v11);
            add_edge(v11, v01);
            add_edge(v01, v00);
            // 注: 対角線 (v00, v11) は基本エッジ（Tension/Compression）には含めない。
            // 織物（布）は縦糸と横糸から構成され、斜め（バイアス）方向には容易に菱形変形できる。
            // 対角線を基本エッジに含めると剛体トラス構造となり布が硬直化するため、
            // 対向頂点間のせん断拘束（Shear Constraint）のみに委ねる。
        }
    }

    let edges = edge_set.into_iter().map(|(a, b)| [a, b]).collect();

    // 三角形面積と面密度 (kg/m2) から物理単位の逆質量 (kg^-1) を計算
    // これにより XPBD 距離拘束と自己衝突ペナルティの物理スケールが正しく調和する
    let inv_masses = areal_inv_masses(&positions, Some(&faces), None, areal_density);

    GridMesh {
        positions,
        faces,
        edges,
        inv_masses,
    }
}

/// 標準的な自己衝突設定の生成
/// - enable_edge: エッジ-エッジ衝突 (E-E) を有効にするか
fn default_self_collision(enable_edge: bool) -> GuiSelfCollisionData {
    GuiSelfCollisionData {
        enabled: true,
        coupled_mode: 1, // RELAXATION
        coupled_collider: false,
        post_relaxation_iters: 2,
        relief_factor: 1.0,
        max_displacement_ratio: 0.5,
        exclude_neighbors: true,
        enable_normal_untangling: true,
        max_iterations: 256,
        substep_interval: 1,
        ee_substep_interval: 1,
        enable_edge_collision: enable_edge,
        edge_margin_scale: 1.0,
        edge_margin_offset: 0.0,
        enable_pair_cache: true,
        pair_cache_margin_mode: 1,
        pair_cache_safety_margin: 0.005,
        pair_cache_horizon_scale: 1.3,
        pair_cache_max_horizon: 0.02,
        pair_cache_max_pairs: 65536,
        enable_pair_cache_final_fallback: true,
        self_collision_algorithm: 2, // 2: Vertex-Vertex (球対球仮想頂点方式, 死角ゼロで最も安定)
    }
}

/// 1. 球体落下・ドレープデモ (Sphere Draping)
/// 球体コライダーの上に水平な布が落ちて覆いかぶさり、自然なドレープとシワを形成する。
pub fn create_sphere_draping() -> SceneInitData {
    let areal_density = 0.20;
    // 35x35 = 1,225 頂点 (約0.75m四方)
    let grid = generate_grid_mesh(35, 35, 0.75, 0.75, [0.0, 0.0, 0.40], 0, areal_density);

    let sphere_collider = GuiColliderData {
        collider_type: 0, // Sphere
        friction: 0.4,
        restitution: 0.0,
        point_a: [0.0, 0.0, 0.18], // 中心位置
        radius: 0.16,               // 半径
        point_b: [0.0, 0.0, 0.0],
    };

    SceneInitData {
        object_name: "Demo_SphereDraping".to_string(),
        positions: grid.positions,
        faces: grid.faces,
        edges: grid.edges,
        sewing_springs: None,
        inv_masses: grid.inv_masses,
        layer_ids: None,
        layer_id: 0,
        thickness: 0.0035, // 3.5mm の自然な布厚
        areal_density,
        enable_coarse_constraints: false,
        stiffness: 1200.0,
        compression_stiffness: 60.0, // 座屈による自然なシワ形成
        shear_stiffness: 60.0,       // 柔軟な斜めせん断変形
        bending_stiffness: 0.4,      // 球面に沿ってしなやかに垂れ下がる曲げ剛性
        air_damping: 0.2,
        tension_damping: 1.2,
        compression_damping: 1.2,
        shear_damping: 1.2,
        bending_damping: 0.2,
        gravity: [0.0, 0.0, -9.81],
        gravity_scale: 1.0,
        sewing_shrink_speed: 0.0,
        sewing_stiffness: 10000.0,
        enable_sewing_lock: false,
        sewing_lock_distance: 0.02,
        sewing_priority_enabled: false,
        sewing_priority_threshold: 0.9,
        sewing_priority_merge_dist: 0.005,
        sewing_priority_ramp_frames: 3,
        sewing_priority_max_frames: 600,
        workgroup_size: 64,
        solver_mode: 0,
        solver_iterations: 1,
        substeps: 40,
        enable_adaptive_substep: true,
        min_substeps: 5,
        max_substeps: 50,
        auto_coupled_on_low_substeps: true,
        auto_compensate_iterations: true,
        enable_strain_adaptive: true,
        strain_tolerance: 0.008,
        self_collision: Some(default_self_collision(false)),
        bone_sdf: None,
        colliders: vec![sphere_collider],
        mesh_triangles: Vec::new(),
        elastic_bands: None,
        fps: Some(60.0),
        voxels: Vec::new(),
        extra_lines: Vec::new(),
    }
}

/// 2. 2点吊りカーテンデモ (Two-Point Curtain)
/// 上端の両端角をピン固定した布が自重で垂れ下がり、自然なカテナリー曲線とシワを作る。
pub fn create_two_point_curtain() -> SceneInitData {
    let nx = 35;
    let ny = 25;
    let areal_density = 0.18;
    let mut grid = generate_grid_mesh(nx, ny, 0.85, 0.60, [0.0, 0.0, 0.45], 1, areal_density);

    // 上端の左右角（各2頂点ずつ）を固定ピン留め (inv_mass = 0.0)
    let top_row = ny - 1;
    let pin_indices = [
        top_row * nx,
        top_row * nx + 1,
        top_row * nx + (nx - 2),
        top_row * nx + (nx - 1),
    ];
    for &idx in &pin_indices {
        grid.inv_masses[idx as usize] = 0.0;
    }

    SceneInitData {
        object_name: "Demo_TwoPointCurtain".to_string(),
        positions: grid.positions,
        faces: grid.faces,
        edges: grid.edges,
        sewing_springs: None,
        inv_masses: grid.inv_masses,
        layer_ids: None,
        layer_id: 0,
        thickness: 0.0035,
        areal_density,
        enable_coarse_constraints: false,
        stiffness: 1500.0,
        compression_stiffness: 50.0,
        shear_stiffness: 60.0,
        bending_stiffness: 0.3,
        air_damping: 0.3,
        tension_damping: 1.5,
        compression_damping: 1.5,
        shear_damping: 1.5,
        bending_damping: 0.2,
        gravity: [0.0, 0.0, -9.81],
        gravity_scale: 1.0,
        sewing_shrink_speed: 0.0,
        sewing_stiffness: 10000.0,
        enable_sewing_lock: false,
        sewing_lock_distance: 0.02,
        sewing_priority_enabled: false,
        sewing_priority_threshold: 0.9,
        sewing_priority_merge_dist: 0.005,
        sewing_priority_ramp_frames: 3,
        sewing_priority_max_frames: 600,
        workgroup_size: 64,
        solver_mode: 0,
        solver_iterations: 1,
        substeps: 40,
        enable_adaptive_substep: true,
        min_substeps: 5,
        max_substeps: 40,
        auto_coupled_on_low_substeps: true,
        auto_compensate_iterations: true,
        enable_strain_adaptive: true,
        strain_tolerance: 0.008,
        self_collision: Some(default_self_collision(false)),
        bone_sdf: None,
        colliders: Vec::new(),
        mesh_triangles: Vec::new(),
        elastic_bands: None,
        fps: Some(60.0),
        voxels: Vec::new(),
        extra_lines: Vec::new(),
    }
}

/// 3. 長い布の地面折り畳みデモ (Ground Folding / Piling)
/// 縦長のリボン状布が地面コライダーに向かって自重で落下し、
/// アコーディオン状（蛇腹）に規則正しく折り重なりながら多層自己衝突を検証する。
pub fn create_ground_folding() -> SceneInitData {
    let nx = 15;
    let ny = 75;
    let width = 0.25;
    let height = 1.35;
    let areal_density = 0.22;
    let num_verts = (nx * ny) as usize;

    let mut positions = Vec::with_capacity(num_verts);
    let dx = width / (nx - 1) as f32;
    let dy = height / (ny - 1) as f32;
    let x_start = -width * 0.5;
    let z_bottom = 0.05; // 地面直上からスタート

    // 均等・規則正しく左右（Y方向）に座屈・折り重なるよう、微小な初期正弦波のたわみを与える
    for j in 0..ny {
        let z = z_bottom + j as f32 * dy;
        let fold_phase = j as f32 * 0.35;
        let y_offset = fold_phase.sin() * 0.015;

        for i in 0..nx {
            let x = x_start + i as f32 * dx;
            positions.push([x, y_offset, z]);
        }
    }

    let mut faces = Vec::with_capacity(((nx - 1) * (ny - 1) * 2) as usize);
    let mut edge_set = std::collections::HashSet::new();

    for j in 0..(ny - 1) {
        for i in 0..(nx - 1) {
            let v00 = j * nx + i;
            let v10 = j * nx + (i + 1);
            let v01 = (j + 1) * nx + i;
            let v11 = (j + 1) * nx + (i + 1);

            faces.push([v00, v10, v11]);
            faces.push([v00, v11, v01]);

            let mut add_edge = |a: u32, b: u32| {
                edge_set.insert((a.min(b), a.max(b)));
            };
            add_edge(v00, v10);
            add_edge(v10, v11);
            add_edge(v11, v01);
            add_edge(v01, v00);
            // 対角線 (v00, v11) は除外し、せん断拘束に任せる
        }
    }

    let edges = edge_set.into_iter().map(|(a, b)| [a, b]).collect();

    // 物理逆質量の算出
    let inv_masses = areal_inv_masses(&positions, Some(&faces), None, areal_density);

    // 地面平面コライダー (Z = 0, 法線 [0, 0, 1])
    let ground_collider = GuiColliderData {
        collider_type: 2, // Plane
        friction: 0.6,    // 適度な摩擦で滑り逃げを防ぎ、綺麗に折り重なる
        restitution: 0.0,
        point_a: [0.0, 0.0, 0.0], // 通過点
        radius: 0.0,
        point_b: [0.0, 0.0, 1.0], // 上向き法線
    };

    SceneInitData {
        object_name: "Demo_GroundFolding".to_string(),
        positions,
        faces,
        edges,
        sewing_springs: None,
        inv_masses,
        layer_ids: None,
        layer_id: 0,
        thickness: 0.0035, // 3.5mm の自然な布厚（鋭角な蛇腹折りが可能）
        areal_density,
        enable_coarse_constraints: false,
        stiffness: 1500.0,
        compression_stiffness: 60.0, // 着地時に容易に座屈して折り目を形成
        shear_stiffness: 60.0,
        bending_stiffness: 0.3,      // しなやかな折り畳み
        air_damping: 0.3,
        tension_damping: 1.2,
        compression_damping: 1.2,
        shear_damping: 1.2,
        bending_damping: 0.2,
        gravity: [0.0, 0.0, -9.81],
        gravity_scale: 1.0,
        sewing_shrink_speed: 0.0,
        sewing_stiffness: 10000.0,
        enable_sewing_lock: false,
        sewing_lock_distance: 0.02,
        sewing_priority_enabled: false,
        sewing_priority_threshold: 0.9,
        sewing_priority_merge_dist: 0.005,
        sewing_priority_ramp_frames: 3,
        sewing_priority_max_frames: 600,
        workgroup_size: 64,
        solver_mode: 0,
        solver_iterations: 1,
        substeps: 50, // 多層接触・自己衝突を死角なく安定処理するため多めのサブステップ
        enable_adaptive_substep: true,
        min_substeps: 5,
        max_substeps: 60,
        auto_coupled_on_low_substeps: true,
        auto_compensate_iterations: true,
        enable_strain_adaptive: true,
        strain_tolerance: 0.008,
        self_collision: Some(default_self_collision(true)), // 多層自己衝突のため E-E 有効
        bone_sdf: None,
        colliders: vec![ground_collider],
        mesh_triangles: Vec::new(),
        elastic_bands: None,
        fps: Some(60.0),
        voxels: Vec::new(),
        extra_lines: Vec::new(),
    }
}

/// 4. 多層レイヤー布デモ (Multi-Layer Cloth)
/// インナー布（Layer 0）とアウター布（Layer 1）を配置し、
/// レイヤー優先度による外側維持・貫通防止（Untangling）を検証する。
pub fn create_multi_layer_cloth() -> SceneInitData {
    let nx = 25;
    let ny = 25;
    let areal_density = 0.18;
    let grid_inner = generate_grid_mesh(nx, ny, 0.50, 0.50, [0.0, 0.0, 0.35], 0, areal_density);
    let grid_outer = generate_grid_mesh(nx, ny, 0.55, 0.55, [0.0, 0.0, 0.38], 0, areal_density);

    let n_inner = grid_inner.positions.len();
    let n_outer = grid_outer.positions.len();

    let mut positions = grid_inner.positions;
    positions.extend(grid_outer.positions);

    let mut inv_masses = grid_inner.inv_masses;
    inv_masses.extend(grid_outer.inv_masses);

    // インナーは Layer 0, アウターは Layer 1
    let mut layer_ids = vec![0u32; n_inner];
    layer_ids.extend(vec![1u32; n_outer]);

    let mut faces = grid_inner.faces;
    for f in grid_outer.faces {
        faces.push([f[0] + n_inner as u32, f[1] + n_inner as u32, f[2] + n_inner as u32]);
    }

    let mut edges = grid_inner.edges;
    for e in grid_outer.edges {
        edges.push([e[0] + n_inner as u32, e[1] + n_inner as u32]);
    }

    // 球体コライダーを中心下に配置
    let sphere = GuiColliderData {
        collider_type: 0,
        friction: 0.3,
        restitution: 0.0,
        point_a: [0.0, 0.0, 0.15],
        radius: 0.15,
        point_b: [0.0, 0.0, 0.0],
    };

    SceneInitData {
        object_name: "Demo_MultiLayerCloth".to_string(),
        positions,
        faces,
        edges,
        sewing_springs: None,
        inv_masses,
        layer_ids: Some(layer_ids),
        layer_id: 0,
        thickness: 0.0035,
        areal_density,
        enable_coarse_constraints: false,
        stiffness: 1200.0,
        compression_stiffness: 60.0,
        shear_stiffness: 60.0,
        bending_stiffness: 0.4,
        air_damping: 0.3,
        tension_damping: 1.2,
        compression_damping: 1.2,
        shear_damping: 1.2,
        bending_damping: 0.2,
        gravity: [0.0, 0.0, -9.81],
        gravity_scale: 1.0,
        sewing_shrink_speed: 0.0,
        sewing_stiffness: 10000.0,
        enable_sewing_lock: false,
        sewing_lock_distance: 0.02,
        sewing_priority_enabled: false,
        sewing_priority_threshold: 0.9,
        sewing_priority_merge_dist: 0.005,
        sewing_priority_ramp_frames: 3,
        sewing_priority_max_frames: 600,
        workgroup_size: 64,
        solver_mode: 0,
        solver_iterations: 1,
        substeps: 50,
        enable_adaptive_substep: true,
        min_substeps: 5,
        max_substeps: 60,
        auto_coupled_on_low_substeps: true,
        auto_compensate_iterations: true,
        enable_strain_adaptive: true,
        strain_tolerance: 0.008,
        self_collision: Some(default_self_collision(true)), // 層間自己衝突のため E-E 有効
        bone_sdf: None,
        colliders: vec![sphere],
        mesh_triangles: Vec::new(),
        elastic_bands: None,
        fps: Some(60.0),
        voxels: Vec::new(),
        extra_lines: Vec::new(),
    }
}

/// 5. 布の絞り・ねじりデモ (Cloth Twisting)
pub fn create_cloth_twisting() -> SceneInitData {
    let nx = 30;
    let ny = 40;
    let width = 0.40;
    let height = 0.70;
    let areal_density = 0.20;
    let mut grid = generate_grid_mesh(nx, ny, width, height, [0.0, 0.0, 0.45], 1, areal_density);

    // 上端と下端をピン留め
    for i in 0..nx {
        grid.inv_masses[i as usize] = 0.0; // 下端ピン
        grid.inv_masses[((ny - 1) * nx + i) as usize] = 0.0; // 上端ピン
    }

    // 上下端で逆方向にねじる初期変位を与える
    for j in 0..ny {
        let t = j as f32 / (ny - 1) as f32; // 0.0 (下) 〜 1.0 (上)
        let angle = (t - 0.5) * std::f32::consts::PI * 1.5; // 最大約270度ねじれ
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        for i in 0..nx {
            let idx = (j * nx + i) as usize;
            let p = grid.positions[idx];
            let x = p[0];
            let y = p[1];
            grid.positions[idx][0] = x * cos_a - y * sin_a;
            grid.positions[idx][1] = x * sin_a + y * cos_a;
        }
    }

    SceneInitData {
        object_name: "Demo_ClothTwisting".to_string(),
        positions: grid.positions,
        faces: grid.faces,
        edges: grid.edges,
        sewing_springs: None,
        inv_masses: grid.inv_masses,
        layer_ids: None,
        layer_id: 0,
        thickness: 0.0035,
        areal_density,
        enable_coarse_constraints: false,
        stiffness: 1500.0,
        compression_stiffness: 60.0,
        shear_stiffness: 80.0,
        bending_stiffness: 0.4,
        air_damping: 0.3,
        tension_damping: 1.5,
        compression_damping: 1.5,
        shear_damping: 1.5,
        bending_damping: 0.2,
        gravity: [0.0, 0.0, 0.0], // ねじれ観察のため無重力または微小重力
        gravity_scale: 0.0,
        sewing_shrink_speed: 0.0,
        sewing_stiffness: 10000.0,
        enable_sewing_lock: false,
        sewing_lock_distance: 0.02,
        sewing_priority_enabled: false,
        sewing_priority_threshold: 0.9,
        sewing_priority_merge_dist: 0.005,
        sewing_priority_ramp_frames: 3,
        sewing_priority_max_frames: 600,
        workgroup_size: 64,
        solver_mode: 0,
        solver_iterations: 1,
        substeps: 50,
        enable_adaptive_substep: true,
        min_substeps: 5,
        max_substeps: 60,
        auto_coupled_on_low_substeps: true,
        auto_compensate_iterations: true,
        enable_strain_adaptive: true,
        strain_tolerance: 0.008,
        self_collision: Some(default_self_collision(true)),
        bone_sdf: None,
        colliders: Vec::new(),
        mesh_triangles: Vec::new(),
        elastic_bands: None,
        fps: Some(60.0),
        voxels: Vec::new(),
        extra_lines: Vec::new(),
    }
}

/// 6. 漏斗・スリット通過デモ (Funnel Pass)
pub fn create_funnel_pass() -> SceneInitData {
    let areal_density = 0.20;
    let grid = generate_grid_mesh(30, 30, 0.50, 0.50, [0.0, 0.0, 0.50], 0, areal_density);

    // 漏斗を模した傾斜カプセル/球コライダーの環
    let mut colliders = Vec::new();
    let num_posts = 6;
    for k in 0..num_posts {
        let angle = k as f32 * std::f32::consts::TAU / num_posts as f32;
        let r_top = 0.25;
        let r_bot = 0.08;
        let p_top = [r_top * angle.cos(), r_top * angle.sin(), 0.35];
        let p_bot = [r_bot * angle.cos(), r_bot * angle.sin(), 0.15];

        colliders.push(GuiColliderData {
            collider_type: 1, // Capsule
            friction: 0.2,
            restitution: 0.0,
            point_a: p_top,
            point_b: p_bot,
            radius: 0.03,
        });
    }

    SceneInitData {
        object_name: "Demo_FunnelPass".to_string(),
        positions: grid.positions,
        faces: grid.faces,
        edges: grid.edges,
        sewing_springs: None,
        inv_masses: grid.inv_masses,
        layer_ids: None,
        layer_id: 0,
        thickness: 0.0035,
        areal_density,
        enable_coarse_constraints: false,
        stiffness: 1200.0,
        compression_stiffness: 50.0,
        shear_stiffness: 60.0,
        bending_stiffness: 0.3,
        air_damping: 0.2,
        tension_damping: 1.2,
        compression_damping: 1.2,
        shear_damping: 1.2,
        bending_damping: 0.2,
        gravity: [0.0, 0.0, -9.81],
        gravity_scale: 1.0,
        sewing_shrink_speed: 0.0,
        sewing_stiffness: 10000.0,
        enable_sewing_lock: false,
        sewing_lock_distance: 0.02,
        sewing_priority_enabled: false,
        sewing_priority_threshold: 0.9,
        sewing_priority_merge_dist: 0.005,
        sewing_priority_ramp_frames: 3,
        sewing_priority_max_frames: 600,
        workgroup_size: 64,
        solver_mode: 0,
        solver_iterations: 1,
        substeps: 50,
        enable_adaptive_substep: true,
        min_substeps: 5,
        max_substeps: 60,
        auto_coupled_on_low_substeps: true,
        auto_compensate_iterations: true,
        enable_strain_adaptive: true,
        strain_tolerance: 0.008,
        self_collision: Some(default_self_collision(false)),
        bone_sdf: None,
        colliders,
        mesh_triangles: Vec::new(),
        elastic_bands: None,
        fps: Some(60.0),
        voxels: Vec::new(),
        extra_lines: Vec::new(),
    }
}

/// 7. 衣服縫合デモ (Garment Sewing)
pub fn create_garment_sewing() -> SceneInitData {
    let nx = 20;
    let ny = 30;
    let width = 0.35;
    let height = 0.55;
    let areal_density = 0.18;

    // 前身頃 (Front) と 後身頃 (Back)
    let grid_front = generate_grid_mesh(nx, ny, width, height, [0.0, 0.15, 0.40], 1, areal_density);
    let grid_back = generate_grid_mesh(nx, ny, width, height, [0.0, -0.15, 0.40], 1, areal_density);

    let n_front = grid_front.positions.len();

    let mut positions = grid_front.positions;
    positions.extend(grid_back.positions);

    let mut inv_masses = grid_front.inv_masses;
    inv_masses.extend(grid_back.inv_masses);

    let mut faces = grid_front.faces;
    for f in grid_back.faces {
        faces.push([f[0] + n_front as u32, f[1] + n_front as u32, f[2] + n_front as u32]);
    }

    let mut edges = grid_front.edges;
    for e in grid_back.edges {
        edges.push([e[0] + n_front as u32, e[1] + n_front as u32]);
    }

    // 左右の側端エッジを縫合バネ (Sewing Springs) で結合
    let mut sewing_springs = Vec::new();
    for j in 0..ny {
        // 左脇
        let f_left = j * nx;
        let b_left = n_front as u32 + j * nx;
        sewing_springs.push([f_left, b_left]);

        // 右脇
        let f_right = j * nx + (nx - 1);
        let b_right = n_front as u32 + j * nx + (nx - 1);
        sewing_springs.push([f_right, b_right]);
    }

    // マネキン胴体コライダー（楕円筒状に2本並べたカプセル）
    let torso_l = GuiColliderData {
        collider_type: 1, // Capsule
        friction: 0.3,
        restitution: 0.0,
        point_a: [-0.06, 0.0, 0.65],
        point_b: [-0.06, 0.0, 0.15],
        radius: 0.11,
    };
    let torso_r = GuiColliderData {
        collider_type: 1, // Capsule
        friction: 0.3,
        restitution: 0.0,
        point_a: [0.06, 0.0, 0.65],
        point_b: [0.06, 0.0, 0.15],
        radius: 0.11,
    };

    SceneInitData {
        object_name: "Demo_GarmentSewing".to_string(),
        positions,
        faces,
        edges,
        sewing_springs: Some(sewing_springs),
        inv_masses,
        layer_ids: None,
        layer_id: 0,
        thickness: 0.0035,
        areal_density,
        enable_coarse_constraints: false,
        stiffness: 1200.0,
        compression_stiffness: 60.0,
        shear_stiffness: 60.0,
        bending_stiffness: 0.4,
        air_damping: 0.3,
        tension_damping: 1.2,
        compression_damping: 1.2,
        shear_damping: 1.2,
        bending_damping: 0.2,
        gravity: [0.0, 0.0, -9.81],
        gravity_scale: 1.0,
        sewing_shrink_speed: 1.0,
        sewing_stiffness: 10000.0,
        enable_sewing_lock: true,
        sewing_lock_distance: 0.02,
        sewing_priority_enabled: true,
        sewing_priority_threshold: 0.9,
        sewing_priority_merge_dist: 0.005,
        sewing_priority_ramp_frames: 3,
        sewing_priority_max_frames: 600,
        workgroup_size: 64,
        solver_mode: 0,
        solver_iterations: 1,
        substeps: 40,
        enable_adaptive_substep: true,
        min_substeps: 5,
        max_substeps: 50,
        auto_coupled_on_low_substeps: true,
        auto_compensate_iterations: true,
        enable_strain_adaptive: true,
        strain_tolerance: 0.008,
        self_collision: Some(default_self_collision(false)),
        bone_sdf: None,
        colliders: vec![torso_l, torso_r],
        mesh_triangles: Vec::new(),
        elastic_bands: None,
        fps: Some(60.0),
        voxels: Vec::new(),
        extra_lines: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate_scene_data(scene: &SceneInitData) {
        let n_verts = scene.positions.len();
        assert!(n_verts >= 4, "{}: 頂点数が少なすぎます ({})", scene.object_name, n_verts);
        assert!(!scene.faces.is_empty(), "{}: 面データが空です", scene.object_name);
        assert!(!scene.edges.is_empty(), "{}: エッジデータが空です", scene.object_name);
        assert_eq!(scene.inv_masses.len(), n_verts, "{}: 質量配列長不整合", scene.object_name);

        // 物理逆質量 (inv_mass) の健全性チェック:
        // 固定ピン (inv_mass == 0.0) 以外の動的頂点は、物理単位 (kg^-1) で正しく計算され、
        // かつ旧バグ (1.0 kg/vert 固定 = 布全体1.2トン) ではなく妥当なスケール (> 50.0 kg^-1) であること。
        let dynamic_invs: Vec<f32> = scene.inv_masses.iter().copied().filter(|&m| m > 0.0).collect();
        assert!(!dynamic_invs.is_empty(), "{}: 動的頂点が存在しません", scene.object_name);
        for &inv_m in &dynamic_invs {
            assert!(
                inv_m > 50.0,
                "{}: 逆質量が物理スケール外です (inv_m={:.2}, 質量が過大でXPBD距離拘束が破綻します)",
                scene.object_name,
                inv_m
            );
        }

        if let Some(ref layers) = scene.layer_ids {
            assert_eq!(layers.len(), n_verts, "{}: レイヤー配列長不整合", scene.object_name);
        }

        for (face_idx, &[v0, v1, v2]) in scene.faces.iter().enumerate() {
            assert!((v0 as usize) < n_verts, "{}: 面 {} の頂点0インデックス範囲外", scene.object_name, face_idx);
            assert!((v1 as usize) < n_verts, "{}: 面 {} の頂点1インデックス範囲外", scene.object_name, face_idx);
            assert!((v2 as usize) < n_verts, "{}: 面 {} の頂点2インデックス範囲外", scene.object_name, face_idx);
            assert!(v0 != v1 && v1 != v2 && v2 != v0, "{}: 面 {} に縮退があります", scene.object_name, face_idx);
        }

        for (edge_idx, &[e0, e1]) in scene.edges.iter().enumerate() {
            assert!((e0 as usize) < n_verts, "{}: エッジ {} の頂点0インデックス範囲外", scene.object_name, edge_idx);
            assert!((e1 as usize) < n_verts, "{}: エッジ {} の頂点1インデックス範囲外", scene.object_name, edge_idx);
            assert!(e0 != e1, "{}: エッジ {} が自己ループです", scene.object_name, edge_idx);
        }

        if let Some(ref sew) = scene.sewing_springs {
            for (sew_idx, &[s0, s1]) in sew.iter().enumerate() {
                assert!((s0 as usize) < n_verts, "{}: 縫合バネ {} の頂点0範囲外", scene.object_name, sew_idx);
                assert!((s1 as usize) < n_verts, "{}: 縫合バネ {} の頂点1範囲外", scene.object_name, sew_idx);
            }
        }
    }

    #[test]
    fn test_all_demo_scenes_valid_topology() {
        for &demo in DemoSceneType::ALL {
            if let Some(scene) = demo.create_scene() {
                validate_scene_data(&scene);
            }
        }
    }

    #[test]
    fn test_ground_folding_has_plane_collider() {
        let scene = create_ground_folding();
        assert_eq!(scene.colliders.len(), 1);
        assert_eq!(scene.colliders[0].collider_type, 2); // Plane
        assert_eq!(scene.colliders[0].point_b, [0.0, 0.0, 1.0]); // Upward normal
    }

    #[test]
    fn test_multi_layer_has_layer_ids() {
        let scene = create_multi_layer_cloth();
        assert!(scene.layer_ids.is_some());
        let layers = scene.layer_ids.unwrap();
        assert!(layers.iter().any(|&l| l == 0));
        assert!(layers.iter().any(|&l| l == 1));
    }
}

