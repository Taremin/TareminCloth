use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use cloth_core::{
    bake_bone_sdf_gpu as core_bake_bone_sdf_gpu, BoneInput, ClothMesh, DynamicBoneSdfSetup,
    GpuBakeParams, GpuBoneInfo, GpuBoneTransform, GpuBoneTriangleSource, GpuClothSimulator,
    GpuContext, GpuMeshTriangle, GpuSkinningVertex,
};

/// GPUが利用可能かどうかを判定する
#[pyfunction]
fn is_gpu_available() -> bool {
    GpuContext::get_or_init().is_ok()
}

/// GPUデバイス名を取得する
#[pyfunction]
fn get_gpu_device_name() -> PyResult<String> {
    match GpuContext::get_or_init() {
        Ok(ctx) => {
            let info = ctx.get_adapter_info();
            Ok(info.name)
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "GPU Contextの初期化に失敗しました: {e}"
        ))),
    }
}

/// 利用可能なGPUデバイスの一覧を取得する
#[pyfunction]
#[pyo3(signature = (backend=None))]
fn get_available_gpu_devices<'py>(
    py: Python<'py>,
    backend: Option<String>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let backend_str = backend.as_deref();
    let devices = GpuContext::enumerate_available_devices(backend_str);
    let mut list = Vec::with_capacity(devices.len());
    for dev in devices {
        let dict = PyDict::new(py);
        dict.set_item("index", dev.index)?;
        dict.set_item("name", dev.name)?;
        dict.set_item("backend", dev.backend)?;
        dict.set_item("device_type", dev.device_type)?;
        dict.set_item("driver", dev.driver)?;
        list.push(dict);
    }
    Ok(list)
}

/// GPUバックエンドおよびデバイスを指定してGPUコンテキストを初期化・再設定する
#[pyfunction]
#[pyo3(signature = (backend=None, device_index=None))]
fn set_gpu_device(backend: Option<String>, device_index: Option<usize>) -> PyResult<()> {
    let backend_str = backend.as_deref();
    GpuContext::init_or_reset(backend_str, device_index).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("GPU再初期化に失敗しました: {e}"))
    })?;
    Ok(())
}

/// 現在アクティブなGPUデバイス情報を取得する
#[pyfunction]
fn get_current_gpu_device<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let dev = GpuContext::get_current_device_info().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("GPUデバイス情報の取得に失敗しました: {e}"))
    })?;
    let dict = PyDict::new(py);
    dict.set_item("index", dev.index)?;
    dict.set_item("name", dev.name)?;
    dict.set_item("backend", dev.backend)?;
    dict.set_item("device_type", dev.device_type)?;
    dict.set_item("driver", dev.driver)?;
    Ok(dict)
}

/// シーン（布メッシュ、縫合スプリング、コライダー）をUnlit/シェーディングで描画してPNG保存する
/// 戻り値: 赤色（裏面露出）ピクセル数
#[pyfunction]
#[pyo3(signature = (
    filepath,
    positions,
    faces,
    sewing_springs=None,
    mesh_colliders=None,
    width=800,
    height=600,
    camera_pos=None,
    camera_target=None,
    fov=45.0,
    draw_wireframe=true,
    wire_width=1.0,
    collider_color=None,
    sewing_color=None,
    vertex_colors=None,
    extra_lines=None
))]
fn render_scene_to_png<'py>(
    _py: Python<'py>,
    filepath: &str,
    positions: PyReadonlyArray2<f32>,
    faces: PyReadonlyArray2<u32>,
    sewing_springs: Option<PyReadonlyArray2<u32>>,
    mesh_colliders: Option<PyReadonlyArray2<f32>>,
    width: u32,
    height: u32,
    camera_pos: Option<[f32; 3]>,
    camera_target: Option<[f32; 3]>,
    fov: f32,
    draw_wireframe: bool,
    wire_width: f32,
    collider_color: Option<[u8; 3]>,
    sewing_color: Option<[u8; 3]>,
    vertex_colors: Option<PyReadonlyArray2<u8>>,
    extra_lines: Option<PyReadonlyArray2<f32>>,
) -> PyResult<usize> {
    let pos_view = positions.as_array();
    let mut pos_vec = Vec::with_capacity(pos_view.shape()[0]);
    for row in pos_view.outer_iter() {
        if row.len() >= 3 {
            pos_vec.push([row[0], row[1], row[2]]);
        }
    }

    let face_view = faces.as_array();
    let mut face_vec = Vec::with_capacity(face_view.shape()[0]);
    for row in face_view.outer_iter() {
        if row.len() >= 3 {
            face_vec.push([row[0], row[1], row[2]]);
        }
    }

    let mut sew_vec = Vec::new();
    if let Some(sew_arr) = sewing_springs {
        let view = sew_arr.as_array();
        sew_vec.reserve(view.shape()[0]);
        for row in view.outer_iter() {
            if row.len() >= 2 {
                sew_vec.push([row[0], row[1]]);
            }
        }
    }

    let mut col_tris = Vec::new();
    if let Some(col_arr) = mesh_colliders {
        let view = col_arr.as_array();
        col_tris.reserve(view.shape()[0]);
        for row in view.outer_iter() {
            if row.len() >= 9 {
                col_tris.push([
                    row[0], row[1], row[2],
                    row[3], row[4], row[5],
                    row[6], row[7], row[8],
                ]);
            }
        }
    }

    let mut v_colors_vec = Vec::new();
    let has_vc = if let Some(vc_arr) = vertex_colors {
        let view = vc_arr.as_array();
        v_colors_vec.reserve(view.shape()[0]);
        for row in view.outer_iter() {
            if row.len() >= 3 {
                v_colors_vec.push([row[0], row[1], row[2]]);
            }
        }
        true
    } else {
        false
    };

    let mut lines_vec = Vec::new();
    if let Some(l_arr) = extra_lines {
        let view = l_arr.as_array();
        lines_vec.reserve(view.shape()[0]);
        for row in view.outer_iter() {
            if row.len() >= 9 {
                lines_vec.push([
                    row[0], row[1], row[2],
                    row[3], row[4], row[5],
                    row[6], row[7], row[8],
                ]);
            }
        }
    }

    let mut options = cloth_core::RenderOptions {
        width,
        height,
        camera_pos,
        camera_target,
        fov_deg: fov,
        bg_color: [35, 35, 35],
        front_color: [255, 255, 255],
        back_color: [255, 0, 0],
        collider_color: [140, 160, 180],
        sewing_color: [0, 204, 255],
        draw_wireframe,
        wire_width_px: wire_width,
    };

    if let Some(c) = collider_color {
        options.collider_color = c;
    }
    if let Some(c) = sewing_color {
        options.sewing_color = c;
    }

    let scene = cloth_core::SceneData {
        positions: &pos_vec,
        faces: &face_vec,
        sewing_springs: &sew_vec,
        mesh_triangles: &col_tris,
        vertex_colors: if has_vc { Some(&v_colors_vec) } else { None },
        extra_lines: &lines_vec,
    };

    let result = cloth_core::render_scene_to_png_file(filepath, &scene, &options)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("PNG描画保存に失敗しました: {e}")))?;

    Ok(result.red_pixels)
}

/// 既存互換用のラッパー
#[pyfunction]
#[pyo3(signature = (filepath, positions, faces, width=800, height=600, camera_pos=None, camera_target=None, fov=45.0))]
fn render_mesh_to_png<'py>(
    py: Python<'py>,
    filepath: &str,
    positions: PyReadonlyArray2<f32>,
    faces: PyReadonlyArray2<u32>,
    width: u32,
    height: u32,
    camera_pos: Option<[f32; 3]>,
    camera_target: Option<[f32; 3]>,
    fov: f32,
) -> PyResult<usize> {
    render_scene_to_png(
        py, filepath, positions, faces,
        None, None,
        width, height, camera_pos, camera_target, fov,
        true, 1.0, None, None,
        None, None,
    )
}

/// GPUコンピュートシェーダーを用いたボーンSDFアトラスの並列ベイク
#[pyfunction]
#[pyo3(signature = (
    mesh_verts,
    mesh_tris,
    bone_names,
    bone_weights,
    bone_bind_matrices,
    resolution=64,
    margin=0.2,
    weight_threshold=0.02,
    blend_k=0.05,
    friction=0.5,
    thickness=0.005,
    restitution=0.0
))]
fn bake_bone_sdf_gpu<'py>(
    py: Python<'py>,
    mesh_verts: PyReadonlyArray2<f32>,
    mesh_tris: PyReadonlyArray2<i32>,
    bone_names: Vec<String>,
    bone_weights: Vec<PyReadonlyArray1<f32>>,
    bone_bind_matrices: PyReadonlyArray3<f32>,
    resolution: usize,
    margin: f32,
    weight_threshold: f32,
    blend_k: f32,
    friction: f32,
    thickness: f32,
    restitution: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let verts_view = mesh_verts.as_array();
    let n_verts = verts_view.shape()[0];
    let mut verts_vec = Vec::with_capacity(n_verts);
    for row in verts_view.outer_iter() {
        if row.len() >= 3 {
            verts_vec.push([row[0], row[1], row[2]]);
        }
    }

    let tris_view = mesh_tris.as_array();
    let n_tris = tris_view.shape()[0];
    let mut tris_vec = Vec::with_capacity(n_tris);
    for row in tris_view.outer_iter() {
        if row.len() >= 3 {
            tris_vec.push([row[0], row[1], row[2]]);
        }
    }

    let bind_mats_view = bone_bind_matrices.as_array();
    let n_bones = bone_names.len();
    let mut bones_input = Vec::with_capacity(n_bones);

    for (b_idx, name) in bone_names.into_iter().enumerate() {
        let w_view = bone_weights[b_idx].as_array();
        let weights = w_view.to_vec();

        let mut bind_mat = [0.0f32; 16];
        if b_idx < bind_mats_view.shape()[0] {
            let mut k = 0;
            for r in 0..4 {
                for c in 0..4 {
                    if k < 16 {
                        bind_mat[k] = bind_mats_view[[b_idx, r, c]];
                        k += 1;
                    }
                }
            }
        }

        bones_input.push(BoneInput {
            name,
            weights,
            bind_matrix: bind_mat,
        });
    }

    let res = core_bake_bone_sdf_gpu(
        &verts_vec,
        &tris_vec,
        &bones_input,
        resolution,
        margin,
        weight_threshold,
        blend_k,
        friction,
        thickness,
        restitution,
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("GPU SDFベイク失敗: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("texture_bytes", PyBytes::new(py, &res.texture_bytes))?;
    dict.set_item("width", res.width)?;
    dict.set_item("height", res.height)?;
    dict.set_item("depth", res.depth)?;

    // bone_infos: [N, 20]
    let py_infos = PyArray2::from_vec2(
        py,
        &res.bone_infos.iter().map(|arr| arr.to_vec()).collect::<Vec<_>>(),
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("bone_infos変換失敗: {e}")))?;
    dict.set_item("bone_infos", py_infos)?;
    dict.set_item("active_bones", res.active_bones)?;

    // bind_matrices: [N, 4, 4]
    let mut mats_3d: Vec<Vec<Vec<f32>>> = Vec::with_capacity(res.bind_matrices.len());
    for mat in &res.bind_matrices {
        let mut mat_2d = Vec::with_capacity(4);
        for r in 0..4 {
            let mut row = Vec::with_capacity(4);
            for c in 0..4 {
                row.push(mat[r * 4 + c]);
            }
            mat_2d.push(row);
        }
        mats_3d.push(mat_2d);
    }
    let py_mats = PyArray3::from_vec3(py, &mats_3d)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("bind_matrices変換失敗: {e}")))?;
    dict.set_item("bind_matrices", py_mats)?;

    Ok(dict)
}

/// GPU Cloth シミュレータ PyClass
#[pyclass]
pub struct ClothSimulator {
    simulator: GpuClothSimulator,
}

#[pymethods]
impl ClothSimulator {
    #[new]
    #[pyo3(signature = (positions, edges, faces=None, inv_masses=None, sewing_springs=None, layer_ids=None, thicknesses=None, layer_id=0, thickness=0.005, stiffness=1000.0, bending_stiffness=10.0, sewing_shrink_speed=1.0, compression_stiffness=None, shear_stiffness=None, workgroup_size=32, solver_mode=0, enable_compact_readback=None))]
    fn new(
        positions: PyReadonlyArray2<f32>,
        edges: PyReadonlyArray2<u32>,
        faces: Option<PyReadonlyArray2<u32>>,
        inv_masses: Option<PyReadonlyArray1<f32>>,
        sewing_springs: Option<PyReadonlyArray2<u32>>,
        layer_ids: Option<PyReadonlyArray1<u32>>,
        thicknesses: Option<PyReadonlyArray1<f32>>,
        layer_id: u32,
        thickness: f32,
        stiffness: f32,
        bending_stiffness: f32,
        sewing_shrink_speed: f32,
        compression_stiffness: Option<f32>,
        shear_stiffness: Option<f32>,
        workgroup_size: u32,
        solver_mode: u32,
        enable_compact_readback: Option<bool>,
    ) -> PyResult<Self> {
        let pos_view = positions.as_array();
        let edge_view = edges.as_array();

        let n_verts = pos_view.shape()[0];
        let mut pos_vec = Vec::with_capacity(n_verts);
        for row in pos_view.outer_iter() {
            if row.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "positions配列の各頂点は3次元 (x, y, z) である必要があります",
                ));
            }
            pos_vec.push([row[0], row[1], row[2]]);
        }

        let n_edges = edge_view.shape()[0];
        let mut edge_vec = Vec::with_capacity(n_edges);
        for row in edge_view.outer_iter() {
            if row.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "edges配列の各エッジは2要素 (v0, v1) である必要があります",
                ));
            }
            edge_vec.push([row[0], row[1]]);
        }

        let mut face_vec = Vec::new();
        if let Some(f_arr) = faces {
            let f_view = f_arr.as_array();
            for row in f_view.outer_iter() {
                if row.len() == 3 {
                    face_vec.push([row[0], row[1], row[2]]);
                }
            }
        }
        let faces_opt = if face_vec.is_empty() {
            None
        } else {
            Some(face_vec.as_slice())
        };

        let mut sew_vec = Vec::new();
        if let Some(s_arr) = sewing_springs {
            let s_view = s_arr.as_array();
            for row in s_view.outer_iter() {
                if row.len() == 2 {
                    sew_vec.push([row[0], row[1]]);
                }
            }
        }
        let sew_opt = if sew_vec.is_empty() {
            None
        } else {
            Some(sew_vec.as_slice())
        };

        let inv_m_vec: Option<Vec<f32>> = inv_masses.map(|arr| arr.to_vec().unwrap());
        let layer_id_vec: Option<Vec<u32>> = layer_ids.map(|arr| arr.to_vec().unwrap());
        let thick_vec: Option<Vec<f32>> = thicknesses.map(|arr| arr.to_vec().unwrap());

        let ctx = GpuContext::get_or_init().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("GPU初期化失敗: {e}"))
        })?;

        let tension_stiffness = stiffness;
        let comp_stiffness = compression_stiffness.unwrap_or(stiffness);
        let sh_stiffness = shear_stiffness.unwrap_or(stiffness * 0.5);

        let mesh = ClothMesh::from_raw(
            &pos_vec,
            &edge_vec,
            faces_opt,
            inv_m_vec.as_deref(),
            sew_opt,
            layer_id_vec.as_deref(),
            thick_vec.as_deref(),
            layer_id,
            thickness,
            tension_stiffness,
            comp_stiffness,
            sh_stiffness,
            bending_stiffness,
            sewing_shrink_speed,
        );

        let mut simulator = GpuClothSimulator::with_options(ctx, mesh, workgroup_size, solver_mode);
        if let Some(compact) = enable_compact_readback {
            simulator.set_enable_compact_readback(compact);
        }

        Ok(Self { simulator })
    }

    /// シミュレーションを 1 フレーム進める（同期）
    #[pyo3(signature = (dt=0.016666667, substeps=20, solver_iterations=None))]
    fn step(&mut self, dt: f32, substeps: u32, solver_iterations: Option<u32>) {
        if let Some(iters) = solver_iterations {
            self.simulator.set_solver_iterations(iters);
        }
        self.simulator.step(dt, substeps);
    }

    /// 単一サブステップのみ計算を進める（オンデマンド・サブステップ顕微鏡解析用）
    #[pyo3(signature = (dt_sub=0.0016666667))]
    fn step_single_substep(&mut self, dt_sub: f32) {
        self.simulator.step_single_substep(dt_sub);
    }

    /// シミュレーションを非同期に 1 フレーム進める（GPU計算を発行しCPUブロックなしで即座に復帰）
    #[pyo3(signature = (dt=0.016666667, substeps=20, solver_iterations=None))]
    fn step_async(&mut self, dt: f32, substeps: u32, solver_iterations: Option<u32>) {
        if let Some(iters) = solver_iterations {
            self.simulator.set_solver_iterations(iters);
        }
        self.simulator.step_async(dt, substeps);
    }

    /// 非同期実行された前フレームの頂点位置を回収する（成功時 True, 未実行/失敗時 False）
    fn fetch_positions<'py>(&mut self, _py: Python<'py>, out_array: Bound<'py, PyArray1<f32>>) -> PyResult<bool> {
        let mut out_slice = unsafe { out_array.as_slice_mut()? };
        let ok = self.simulator.fetch_positions_flat(&mut out_slice);
        Ok(ok)
    }

    /// シミュレーションを 1 フレーム進め、計算結果座標をGPU内部のリングバッファに保存する（待機・転送ゼロで即座に復帰）
    /// 戻り値: 現在バッファリングされているフレーム数 (1 ..= 8)
    #[pyo3(signature = (dt=0.016666667, substeps=20, solver_iterations=None))]
    fn step_buffered(&mut self, dt: f32, substeps: u32, solver_iterations: Option<u32>) -> u32 {
        if let Some(iters) = solver_iterations {
            self.simulator.set_solver_iterations(iters);
        }
        self.simulator.step_buffered(dt, substeps)
    }

    /// リングバッファにたまっている全フレームの頂点座標を一度のPCIe転送でまとめて取得する
    /// out_array の長さは (buffered_frame_count * num_vertices * 3) 以上である必要があります
    /// 戻り値: 取得したフレーム数
    fn fetch_buffered_positions<'py>(&mut self, _py: Python<'py>, out_array: Bound<'py, PyArray1<f32>>) -> PyResult<u32> {
        let mut out_slice = unsafe { out_array.as_slice_mut()? };
        let count = self.simulator.fetch_buffered_positions(&mut out_slice);
        Ok(count)
    }

    /// 現在リングバッファにたまっているフレーム数を取得
    fn get_buffered_frame_count(&self) -> u32 {
        self.simulator.get_buffered_frame_count()
    }

    /// リングバッファの蓄積カウントをクリアする
    fn clear_frame_buffer(&mut self) {
        self.simulator.clear_frame_buffer();
    }

    /// 拘束解決の反復回数を設定 (1サブステップあたり)
    fn set_solver_iterations(&mut self, iterations: u32) {
        self.simulator.set_solver_iterations(iterations);
    }

    /// ソルバーモードを設定 (0: Coloring, 1: Atomic Jacobi)
    fn set_solver_mode(&mut self, mode: u32) {
        self.simulator.set_solver_mode(mode);
    }

    /// 現在のソルバーモードを取得
    fn get_solver_mode(&self) -> u32 {
        self.simulator.solver_mode
    }

    /// ワークグループサイズを設定 (32 または 64)
    fn set_workgroup_size(&mut self, wg_size: u32) {
        self.simulator.set_workgroup_size(wg_size);
    }

    /// 現在のワークグループサイズを取得
    fn get_workgroup_size(&self) -> u32 {
        self.simulator.workgroup_size
    }

    /// 自己衝突・レイヤー衝突処理の有効/無効を設定
    fn set_enable_self_collision(&mut self, enable: bool) {
        self.simulator.set_enable_self_collision(enable);
    }

    /// 自己衝突および貫通解消 (Untangling) オプションを設定
    #[pyo3(signature = (relief_factor=0.2, max_displacement_ratio=0.2, exclude_neighbors=true, enable_normal_untangling=true, max_iterations=128))]
    fn set_self_collision_options(
        &mut self,
        relief_factor: f32,
        max_displacement_ratio: f32,
        exclude_neighbors: bool,
        enable_normal_untangling: bool,
        max_iterations: u32,
    ) {
        self.simulator.set_self_collision_options(
            relief_factor,
            max_displacement_ratio,
            exclude_neighbors,
            enable_normal_untangling,
            max_iterations,
        );
    }

    /// コンパクトリードバックの有効/無効を設定 (true: 12B/頂点, false: 48B/頂点)
    fn set_enable_compact_readback(&mut self, enable: bool) {
        self.simulator.set_enable_compact_readback(enable);
    }

    /// 現在のコンパクトリードバック設定を取得
    fn get_enable_compact_readback(&self) -> bool {
        self.simulator.enable_compact_readback
    }

    /// 頂点位置を NumPy フラット配列 (len = num_vertices * 3) に同期的に書き戻す
    fn get_positions<'py>(&self, _py: Python<'py>, out_array: Bound<'py, PyArray1<f32>>) -> PyResult<()> {
        let mut out_slice = unsafe { out_array.as_slice_mut()? };
        self.simulator.get_positions_flat(&mut out_slice);
        Ok(())
    }

    /// 頂点座標および速度ベクトルを直接設定し、シミュレーション状態を任意フレームへ復元する
    #[pyo3(signature = (positions, velocities=None))]
    fn set_positions_and_velocities<'py>(
        &mut self,
        _py: Python<'py>,
        positions: PyReadonlyArray2<f32>,
        velocities: Option<PyReadonlyArray2<f32>>,
    ) -> PyResult<()> {
        let pos_view = positions.as_array();
        let mut pos_vec = Vec::with_capacity(pos_view.shape()[0]);
        for row in pos_view.outer_iter() {
            if row.len() >= 3 {
                pos_vec.push([row[0], row[1], row[2]]);
            }
        }

        let vel_vec = if let Some(vels) = velocities {
            let vel_view = vels.as_array();
            let mut v_vec = Vec::with_capacity(vel_view.shape()[0]);
            for row in vel_view.outer_iter() {
                if row.len() >= 3 {
                    v_vec.push([row[0], row[1], row[2]]);
                }
            }
            Some(v_vec)
        } else {
            None
        };

        self.simulator.set_positions_and_velocities(&pos_vec, vel_vec.as_deref());
        Ok(())
    }

    /// 球コライダーを追加
    #[pyo3(signature = (center, radius, friction=0.0, restitution=0.0))]
    fn add_sphere_collider(&mut self, center: [f32; 3], radius: f32, friction: f32, restitution: f32) {
        self.simulator.add_sphere_collider(center, radius, friction, restitution);
    }

    /// カプセルコライダーを追加
    #[pyo3(signature = (point_a, point_b, radius, friction=0.0, restitution=0.0))]
    fn add_capsule_collider(&mut self, point_a: [f32; 3], point_b: [f32; 3], radius: f32, friction: f32, restitution: f32) {
        self.simulator.add_capsule_collider(point_a, point_b, radius, friction, restitution);
    }

    /// 平面コライダーを追加
    #[pyo3(signature = (point, normal, friction=0.0, restitution=0.0))]
    fn add_plane_collider(&mut self, point: [f32; 3], normal: [f32; 3], friction: f32, restitution: f32) {
        self.simulator.add_plane_collider(point, normal, friction, restitution);
    }

    /// メッシュ三角形コライダーを設定 (triangles: shape [N, 3, 3], attributes: shape [N, 4] optional)
    #[pyo3(signature = (triangles, friction=0.3, thickness=0.01, restitution=0.0, single_sided=true, attributes=None))]
    fn set_mesh_collider_triangles(
        &mut self,
        triangles: PyReadonlyArray3<f32>,
        friction: f32,
        thickness: f32,
        restitution: f32,
        single_sided: bool,
        attributes: Option<PyReadonlyArray2<f32>>,
    ) -> PyResult<()> {
        let tri_view = triangles.as_array();
        let n_triangles = tri_view.shape()[0];
        let mut mesh_triangles = Vec::with_capacity(n_triangles);
        let default_flags = if single_sided { 1u32 } else { 0u32 };

        let attr_opt = attributes.as_ref().map(|a| a.as_array());

        for (i, tri) in tri_view.outer_iter().enumerate() {
            if tri.shape() != [3, 3] {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "各三角形は 3 頂点 x 3 座標 (shape [3, 3]) である必要があります",
                ));
            }
            let p0 = [tri[[0, 0]], tri[[0, 1]], tri[[0, 2]]];
            let p1 = [tri[[1, 0]], tri[[1, 1]], tri[[1, 2]]];
            let p2 = [tri[[2, 0]], tri[[2, 1]], tri[[2, 2]]];

            let (fric, thick, rest, flags) = if let Some(ref attr_view) = attr_opt {
                let row = attr_view.row(i);
                let f = row[0];
                let t = row[1];
                let r = row[2];
                let s = if row[3] > 0.5 { 1u32 } else { 0u32 };
                (f, t, r, s)
            } else {
                (friction, thickness, restitution, default_flags)
            };

            mesh_triangles.push(GpuMeshTriangle {
                p0,
                friction: fric,
                p1,
                thickness: thick,
                p2,
                restitution: rest,
                flags,
                _pad: [0.0; 3],
            });
        }

        self.simulator.set_mesh_triangles(&mesh_triangles);
        Ok(())
    }

    /// すべてのコライダーをクリア
    fn clear_colliders(&mut self) {
        self.simulator.clear_colliders();
    }

    /// ボーンSDFコライダーを設定
    /// - width, height, depth: 3Dテクスチャ解像度
    /// - texture_bytes: 生バイト列 (Rg16Float)
    /// - bone_infos: shape [N, 20] (各ボーンの aabb_min:4, aabb_max:4, uvw_scale:4, uvw_offset:4, params:4)
    #[pyo3(signature = (width, height, depth, texture_bytes, bone_infos))]
    fn set_bone_sdf_colliders(
        &mut self,
        width: u32,
        height: u32,
        depth: u32,
        texture_bytes: &[u8],
        bone_infos: PyReadonlyArray2<f32>,
    ) -> PyResult<()> {
        let info_view = bone_infos.as_array();
        let n_bones = info_view.shape()[0];
        let mut gpu_bone_infos = Vec::with_capacity(n_bones);

        for row in info_view.outer_iter() {
            if row.len() < 20 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "bone_infos の各行は少なくとも 20 要素必要です",
                ));
            }
            gpu_bone_infos.push(GpuBoneInfo {
                aabb_min: [row[0], row[1], row[2], row[3]],
                aabb_max: [row[4], row[5], row[6], row[7]],
                uvw_scale: [row[8], row[9], row[10], row[11]],
                uvw_offset: [row[12], row[13], row[14], row[15]],
                params: [row[16], row[17], row[18], row[19]],
            });
        }

        self.simulator.set_bone_sdf_colliders(
            width,
            height,
            depth,
            texture_bytes,
            &gpu_bone_infos,
        );
        Ok(())
    }

    /// 各ボーンのワールド変換行列を更新（毎フレーム実行）
    /// - world_matrices: shape [N, 4, 4]
    /// - inv_world_matrices: shape [N, 4, 4]
    #[pyo3(signature = (world_matrices, inv_world_matrices))]
    fn update_bone_transforms(
        &mut self,
        world_matrices: PyReadonlyArray3<f32>,
        inv_world_matrices: PyReadonlyArray3<f32>,
    ) -> PyResult<()> {
        let w_view = world_matrices.as_array();
        let inv_view = inv_world_matrices.as_array();
        let n_bones = w_view.shape()[0];
        if inv_view.shape()[0] != n_bones {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "world_matrices と inv_world_matrices のボーン数が一致しません",
            ));
        }

        let mut transforms = Vec::with_capacity(n_bones);
        for i in 0..n_bones {
            let mut w_mat = [[0.0f32; 4]; 4];
            let mut inv_mat = [[0.0f32; 4]; 4];

            for r in 0..4 {
                for c in 0..4 {
                    w_mat[r][c] = w_view[[i, r, c]];
                    inv_mat[r][c] = inv_view[[i, r, c]];
                }
            }

            transforms.push(GpuBoneTransform {
                world_matrix: w_mat,
                inv_world_matrix: inv_mat,
            });
        }

        self.simulator.update_bone_transforms(&transforms);
        Ok(())
    }

    /// ボーンSDFコライダーをクリア
    fn clear_bone_sdf_colliders(&mut self) {
        self.simulator.clear_bone_sdf_colliders();
    }

    /// フルGPU動的SDFコライダーを初期化・設定する
    #[pyo3(signature = (
        width,
        height,
        depth,
        res,
        bone_infos,
        rest_verts,
        bone_indices,
        bone_weights,
        tri_sources,
        tri_weights,
        bone_bind_inv_matrices,
        update_interval=1
    ))]
    fn setup_dynamic_bone_sdf(
        &mut self,
        width: u32,
        height: u32,
        depth: u32,
        res: u32,
        bone_infos: PyReadonlyArray2<f32>,
        rest_verts: PyReadonlyArray2<f32>,
        bone_indices: PyReadonlyArray2<u32>,
        bone_weights: PyReadonlyArray2<f32>,
        tri_sources: PyReadonlyArray2<u32>,
        tri_weights: PyReadonlyArray2<f32>,
        bone_bind_inv_matrices: PyReadonlyArray3<f32>,
        update_interval: u32,
    ) -> PyResult<()> {
        let info_view = bone_infos.as_array();
        let n_bones = info_view.shape()[0];
        let mut gpu_bone_infos = Vec::with_capacity(n_bones);
        for row in info_view.outer_iter() {
            if row.len() < 20 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "bone_infos の各行は少なくとも 20 要素必要です",
                ));
            }
            gpu_bone_infos.push(GpuBoneInfo {
                aabb_min: [row[0], row[1], row[2], row[3]],
                aabb_max: [row[4], row[5], row[6], row[7]],
                uvw_scale: [row[8], row[9], row[10], row[11]],
                uvw_offset: [row[12], row[13], row[14], row[15]],
                params: [row[16], row[17], row[18], row[19]],
            });
        }

        let rv_view = rest_verts.as_array();
        let bi_view = bone_indices.as_array();
        let bw_view = bone_weights.as_array();
        let n_verts = rv_view.shape()[0];

        if bi_view.shape()[0] != n_verts || bw_view.shape()[0] != n_verts {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rest_verts, bone_indices, bone_weights の頂点数が一致しません",
            ));
        }

        let mut gpu_rest_verts = Vec::with_capacity(n_verts);
        for i in 0..n_verts {
            let r_row = rv_view.row(i);
            let b_idx = bi_view.row(i);
            let b_w = bw_view.row(i);

            let pos = [r_row[0], r_row[1], r_row[2]];
            let normal = if r_row.len() >= 6 {
                [r_row[3], r_row[4], r_row[5]]
            } else {
                [0.0, 1.0, 0.0]
            };

            gpu_rest_verts.push(GpuSkinningVertex {
                pos,
                _pad0: 0.0,
                normal,
                _pad1: 0.0,
                bone_indices: [b_idx[0], b_idx[1], b_idx[2], b_idx[3]],
                bone_weights: [b_w[0], b_w[1], b_w[2], b_w[3]],
            });
        }

        let ts_view = tri_sources.as_array();
        let tw_view = tri_weights.as_array();
        let n_tris = ts_view.shape()[0];
        if tw_view.shape()[0] != n_tris {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tri_sources と tri_weights の三角形数が一致しません",
            ));
        }

        let mut gpu_tri_sources = Vec::with_capacity(n_tris);
        for i in 0..n_tris {
            let ts_row = ts_view.row(i);
            let tw_row = tw_view.row(i);
            if ts_row.len() < 4 || tw_row.len() < 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "tri_sources は 4要素 (i0, i1, i2, bone_idx)、tri_weights は 3要素 (w0, w1, w2) 必要です",
                ));
            }
            gpu_tri_sources.push(GpuBoneTriangleSource {
                i0: ts_row[0],
                i1: ts_row[1],
                i2: ts_row[2],
                bone_idx: ts_row[3],
                w0: tw_row[0],
                w1: tw_row[1],
                w2: tw_row[2],
                _pad: 0.0,
            });
        }

        let inv_view = bone_bind_inv_matrices.as_array();
        if inv_view.shape()[0] != n_bones {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "bone_bind_inv_matrices のボーン数が bone_infos と一致しません",
            ));
        }
        let mut bind_inv_mats = Vec::with_capacity(n_bones);
        for i in 0..n_bones {
            let mut mat = [[0.0f32; 4]; 4];
            for r in 0..4 {
                for c in 0..4 {
                    mat[r][c] = inv_view[[i, r, c]];
                }
            }
            bind_inv_mats.push(mat);
        }

        // 各ボーンの三角形範囲 (tri_start, tri_count) を算出
        let mut bone_tri_counts = vec![0u32; n_bones];
        let mut bone_tri_starts = vec![0u32; n_bones];
        for (idx, ts) in gpu_tri_sources.iter().enumerate() {
            let b = ts.bone_idx as usize;
            if b < n_bones {
                if bone_tri_counts[b] == 0 {
                    bone_tri_starts[b] = idx as u32;
                }
                bone_tri_counts[b] += 1;
            }
        }

        let n_cols = (width / res).max(1);
        let n_rows = (height / res).max(1);
        let b_per_layer = n_cols * n_rows;

        let mut bake_params = Vec::with_capacity(n_bones);
        for b in 0..n_bones {
            let l = b as u32 / b_per_layer;
            let rem = b as u32 % b_per_layer;
            let r = rem / n_cols;
            let c = rem % n_cols;

            let info = &gpu_bone_infos[b];
            bake_params.push(GpuBakeParams {
                local_min: [info.aabb_min[0], info.aabb_min[1], info.aabb_min[2]],
                tri_start: bone_tri_starts[b],
                local_max: [info.aabb_max[0], info.aabb_max[1], info.aabb_max[2]],
                tri_count: bone_tri_counts[b],
                tile_col: c,
                tile_row: r,
                tile_layer: l,
                res,
                total_width: width,
                total_height: height,
                total_depth: depth,
                row_pitch: 0,
            });
        }

        let setup = DynamicBoneSdfSetup {
            width,
            height,
            depth,
            res,
            bone_infos: gpu_bone_infos,
            rest_verts: gpu_rest_verts,
            tri_sources: gpu_tri_sources,
            bone_bind_inv_matrices: bind_inv_mats,
            bake_params,
        };

        self.simulator.setup_dynamic_bone_sdf(setup);
        self.simulator.set_dynamic_bone_sdf_update_interval(update_interval);
        Ok(())
    }

    /// 動的ボーンSDFの有効/無効を切り替え
    fn set_dynamic_bone_sdf_enabled(&mut self, enabled: bool) {
        self.simulator.set_dynamic_bone_sdf_enabled(enabled);
    }

    /// 動的ボーンSDFの更新間隔（フレーム数）を設定
    fn set_dynamic_bone_sdf_update_interval(&mut self, interval: u32) {
        self.simulator.set_dynamic_bone_sdf_update_interval(interval);
    }

    /// コライダー最適化およびリカバリーのオプションを設定する
    #[pyo3(signature = (enable_cluster_culling=false, enable_single_sided_recovery=true, sweep_margin=0.05))]
    fn set_collider_options(
        &mut self,
        enable_cluster_culling: bool,
        enable_single_sided_recovery: bool,
        sweep_margin: f32,
    ) {
        self.simulator.set_collider_options(
            enable_cluster_culling,
            enable_single_sided_recovery,
            sweep_margin,
        );
    }

    /// 動的ピンを設定する
    #[pyo3(signature = (vertex_idx, target_pos, weight=1.0))]
    fn set_pin(&mut self, vertex_idx: u32, target_pos: [f32; 3], weight: f32) {
        self.simulator.set_pin_target(vertex_idx, target_pos, weight);
    }

    /// 指定頂点の動的ピンを解除する
    fn release_pin(&mut self, vertex_idx: u32) {
        self.simulator.release_pin(vertex_idx);
    }

    /// すべての動的ピンを解除する
    fn clear_pins(&mut self) {
        self.simulator.clear_dynamic_pins();
    }

    /// 初期状態にリセット
    fn reset(&mut self) {
        self.simulator.reset();
    }

    /// 指定された元エッジインデックス群に対して自然長スケール（倍率）を設定
    #[pyo3(signature = (edge_indices, scales))]
    fn set_edge_rest_length_scales(
        &mut self,
        edge_indices: PyReadonlyArray1<u32>,
        scales: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        let indices_slice = edge_indices.as_slice()?;
        let scales_slice = scales.as_slice()?;
        self.simulator.set_edge_rest_length_scales(indices_slice, scales_slice);
        Ok(())
    }

    /// 全エッジの自然長を初期状態（スケール 1.0）にリセット
    fn reset_edge_rest_lengths(&mut self) {
        self.simulator.reset_edge_rest_lengths();
    }

    /// 指定エッジの現在の自然長を取得
    fn get_edge_rest_length(&self, edge_idx: u32) -> f32 {
        self.simulator.get_edge_rest_length(edge_idx)
    }

    /// 指定エッジの初期自然長を取得
    fn get_edge_initial_rest_length(&self, edge_idx: u32) -> f32 {
        self.simulator.get_edge_initial_rest_length(edge_idx)
    }

    /// エッジ詳細接触判定の有効/無効を設定
    fn set_enable_edge_collision(&mut self, enable: bool) {
        self.simulator.set_enable_edge_collision(enable);
    }

    /// エッジ詳細接触マージン倍率を設定
    fn set_edge_margin_scale(&mut self, scale: f32) {
        self.simulator.set_edge_margin_scale(scale);
    }

    /// エッジ詳細接触マージン固定加算値を設定 (m単位)
    fn set_edge_margin_offset(&mut self, offset: f32) {
        self.simulator.set_edge_margin_offset(offset);
    }

    /// 重力の設定
    fn set_gravity(&mut self, gx: f32, gy: f32, gz: f32) {
        self.simulator.gravity = [gx, gy, gz];
    }

    /// 減衰率の設定
    /// 減衰率を動的に更新 (Air Damping / Velocity Damping)
    fn set_damping(&mut self, damping: f32) {
        self.simulator.set_damping(damping);
    }

    /// 剛性パラメータ（伸縮剛性・曲げ剛性）を動的に更新 (後方互換用)
    fn set_stiffness(&mut self, stretch_stiffness: f32, bending_stiffness: f32) {
        self.simulator.set_stiffness(stretch_stiffness, bending_stiffness);
    }

    /// 剛性4種（引張・圧縮・せん断・曲げ）を動的に更新
    fn set_stiffness_all(
        &mut self,
        tension_stiffness: f32,
        compression_stiffness: f32,
        shear_stiffness: f32,
        bending_stiffness: f32,
    ) {
        self.simulator.set_stiffness_all(
            tension_stiffness,
            compression_stiffness,
            shear_stiffness,
            bending_stiffness,
        );
    }

    /// 減衰4種（引張・圧縮・せん断・曲げ）を動的に反映
    fn set_damping_all(
        &mut self,
        tension_damp: f32,
        compression_damp: f32,
        shear_damp: f32,
        bending_damp: f32,
    ) {
        self.simulator.set_damping_all(
            tension_damp,
            compression_damp,
            shear_damp,
            bending_damp,
        );
    }

    /// 頂点数を取得
    #[getter]
    fn get_num_vertices(&self) -> u32 {
        self.simulator.num_vertices
    }

    /// 距離拘束数を取得
    #[getter]
    fn get_num_distance_constraints(&self) -> u32 {
        self.simulator.num_distance_constraints
    }

    /// 曲げ拘束数を取得
    #[getter]
    fn get_num_bending_constraints(&self) -> u32 {
        self.simulator.num_bending_constraints
    }

    /// 縫合拘束数を取得
    #[getter]
    fn get_num_sewing_constraints(&self) -> u32 {
        self.simulator.num_sewing_constraints
    }

    // ==========================================
    // デバッグ記録 (Debug Recording)
    // ==========================================

    /// デバッグ状態のフレーム単位記録を開始する
    #[pyo3(signature = (object_name="Cloth", max_frames=None))]
    fn start_debug_recording(&mut self, object_name: &str, max_frames: Option<usize>) {
        self.simulator.start_debug_recording(object_name, max_frames);
    }

    /// 現在デバッグ状態記録中かどうか
    fn is_debug_recording(&self) -> bool {
        self.simulator.is_debug_recording()
    }

    /// 現在記録されているデバッグフレーム数を取得
    fn get_debug_frame_count(&self) -> usize {
        self.simulator.get_debug_frame_count()
    }

    /// 記録されたデバッグトレースをgzip圧縮ファイルとして保存する
    fn save_debug_recording(&self, file_path: &str) -> PyResult<String> {
        self.simulator.save_debug_recording(file_path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("デバッグ記録の保存に失敗しました: {e}"))
        })
    }

    /// デバッグ状態記録を停止しメモリバッファを解放する
    fn stop_debug_recording(&mut self) {
        self.simulator.stop_debug_recording();
    }
}

/// taremin_cloth_core Python モジュール
#[pymodule]
fn taremin_cloth_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(get_gpu_device_name, m)?)?;
    m.add_function(wrap_pyfunction!(get_available_gpu_devices, m)?)?;
    m.add_function(wrap_pyfunction!(set_gpu_device, m)?)?;
    m.add_function(wrap_pyfunction!(get_current_gpu_device, m)?)?;
    m.add_function(wrap_pyfunction!(render_mesh_to_png, m)?)?;
    m.add_function(wrap_pyfunction!(render_scene_to_png, m)?)?;
    m.add_function(wrap_pyfunction!(bake_bone_sdf_gpu, m)?)?;
    m.add_class::<ClothSimulator>()?;
    Ok(())
}
