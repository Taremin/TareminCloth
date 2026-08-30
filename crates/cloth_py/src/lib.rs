use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use cloth_core::{ClothMesh, GpuClothSimulator, GpuContext, GpuMeshTriangle};

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
    #[pyo3(signature = (relief_factor=0.2, max_displacement_ratio=0.2, exclude_neighbors=true, enable_normal_untangling=true))]
    fn set_self_collision_options(
        &mut self,
        relief_factor: f32,
        max_displacement_ratio: f32,
        exclude_neighbors: bool,
        enable_normal_untangling: bool,
    ) {
        self.simulator.set_self_collision_options(
            relief_factor,
            max_displacement_ratio,
            exclude_neighbors,
            enable_normal_untangling,
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

    /// メッシュ三角形コライダーを設定 (triangles: shape [N, 3, 3])
    #[pyo3(signature = (triangles, friction=0.3, thickness=0.01, restitution=0.0, single_sided=true))]
    fn set_mesh_collider_triangles(
        &mut self,
        triangles: PyReadonlyArray3<f32>,
        friction: f32,
        thickness: f32,
        restitution: f32,
        single_sided: bool,
    ) -> PyResult<()> {
        let tri_view = triangles.as_array();
        let n_triangles = tri_view.shape()[0];
        let mut mesh_triangles = Vec::with_capacity(n_triangles);
        let flags = if single_sided { 1u32 } else { 0u32 };

        for tri in tri_view.outer_iter() {
            if tri.shape() != [3, 3] {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "各三角形は 3 頂点 x 3 座標 (shape [3, 3]) である必要があります",
                ));
            }
            let p0 = [tri[[0, 0]], tri[[0, 1]], tri[[0, 2]]];
            let p1 = [tri[[1, 0]], tri[[1, 1]], tri[[1, 2]]];
            let p2 = [tri[[2, 0]], tri[[2, 1]], tri[[2, 2]]];

            mesh_triangles.push(GpuMeshTriangle {
                p0,
                friction,
                p1,
                thickness,
                p2,
                restitution,
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
    m.add_class::<ClothSimulator>()?;
    Ok(())
}
