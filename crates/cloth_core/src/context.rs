use std::sync::{Arc, RwLock};
use thiserror::Error;

static GLOBAL_CONTEXT: RwLock<Option<Arc<GpuContext>>> = RwLock::new(None);

#[derive(Error, Debug)]
pub enum GpuContextError {
    #[error("適合するGPUアダプタが見つかりませんでした")]
    AdapterNotFound,
    #[error("GPUデバイスの要求に失敗しました: {0}")]
    DeviceRequestFailed(#[from] wgpu::RequestDeviceError),
}

/// 利用可能なGPUデバイス（アダプタ）の情報
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDeviceInfo {
    pub index: usize,
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub driver: String,
}

pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

fn format_backend(b: wgpu::Backend) -> String {
    match b {
        wgpu::Backend::Vulkan => "Vulkan".to_string(),
        wgpu::Backend::Metal => "Metal".to_string(),
        wgpu::Backend::Dx12 => "DirectX 12".to_string(),
        wgpu::Backend::Gl => "OpenGL".to_string(),
        wgpu::Backend::BrowserWebGpu => "WebGPU".to_string(),
        _ => format!("{:?}", b),
    }
}

fn format_device_type(d: wgpu::DeviceType) -> String {
    match d {
        wgpu::DeviceType::DiscreteGpu => "DiscreteGPU".to_string(),
        wgpu::DeviceType::IntegratedGpu => "IntegratedGPU".to_string(),
        wgpu::DeviceType::Cpu => "CPU".to_string(),
        wgpu::DeviceType::VirtualGpu => "VirtualGPU".to_string(),
        wgpu::DeviceType::Other => "Other".to_string(),
    }
}

fn parse_backends(backend_str: Option<&str>) -> wgpu::Backends {
    match backend_str.map(|s| s.to_lowercase()).as_deref() {
        Some("dx12") => wgpu::Backends::DX12,
        Some("vulkan") => wgpu::Backends::VULKAN,
        Some("metal") => wgpu::Backends::METAL,
        Some("gl") => wgpu::Backends::GL,
        Some("all") => wgpu::Backends::all(),
        _ => {
            // Windows の場合: Blender本体描画との競合を防止するため、DirectX 12 を最優先
            // その他 (Linux/macOS): PRIMARY (Vulkan / Metal)
            if cfg!(target_os = "windows") {
                wgpu::Backends::DX12
            } else {
                wgpu::Backends::PRIMARY
            }
        }
    }
}

impl GpuContext {
    /// プロセス内で共有されるグローバルな GPU Context を取得または初期化する
    pub fn get_or_init() -> Result<Arc<Self>, GpuContextError> {
        {
            let read_lock = GLOBAL_CONTEXT.read().unwrap();
            if let Some(ctx) = read_lock.as_ref() {
                return Ok(Arc::clone(ctx));
            }
        }
        Self::init_or_reset(None, None)
    }

    /// 利用可能なGPUデバイスの一覧を取得する（指定バックエンド対応デバイスのみを走査）
    pub fn enumerate_available_devices(backend_str: Option<&str>) -> Vec<GpuDeviceInfo> {
        let backends = parse_backends(backend_str);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        instance
            .enumerate_adapters(backends)
            .into_iter()
            .enumerate()
            .map(|(idx, adapter)| {
                let info = adapter.get_info();
                GpuDeviceInfo {
                    index: idx,
                    name: info.name,
                    backend: format_backend(info.backend),
                    device_type: format_device_type(info.device_type),
                    driver: info.driver,
                }
            })
            .collect()
    }

    /// 新規 GPU Context を同期的に作成する
    pub fn new() -> Result<Self, GpuContextError> {
        pollster::block_on(Self::new_async())
    }

    /// 新規 GPU Context を非同期に作成する
    pub async fn new_async() -> Result<Self, GpuContextError> {
        Self::create_device_async(None, None).await
    }

    /// 指定されたバックエンドおよびデバイスで新規 GPU Context を作成する
    pub async fn create_device_async(
        backend_str: Option<&str>,
        device_index: Option<usize>,
    ) -> Result<Self, GpuContextError> {
        let backends = parse_backends(backend_str);
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter = if let Some(idx) = device_index {
            let adapters = instance.enumerate_adapters(backends);
            adapters
                .into_iter()
                .nth(idx)
                .ok_or(GpuContextError::AdapterNotFound)?
        } else {
            let adapter_opt = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await;

            match adapter_opt {
                Some(a) => a,
                None if cfg!(target_os = "windows") && backends == wgpu::Backends::DX12 => {
                    // WindowsでDX12が取得できなかった場合はPRIMARYにフォールバック
                    let fallback_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                        backends: wgpu::Backends::PRIMARY,
                        ..Default::default()
                    });
                    fallback_instance
                        .request_adapter(&wgpu::RequestAdapterOptions {
                            power_preference: wgpu::PowerPreference::HighPerformance,
                            compatible_surface: None,
                            force_fallback_adapter: false,
                        })
                        .await
                        .ok_or(GpuContextError::AdapterNotFound)?
                }
                None => return Err(GpuContextError::AdapterNotFound),
            }
        };

        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::default();
        // アダプタがサポートする限界値までバッファサイズとストレージバインディングサイズを引き上げる
        required_limits.max_buffer_size = adapter_limits.max_buffer_size;
        required_limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        required_limits.max_storage_buffers_per_shader_stage = adapter_limits.max_storage_buffers_per_shader_stage.min(16).max(8);
        required_limits.max_compute_workgroup_storage_size = adapter_limits.max_compute_workgroup_storage_size;
        required_limits.max_compute_invocations_per_workgroup = adapter_limits.max_compute_invocations_per_workgroup;
        required_limits.max_compute_workgroup_size_x = adapter_limits.max_compute_workgroup_size_x;
        required_limits.max_compute_workgroup_size_y = adapter_limits.max_compute_workgroup_size_y;
        required_limits.max_compute_workgroup_size_z = adapter_limits.max_compute_workgroup_size_z;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("TareminCloth Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// 同期的に指定バックエンド・デバイスで GPU Context を初期化・再設定する
    pub fn init_or_reset(
        backend_str: Option<&str>,
        device_index: Option<usize>,
    ) -> Result<Arc<Self>, GpuContextError> {
        pollster::block_on(Self::init_or_reset_async(backend_str, device_index))
    }

    /// 非同期に指定バックエンド・デバイスで GPU Context を初期化・再設定する
    pub async fn init_or_reset_async(
        backend_str: Option<&str>,
        device_index: Option<usize>,
    ) -> Result<Arc<Self>, GpuContextError> {
        let ctx = Arc::new(Self::create_device_async(backend_str, device_index).await?);
        {
            let mut write_lock = GLOBAL_CONTEXT.write().unwrap();
            *write_lock = Some(Arc::clone(&ctx));
        }
        Ok(ctx)
    }

    /// 現在のGPUデバイス情報を取得する
    pub fn get_current_device_info() -> Result<GpuDeviceInfo, GpuContextError> {
        let ctx = Self::get_or_init()?;
        let info = ctx.get_adapter_info();
        Ok(GpuDeviceInfo {
            index: 0,
            name: info.name,
            backend: format_backend(info.backend),
            device_type: format_device_type(info.device_type),
            driver: info.driver,
        })
    }

    /// アダプタの情報を取得する
    pub fn get_adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        let context_result = GpuContext::get_or_init();
        match context_result {
            Ok(ctx) => {
                let info = ctx.get_adapter_info();
                println!("GPU Adapter Name: {}", info.name);
                println!("GPU Backend: {:?}", info.backend);
                let dev_limits = ctx.device.limits();
                println!("Device max_buffer_size: {} bytes ({} MB)", dev_limits.max_buffer_size, dev_limits.max_buffer_size / (1024 * 1024));
                assert!(!info.name.is_empty(), "アダプタ名が空であってはならない");
                assert!(dev_limits.max_buffer_size >= 256 * 1024 * 1024, "max_buffer_size は256MB以上");
            }
            Err(e) => {
                eprintln!("GPU Context 初期化エラー: {:?}", e);
                panic!("GPU Context の作成に失敗しました: {}", e);
            }
        }
    }

    #[test]
    fn test_enumerate_devices() {
        let devices = GpuContext::enumerate_available_devices(None);
        println!("検出されたGPUデバイス一覧: {:?}", devices);
        assert!(!devices.is_empty(), "利用可能なデバイスが1件以上検出される必要があります");
    }

    #[test]
    fn test_reinit_gpu_context() {
        // DX12で初期化
        let ctx_dx12 = GpuContext::init_or_reset(Some("dx12"), None);
        assert!(ctx_dx12.is_ok(), "DX12初期化テスト");

        let dev_info = GpuContext::get_current_device_info().unwrap();
        println!("現在のデバイス情報: {:?}", dev_info);
        assert!(!dev_info.name.is_empty());
    }
}
