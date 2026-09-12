//! Configuration system for model inference and processing pipelines
//!
//! A flexible, type-safe configuration system for managing model engines, processors, and inference parameters.

use std::collections::HashMap;

use aksr::Builder;

use crate::{ImageProcessorConfig, InferenceParams, Module, ORTConfig, Scale, Task, Version};

/// Flexible, type-safe configuration system for model engines, processors, and inference parameters.
///
/// # 🔩 Configuration System
///
/// A comprehensive configuration management system that handles model inference setups,
/// processing pipelines, and runtime parameters. Provides a unified interface for
/// configuring vision and vision-language models with their respective processors.
///
/// ## Features
///
/// - **Dynamic Module Management**: Store modules in `HashMap<Module, ORTConfig>` for flexible configuration
/// - **Consolidated Parameters**: All inference parameters unified in `InferenceParams` struct
/// - **Move Semantics**: Zero-copy module consumption with `take_module()` for efficient resource management
/// - **Auto-resolution**: Automatic model file naming and path resolution
/// - **Validation**: Configuration validation during commit
/// - **Model Downloads**: Downloads models from remote repositories if needed
/// - **YOLO Models (Special Case)**: Auto-generates filenames from version/scale/task
///   (e.g., `v8-n-det.onnx`) - **Note**: Only YOLO uses this naming pattern
///
/// ## Architecture
///
/// ```text
/// Config
/// ├── Basics
/// │   ├── name: &'static str           // Model identifier
/// │   ├── version: Option<Version>     // Model version
/// │   ├── task: Option<Task>           // Inference task type
/// │   └── scale: Option<Scale>         // Model scale/size
/// ├── Modules
/// │   └── HashMap<Module, ORTConfig>   // Dynamic module registry
/// ├── Processors
/// │   ├── image_processor: ImageProcessorConfig
/// │   └── text_processor: TextProcessorConfig (VLM feature)
/// └── Inference
///     └── inference: InferenceParams   // Runtime parameters
/// ```
///
/// ## Pre-configured Models
///
/// The system provides pre-defined configurations for popular models:
///
/// ### YOLO Models
/// - `Config::yolo().with_task(Task::ObjectDetection).with_version(8).with_scale(Scale::N)` - YOLOv8n Detect
/// - `Config::yolo().with_task(Task::InstanceSegmentation).with_version(11).with_scale(Scale::M)` - YOLO11m Segement
///
/// ### Vision Models
/// - `Config::rfdetr_nano()` - Real-time detection
/// - `Config::depth_anything_small()` - Depth estimation
///
/// ### Vision-Language Models
/// - `Config::smolvlm()` - Small VLM
/// - `Config::moondream2()` - Efficient VLM
///
/// # Examples
///
/// ## Basic Usage
/// ```no_run
/// use usls::{Config, Device, DType};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = Config::rfdetr_nano()
///     .with_model_dtype(DType::Fp16)
///     .with_model_device(Device::Cuda(0))
///     .with_image_processor_device(Device::Cuda(0))
///     .with_batch_size_min_opt_max_all(1, 1, 8)
///     .commit()?;  // Resolve paths and finalize
/// # Ok(())
/// # }
/// ```
///
/// ## YOLO Special Case
///
/// ```no_run
/// use usls::{Config, Device, Task, DType, Scale};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = Config::yolo()
///     .with_task(Task::ObjectDetection)
///     .with_version(8.into())
///     .with_scale(Scale::N)
///     .with_model_dtype(DType::Fp16)
///     .with_model_device(Device::Cuda(0))
///     .with_image_processor_device(Device::Cuda(0))
///     .commit()?;
/// # Ok(())
/// # }
/// ```
///
#[derive(Builder, Debug, Clone, Default)]
pub struct Config {
    // Basics
    pub(crate) name: &'static str,
    pub(crate) version: Option<Version>,
    pub(crate) task: Option<Task>,
    pub(crate) scale: Option<Scale>,

    // Modules (dynamic registry)
    pub(crate) modules: HashMap<Module, ORTConfig>,

    // Processors
    pub(crate) image_processor: ImageProcessorConfig,
    #[cfg(feature = "vlm")]
    pub(crate) text_processor: crate::TextProcessorConfig,

    // Inference Parameters
    pub(crate) inference: InferenceParams,
}

/// Emit `msg` at most once per flag, without ever poisoning or panicking.
///
/// Deliberately NOT `std::sync::Once` + `eprintln!`: `eprintln!` panics when
/// stderr is a pipe whose reader is gone (Windows os error 232), and a panic
/// inside `Once::call_once` poisons the `Once`, so every later caller panics
/// too — on 2026-09-12 that killed OCR for a whole Javi process. An
/// `AtomicBool` cannot be poisoned, the write cannot panic, and a failed
/// emission is returned as `Err` for the caller to ignore or report.
pub(crate) fn emit_once<F>(flag: &std::sync::atomic::AtomicBool, emit: F) -> Result<(), String>
where
    F: FnOnce() -> std::io::Result<()> + std::panic::UnwindSafe,
{
    if flag.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return Ok(());
    }
    match std::panic::catch_unwind(emit) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(format!("warn-once write failed: {e}")),
        Err(_) => Err("warn-once writer panicked".to_string()),
    }
}

/// Write one shim warning to stderr; a closed pipe is an `io::Error`, not a panic.
fn warn_once(flag: &std::sync::atomic::AtomicBool, msg: &'static str) -> Result<(), String> {
    emit_once(flag, move || {
        use std::io::Write;
        writeln!(std::io::stderr(), "{msg}")
    })
}

impl Config {
    /// TEMP SHIM (2026-08-19, see mylm/USLS_VENDOR_SHIMS.md): the local
    /// patches adding SVTR class-mask filtering were lost when this fork
    /// was re-cloned; accept the call and warn once so mylm-perception
    /// builds. The mask is NOT applied until the real patch is restored.
    pub fn with_class_mask(self, _mask: Option<Vec<bool>>) -> Self {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let _ = warn_once(
            &WARNED,
            "[usls-shim] with_class_mask is a no-op — lost local patch, see mylm/USLS_VENDOR_SHIMS.md",
        );
        self
    }

    /// TEMP SHIM (2026-08-19, see mylm/USLS_VENDOR_SHIMS.md): DB min-area
    /// box filtering was lost with the re-clone; no-op with a single warn.
    pub fn with_db_min_area(self, _min_area: f32) -> Self {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let _ = warn_once(
            &WARNED,
            "[usls-shim] with_db_min_area is a no-op — lost local patch, see mylm/USLS_VENDOR_SHIMS.md",
        );
        self
    }

    /// Finalize and validate all module configurations.
    ///
    /// Resolves model file paths, downloads missing models, and validates the setup.
    /// Must be called before using the configuration for inference.
    pub fn commit(mut self) -> anyhow::Result<Self> {
        // Special case for YOLO: generate file name from version/scale/task
        if self.name == "yolo" {
            if let Some(model_config) = self.modules.get_mut(&Module::Model) {
                if model_config.file.is_empty() {
                    let mut y = String::new();
                    if let Some(x) = self.version {
                        y.push_str(&x.to_string());
                    }
                    if let Some(ref x) = self.scale {
                        y.push_str(&format!("-{x}"));
                    }
                    if let Some(ref x) = self.task {
                        y.push_str(&format!("-{}", x.yolo_str()));
                    }
                    y.push_str(".onnx");
                    model_config.file = y;
                }
            }
        }

        // Commit each module config
        let name = self.name;
        for (module, config) in self.modules.iter_mut() {
            if !config.file.is_empty() {
                // 1. Feature availability pre-check
                match config.device {
                    crate::Device::Cuda(_) => {
                        if !cfg!(feature = "cuda") {
                            anyhow::bail!("CUDA execution provider for module ({module:?}) requires the 'cuda' feature. \n\
                            Consider enabling it by adding '-F cuda'.\n\
                            If you also need CUDA image processing, use '-F cuda-full' or a version-specific feature like '-F cuda-12040'.");
                        }
                    }
                    crate::Device::TensorRt(_) => {
                        if !cfg!(feature = "tensorrt") {
                            anyhow::bail!("TensorRT execution provider for module ({module:?}) requires the 'tensorrt' feature. \n\
                            Consider enabling it by adding '-F tensorrt'. \n\
                            If you also need CUDA image processing, use '-F tensorrt-full' or a version-specific feature like '-F tensorrt-cuda-12040'.");
                        }
                    }
                    crate::Device::NvRtx(_) => {
                        if !cfg!(feature = "nvrtx") {
                            anyhow::bail!("NvTensorRT-RTX execution provider for module ({module:?}) requires the 'nvrtx' feature. \n\
                            Consider enabling it by adding '-F nvrtx'. \n\
                            If you also need CUDA image processing, enable both '-F nvrtx' and a CUDA runtime feature like '-F cuda-12040'.");
                        }
                    }
                    _ => {}
                }

                *config = std::mem::take(config).try_commit(name)?;

                if !module.is_vision() {
                    continue;
                }

                // 2. Device compatibility check
                let p_dev = self.image_processor.device;
                let m_dev = config.device;

                // ImageProcessor only supports CPU or CUDA
                match p_dev {
                    crate::Device::Cpu(_) => {
                        // ImageProcessor CPU - no special checks needed
                    }
                    crate::Device::Cuda(cuda_id) => {
                        // ImageProcessor CUDA requires 'cuda-runtime' feature
                        if !cfg!(feature = "cuda-runtime") {
                            anyhow::bail!("GPU image processing requires CUDA runtime libraries. \n\
                            Consider enabling it by adding '-F cuda-full', '-F tensorrt-full', or a version-specific feature like '-F cuda-12040' or '-F tensorrt-cuda-12040'.");
                        }

                        // ImageProcessor CUDA requires ORT GPU
                        match m_dev {
                            crate::Device::Cpu(_) => {
                                anyhow::bail!(
                                    "Device mismatch: ImageProcessor is on CUDA({cuda_id}) but Module ({module:?}) is on CPU. \n\
                                    CUDA image processing requires CUDA or TensorRT execution provider for model inference. \n\
                                    Consider enabling it by adding '-F cuda-full', '-F tensorrt-full', or a version-specific feature like '-F cuda-12040' or '-F tensorrt-cuda-12040'."
                                );
                            }
                            crate::Device::Cuda(model_cuda_id) => {
                                // Check GPU ID consistency
                                if cuda_id != model_cuda_id {
                                    anyhow::bail!(
                                        "Device ID mismatch: ImageProcessor is on CUDA({cuda_id}) but Module ({module:?}) is on CUDA({model_cuda_id}). \n\
                                        They must use the same GPU ID to avoid performance loss and illegal memory access."
                                    );
                                }
                            }
                            crate::Device::TensorRt(tensorrt_id) => {
                                // Check GPU ID consistency
                                if cuda_id != tensorrt_id {
                                    anyhow::bail!(
                                        "Device ID mismatch: ImageProcessor is on CUDA({cuda_id}) but Module ({module:?}) is on TensorRT({tensorrt_id}). \n\
                                        They must use the same GPU ID to avoid performance loss and illegal memory access."
                                    );
                                }
                            }
                            crate::Device::NvRtx(nvrtx_id) => {
                                // Check GPU ID consistency
                                if cuda_id != nvrtx_id {
                                    anyhow::bail!(
                                        "Device ID mismatch: ImageProcessor is on CUDA({cuda_id}) but Module ({module:?}) is on NvTensorRT-RTX({nvrtx_id}). \n\
                                        They must use the same GPU ID to avoid performance loss and illegal memory access."
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {
                        anyhow::bail!("Unsupported ImageProcessor device: {p_dev:?}. Only CUDA and CPU are supported.");
                    }
                }
            }
        }

        Ok(self)
    }

    /// Utility function to load vocabulary from text file
    pub fn load_txt_into_vec(f: &str) -> Vec<String> {
        crate::Hub::default()
            .try_fetch(f)
            .ok()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .map(|content| content.lines().map(|line| line.to_string()).collect())
            .unwrap_or_else(Vec::new)
    }
}

#[cfg(test)]
mod shim_warn_once_tests {
    use super::{emit_once, Config};
    use std::sync::atomic::AtomicBool;

    /// A panicking emitter must come back as `Err`, never as an unwind.
    #[test]
    fn panicking_emitter_is_reported_as_err() {
        static FLAG: AtomicBool = AtomicBool::new(false);
        let out = emit_once(&FLAG, || panic!("failed printing to stderr"));
        assert!(out.is_err(), "a panicking emitter must yield Err");
    }

    /// The old `Once` poisoned itself when its initialiser panicked and then
    /// panicked in every later caller. The flag must survive instead.
    #[test]
    fn second_call_after_a_panicking_first_never_panics() {
        static FLAG: AtomicBool = AtomicBool::new(false);
        let first = emit_once(&FLAG, || panic!("the pipe is being closed"));
        assert!(first.is_err(), "first call reports the failure");

        // Would have been "Once instance has previously been poisoned".
        let second = emit_once(&FLAG, || Ok(()));
        assert!(second.is_ok(), "second call must not be poisoned: {second:?}");
    }

    /// A write error (closed stderr pipe) is an `Err`, not a panic.
    #[test]
    fn write_error_is_reported_as_err() {
        static FLAG: AtomicBool = AtomicBool::new(false);
        let out = emit_once(&FLAG, || {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "The pipe is being closed. (os error 232)",
            ))
        });
        assert!(out.is_err(), "a failed write must yield Err");
    }

    /// The shims themselves stay callable for the life of the process.
    #[test]
    fn shims_can_be_called_repeatedly() {
        let _ = Config::default().with_class_mask(None);
        let _ = Config::default().with_class_mask(Some(vec![true, false]));
        let _ = Config::default().with_db_min_area(1.0);
        let _ = Config::default().with_db_min_area(2.0);
    }
}
