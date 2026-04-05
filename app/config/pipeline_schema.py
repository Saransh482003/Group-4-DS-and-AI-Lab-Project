from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class PipelineSettings:
    source_type: Literal["webcam", "video", "frame_folder"] = "webcam"
    source_path: Optional[str] = "1"  # ID for webcam, path for others
    execution_mode: Literal["sequential", "threaded_parallel"] = "sequential"
    enable_detection: bool = True
    enable_depth: bool = True
    enable_navigation: bool = True
    enable_tts: bool = True
    enable_visualization: bool = True
    save_annotated_video: bool = False
    show_windows: bool = True
    warmup_frames: int = 15
    sample_resources_every_n_frames: int = 10
    plot_stream_mode: int = 0 # 0 for RGB, 1 for depth map
    nav_logic_mode: int = 0 # 0 for deterministic, 1 for SLM


@dataclass
class ModelSettings:
    yolo_weights_path: str = ""
    depth_model_path: str = ""
    device: str = "cuda"
    confidence_threshold: float = 0.5


@dataclass
class BenchmarkSettings:
    export_csv: bool = True
    export_json: bool = True
    export_plots: bool = True
    collect_resource_metrics: bool = True
    max_frames: Optional[int] = None
    stride: int = 1
    enable_component_timing: bool = True


@dataclass
class KaggleSettings:
    dataset_slug: str = "quackphuc/egoblind-short-context-frames"
    local_cache_root: str = "data_cache/kaggle"
    extracted_folder_name: str = "egoblind-short-context-frames"
    auto_download_if_missing: bool = True


@dataclass
class AppConfig:
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    kaggle: KaggleSettings = field(default_factory=KaggleSettings)
