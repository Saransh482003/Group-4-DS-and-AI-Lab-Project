import os
import argparse
from app.config.settings import load_config
from app.runners.live_runner import LiveRunner
from app.runners.dataset_runner import DatasetRunner
from app.components.detector import DetectorComponent
from app.components.depth import DepthComponent
from app.components.navigation import NavigationComponent
from app.components.visualization import VisualizationComponent
from app.components.tts import TTSComponent
from app.pipeline.executor import SequentialExecutor
from app.pipeline.orchestrator import PipelineOrchestrator
from app.metrics.tracker import MetricsTracker
from app.utils.paths import get_base_dir
from app.sources.frame_folder_source import FrameFolderSource
from app.sources.webcam_source import WebcamFrameSource
from app.sources.scannet_source import ScanNetSource


def setup_components(config):
    components = {}

    if config.pipeline.enable_detection:
        components["detector"] = DetectorComponent(config.models.yolo_weights_path)

    if config.pipeline.enable_depth:
        components["depth"] = DepthComponent(
            config.models.depth_model_path, config.models.device
        )

    if config.pipeline.enable_visualization:
        components["visualization"] = VisualizationComponent(
            plot_stream_mode=config.pipeline.plot_stream_mode
        )

    tts_comp = None
    if config.pipeline.enable_tts:
        from app.utils.paths import env_rel_path

        tts_comp = TTSComponent(
            piper_exe=str(
                env_rel_path("PIPER_EXE_REL", os.path.join("piper", "piper.exe"))
            ),
            voice_model_path=str(
                env_rel_path(
                    "PIPER_VOICE_MODEL_REL",
                    os.path.join("piper_voices", "en_US-amy-medium.onnx"),
                )
            ),
            voice_config_path=str(
                env_rel_path(
                    "PIPER_VOICE_CONFIG_REL",
                    os.path.join("piper_voices", "en_US-amy-medium.onnx.json"),
                )
            ),
        )
        components["tts"] = tts_comp

    if config.pipeline.enable_navigation:
        # Assuming frame width from typical webcam 640
        components["navigation"] = NavigationComponent(
            frame_width=640, tts_component=tts_comp
        )

    return components


def main():
    parser = argparse.ArgumentParser(
        description="Indoor Navigation Evaluation Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "dataset_eval", "benchmark"],
        default="live",
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset slug for evaluation/benchmark mode"
    )
    parser.add_argument(
        "--source-path",
        type=str,
        help="Path to input source (camera id, video file, or image folder)",
    )
    parser.add_argument(
        "--execution-mode",
        type=str,
        choices=["sequential", "threaded_parallel"],
        help="Execution mode",
    )
    parser.add_argument("--max-frames", type=int, help="Max frames to process")
    parser.add_argument("--stride", type=int, help="Frame stride for dataset eval")
    parser.add_argument("--enable-tts", action="store_true", help="Enable TTS output")
    parser.add_argument(
        "--save-annotated-video",
        action="store_true",
        help="Save annotated video output",
    )
    parser.add_argument(
        "--show-windows", action="store_true", help="Show display windows"
    )
    parser.add_argument("--device", type=str, help="Device to run models on")
    parser.add_argument(
        "--yolo-weights", type=str, help="Path to custom YOLO weights (.pt)"
    )
    parser.add_argument("--output-dir", type=str, help="Output directory for artifacts")
    parser.add_argument(
        "--benchmark-config", type=str, help="Path to benchmark suite YAML config"
    )
    parser.add_argument(
        "--runner-config", type=str, help="Path to a unified runner config YAML"
    )
    parser.add_argument(
        "--benchmark-target",
        nargs="+",
        help="Specific benchmark suite names to run (filters the YAML list)",
    )

    args = parser.parse_args()

    # If a runner config is provided, load it and override defaults
    if args.runner_config and os.path.exists(args.runner_config):
        import yaml

        print(f"Loading runner configuration from: {args.runner_config}")
        with open(args.runner_config, "r") as f:
            runner_data = yaml.safe_load(f)
            if runner_data:
                # Update args namespace with values from YAML if they exist
                for key, value in runner_data.items():
                    if hasattr(args, key):
                        # Use CLI value if it's not the default, otherwise use YAML
                        # For simplicity, we'll let YAML override unless explicitly overridden via CLI
                        # (Checking CLI overrides is complex with argparse,
                        # so we'll just prioritize YAML for now if explicitly requested)
                        setattr(args, key, value)

    config = load_config()

    # Overrides
    if args.source_path is not None:
        config.pipeline.source_path = args.source_path
    if args.execution_mode is not None:
        config.pipeline.execution_mode = args.execution_mode
    if args.max_frames is not None:
        config.benchmark.max_frames = args.max_frames
    if args.stride is not None:
        config.benchmark.stride = args.stride

    if args.mode == "dataset_eval":
        # In dataset eval, we default GUI/TTS off unless explicitly turned on
        config.pipeline.show_windows = args.show_windows
        config.pipeline.enable_tts = args.enable_tts
    else:
        if args.enable_tts:
            config.pipeline.enable_tts = True
        if args.show_windows:
            config.pipeline.show_windows = True

    if args.save_annotated_video:
        config.pipeline.save_annotated_video = True
    if args.device is not None:
        config.models.device = args.device
    if args.yolo_weights is not None:
        config.models.yolo_weights_path = args.yolo_weights

    base_dir = args.output_dir if args.output_dir else str(get_base_dir())

    # Resolve Frame Source
    source = None
    if args.mode == "live":
        source = WebcamFrameSource(config.pipeline.source_path)
    elif args.mode == "dataset_eval" or args.mode == "benchmark":
        if args.dataset == "egoblind":
            from app.utils.kaggle_data import ensure_egoblind_dataset

            resolved_path = ensure_egoblind_dataset(config.kaggle)
            if resolved_path:
                source = FrameFolderSource(resolved_path)
        elif args.dataset == "scannet":
            if config.pipeline.source_path and os.path.isdir(
                config.pipeline.source_path
            ):
                source = ScanNetSource(config.pipeline.source_path)
            else:
                print(
                    "Error: ScanNet requires --source-path to be a directory containing scene folders."
                )
                return
        elif args.dataset == "ego4d":
            # Ego4D is typically video files or frame folders
            if config.pipeline.source_path and os.path.isfile(
                config.pipeline.source_path
            ):
                source = WebcamFrameSource(
                    config.pipeline.source_path
                )  # Handles video files
            elif config.pipeline.source_path and os.path.isdir(
                config.pipeline.source_path
            ):
                source = FrameFolderSource(config.pipeline.source_path)
            else:
                print(
                    "Error: Ego4D requires --source-path to be a video file or frame directory."
                )
                return
        else:
            # Default to frame folder if source_path is a dir, or video if it's a file/id
            if config.pipeline.source_path and os.path.isdir(
                config.pipeline.source_path
            ):
                source = FrameFolderSource(config.pipeline.source_path)
            else:
                source = WebcamFrameSource(config.pipeline.source_path)

    if source is None:
        print(
            "Error: Could not determine frame source. Check --mode and --source-path."
        )
        return

    print(f"Starting pipeline in {args.mode} mode.")
    print(f"YOLO Weights Path: {config.models.yolo_weights_path}")

    components = setup_components(config)

    if config.pipeline.execution_mode == "threaded_parallel":
        from app.pipeline.executor import ThreadedParallelExecutor

        executor = ThreadedParallelExecutor()
    else:
        executor = SequentialExecutor()

    orchestrator = PipelineOrchestrator(executor, components)

    if args.mode == "live":
        tracker = MetricsTracker(base_dir=base_dir, run_name="live_run", config=config)
        runner = LiveRunner(config, orchestrator, tracker, source)
        runner.run()
    elif args.mode == "dataset_eval":
        tracker = MetricsTracker(
            base_dir=base_dir, run_name="dataset_eval", config=config
        )
        runner = DatasetRunner(config, orchestrator, tracker, source)
        runner.run()
    elif args.mode == "benchmark":
        from app.runners.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner(
            config,
            components,
            base_dir,
            source,
            suite_config_path=args.benchmark_config,
            suite_targets=args.benchmark_target,
        )
        runner.run()
    else:
        print(f"Mode {args.mode} not fully implemented yet.")

    # Teardown TTS
    if "tts" in components:
        components["tts"].stop()


if __name__ == "__main__":
    main()
