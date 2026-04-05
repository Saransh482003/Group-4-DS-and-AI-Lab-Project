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

def setup_components(config):
    components = {}

    if config.pipeline.enable_detection:
        components["detector"] = DetectorComponent(config.models.yolo_weights_path)

    if config.pipeline.enable_depth:
        components["depth"] = DepthComponent(config.models.depth_model_path, config.models.device)

    if config.pipeline.enable_navigation:
        # Assuming frame width from typical webcam 640
        components["navigation"] = NavigationComponent(frame_width=640)

    if config.pipeline.enable_visualization:
        components["visualization"] = VisualizationComponent(plot_stream_mode=config.pipeline.plot_stream_mode)

    if config.pipeline.enable_tts:
        from app.utils.paths import env_rel_path
        components["tts"] = TTSComponent(
            piper_exe=str(env_rel_path("PIPER_EXE_REL", os.path.join("piper", "piper.exe"))),
            voice_model_path=str(env_rel_path("PIPER_VOICE_MODEL_REL", os.path.join("piper_voices", "en_US-amy-medium.onnx"))),
            voice_config_path=str(env_rel_path("PIPER_VOICE_CONFIG_REL", os.path.join("piper_voices", "en_US-amy-medium.onnx.json")))
        )

    return components

def main():
    parser = argparse.ArgumentParser(description="Indoor Navigation Evaluation Pipeline")
    parser.add_argument("--mode", type=str, choices=["live", "dataset_eval", "benchmark"], default="live")
    parser.add_argument("--dataset", type=str, help="Dataset slug for evaluation/benchmark mode")
    parser.add_argument("--source-path", type=str, help="Path to input source (camera id, video file, or image folder)")
    parser.add_argument("--execution-mode", type=str, choices=["sequential", "threaded_parallel"], help="Execution mode")
    parser.add_argument("--max-frames", type=int, help="Max frames to process")
    parser.add_argument("--stride", type=int, help="Frame stride for dataset eval")
    parser.add_argument("--enable-tts", action="store_true", help="Enable TTS output")
    parser.add_argument("--save-annotated-video", action="store_true", help="Save annotated video output")
    parser.add_argument("--show-windows", action="store_true", help="Show display windows")
    parser.add_argument("--device", type=str, help="Device to run models on")
    parser.add_argument("--output-dir", type=str, help="Output directory for artifacts")

    args = parser.parse_args()

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

    # Dataset resolution if kaggle util is available
    if args.dataset == "egoblind":
        from app.utils.kaggle_data import ensure_egoblind_dataset
        resolved_path = ensure_egoblind_dataset(config.kaggle)
        if resolved_path:
            config.pipeline.source_path = resolved_path
        else:
            print("Failed to resolve Kaggle dataset.")
            return

    print(f"Starting pipeline in {args.mode} mode.")

    components = setup_components(config)

    if config.pipeline.execution_mode == "threaded_parallel":
        from app.pipeline.executor import ThreadedParallelExecutor
        executor = ThreadedParallelExecutor()
    else:
        executor = SequentialExecutor()

    orchestrator = PipelineOrchestrator(executor, components)

    base_dir = args.output_dir if args.output_dir else str(get_base_dir())

    if args.mode == "live":
        tracker = MetricsTracker(base_dir=base_dir, run_name="live_run", config=config)
        runner = LiveRunner(config, orchestrator, tracker)
        runner.run()
    elif args.mode == "dataset_eval":
        tracker = MetricsTracker(base_dir=base_dir, run_name="dataset_eval", config=config)
        runner = DatasetRunner(config, orchestrator, tracker)
        runner.run()
    elif args.mode == "benchmark":
        from app.runners.benchmark_runner import BenchmarkRunner
        runner = BenchmarkRunner(config, components, base_dir)
        runner.run()
    else:
        print(f"Mode {args.mode} not fully implemented yet.")

    # Teardown TTS
    if "tts" in components:
        components["tts"].stop()

if __name__ == "__main__":
    main()
