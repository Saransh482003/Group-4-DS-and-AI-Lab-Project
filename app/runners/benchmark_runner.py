import time
import os
import copy
from typing import List, Dict, Any

from app.config.pipeline_schema import AppConfig
from app.runners.dataset_runner import DatasetRunner
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.executor import SequentialExecutor, ThreadedParallelExecutor
from app.metrics.tracker import MetricsTracker
from app.metrics.exporters import export_to_csv


class BenchmarkRunner:
    """Runs multiple configurations on the same source and exports a comparison."""

    def __init__(self, base_config: AppConfig, base_components: dict, base_dir: str):
        self.base_config = base_config
        self.base_components = base_components
        self.base_dir = base_dir

    def run(self):
        # The 8 modes we want to benchmark
        configs_to_test = [
            {"name": "detection_only", "det": True, "dep": False, "tts": False, "exec": "sequential"},
            {"name": "depth_only", "det": False, "dep": True, "tts": False, "exec": "sequential"},
            {"name": "sequential_det_dep", "det": True, "dep": True, "tts": False, "exec": "sequential"},
            {"name": "parallel_det_dep", "det": True, "dep": True, "tts": False, "exec": "threaded_parallel"},
            {"name": "sequential_full_no_tts", "det": True, "dep": True, "tts": False, "exec": "sequential"},
            {"name": "sequential_full_with_tts", "det": True, "dep": True, "tts": True, "exec": "sequential"},
            {"name": "parallel_full_no_tts", "det": True, "dep": True, "tts": False, "exec": "threaded_parallel"},
            {"name": "parallel_full_with_tts", "det": True, "dep": True, "tts": True, "exec": "threaded_parallel"},
        ]

        results: List[Dict[str, Any]] = []

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        benchmark_dir = os.path.join(self.base_dir, "outputs", f"benchmark_{timestamp}")
        os.makedirs(benchmark_dir, exist_ok=True)

        for cfg in configs_to_test:
            print(f"\n--- Running benchmark config: {cfg['name']} ---")

            run_config = copy.deepcopy(self.base_config)
            run_config.pipeline.enable_detection = cfg["det"]
            run_config.pipeline.enable_depth = cfg["dep"]
            run_config.pipeline.enable_tts = cfg["tts"]
            run_config.pipeline.execution_mode = cfg["exec"]

            # Setup executor
            executor = SequentialExecutor() if cfg["exec"] == "sequential" else ThreadedParallelExecutor()

            # Filter components
            run_components = {}
            if cfg["det"] and "detector" in self.base_components:
                run_components["detector"] = self.base_components["detector"]
            if cfg["dep"] and "depth" in self.base_components:
                run_components["depth"] = self.base_components["depth"]
            if "navigation" in self.base_components:
                run_components["navigation"] = self.base_components["navigation"]
            if "visualization" in self.base_components:
                run_components["visualization"] = self.base_components["visualization"]
            if cfg["tts"] and "tts" in self.base_components:
                run_components["tts"] = self.base_components["tts"]

            orchestrator = PipelineOrchestrator(executor, run_components)

            # We don't want windows or video saving during benchmark to isolate performance
            run_config.pipeline.show_windows = False
            run_config.pipeline.save_annotated_video = False

            tracker = MetricsTracker(base_dir=self.base_dir, run_name=f"bench_{cfg['name']}", config=run_config)
            runner = DatasetRunner(run_config, orchestrator, tracker)

            # RUN
            runner.run()

            # Collect summarized metrics
            from app.metrics.aggregator import MetricsAggregator
            agg = MetricsAggregator(tracker.frames)
            summary = agg.get_latency_summary()

            res_entry = {
                "config_name": cfg["name"],
                "execution_mode": cfg["exec"],
                "total_frames": len(tracker.frames),
                "total_runtime_sec": tracker._build_summary()["duration_seconds"],
                "avg_fps": summary.get("avg_fps_instant"),
                "avg_frame_latency_ms": summary.get("avg_frame_total_latency_ms"),
                "p95_frame_latency_ms": summary.get("p95_frame_total_latency_ms"),
                "avg_yolo_latency_ms": summary.get("avg_yolo_latency_ms"),
                "avg_depth_latency_ms": summary.get("avg_depth_latency_ms"),
                "avg_tts_latency_ms": summary.get("avg_tts_latency_ms"),
            }
            results.append(res_entry)

        # Export comparison
        comparison_csv = os.path.join(benchmark_dir, "benchmark_comparison.csv")
        export_to_csv(results, comparison_csv)
        print(f"\nBenchmark complete. Comparison saved to: {comparison_csv}")

        # Markdown export
        md_path = os.path.join(benchmark_dir, "benchmark_comparison.md")
        with open(md_path, "w") as f:
            f.write("# Benchmark Comparison\n\n")
            f.write("| Config | Mode | FPS | Avg Latency (ms) | P95 Latency | YOLO (ms) | Depth (ms) | TTS (ms) |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in results:
                f.write(f"| {r['config_name']} | {r['execution_mode']} | {r['avg_fps']:.2f} | "
                        f"{r['avg_frame_latency_ms']:.2f} | {r.get('p95_frame_latency_ms', 0):.2f} | "
                        f"{r.get('avg_yolo_latency_ms') or 0:.2f} | {r.get('avg_depth_latency_ms') or 0:.2f} | "
                        f"{r.get('avg_tts_latency_ms') or 0:.2f} |\n")
