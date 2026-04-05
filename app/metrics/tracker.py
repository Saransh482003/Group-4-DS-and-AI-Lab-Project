import os
import time
import json
from datetime import datetime
from typing import Dict, Any

from app.config.pipeline_schema import AppConfig
from app.pipeline.frame_context import FrameContext

class MetricsTracker:
    """Collects frame metrics during a run and exports them on completion."""

    def __init__(self, base_dir: str, run_name: str, config: AppConfig):
        self.base_dir = base_dir
        self.run_name = run_name
        self.config = config

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, "outputs", f"{run_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.started_at_epoch = time.time()
        self.frames = []
        self.resource_samples = []

    def sample_resources(self):
        sample = {
            "timestamp_epoch": time.time(),
            "cpu_percent": None,
            "ram_percent": None,
            "gpu_mem_allocated_mb": None,
            "gpu_mem_reserved_mb": None,
        }

        try:
            import psutil
            sample["cpu_percent"] = psutil.cpu_percent(interval=None)
            sample["ram_percent"] = psutil.virtual_memory().percent
        except ImportError:
            pass

        try:
            import torch
            if self.config.models.device == "cuda" and torch.cuda.is_available():
                sample["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024.0 * 1024.0), 2)
                sample["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024.0 * 1024.0), 2)
        except ImportError:
            pass

        self.resource_samples.append(sample)
        return sample

    def log_frame(self, ctx: FrameContext):
        entry = {
            "frame_idx": ctx.frame_idx,
            "source_id": ctx.source_id,
            "timestamp_epoch": time.time(),
            "metrics": ctx.metrics,
        }

        # Sample resources periodically
        if ctx.frame_idx % max(1, self.config.pipeline.sample_resources_every_n_frames) == 0:
            resource = self.sample_resources()
            entry["resource"] = resource
            ctx.metrics["resource_sampled"] = True
        else:
            ctx.metrics["resource_sampled"] = False

        self.frames.append(entry)

    def _build_summary(self):
        finished_at = time.time()
        return {
            "started_at_epoch": self.started_at_epoch,
            "finished_at_epoch": finished_at,
            "duration_seconds": finished_at - self.started_at_epoch,
            "total_frames": len(self.frames),
            "resource_samples": len(self.resource_samples),
        }

    def finalize(self):
        summary = self._build_summary()

        with open(os.path.join(self.run_dir, "run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        with open(os.path.join(self.run_dir, "frame_metrics.json"), "w") as f:
            json.dump(self.frames, f, indent=2)

        with open(os.path.join(self.run_dir, "resource_samples.json"), "w") as f:
            json.dump(self.resource_samples, f, indent=2)

        # Export CSVs using aggregator
        self._export_csvs()

    def _export_csvs(self):
        from app.metrics.aggregator import MetricsAggregator
        from app.metrics.exporters import export_to_csv

        if not self.frames:
            return

        aggregator = MetricsAggregator(self.frames)

        frame_metrics_flat = aggregator.get_flat_frame_metrics()
        export_to_csv(frame_metrics_flat, os.path.join(self.run_dir, "frame_metrics.csv"))

        command_dist = aggregator.get_command_distribution()
        export_to_csv([{"command": k, "count": v} for k,v in command_dist.items()], os.path.join(self.run_dir, "command_distribution.csv"))

        latency_summary = aggregator.get_latency_summary()
        export_to_csv([latency_summary], os.path.join(self.run_dir, "latency_summary.csv"))

        top_risk = aggregator.get_top_risk_frames()
        export_to_csv(top_risk, os.path.join(self.run_dir, "top_risk_frames.csv"))

        slowest = aggregator.get_slowest_frames()
        export_to_csv(slowest, os.path.join(self.run_dir, "slowest_frames.csv"))

        if self.resource_samples:
            export_to_csv(self.resource_samples, os.path.join(self.run_dir, "resource_samples.csv"))
