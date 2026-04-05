import json
import os
import platform
import time
from datetime import datetime


class PerformanceMetricsLogger:
	"""Simple run-level metrics logger used by the pipeline tracker."""

	def __init__(
		self,
		base_dir,
		nav_logic_mode,
		plot_stream_mode,
		video_source,
		show_windows,
		tts_enabled,
		yolo_weights,
		depth_model_dir,
	):
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.run_dir = os.path.join(base_dir, "performance_runs", f"run_{timestamp}")
		os.makedirs(self.run_dir, exist_ok=True)

		self.started_at_epoch = time.time()
		self.model_load_ms = None
		self.frames = []
		self.resource_samples = []

		self.run_config = {
			"nav_logic_mode": nav_logic_mode,
			"plot_stream_mode": plot_stream_mode,
			"video_source": str(video_source),
			"show_windows": bool(show_windows),
			"tts_enabled": bool(tts_enabled),
			"yolo_weights": yolo_weights,
			"depth_model_dir": depth_model_dir,
			"platform": platform.platform(),
		}

	def set_model_load_ms(self, value_ms):
		self.model_load_ms = float(value_ms) if value_ms is not None else None

	def sample_resources(self, device):
		sample = {
			"timestamp_epoch": time.time(),
			"device": device,
			"cpu_percent": None,
			"ram_percent": None,
			"gpu_mem_allocated_mb": None,
			"gpu_mem_reserved_mb": None,
		}

		try:
			import psutil

			sample["cpu_percent"] = psutil.cpu_percent(interval=None)
			sample["ram_percent"] = psutil.virtual_memory().percent
		except Exception:
			pass

		try:
			import torch

			if device == "cuda" and torch.cuda.is_available():
				sample["gpu_mem_allocated_mb"] = round(
					torch.cuda.memory_allocated() / (1024.0 * 1024.0),
					2,
				)
				sample["gpu_mem_reserved_mb"] = round(
					torch.cuda.memory_reserved() / (1024.0 * 1024.0),
					2,
				)
		except Exception:
			pass

		self.resource_samples.append(sample)
		return sample

	def log_frame(self, frame_idx, frame_metrics, resource_snapshot):
		entry = {
			"frame_idx": int(frame_idx),
			"timestamp_epoch": time.time(),
			"metrics": frame_metrics,
			"resource": resource_snapshot,
		}
		self.frames.append(entry)

	def _build_summary(self):
		finished_at = time.time()
		total_frames = len(self.frames)

		return {
			"started_at_epoch": self.started_at_epoch,
			"finished_at_epoch": finished_at,
			"duration_seconds": finished_at - self.started_at_epoch,
			"model_load_ms": self.model_load_ms,
			"total_frames": total_frames,
			"resource_samples": len(self.resource_samples),
			"run_config": self.run_config,
		}

	def finalize(self, output_json=None, output_video=None):
		_ = output_video  # Reserved for compatibility with the previous API.

		summary = self._build_summary()
		summary_path = os.path.join(self.run_dir, "run_summary.json")
		with open(summary_path, "w", encoding="utf-8") as f:
			json.dump(summary, f, indent=2)

		frames_path = output_json if output_json else os.path.join(self.run_dir, "frame_metrics.json")
		with open(frames_path, "w", encoding="utf-8") as f:
			json.dump(self.frames, f, indent=2)

		resources_path = os.path.join(self.run_dir, "resource_samples.json")
		with open(resources_path, "w", encoding="utf-8") as f:
			json.dump(self.resource_samples, f, indent=2)
