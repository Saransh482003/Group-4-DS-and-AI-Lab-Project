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
		plot_stream_mode,
		video_source,
		show_windows,
		tts_enabled,
		yolo_weights,
		depth_model_dir,
		process_every_n_frames=1,
	):
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.run_dir = os.path.join(base_dir, "performance_runs", f"run_{timestamp}")
		os.makedirs(self.run_dir, exist_ok=True)

		self.started_at_epoch = time.time()
		self.model_load_ms = None
		self.frames = []
		self.resource_samples = []

		self.run_config = {
			"plot_stream_mode": plot_stream_mode,
			"video_source": str(video_source),
			"show_windows": bool(show_windows),
			"tts_enabled": bool(tts_enabled),
			"yolo_weights": yolo_weights,
			"depth_model_dir": depth_model_dir,
			"process_every_n_frames": int(max(1, process_every_n_frames)),
			"frame_skipping_enabled": int(max(1, process_every_n_frames)) > 1,
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

	def _safe_mean(self, values):
		if not values:
			return None
		return sum(values) / float(len(values))

	def _safe_percentile(self, values, percentile):
		if not values:
			return None
		sorted_values = sorted(values)
		if len(sorted_values) == 1:
			return sorted_values[0]
		index = int(round((percentile / 100.0) * (len(sorted_values) - 1)))
		index = max(0, min(index, len(sorted_values) - 1))
		return sorted_values[index]

	def _build_summary(self):
		finished_at = time.time()
		duration_seconds = finished_at - self.started_at_epoch
		total_frames_captured = len(self.frames)
		total_frames_skipped = sum(
			1 for frame in self.frames if frame.get("metrics", {}).get("frame_skipped", False)
		)
		total_frames_processed = total_frames_captured - total_frames_skipped

		app_loop_values = []
		for frame in self.frames:
			value = frame.get("metrics", {}).get("app_loop_latency_ms")
			if value is None:
				continue
			try:
				app_loop_values.append(float(value))
			except (TypeError, ValueError):
				continue

		skip_ratio = (
			(total_frames_skipped / float(total_frames_captured))
			if total_frames_captured > 0
			else 0.0
		)
		effective_capture_fps = (
			(total_frames_captured / duration_seconds)
			if duration_seconds > 0
			else None
		)
		processed_fps = (
			(total_frames_processed / duration_seconds)
			if duration_seconds > 0
			else None
		)

		summary = {
			"started_at_epoch": self.started_at_epoch,
			"finished_at_epoch": finished_at,
			"duration_seconds": duration_seconds,
			"model_load_ms": self.model_load_ms,
			"total_frames": total_frames_captured,
			"total_frames_captured": total_frames_captured,
			"total_frames_processed": total_frames_processed,
			"total_frames_skipped": total_frames_skipped,
			"skip_ratio": skip_ratio,
			"effective_capture_fps": effective_capture_fps,
			"processed_fps": processed_fps,
			"resource_samples": len(self.resource_samples),
			"run_config": self.run_config,
			"module_latencies": {},
		}

		modules = [
			"yolo_latency_ms",
			"depth_latency_ms",
			"depth_hazard_latency_ms",
			"navigation_latency_ms",
			"tts_latency_ms",
			"frame_total_latency_ms",
			"app_loop_latency_ms",
			"fps_instant"
		]

		for mod in modules:
			# Gather non-None values for this module
			vals = [
				float(f["metrics"][mod]) for f in self.frames
				if f.get("metrics") and f["metrics"].get(mod) is not None
			]
			summary["module_latencies"][mod] = {
				"mean": self._safe_mean(vals),
				"p50": self._safe_percentile(vals, 50),
				"p95": self._safe_percentile(vals, 95),
			}

		return summary

	def _generate_plots(self):
		"""Generates latency and resource usage plots for the run."""
		try:
			import matplotlib.pyplot as plt
			import numpy as np
		except ImportError:
			print("matplotlib or numpy not found. Skipping performance plot generation.")
			return

		plots_dir = os.path.join(self.run_dir, "plots")
		os.makedirs(plots_dir, exist_ok=True)

		# Filter to only processed frames for cleaner metrics over time
		processed_frames = [f for f in self.frames if not f.get("metrics", {}).get("frame_skipped", False)]
		if not processed_frames:
			return

		frame_indices = [f["frame_idx"] for f in processed_frames]

		# 1. Module Latency Over Time (Line Plot)
		plt.figure(figsize=(12, 6))
		modules_to_plot = {
			"YOLO": "yolo_latency_ms",
			"Depth": "depth_latency_ms",
			"Hazard": "depth_hazard_latency_ms",
			"Nav": "navigation_latency_ms",
			"TTS": "tts_latency_ms",
			"Total Frame": "frame_total_latency_ms"
		}
		for label, key in modules_to_plot.items():
			vals = [f["metrics"].get(key, 0) or 0 for f in processed_frames]
			plt.plot(frame_indices, vals, label=label, alpha=0.8, linewidth=1.5)
		
		plt.xlabel("Frame Index")
		plt.ylabel("Latency (ms)")
		plt.title("Module Latency Over Time (Live Pipeline)")
		plt.legend()
		plt.grid(alpha=0.3)
		plt.tight_layout()
		plt.savefig(os.path.join(plots_dir, "latency_over_time.png"), dpi=150)
		plt.close()

		# 2. Mean vs P95 Latency Summary (Bar Chart)
		summary = self._build_summary()
		labels = list(modules_to_plot.keys())
		keys = list(modules_to_plot.values())
		
		means = [summary["module_latencies"][k]["mean"] or 0 for k in keys]
		p95s = [summary["module_latencies"][k]["p95"] or 0 for k in keys]

		x = np.arange(len(labels))
		width = 0.35
		
		plt.figure(figsize=(10, 6))
		plt.bar(x - width/2, means, width, label='Mean', color='#4C72B0')
		plt.bar(x + width/2, p95s, width, label='P95', color='#C44E52')
		plt.xticks(x, labels)
		plt.ylabel("Latency (ms)")
		plt.title("Mean vs P95 Latency by Module")
		plt.legend()
		plt.grid(axis="y", alpha=0.3)
		plt.tight_layout()
		plt.savefig(os.path.join(plots_dir, "latency_summary_bar.png"), dpi=150)
		plt.close()

		# 3. FPS Over Time (Line Plot)
		fps_vals = [f["metrics"].get("fps_instant") or 0 for f in processed_frames]
		plt.figure(figsize=(10, 4))
		plt.plot(frame_indices, fps_vals, color='#55A868', linewidth=2)
		plt.xlabel("Frame Index")
		plt.ylabel("FPS")
		plt.title("Instantaneous FPS Over Time")
		plt.grid(alpha=0.3)
		plt.tight_layout()
		plt.savefig(os.path.join(plots_dir, "fps_over_time.png"), dpi=150)
		plt.close()

		# 4. Subsystem Resources Over Time
		if self.resource_samples:
			times = [r["timestamp_epoch"] - self.started_at_epoch for r in self.resource_samples]
			cpu = [r.get("cpu_percent") or 0 for r in self.resource_samples]
			ram = [r.get("ram_percent") or 0 for r in self.resource_samples]
			
			fig, ax1 = plt.subplots(figsize=(10, 5))
			ax1.plot(times, cpu, label='CPU %', color='#333333', linewidth=2)
			ax1.plot(times, ram, label='RAM %', color='#8172B2', linewidth=2)
			ax1.set_xlabel("Time (Seconds from Start)")
			ax1.set_ylabel("Utilization (%)")
			ax1.tick_params(axis='y')
			
			has_gpu = any(r.get("gpu_mem_allocated_mb") for r in self.resource_samples)
			if has_gpu:
                # Secondary axis for GPU memory
				ax2 = ax1.twinx()
				gpu = [r.get("gpu_mem_allocated_mb") or 0 for r in self.resource_samples]
				ax2.plot(times, gpu, label='GPU Mem (MB)', color='#DD8452', linewidth=2, linestyle='--')
				ax2.set_ylabel("GPU Memory (MB)", color='#DD8452')
				ax2.tick_params(axis='y', labelcolor='#DD8452')
				lines_1, labels_1 = ax1.get_legend_handles_labels()
				lines_2, labels_2 = ax2.get_legend_handles_labels()
				ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
			else:
				ax1.legend(loc='upper right')

			plt.title("System Resource Utilization")
			plt.grid(alpha=0.3)
			plt.tight_layout()
			plt.savefig(os.path.join(plots_dir, "resource_utilization.png"), dpi=150)
			plt.close()

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
		
		# Generate the visualizations
		self._generate_plots()
