import time

from performance_metrics import PerformanceMetricsLogger


class PipelinePerformanceTracker:
	"""Owns frame/app performance metric collection and persistence."""

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
		process_every_n_frames=1,
	):
		self.logger = PerformanceMetricsLogger(
			base_dir=base_dir,
			nav_logic_mode=nav_logic_mode,
			plot_stream_mode=plot_stream_mode,
			video_source=video_source,
			show_windows=show_windows,
			tts_enabled=tts_enabled,
			yolo_weights=yolo_weights,
			depth_model_dir=depth_model_dir,
			process_every_n_frames=process_every_n_frames,
		)

	def set_model_load_ms(self, value_ms):
		self.logger.set_model_load_ms(value_ms)

	def init_frame_metrics(self, frame_idx, nav_logic_mode, warmup_frames):
		return {
			"yolo_latency_ms": None,
			"depth_latency_ms": None,
			"depth_hazard_latency_ms": None,
			"spatial_latency_ms": None,
			"navigation_latency_ms": None,
			"deterministic_nav_latency_ms": None,
			"slm_nav_latency_ms": None,
			"visualization_latency_ms": None,
			"tts_latency_ms": 0.0,
			"tts_should_speak": False,
			"tts_skip_reason": None,
			"tts_mode": None,
			"tts_error": None,
			"detection_count": 0,
			"depth_hazard_danger_pixel_count": 0,
			"depth_hazard_warning_pixel_count": 0,
			"depth_hazard_near_pixel_count": 0,
			"depth_hazard_min_depth_m": None,
			"frame_total_latency_ms": None,
			"nav_mode_name": "deterministic" if nav_logic_mode == 0 else "slm",
			"is_warmup": frame_idx <= warmup_frames,
			"resource_sampled": False,
			"tts_enqueue_ok": None,
			"captured_frame_idx": None,
			"processed_frame_idx": None,
			"process_every_n_frames": 1,
			"frame_skipped": False,
			"skip_reason": None,
			"inference_reused": False,
		}

	def attach_loop_metrics(self, frame_metrics, loop_start, capture_ms, video_write_ms, display_ms):
		frame_metrics["capture_latency_ms"] = capture_ms
		frame_metrics["video_write_latency_ms"] = video_write_ms
		frame_metrics["display_latency_ms"] = display_ms
		frame_metrics["app_loop_latency_ms"] = (time.perf_counter() - loop_start) * 1000.0
		frame_metrics["fps_instant"] = (
			1000.0 / frame_metrics["app_loop_latency_ms"]
			if frame_metrics["app_loop_latency_ms"] > 0
			else None
		)

	def maybe_sample_resources(self, frame_idx, frame_metrics, device, sample_every_n_frames):
		if frame_idx % max(1, sample_every_n_frames) == 0:
			frame_metrics["resource_sampled"] = True
			return self.logger.sample_resources(device)
		return None

	def sample_resources(self, device):
		return self.logger.sample_resources(device)

	def log_frame(self, frame_idx, frame_metrics, resource_snapshot):
		self.logger.log_frame(frame_idx, frame_metrics, resource_snapshot)

	def finalize_run(self, output_json=None, output_video=None, all_frames=None):
		if output_json and all_frames is not None:
			with open(output_json, "w", encoding="utf-8") as f:
				import json

				json.dump(all_frames, f, indent=2)
		self.logger.finalize(output_json, output_video)

	@property
	def run_dir(self):
		return self.logger.run_dir
