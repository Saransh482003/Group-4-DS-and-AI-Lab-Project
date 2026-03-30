import json
import importlib
import os
import platform
import sys
import time
from datetime import datetime

import torch

try:
    import psutil
except ImportError:
    psutil = None


class PerformanceMetricsLogger:
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
        mode_name = "deterministic" if nav_logic_mode == 0 else "slm"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{timestamp}_{mode_name}"
        self.run_dir = os.path.join(base_dir, "perf_logs", self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.log_paths = {
            "yolo": os.path.join(self.run_dir, "yolo_metrics.jsonl"),
            "depth": os.path.join(self.run_dir, "depth_metrics.jsonl"),
            "navigation": os.path.join(self.run_dir, "navigation_metrics.jsonl"),
            "tts": os.path.join(self.run_dir, "tts_metrics.jsonl"),
            "overall": os.path.join(self.run_dir, "overall_metrics.jsonl"),
            "resources": os.path.join(self.run_dir, "resources_metrics.jsonl"),
            "module_summary": os.path.join(self.run_dir, "module_summary.json"),
            "architecture": os.path.join(self.run_dir, "architecture_metrics.json"),
            "run_metadata": os.path.join(self.run_dir, "run_metadata.json"),
        }

        self._handles = {
            "yolo": open(self.log_paths["yolo"], "w", encoding="utf-8", buffering=65536),
            "depth": open(self.log_paths["depth"], "w", encoding="utf-8", buffering=65536),
            "navigation": open(self.log_paths["navigation"], "w", encoding="utf-8", buffering=65536),
            "tts": open(self.log_paths["tts"], "w", encoding="utf-8", buffering=65536),
            "overall": open(self.log_paths["overall"], "w", encoding="utf-8", buffering=65536),
            "resources": open(self.log_paths["resources"], "w", encoding="utf-8", buffering=65536),
        }

        self.start_wall_time = time.time()
        self.start_perf_counter = time.perf_counter()
        self.model_load_ms = None

        self.config = {
            "nav_logic_mode": nav_logic_mode,
            "plot_stream_mode": plot_stream_mode,
            "video_source": video_source,
            "show_windows": show_windows,
            "tts_enabled": tts_enabled,
            "yolo_weights": yolo_weights,
            "depth_model_dir": depth_model_dir,
        }

        self.module_latency_samples = {
            "yolo": [],
            "depth": [],
            "navigation": [],
            "tts": [],
            "overall": [],
            "capture": [],
            "video_write": [],
            "display": [],
            "deterministic_nav": [],
            "slm_nav": [],
            "visualization": [],
            "spatial": [],
        }

        self.frame_count = 0
        self.warmup_frame_count = 0
        self.tts_event_count = 0
        self.nav_command_counts = {}
        self.total_detections = 0
        self.resource_sample_count = 0

        self.process = psutil.Process(os.getpid()) if psutil else None
        if self.process is not None:
            self.process.cpu_percent(interval=None)

        self.peak_process_rss_mb = 0.0
        self.peak_process_cpu_percent = 0.0
        self.peak_system_memory_percent = 0.0
        self.peak_gpu_memory_allocated_mb = 0.0
        self.peak_gpu_memory_reserved_mb = 0.0
        self.peak_gpu_utilization_percent = 0.0

        self._nvml = None
        self._nvml_handle = None
        self._init_nvml()

    def _init_nvml(self):
        try:
            pynvml = importlib.import_module("pynvml")
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None
            self._nvml_handle = None

    def set_model_load_ms(self, value_ms):
        self.model_load_ms = float(value_ms)

    def _append_jsonl(self, stream_name, payload):
        handle = self._handles[stream_name]
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _append_sample(self, name, value):
        if value is None:
            return
        self.module_latency_samples[name].append(float(value))

    def _percentile(self, values, percentile):
        if not values:
            return None
        data = sorted(values)
        if len(data) == 1:
            return data[0]
        rank = (len(data) - 1) * (percentile / 100.0)
        lo = int(rank)
        hi = min(lo + 1, len(data) - 1)
        frac = rank - lo
        return data[lo] * (1.0 - frac) + data[hi] * frac

    def _make_stats(self, values):
        if not values:
            return {
                "count": 0,
                "mean_ms": None,
                "min_ms": None,
                "max_ms": None,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
            }
        return {
            "count": len(values),
            "mean_ms": sum(values) / len(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "p50_ms": self._percentile(values, 50),
            "p95_ms": self._percentile(values, 95),
            "p99_ms": self._percentile(values, 99),
        }

    def sample_resources(self, device):
        payload = {
            "timestamp": time.time(),
            "process_cpu_percent": None,
            "process_rss_mb": None,
            "process_vms_mb": None,
            "system_memory_percent": None,
            "gpu_available": bool(torch.cuda.is_available()),
            "gpu_utilization_percent": None,
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_memory_used_mb": None,
            "gpu_source": "none",
        }

        if self.process is not None:
            try:
                mem_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent(interval=None)
                payload["process_cpu_percent"] = float(cpu_percent)
                payload["process_rss_mb"] = float(mem_info.rss / (1024.0 * 1024.0))
                payload["process_vms_mb"] = float(mem_info.vms / (1024.0 * 1024.0))
                self.peak_process_rss_mb = max(self.peak_process_rss_mb, payload["process_rss_mb"])
                self.peak_process_cpu_percent = max(self.peak_process_cpu_percent, payload["process_cpu_percent"])
            except Exception:
                pass

            try:
                vm = psutil.virtual_memory()
                payload["system_memory_percent"] = float(vm.percent)
                self.peak_system_memory_percent = max(self.peak_system_memory_percent, payload["system_memory_percent"])
            except Exception:
                pass

        if device == "cuda" and torch.cuda.is_available():
            try:
                payload["gpu_memory_allocated_mb"] = float(torch.cuda.memory_allocated() / (1024.0 * 1024.0))
                payload["gpu_memory_reserved_mb"] = float(torch.cuda.memory_reserved() / (1024.0 * 1024.0))
                self.peak_gpu_memory_allocated_mb = max(
                    self.peak_gpu_memory_allocated_mb,
                    payload["gpu_memory_allocated_mb"],
                )
                self.peak_gpu_memory_reserved_mb = max(
                    self.peak_gpu_memory_reserved_mb,
                    payload["gpu_memory_reserved_mb"],
                )
                payload["gpu_source"] = "torch"
            except Exception:
                pass

            if self._nvml is not None and self._nvml_handle is not None:
                try:
                    util = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    mem = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    payload["gpu_utilization_percent"] = float(util.gpu)
                    payload["gpu_memory_total_mb"] = float(mem.total / (1024.0 * 1024.0))
                    payload["gpu_memory_used_mb"] = float(mem.used / (1024.0 * 1024.0))
                    payload["gpu_source"] = "nvml"
                    self.peak_gpu_utilization_percent = max(
                        self.peak_gpu_utilization_percent,
                        payload["gpu_utilization_percent"],
                    )
                except Exception:
                    pass

        return payload

    def log_frame(self, frame_idx, frame_metrics, resource_snapshot):
        now = time.time()
        detection_count = int(frame_metrics.get("detection_count", 0))
        nav_command = frame_metrics.get("nav_command")
        is_warmup = bool(frame_metrics.get("is_warmup", False))

        self.frame_count += 1
        if is_warmup:
            self.warmup_frame_count += 1
        self.total_detections += detection_count
        if nav_command:
            self.nav_command_counts[nav_command] = self.nav_command_counts.get(nav_command, 0) + 1

        yolo_payload = {
            "timestamp": now,
            "frame": frame_idx,
            "latency_ms": frame_metrics.get("yolo_latency_ms"),
            "detection_count": detection_count,
            "mode": frame_metrics.get("nav_mode_name"),
        }
        self._append_jsonl("yolo", yolo_payload)

        depth_payload = {
            "timestamp": now,
            "frame": frame_idx,
            "latency_ms": frame_metrics.get("depth_latency_ms"),
            "mode": frame_metrics.get("nav_mode_name"),
        }
        self._append_jsonl("depth", depth_payload)

        nav_payload = {
            "timestamp": now,
            "frame": frame_idx,
            "latency_ms": frame_metrics.get("navigation_latency_ms"),
            "deterministic_latency_ms": frame_metrics.get("deterministic_nav_latency_ms"),
            "slm_latency_ms": frame_metrics.get("slm_nav_latency_ms"),
            "command": nav_command,
            "zone_risks": frame_metrics.get("zone_risks"),
            "mode": frame_metrics.get("nav_mode_name"),
        }
        self._append_jsonl("navigation", nav_payload)

        tts_payload = {
            "timestamp": now,
            "frame": frame_idx,
            "latency_ms": frame_metrics.get("tts_latency_ms"),
            "should_speak": frame_metrics.get("tts_should_speak"),
            "skip_reason": frame_metrics.get("tts_skip_reason"),
            "mode": frame_metrics.get("tts_mode"),
            "error": frame_metrics.get("tts_error"),
        }
        self._append_jsonl("tts", tts_payload)

        overall_payload = {
            "timestamp": now,
            "frame": frame_idx,
            "frame_total_latency_ms": frame_metrics.get("frame_total_latency_ms"),
            "app_loop_latency_ms": frame_metrics.get("app_loop_latency_ms"),
            "capture_latency_ms": frame_metrics.get("capture_latency_ms"),
            "video_write_latency_ms": frame_metrics.get("video_write_latency_ms"),
            "display_latency_ms": frame_metrics.get("display_latency_ms"),
            "visualization_latency_ms": frame_metrics.get("visualization_latency_ms"),
            "spatial_latency_ms": frame_metrics.get("spatial_latency_ms"),
            "fps_instant": frame_metrics.get("fps_instant"),
            "is_warmup": is_warmup,
            "mode": frame_metrics.get("nav_mode_name"),
        }
        self._append_jsonl("overall", overall_payload)

        if resource_snapshot is None:
            resource_snapshot = {"sampled": False}
        else:
            resource_snapshot["sampled"] = True
            self.resource_sample_count += 1

        resource_payload = {
            "timestamp": now,
            "frame": frame_idx,
            "is_warmup": is_warmup,
            **resource_snapshot,
        }
        self._append_jsonl("resources", resource_payload)

        if not is_warmup:
            self._append_sample("yolo", frame_metrics.get("yolo_latency_ms"))
            self._append_sample("depth", frame_metrics.get("depth_latency_ms"))
            self._append_sample("navigation", frame_metrics.get("navigation_latency_ms"))
            self._append_sample("tts", frame_metrics.get("tts_latency_ms"))
            self._append_sample("overall", frame_metrics.get("app_loop_latency_ms"))
            self._append_sample("capture", frame_metrics.get("capture_latency_ms"))
            self._append_sample("video_write", frame_metrics.get("video_write_latency_ms"))
            self._append_sample("display", frame_metrics.get("display_latency_ms"))
            self._append_sample("deterministic_nav", frame_metrics.get("deterministic_nav_latency_ms"))
            self._append_sample("slm_nav", frame_metrics.get("slm_nav_latency_ms"))
            self._append_sample("visualization", frame_metrics.get("visualization_latency_ms"))
            self._append_sample("spatial", frame_metrics.get("spatial_latency_ms"))

        if frame_metrics.get("tts_should_speak"):
            self.tts_event_count += 1

    def finalize(self, output_json_path, output_video_path):
        end_wall_time = time.time()
        end_perf_counter = time.perf_counter()
        duration_sec = max(0.0, end_perf_counter - self.start_perf_counter)

        for handle in self._handles.values():
            handle.flush()
            handle.close()

        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

        module_summary = {
            "yolo": self._make_stats(self.module_latency_samples["yolo"]),
            "depth": self._make_stats(self.module_latency_samples["depth"]),
            "navigation": self._make_stats(self.module_latency_samples["navigation"]),
            "tts": self._make_stats(self.module_latency_samples["tts"]),
            "overall": self._make_stats(self.module_latency_samples["overall"]),
            "capture": self._make_stats(self.module_latency_samples["capture"]),
            "video_write": self._make_stats(self.module_latency_samples["video_write"]),
            "display": self._make_stats(self.module_latency_samples["display"]),
            "deterministic_nav": self._make_stats(self.module_latency_samples["deterministic_nav"]),
            "slm_nav": self._make_stats(self.module_latency_samples["slm_nav"]),
            "visualization": self._make_stats(self.module_latency_samples["visualization"]),
            "spatial": self._make_stats(self.module_latency_samples["spatial"]),
        }

        architecture_metrics = {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "start_time_epoch": self.start_wall_time,
            "end_time_epoch": end_wall_time,
            "duration_sec": duration_sec,
            "frames_processed": self.frame_count,
            "avg_fps": (self.frame_count / duration_sec) if duration_sec > 0 else None,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": (
                self.total_detections / self.frame_count if self.frame_count > 0 else None
            ),
            "tts_event_count": self.tts_event_count,
            "nav_command_distribution": self.nav_command_counts,
            "resource_peaks": {
                "peak_process_rss_mb": self.peak_process_rss_mb,
                "peak_process_cpu_percent": self.peak_process_cpu_percent,
                "peak_system_memory_percent": self.peak_system_memory_percent,
                "peak_gpu_memory_allocated_mb": self.peak_gpu_memory_allocated_mb,
                "peak_gpu_memory_reserved_mb": self.peak_gpu_memory_reserved_mb,
                "peak_gpu_utilization_percent": self.peak_gpu_utilization_percent,
            },
            "latency_summary_ms": module_summary,
            "model_load_ms": self.model_load_ms,
            "warmup_frames_excluded": self.warmup_frame_count,
            "steady_state_frames": max(0, self.frame_count - self.warmup_frame_count),
            "resource_samples_taken": self.resource_sample_count,
            "output_json_path": output_json_path,
            "output_video_path": output_video_path,
            "system": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "processor": platform.processor(),
                "machine": platform.machine(),
                "cpu_count_logical": os.cpu_count(),
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "psutil_available": psutil is not None,
                "nvml_available": self._nvml_handle is not None,
            },
            "model_artifacts": {
                "yolo_weights": self.config["yolo_weights"],
                "yolo_weights_size_mb": self._safe_file_size_mb(self.config["yolo_weights"]),
                "depth_model_dir": self.config["depth_model_dir"],
                "depth_model_dir_size_mb": self._safe_dir_size_mb(self.config["depth_model_dir"]),
            },
            "config": self.config,
        }

        run_metadata = {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "files": self.log_paths,
            "duration_sec": duration_sec,
            "frames_processed": self.frame_count,
            "warmup_frames_excluded": self.warmup_frame_count,
            "steady_state_frames": max(0, self.frame_count - self.warmup_frame_count),
            "resource_samples_taken": self.resource_sample_count,
        }

        self._write_json(self.log_paths["module_summary"], module_summary)
        self._write_json(self.log_paths["architecture"], architecture_metrics)
        self._write_json(self.log_paths["run_metadata"], run_metadata)

    def _safe_file_size_mb(self, path):
        try:
            if os.path.isfile(path):
                return float(os.path.getsize(path) / (1024.0 * 1024.0))
        except Exception:
            return None
        return None

    def _safe_dir_size_mb(self, directory):
        try:
            total = 0
            for root, _, files in os.walk(directory):
                for name in files:
                    full_path = os.path.join(root, name)
                    try:
                        total += os.path.getsize(full_path)
                    except OSError:
                        continue
            return float(total / (1024.0 * 1024.0))
        except Exception:
            return None

    def _write_json(self, path, payload):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
