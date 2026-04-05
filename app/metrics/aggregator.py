import numpy as np
from typing import List, Dict, Any

class MetricsAggregator:
    def __init__(self, frames_data: List[Dict[str, Any]]):
        self.frames = frames_data

    def get_flat_frame_metrics(self) -> List[Dict[str, Any]]:
        flat_data = []
        for f in self.frames:
            flat = {"frame_idx": f["frame_idx"], "source_id": f["source_id"], "timestamp_epoch": f["timestamp_epoch"]}
            flat.update(f.get("metrics", {}))
            flat_data.append(flat)
        return flat_data

    def get_command_distribution(self) -> Dict[str, int]:
        dist = {}
        for f in self.frames:
            cmd = f.get("metrics", {}).get("nav_command", "None")
            if not cmd:
                cmd = "None"
            dist[cmd] = dist.get(cmd, 0) + 1
        return dist

    def get_latency_summary(self) -> Dict[str, Any]:
        metrics = self.get_flat_frame_metrics()
        if not metrics:
            return {}

        summary = {}
        keys_to_agg = [
            "yolo_latency_ms", "depth_latency_ms", "fusion_latency_ms",
            "navigation_latency_ms", "visualization_latency_ms", "tts_latency_ms",
            "frame_total_latency_ms", "loop_total_latency_ms", "fps_instant"
        ]

        for key in keys_to_agg:
            values = [m[key] for m in metrics if m.get(key) is not None]
            if values:
                summary[f"avg_{key}"] = np.mean(values)
                summary[f"median_{key}"] = np.median(values)
                summary[f"p95_{key}"] = np.percentile(values, 95)
                summary[f"max_{key}"] = np.max(values)
            else:
                summary[f"avg_{key}"] = None
                summary[f"median_{key}"] = None
                summary[f"p95_{key}"] = None
                summary[f"max_{key}"] = None

        return summary

    def get_top_risk_frames(self, top_n=10) -> List[Dict[str, Any]]:
        metrics = self.get_flat_frame_metrics()

        def risk_score(m):
            score = 0
            # Higher score for blocked zones
            if m.get("center_risk") == "Blocked": score += 3
            elif m.get("center_risk") == "Warning": score += 1
            if m.get("left_risk") == "Blocked": score += 2
            if m.get("right_risk") == "Blocked": score += 2

            # Lower distance is higher risk
            dist = m.get("nearest_object_depth_m")
            if dist is not None:
                score += max(0, 5 - dist)
            return score

        sorted_frames = sorted(metrics, key=risk_score, reverse=True)
        return sorted_frames[:top_n]

    def get_slowest_frames(self, top_n=10) -> List[Dict[str, Any]]:
        metrics = self.get_flat_frame_metrics()
        sorted_frames = sorted(
            [m for m in metrics if m.get("frame_total_latency_ms") is not None],
            key=lambda x: x["frame_total_latency_ms"],
            reverse=True
        )
        return sorted_frames[:top_n]
