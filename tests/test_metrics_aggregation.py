import pytest
from app.metrics.aggregator import MetricsAggregator

def test_metrics_aggregation():
    frames = [
        {"frame_idx": 1, "source_id": "s1", "timestamp_epoch": 100, "metrics": {"yolo_latency_ms": 10, "nav_command": "Move Left"}},
        {"frame_idx": 2, "source_id": "s1", "timestamp_epoch": 101, "metrics": {"yolo_latency_ms": 20, "nav_command": "Move Left"}},
        {"frame_idx": 3, "source_id": "s1", "timestamp_epoch": 102, "metrics": {"yolo_latency_ms": 30, "nav_command": "Stop"}},
    ]

    agg = MetricsAggregator(frames)

    # Check flat metrics
    flat = agg.get_flat_frame_metrics()
    assert len(flat) == 3
    assert flat[0]["yolo_latency_ms"] == 10

    # Check command distribution
    dist = agg.get_command_distribution()
    assert dist["Move Left"] == 2
    assert dist["Stop"] == 1

    # Check latency summary
    summary = agg.get_latency_summary()
    assert summary["avg_yolo_latency_ms"] == 20.0
    assert summary["max_yolo_latency_ms"] == 30.0
