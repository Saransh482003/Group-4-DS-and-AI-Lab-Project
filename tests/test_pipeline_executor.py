import pytest
from unittest.mock import Mock
from app.pipeline.frame_context import FrameContext
from app.pipeline.executor import SequentialExecutor, ThreadedParallelExecutor

class DummyDetector:
    def run(self, ctx):
        ctx.detections = [{"x1": 0, "y1": 0, "x2": 10, "y2": 10, "class_name": "person", "confidence": 0.9}]

class DummyDepth:
    def run(self, ctx):
        import numpy as np
        ctx.depth_map = np.ones((100, 100))

def test_sequential_executor():
    ctx = FrameContext(frame_idx=1, source_id="test", frame_bgr=None)
    components = {"detector": DummyDetector(), "depth": DummyDepth()}

    executor = SequentialExecutor()
    executor.execute(ctx, components)

    assert len(ctx.detections) == 1
    assert ctx.depth_map is not None
    assert len(ctx.nav_detections) == 1
    assert ctx.nav_detections[0]["depth"] is not None

def test_threaded_parallel_executor():
    ctx = FrameContext(frame_idx=1, source_id="test", frame_bgr=None)
    components = {"detector": DummyDetector(), "depth": DummyDepth()}

    executor = ThreadedParallelExecutor()
    executor.execute(ctx, components)

    assert len(ctx.detections) == 1
    assert ctx.depth_map is not None
    assert len(ctx.nav_detections) == 1
    assert ctx.nav_detections[0]["depth"] is not None
