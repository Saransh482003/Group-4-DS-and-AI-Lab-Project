import abc
import threading
from app.pipeline.frame_context import FrameContext
from app.pipeline.fusion import fuse_detections_and_depth

class PipelineExecutor(abc.ABC):
    """
    Abstract base class defining the interface for executing pipeline components.
    """
    @abc.abstractmethod
    def execute(self, ctx: FrameContext, components: dict) -> None:
        pass

class SequentialExecutor(PipelineExecutor):
    """
    Executes detector and depth estimator components sequentially within the pipeline.
    """

    def execute(self, ctx: FrameContext, components: dict) -> None:
        detector = components.get("detector")
        depth_estimator = components.get("depth")

        if detector:
            detector.run(ctx)

        if depth_estimator:
            depth_estimator.run(ctx)

        fuse_detections_and_depth(ctx)

class ThreadedParallelExecutor(PipelineExecutor):
    """
    Executes detector and depth estimator components in parallel threads to optimize pipeline performance.
    """

    def execute(self, ctx: FrameContext, components: dict) -> None:
        detector = components.get("detector")
        depth_estimator = components.get("depth")

        threads = []

        if detector:
            t_det = threading.Thread(target=detector.run, args=(ctx,))
            threads.append(t_det)
            t_det.start()

        if depth_estimator:
            t_dep = threading.Thread(target=depth_estimator.run, args=(ctx,))
            threads.append(t_dep)
            t_dep.start()

        for t in threads:
            t.join()

        fuse_detections_and_depth(ctx)
