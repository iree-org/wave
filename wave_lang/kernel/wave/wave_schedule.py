from typing import Callable, Any
from .._support.tracing import ScheduleContext, CapturedTrace
from ..ops import wave_schedule_ops
from .._support.regions import ScheduleRegionGraph
from .constraints import Constraint
from .._support.location import capture_function_location
from ...support.location_config import LocationCaptureConfig, LocationCaptureLevel
import inspect


def wave_schedule():
    def decorator(f: Callable[..., Any]) -> "WaveSchedule":
        return WaveSchedule(f.__name__, f)

    return decorator


class WaveSchedule:
    def __init__(self, schedule_function: str, f: Callable[..., Any]):
        self.schedule_function = schedule_function
        self.f = f

    def _trace(self, kernel_trace=None, constraints: list[Constraint] = None):
        region_graph = ScheduleRegionGraph()
        with ScheduleContext(region_graph, kernel_trace, constraints) as context:
            # Get all explicitly defined custom schedule ops (dataclasses)
            schedule_ops: dict[str, wave_schedule_ops.CustomScheduleOp] = {
                cls.schedule_op_name: cls
                for _, cls in inspect.getmembers(wave_schedule_ops, inspect.isclass)
                if issubclass(cls, wave_schedule_ops.CustomScheduleOp)
                and hasattr(cls, "schedule_op_name")
            }

            # Register schedule ops
            for name, op in schedule_ops.items():
                context.register_custom_op(name, op)

            with region_graph.subtracer() as subtracer:
                root_name, _ = subtracer.trace(self.f, kernel_trace)
                # Create a location capture config with minimal level to avoid overhead
                location_capture_config = LocationCaptureConfig(
                    level=LocationCaptureLevel.NONE
                )
                kernel_location = capture_function_location(
                    self.f, location_capture_config
                )
                trace = CapturedTrace(region_graph, root_name, kernel_location)

        return trace

    def trace(self, kernel_trace: CapturedTrace, constraints: list[Constraint]):
        """Run the schedule"""
        trace = self._trace(kernel_trace, constraints)
        return trace
