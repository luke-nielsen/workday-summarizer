"""Turn a screen recording of a workday into a structured task/distraction summary."""

from workday_summarizer.models import (
    Distraction,
    SegmentObservation,
    Task,
    WorkdaySummary,
)

__all__ = [
    "Distraction",
    "SegmentObservation",
    "Task",
    "WorkdaySummary",
    "__version__",
]

__version__ = "0.1.0"
