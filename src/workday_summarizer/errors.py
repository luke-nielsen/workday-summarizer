"""Exception hierarchy for the package.

A single base exception lets callers (and the CLI) catch every expected failure mode
without swallowing genuine programming errors.
"""

from __future__ import annotations


class WorkdaySummarizerError(Exception):
    """Base class for all errors raised by this package."""


class FfmpegNotFoundError(WorkdaySummarizerError):
    """Raised when the ``ffmpeg``/``ffprobe`` binaries are not on the PATH."""


class FrameExtractionError(WorkdaySummarizerError):
    """Raised when frames cannot be extracted from the source video."""


class AnalysisError(WorkdaySummarizerError):
    """Raised when the model fails to return a usable analysis."""


__all__ = [
    "AnalysisError",
    "FfmpegNotFoundError",
    "FrameExtractionError",
    "WorkdaySummarizerError",
]
