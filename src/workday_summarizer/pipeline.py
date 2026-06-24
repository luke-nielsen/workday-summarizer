"""High-level orchestration: video file in, :class:`WorkdaySummary` out."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from workday_summarizer.analyzer import OpenAIWorkdayAnalyzer, WorkdayAnalyzer
from workday_summarizer.config import Settings
from workday_summarizer.frames import FfmpegFrameExtractor, FrameExtractor
from workday_summarizer.models import WorkdaySummary

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SummarizationResult:
    """The summary plus light metadata about how it was produced."""

    summary: WorkdaySummary
    frame_count: int
    recording_seconds: float


class WorkdayPipeline:
    """Extract frames from a recording and analyze them into a summary.

    The collaborators are injected, so the pipeline is fully testable without ffmpeg or a
    network connection. Use :meth:`from_settings` for the production wiring.
    """

    def __init__(self, extractor: FrameExtractor, analyzer: WorkdayAnalyzer) -> None:
        self._extractor = extractor
        self._analyzer = analyzer

    @classmethod
    def from_settings(cls, settings: Settings) -> WorkdayPipeline:
        client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout_seconds,
            max_retries=0,  # retries are handled in the analyzer with structured backoff
        )
        extractor = FfmpegFrameExtractor(
            interval_seconds=settings.frame_interval_seconds,
            max_dimension=settings.max_frame_dimension,
        )
        analyzer = OpenAIWorkdayAnalyzer(
            client,
            model=settings.model,
            batch_size=settings.batch_size,
            image_detail=settings.image_detail,
            max_retries=settings.max_retries,
        )
        return cls(extractor, analyzer)

    def run(self, video_path: Path) -> SummarizationResult:
        logger.info("Extracting frames from %s", video_path)
        frames = self._extractor.extract(video_path)
        logger.info("Extracted %d frames", len(frames))

        summary = self._analyzer.analyze(frames)
        recording_seconds = frames[-1].offset_seconds if frames else 0.0
        return SummarizationResult(
            summary=summary,
            frame_count=len(frames),
            recording_seconds=recording_seconds,
        )


__all__ = ["SummarizationResult", "WorkdayPipeline"]
