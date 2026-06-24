"""High-level orchestration: video file in, :class:`WorkdaySummary` out."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from workday_summarizer.analyzer import OpenAIWorkdayAnalyzer, WorkdayAnalyzer
from workday_summarizer.config import Settings
from workday_summarizer.frames import FfmpegFrameExtractor, FrameExtractor
from workday_summarizer.models import WorkdaySummary
from workday_summarizer.pricing import estimate_cost
from workday_summarizer.usage import TokenUsage

logger = logging.getLogger(__name__)

# Given the tokens a run consumed, return its estimated USD cost (or None if unknown).
CostEstimator = Callable[[TokenUsage], "float | None"]


@dataclass(frozen=True, slots=True)
class SummarizationResult:
    """The summary plus light metadata about how it was produced."""

    summary: WorkdaySummary
    frame_count: int
    recording_seconds: float
    usage: TokenUsage
    estimated_cost: float | None


class WorkdayPipeline:
    """Extract frames from a recording and analyze them into a summary.

    The collaborators are injected, so the pipeline is fully testable without ffmpeg or a
    network connection. Use :meth:`from_settings` for the production wiring.
    """

    def __init__(
        self,
        extractor: FrameExtractor,
        analyzer: WorkdayAnalyzer,
        *,
        cost_estimator: CostEstimator | None = None,
    ) -> None:
        self._extractor = extractor
        self._analyzer = analyzer
        self._cost_estimator = cost_estimator

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
            frame_interval_seconds=settings.frame_interval_seconds,
            max_concurrency=settings.max_concurrency,
        )

        def cost_estimator(usage: TokenUsage) -> float | None:
            return estimate_cost(
                settings.model,
                usage,
                input_per_mtok=settings.input_cost_per_mtok,
                output_per_mtok=settings.output_cost_per_mtok,
            )

        return cls(extractor, analyzer, cost_estimator=cost_estimator)

    def run(self, video_path: Path) -> SummarizationResult:
        logger.info("Extracting frames from %s", video_path)
        frames = self._extractor.extract(video_path)
        logger.info("Extracted %d frames", len(frames))

        analysis = self._analyzer.analyze(frames)
        recording_seconds = frames[-1].offset_seconds if frames else 0.0
        estimated_cost = (
            self._cost_estimator(analysis.usage) if self._cost_estimator is not None else None
        )
        logger.info(
            "Used %d tokens across %d request(s)%s.",
            analysis.usage.total_tokens,
            analysis.usage.requests,
            f" (~${estimated_cost:.2f})" if estimated_cost is not None else "",
        )
        return SummarizationResult(
            summary=analysis.summary,
            frame_count=len(frames),
            recording_seconds=recording_seconds,
            usage=analysis.usage,
            estimated_cost=estimated_cost,
        )


__all__ = ["CostEstimator", "SummarizationResult", "WorkdayPipeline"]
