from __future__ import annotations

from pathlib import Path

from workday_summarizer.analyzer import OpenAIWorkdayAnalyzer
from workday_summarizer.models import WorkdaySummary
from workday_summarizer.pipeline import WorkdayPipeline
from workday_summarizer.usage import TokenUsage


class _StaticExtractor:
    def __init__(self, frames) -> None:
        self._frames = frames

    def extract(self, video_path: Path):
        return self._frames


def test_pipeline_runs_end_to_end(fake_client, frames) -> None:
    analyzer = OpenAIWorkdayAnalyzer(fake_client, model="m", batch_size=10)
    pipeline = WorkdayPipeline(_StaticExtractor(frames), analyzer)

    result = pipeline.run(Path("ignored.mp4"))

    assert isinstance(result.summary, WorkdaySummary)
    assert result.frame_count == len(frames)
    assert result.recording_seconds == frames[-1].offset_seconds
    assert result.usage.requests > 0
    # No estimator supplied via the DI constructor, so cost is unknown.
    assert result.estimated_cost is None


def test_pipeline_reports_estimated_cost(fake_client, frames) -> None:
    analyzer = OpenAIWorkdayAnalyzer(fake_client, model="m", batch_size=10)
    captured: dict[str, TokenUsage] = {}

    def estimator(usage: TokenUsage) -> float:
        captured["usage"] = usage
        return 1.23

    pipeline = WorkdayPipeline(_StaticExtractor(frames), analyzer, cost_estimator=estimator)
    result = pipeline.run(Path("ignored.mp4"))

    assert result.estimated_cost == 1.23
    assert captured["usage"] is result.usage
