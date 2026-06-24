from __future__ import annotations

from pathlib import Path

from workday_summarizer.analyzer import OpenAIWorkdayAnalyzer
from workday_summarizer.models import WorkdaySummary
from workday_summarizer.pipeline import WorkdayPipeline


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
