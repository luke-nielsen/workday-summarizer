from __future__ import annotations

import pytest

from workday_summarizer.analyzer import OpenAIWorkdayAnalyzer, _batched
from workday_summarizer.errors import AnalysisError
from workday_summarizer.frames import ExtractedFrame
from workday_summarizer.models import SegmentObservation, WorkdaySummary


def test_batched_splits_evenly() -> None:
    items = list(range(7))
    assert _batched(items, 3) == [[0, 1, 2], [3, 4, 5], [6]]  # type: ignore[arg-type]


def test_analyze_empty_raises(fake_client) -> None:
    analyzer = OpenAIWorkdayAnalyzer(fake_client, model="m")
    with pytest.raises(AnalysisError):
        analyzer.analyze([])


def test_analyze_maps_then_reduces(fake_client, frames) -> None:
    analyzer = OpenAIWorkdayAnalyzer(fake_client, model="gpt-test", batch_size=2)
    summary = analyzer.analyze(frames)

    assert isinstance(summary, WorkdaySummary)
    # 5 frames / batch size 2 -> 3 segment calls + 1 synthesis call.
    schemas = [call["schema"] for call in fake_client.calls]
    assert schemas.count(SegmentObservation) == 3
    assert schemas.count(WorkdaySummary) == 1


def test_segment_times_come_from_frames(fake_client) -> None:
    # Frames whose offsets differ from the fake model's hard-coded observation times.
    frames = [
        ExtractedFrame(index=i, offset_seconds=300.0 + i * 30, jpeg_bytes=b"x") for i in range(3)
    ]
    analyzer = OpenAIWorkdayAnalyzer(fake_client, model="m", batch_size=10)
    analyzer.analyze(frames)

    # The synthesis call receives observations stamped with the true frame offsets.
    synthesis_call = next(c for c in fake_client.calls if c["schema"] is WorkdaySummary)
    user_text = synthesis_call["messages"][1]["content"]
    assert '"start_seconds":300.0' in user_text
    assert '"end_seconds":360.0' in user_text


def test_images_are_attached_to_segment_request(fake_client, frames) -> None:
    analyzer = OpenAIWorkdayAnalyzer(fake_client, model="m", batch_size=10, image_detail="low")
    analyzer.analyze(frames)

    segment_call = next(c for c in fake_client.calls if c["schema"] is SegmentObservation)
    content = segment_call["messages"][1]["content"]
    images = [part for part in content if part["type"] == "image_url"]
    assert len(images) == len(frames)
    assert all(img["image_url"]["detail"] == "low" for img in images)
    assert all(img["image_url"]["url"].startswith("data:image/jpeg;base64,") for img in images)
