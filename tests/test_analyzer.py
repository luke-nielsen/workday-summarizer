from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from workday_summarizer.analyzer import (
    AnalysisResult,
    OpenAIWorkdayAnalyzer,
    _batched,
    _retry_after_seconds,
    compute_focus_metrics,
)
from workday_summarizer.errors import AnalysisError
from workday_summarizer.frames import ExtractedFrame
from workday_summarizer.models import SegmentObservation, WorkdaySummary, WorkdaySummaryDraft


def _segment(start: float, end: float, *, distracted: bool) -> SegmentObservation:
    return SegmentObservation(
        start_seconds=start,
        end_seconds=end,
        summary="",
        activities=[],
        applications=[],
        appears_distracted=distracted,
    )


def test_batched_splits_evenly() -> None:
    items = list(range(7))
    assert _batched(items, 3) == [[0, 1, 2], [3, 4, 5], [6]]  # type: ignore[arg-type]


def test_analyze_empty_raises(fake_client) -> None:
    analyzer = OpenAIWorkdayAnalyzer(fake_client, model="m")
    with pytest.raises(AnalysisError):
        analyzer.analyze([])


def test_analyze_maps_then_reduces(fake_client, frames) -> None:
    analyzer = OpenAIWorkdayAnalyzer(fake_client, model="gpt-test", batch_size=2)
    result = analyzer.analyze(frames)

    assert isinstance(result, AnalysisResult)
    assert isinstance(result.summary, WorkdaySummary)
    # 5 frames / batch size 2 -> 3 segment calls + 1 synthesis call.
    schemas = [call["schema"] for call in fake_client.calls]
    assert schemas.count(SegmentObservation) == 3
    assert schemas.count(WorkdaySummaryDraft) == 1


def test_usage_is_summed_across_requests(fake_client, frames) -> None:
    analyzer = OpenAIWorkdayAnalyzer(fake_client, model="m", batch_size=2)
    result = analyzer.analyze(frames)

    # 3 segment + 1 synthesis = 4 requests, each reporting 10 in / 5 out.
    assert result.usage.requests == 4
    assert result.usage.prompt_tokens == 40
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 60


def test_segment_times_come_from_frames(fake_client) -> None:
    # Frames whose offsets differ from the fake model's hard-coded observation times.
    frames = [
        ExtractedFrame(index=i, offset_seconds=300.0 + i * 30, jpeg_bytes=b"x") for i in range(3)
    ]
    analyzer = OpenAIWorkdayAnalyzer(fake_client, model="m", batch_size=10)
    analyzer.analyze(frames)

    # The synthesis call receives observations stamped with the true frame offsets.
    synthesis_call = next(c for c in fake_client.calls if c["schema"] is WorkdaySummaryDraft)
    user_text = synthesis_call["messages"][1]["content"]
    assert '"start_seconds":300.0' in user_text
    assert '"end_seconds":360.0' in user_text


def test_concurrent_map_preserves_chronological_order(fake_client) -> None:
    # One frame per batch over several batches forces the concurrent code path.
    frames = [
        ExtractedFrame(index=i, offset_seconds=i * 30.0, jpeg_bytes=b"x") for i in range(6)
    ]
    analyzer = OpenAIWorkdayAnalyzer(fake_client, model="m", batch_size=1, max_concurrency=4)
    analyzer.analyze(frames)

    synthesis_call = next(c for c in fake_client.calls if c["schema"] is WorkdaySummaryDraft)
    payload = synthesis_call["messages"][1]["content"]
    # The observation JSON is appended after a blank line; parse each observation line.
    observation_lines = [line for line in payload.splitlines() if line.startswith("{")]
    starts = [json.loads(line)["start_seconds"] for line in observation_lines]
    # Observations must reach the reducer in ascending time order regardless of which
    # worker finished first.
    assert starts == [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    assert starts == sorted(starts)


def test_images_are_attached_to_segment_request(fake_client, frames) -> None:
    analyzer = OpenAIWorkdayAnalyzer(fake_client, model="m", batch_size=10, image_detail="low")
    analyzer.analyze(frames)

    segment_call = next(c for c in fake_client.calls if c["schema"] is SegmentObservation)
    content = segment_call["messages"][1]["content"]
    images = [part for part in content if part["type"] == "image_url"]
    assert len(images) == len(frames)
    assert all(img["image_url"]["detail"] == "low" for img in images)
    assert all(img["image_url"]["url"].startswith("data:image/jpeg;base64,") for img in images)


# -- deterministic focus metrics --------------------------------------------


def test_focus_metrics_all_productive() -> None:
    observations = [_segment(0, 30, distracted=False), _segment(60, 90, distracted=False)]
    metrics = compute_focus_metrics(observations, interval_seconds=30)

    # Each segment spans 30s of frames + one 30s interval = 60s of screen time.
    assert metrics.productive_seconds == 120
    assert metrics.distracted_seconds == 0
    assert metrics.focus_score == 100


def test_focus_metrics_mixes_productive_and_distracted() -> None:
    observations = [
        _segment(0, 30, distracted=False),  # 60s productive
        _segment(60, 90, distracted=True),  # 60s distracted
        _segment(120, 120, distracted=True),  # single frame -> 30s distracted
    ]
    metrics = compute_focus_metrics(observations, interval_seconds=30)

    assert metrics.productive_seconds == 60
    assert metrics.distracted_seconds == 90
    assert metrics.focus_score == 40  # round(100 * 60 / 150)


def test_focus_metrics_empty_is_neutral() -> None:
    metrics = compute_focus_metrics([], interval_seconds=30)
    assert metrics.focus_score == 100
    assert metrics.productive_seconds == 0
    assert metrics.distracted_seconds == 0


def test_focus_metrics_drive_the_summary(make_fake_client, frames) -> None:
    # Every segment is flagged distracted, so the final summary's metrics must reflect
    # that — regardless of what the reduce model returns for the qualitative draft.
    client = make_fake_client(_segment(0, 120, distracted=True))
    analyzer = OpenAIWorkdayAnalyzer(client, model="m", batch_size=10)
    result = analyzer.analyze(frames)

    assert result.summary.distracted_seconds > 0
    assert result.summary.productive_seconds == 0
    assert result.summary.focus_score == 0


# -- retry-after parsing -----------------------------------------------------


@dataclass
class _Outcome:
    _exc: Any

    def exception(self) -> Any:
        return self._exc


@dataclass
class _RetryState:
    outcome: Any


class _Resp:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class _RateLimited(Exception):
    def __init__(self, headers: dict[str, str]) -> None:
        self.response = _Resp(headers)


def test_retry_after_reads_numeric_header() -> None:
    state = _RetryState(outcome=_Outcome(_RateLimited({"retry-after": "12"})))
    assert _retry_after_seconds(state) == 12.0


def test_retry_after_ignores_http_date() -> None:
    headers = {"retry-after": "Wed, 21 Oct 2099 07:28:00 GMT"}
    state = _RetryState(outcome=_Outcome(_RateLimited(headers)))
    assert _retry_after_seconds(state) is None


def test_retry_after_absent_header() -> None:
    state = _RetryState(outcome=_Outcome(_RateLimited({})))
    assert _retry_after_seconds(state) is None
