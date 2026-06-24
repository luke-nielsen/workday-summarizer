"""Shared test fixtures and lightweight fakes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from workday_summarizer.frames import ExtractedFrame
from workday_summarizer.models import (
    Distraction,
    SegmentObservation,
    Task,
    WorkdaySummary,
)

# A 1x1 white JPEG, enough to stand in for a real frame in tests.
TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
    "07090908"
)


@pytest.fixture
def frames() -> list[ExtractedFrame]:
    return [
        ExtractedFrame(index=i, offset_seconds=i * 30.0, jpeg_bytes=TINY_JPEG) for i in range(5)
    ]


@dataclass
class _Message:
    parsed: Any
    refusal: str | None = None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Completion:
    choices: list[_Choice]


@dataclass
class _ParseEndpoint:
    """Records calls and returns queued, schema-appropriate fakes."""

    segment: SegmentObservation
    summary: WorkdaySummary
    calls: list[dict[str, Any]] = field(default_factory=list)

    def parse(self, *, model: str, messages: Any, response_format: type) -> _Completion:
        self.calls.append({"model": model, "messages": messages, "schema": response_format})
        result = self.segment if response_format is SegmentObservation else self.summary
        return _Completion(choices=[_Choice(message=_Message(parsed=result))])


class FakeOpenAI:
    """Minimal stand-in exposing ``client.beta.chat.completions.parse``."""

    def __init__(self, segment: SegmentObservation, summary: WorkdaySummary) -> None:
        self._endpoint = _ParseEndpoint(segment=segment, summary=summary)
        self.beta = type(
            "beta",
            (),
            {"chat": type("chat", (), {"completions": self._endpoint})()},
        )()

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._endpoint.calls


@pytest.fixture
def fake_segment() -> SegmentObservation:
    return SegmentObservation(
        start_seconds=0.0,
        end_seconds=120.0,
        summary="Coding in an editor.",
        activities=["editing code"],
        applications=["VS Code"],
        appears_distracted=False,
    )


@pytest.fixture
def fake_summary() -> WorkdaySummary:
    return WorkdaySummary(
        headline="A focused day of coding.",
        narrative="The user spent the day building a feature.",
        tasks=[
            Task(
                title="Build feature",
                description="Implemented the summarizer.",
                category="coding",
                start_seconds=0.0,
                end_seconds=120.0,
                applications=["VS Code"],
                evidence=["editor visible"],
            )
        ],
        distractions=[
            Distraction(
                description="Checked social media.",
                category="social_media",
                start_seconds=60.0,
                end_seconds=90.0,
                applications=["Browser"],
            )
        ],
        key_accomplishments=["Shipped the feature"],
        recommendations=["Batch notifications"],
        focus_score=82,
        productive_seconds=90.0,
        distracted_seconds=30.0,
    )


@pytest.fixture
def fake_client(fake_segment: SegmentObservation, fake_summary: WorkdaySummary) -> FakeOpenAI:
    return FakeOpenAI(segment=fake_segment, summary=fake_summary)
