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
    WorkdaySummaryDraft,
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
class _Usage:
    """Stands in for the OpenAI ``completion.usage`` object."""

    prompt_tokens: int = 10
    completion_tokens: int = 5


_DEFAULT_USAGE = _Usage()


@dataclass
class _Completion:
    choices: list[_Choice]
    usage: _Usage | None = None


@dataclass
class _ParseEndpoint:
    """Records calls and returns queued, schema-appropriate fakes."""

    segment: SegmentObservation
    summary_draft: WorkdaySummaryDraft
    usage: _Usage | None = field(default_factory=_Usage)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def parse(self, *, model: str, messages: Any, response_format: type) -> _Completion:
        self.calls.append({"model": model, "messages": messages, "schema": response_format})
        result = self.segment if response_format is SegmentObservation else self.summary_draft
        return _Completion(choices=[_Choice(message=_Message(parsed=result))], usage=self.usage)


class FakeOpenAI:
    """Minimal stand-in exposing ``client.beta.chat.completions.parse``."""

    def __init__(
        self,
        segment: SegmentObservation,
        summary_draft: WorkdaySummaryDraft,
        usage: _Usage | None = _DEFAULT_USAGE,
    ) -> None:
        self._endpoint = _ParseEndpoint(segment=segment, summary_draft=summary_draft, usage=usage)
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
def fake_summary_draft() -> WorkdaySummaryDraft:
    return WorkdaySummaryDraft(
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
    )


@pytest.fixture
def fake_client(
    fake_segment: SegmentObservation, fake_summary_draft: WorkdaySummaryDraft
) -> FakeOpenAI:
    return FakeOpenAI(segment=fake_segment, summary_draft=fake_summary_draft)


@pytest.fixture
def make_fake_client(fake_summary_draft: WorkdaySummaryDraft):
    """Factory for a client whose every segment observation is the one provided."""

    def _make(segment: SegmentObservation) -> FakeOpenAI:
        return FakeOpenAI(segment=segment, summary_draft=fake_summary_draft)

    return _make
