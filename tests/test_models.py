from __future__ import annotations

import pytest
from pydantic import ValidationError

from workday_summarizer.models import Task, WorkdaySummary, _format_clock


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00:00"), (59, "00:00:59"), (90, "00:01:30"), (3661, "01:01:01"), (29.6, "00:00:30")],
)
def test_format_clock(seconds: float, expected: str) -> None:
    assert _format_clock(seconds) == expected


def test_task_clock_range() -> None:
    task = Task(
        title="t",
        description="d",
        category="coding",
        start_seconds=30.0,
        end_seconds=150.0,
        applications=[],
        evidence=[],
    )
    assert task.clock_range == "00:00:30 – 00:02:30"


def test_focus_score_is_bounded() -> None:
    with pytest.raises(ValidationError):
        WorkdaySummary(
            headline="h",
            narrative="n",
            tasks=[],
            distractions=[],
            key_accomplishments=[],
            recommendations=[],
            focus_score=140,
            productive_seconds=0.0,
            distracted_seconds=0.0,
        )
