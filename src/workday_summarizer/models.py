"""Pydantic schemas for the model's structured output.

These types are the contract between the application and the OpenAI model. They are
intentionally flat and self-describing: every field carries a ``description`` so the
schema doubles as the prompt's instructions when sent as a structured-output spec.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


def _format_clock(seconds: float) -> str:
    """Render a recording offset in seconds as ``HH:MM:SS``."""
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class SegmentObservation(BaseModel):
    """What the model observed across one batch of consecutive frames.

    Produced in the *map* stage of analysis, one per batch, before the final summary
    is synthesised. Times are offsets in seconds from the start of the recording.
    """

    start_seconds: float = Field(description="Offset of this segment's first frame, in seconds.")
    end_seconds: float = Field(description="Offset of this segment's last frame, in seconds.")
    summary: str = Field(description="One or two sentences describing what the user was doing.")
    activities: list[str] = Field(
        description="Concrete activities observed, e.g. 'editing a Python file'."
    )
    applications: list[str] = Field(description="Applications, sites, or tools visible on screen.")
    appears_distracted: bool = Field(
        description="True if the on-screen content looks unrelated to focused work."
    )


class Task(BaseModel):
    """A coherent unit of work the user performed during the day."""

    title: str = Field(description="Short, action-oriented title, e.g. 'Implement login API'.")
    description: str = Field(description="What the user actually did, grounded in the frames.")
    category: str = Field(
        description="A bucket such as coding, meetings, email, research, design, or admin."
    )
    start_seconds: float = Field(description="When the task began, in seconds from the start.")
    end_seconds: float = Field(description="When the task ended, in seconds from the start.")
    applications: list[str] = Field(description="Primary applications or tools used for this task.")
    evidence: list[str] = Field(
        description="Specific on-screen details that support this conclusion."
    )

    @property
    def clock_range(self) -> str:
        """Human-readable ``HH:MM:SS – HH:MM:SS`` time range."""
        return f"{_format_clock(self.start_seconds)} – {_format_clock(self.end_seconds)}"


class Distraction(BaseModel):
    """A stretch of time spent on something other than focused work."""

    description: str = Field(description="What pulled the user off-task.")
    category: str = Field(
        description="A bucket such as social_media, news, entertainment, browsing, or idle."
    )
    start_seconds: float = Field(description="When the distraction began, in seconds.")
    end_seconds: float = Field(description="When the distraction ended, in seconds.")
    applications: list[str] = Field(description="Applications or sites involved.")

    @property
    def clock_range(self) -> str:
        """Human-readable ``HH:MM:SS – HH:MM:SS`` time range."""
        return f"{_format_clock(self.start_seconds)} – {_format_clock(self.end_seconds)}"


class WorkdaySummary(BaseModel):
    """The final, synthesised report for an entire workday."""

    headline: str = Field(description="A single sentence capturing the day at a glance.")
    narrative: str = Field(description="A short paragraph narrating how the day unfolded.")
    tasks: list[Task] = Field(
        description="Distinct tasks the user worked on, in chronological order."
    )
    distractions: list[Distraction] = Field(
        description="Distinct distractions, in chronological order. Empty if none were observed."
    )
    key_accomplishments: list[str] = Field(
        description="The most meaningful things shipped or completed."
    )
    recommendations: list[str] = Field(
        description="Actionable suggestions for a more focused, productive day."
    )
    focus_score: int = Field(
        ge=0,
        le=100,
        description="Overall focus from 0 (constantly distracted) to 100 (deep focus).",
    )
    productive_seconds: float = Field(description="Estimated seconds spent on focused work.")
    distracted_seconds: float = Field(description="Estimated seconds spent distracted.")
