"""Vision analysis of sampled frames via the OpenAI API.

The analysis is a small map/reduce:

* **map** — frames are grouped into batches and each batch is described by the model as a
  :class:`~workday_summarizer.models.SegmentObservation`. Batching keeps any single
  request well within the model's context window and bounds cost for long recordings.
* **reduce** — the observations are synthesised into one
  :class:`~workday_summarizer.models.WorkdaySummary`.

Both stages use OpenAI structured outputs, so the model is constrained to our schema and
the result needs no fragile post-hoc parsing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from workday_summarizer.errors import AnalysisError
from workday_summarizer.frames import ExtractedFrame
from workday_summarizer.models import SegmentObservation, WorkdaySummary, _format_clock
from workday_summarizer.prompts import (
    SEGMENT_SYSTEM_PROMPT,
    SEGMENT_USER_PREFIX,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PREFIX,
)

logger = logging.getLogger(__name__)

# Errors worth retrying: transient network blips and rate limits.
_RETRYABLE = (APIConnectionError, RateLimitError)

_T = TypeVar("_T", bound=BaseModel)


class WorkdayAnalyzer(Protocol):
    """Turns sampled frames into a structured workday summary."""

    def analyze(self, frames: Sequence[ExtractedFrame]) -> WorkdaySummary:
        ...


def _batched(frames: Sequence[ExtractedFrame], size: int) -> list[list[ExtractedFrame]]:
    return [list(frames[i : i + size]) for i in range(0, len(frames), size)]


class OpenAIWorkdayAnalyzer:
    """Analyze frames using an OpenAI vision model with structured outputs."""

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        batch_size: int = 12,
        image_detail: str = "auto",
        max_retries: int = 4,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._client = client
        self._model = model
        self._batch_size = batch_size
        self._image_detail = image_detail
        self._max_retries = max_retries

    def analyze(self, frames: Sequence[ExtractedFrame]) -> WorkdaySummary:
        if not frames:
            raise AnalysisError("No frames to analyze.")

        batches = _batched(frames, self._batch_size)
        logger.info("Analyzing %d frames in %d batch(es).", len(frames), len(batches))

        observations: list[SegmentObservation] = []
        for number, batch in enumerate(batches, start=1):
            logger.info("Observing segment %d/%d (%d frames).", number, len(batches), len(batch))
            observations.append(self._observe_segment(batch))

        duration = frames[-1].offset_seconds
        return self._synthesize(observations, duration_seconds=duration)

    # -- map -----------------------------------------------------------------

    def _observe_segment(self, frames: Sequence[ExtractedFrame]) -> SegmentObservation:
        prefix = SEGMENT_USER_PREFIX.format(
            count=len(frames),
            start=_format_clock(frames[0].offset_seconds),
            end=_format_clock(frames[-1].offset_seconds),
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prefix}]
        for frame in frames:
            content.append({"type": "text", "text": f"[{_format_clock(frame.offset_seconds)}]"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": frame.as_data_url(), "detail": self._image_detail},
                }
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        observation = self._parse(messages, SegmentObservation)
        # Trust our own frame timestamps over the model's arithmetic.
        return observation.model_copy(
            update={
                "start_seconds": frames[0].offset_seconds,
                "end_seconds": frames[-1].offset_seconds,
            }
        )

    # -- reduce --------------------------------------------------------------

    def _synthesize(
        self, observations: Sequence[SegmentObservation], *, duration_seconds: float
    ) -> WorkdaySummary:
        payload = "\n".join(obs.model_dump_json() for obs in observations)
        user_text = (
            SUMMARY_USER_PREFIX.format(duration=_format_clock(duration_seconds))
            + "\n\n"
            + payload
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        return self._parse(messages, WorkdaySummary)

    # -- shared --------------------------------------------------------------

    def _parse(self, messages: list[dict[str, Any]], schema: type[_T]) -> _T:
        @retry(
            retry=retry_if_exception_type(_RETRYABLE),
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            reraise=True,
        )
        def _call() -> _T:
            completion = self._client.beta.chat.completions.parse(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                response_format=schema,
            )
            message = completion.choices[0].message
            if message.refusal:
                raise AnalysisError(f"Model refused the request: {message.refusal}")
            if message.parsed is None:
                raise AnalysisError("Model returned no parsable structured output.")
            return message.parsed

        try:
            return _call()
        except APIStatusError as exc:  # non-retryable API error (4xx other than rate limit)
            raise AnalysisError(f"OpenAI API error ({exc.status_code}): {exc.message}") from exc


__all__ = ["OpenAIWorkdayAnalyzer", "WorkdayAnalyzer"]
