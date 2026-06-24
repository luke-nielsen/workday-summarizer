"""Vision analysis of sampled frames via the OpenAI API.

The analysis is a small map/reduce:

* **map** — frames are grouped into batches and each batch is described by the model as a
  :class:`~workday_summarizer.models.SegmentObservation`. Batching keeps any single
  request well within the model's context window and bounds cost for long recordings.
  Batches are independent, so they are observed concurrently (bounded by
  ``max_concurrency``) — the dominant cost of a long recording is wall-clock spent waiting
  on the API, and that parallelises cleanly.
* **reduce** — the observations are synthesised into one
  :class:`~workday_summarizer.models.WorkdaySummaryDraft`, and the quantitative focus
  metrics are computed deterministically from the observations (never guessed by the
  model) to produce the final :class:`~workday_summarizer.models.WorkdaySummary`.

Both stages use OpenAI structured outputs, so the model is constrained to our schema and
the result needs no fragile post-hoc parsing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from workday_summarizer.errors import AnalysisError
from workday_summarizer.frames import ExtractedFrame
from workday_summarizer.models import (
    SegmentObservation,
    WorkdaySummary,
    WorkdaySummaryDraft,
    _format_clock,
)
from workday_summarizer.prompts import (
    SEGMENT_SYSTEM_PROMPT,
    SEGMENT_USER_PREFIX,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PREFIX,
)
from workday_summarizer.usage import TokenUsage

logger = logging.getLogger(__name__)

# Errors worth retrying: transient network blips and rate limits.
_RETRYABLE = (APIConnectionError, RateLimitError)

# Never honour an absurd Retry-After; fall back to our own ceiling instead.
_MAX_RETRY_AFTER_SECONDS = 60.0

_T = TypeVar("_T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class FocusMetrics:
    """Time accounting derived deterministically from the segment observations."""

    productive_seconds: float
    distracted_seconds: float
    focus_score: int


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """A finished summary plus the token usage it took to produce it."""

    summary: WorkdaySummary
    usage: TokenUsage


class WorkdayAnalyzer(Protocol):
    """Turns sampled frames into a structured workday summary."""

    def analyze(self, frames: Sequence[ExtractedFrame]) -> AnalysisResult:
        ...


def _batched(frames: Sequence[ExtractedFrame], size: int) -> list[list[ExtractedFrame]]:
    return [list(frames[i : i + size]) for i in range(0, len(frames), size)]


def compute_focus_metrics(
    observations: Sequence[SegmentObservation], interval_seconds: float
) -> FocusMetrics:
    """Derive productive/distracted seconds and a focus score from the observations.

    Each observation covers the span of its frames; since the last frame still represents
    ``interval_seconds`` of screen time, a segment's duration is its frame span plus one
    interval. Time is attributed to "productive" or "distracted" purely by the segment's
    own ``appears_distracted`` flag, so the headline numbers are a direct, reproducible
    function of the segments — not a separate estimate the model could contradict.
    """
    productive = 0.0
    distracted = 0.0
    for obs in observations:
        seconds = max(0.0, obs.end_seconds - obs.start_seconds) + interval_seconds
        if obs.appears_distracted:
            distracted += seconds
        else:
            productive += seconds

    total = productive + distracted
    focus_score = round(100 * productive / total) if total > 0 else 100
    return FocusMetrics(
        productive_seconds=productive,
        distracted_seconds=distracted,
        focus_score=focus_score,
    )


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
        frame_interval_seconds: float = 30.0,
        max_concurrency: int = 4,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if frame_interval_seconds <= 0:
            raise ValueError("frame_interval_seconds must be positive")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._client = client
        self._model = model
        self._batch_size = batch_size
        self._image_detail = image_detail
        self._max_retries = max_retries
        self._interval = frame_interval_seconds
        self._max_concurrency = max_concurrency

    def analyze(self, frames: Sequence[ExtractedFrame]) -> AnalysisResult:
        if not frames:
            raise AnalysisError("No frames to analyze.")

        batches = _batched(frames, self._batch_size)
        observations, map_usage = self._observe_segments(batches)

        duration = frames[-1].offset_seconds
        draft, reduce_usage = self._synthesize(observations, duration_seconds=duration)
        metrics = compute_focus_metrics(observations, self._interval)
        summary = WorkdaySummary(
            **draft.model_dump(),
            focus_score=metrics.focus_score,
            productive_seconds=metrics.productive_seconds,
            distracted_seconds=metrics.distracted_seconds,
        )
        return AnalysisResult(summary=summary, usage=map_usage + reduce_usage)

    # -- map -----------------------------------------------------------------

    def _observe_segments(
        self, batches: Sequence[Sequence[ExtractedFrame]]
    ) -> tuple[list[SegmentObservation], TokenUsage]:
        total = len(batches)
        workers = min(self._max_concurrency, total)
        logger.info("Analyzing %d batch(es) with concurrency %d.", total, workers)

        if workers == 1:
            observations: list[SegmentObservation] = []
            usage = TokenUsage()
            for number, batch in enumerate(batches, start=1):
                logger.info("Observing segment %d/%d (%d frames).", number, total, len(batch))
                observation, call_usage = self._observe_segment(batch)
                observations.append(observation)
                usage += call_usage
            return observations, usage

        # Preserve chronological order by writing each result back into its slot; the
        # observation timestamps drive the reduce stage, so order must be deterministic.
        results: list[SegmentObservation | None] = [None] * total
        usages: list[TokenUsage] = [TokenUsage()] * total
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._observe_segment, batch): i
                for i, batch in enumerate(batches)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                observation, call_usage = future.result()
                results[index] = observation
                usages[index] = call_usage
                logger.info("Observed segment %d/%d.", done, total)

        return [obs for obs in results if obs is not None], sum(usages, TokenUsage())

    def _observe_segment(
        self, frames: Sequence[ExtractedFrame]
    ) -> tuple[SegmentObservation, TokenUsage]:
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
        observation, usage = self._parse(messages, SegmentObservation)
        # Trust our own frame timestamps over the model's arithmetic.
        stamped = observation.model_copy(
            update={
                "start_seconds": frames[0].offset_seconds,
                "end_seconds": frames[-1].offset_seconds,
            }
        )
        return stamped, usage

    # -- reduce --------------------------------------------------------------

    def _synthesize(
        self, observations: Sequence[SegmentObservation], *, duration_seconds: float
    ) -> tuple[WorkdaySummaryDraft, TokenUsage]:
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
        return self._parse(messages, WorkdaySummaryDraft)

    # -- shared --------------------------------------------------------------

    def _parse(self, messages: list[dict[str, Any]], schema: type[_T]) -> tuple[_T, TokenUsage]:
        @retry(
            retry=retry_if_exception_type(_RETRYABLE),
            stop=stop_after_attempt(self._max_retries + 1),
            wait=_wait_respecting_retry_after(),
            reraise=True,
        )
        def _call() -> tuple[_T, TokenUsage]:
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
            usage = TokenUsage.from_completion_usage(getattr(completion, "usage", None))
            return message.parsed, usage

        try:
            return _call()
        except APIStatusError as exc:  # non-retryable API error (4xx other than rate limit)
            raise AnalysisError(f"OpenAI API error ({exc.status_code}): {exc.message}") from exc


def _wait_respecting_retry_after() -> Callable[[RetryCallState], float]:
    """A tenacity wait strategy that honours a server ``Retry-After`` header.

    Rate-limit responses tell us exactly how long to wait; respecting that header avoids
    both hammering the API too soon and backing off longer than necessary. When no header
    is present (or it is an HTTP-date we don't parse), we fall back to exponential backoff.
    """
    exponential = wait_exponential(multiplier=1, min=1, max=30)

    def _wait(retry_state: RetryCallState) -> float:
        retry_after = _retry_after_seconds(retry_state)
        if retry_after is not None:
            return min(retry_after, _MAX_RETRY_AFTER_SECONDS)
        return exponential(retry_state)

    return _wait


def _retry_after_seconds(retry_state: RetryCallState) -> float | None:
    outcome = retry_state.outcome
    if outcome is None:
        return None
    headers = getattr(getattr(outcome.exception(), "response", None), "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None  # HTTP-date form — let exponential backoff take over.


__all__ = [
    "AnalysisResult",
    "FocusMetrics",
    "OpenAIWorkdayAnalyzer",
    "WorkdayAnalyzer",
    "compute_focus_metrics",
]
