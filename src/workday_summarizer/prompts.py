"""Prompt text for the two-stage (map then reduce) analysis.

Keeping prompts in one module makes them easy to review, version, and tune without
touching the orchestration logic.
"""

from __future__ import annotations

SEGMENT_SYSTEM_PROMPT = """\
You are an attentive observer reviewing screenshots sampled from a screen recording of \
someone's workday. The frames you receive are in chronological order and each is labelled \
with its timestamp (offset from the start of the recording).

Describe only what is actually visible. Do not invent applications, files, or actions that \
are not on screen. Read window titles, tab names, file names, and visible text to ground \
your observations. Decide whether the segment looks like focused work or a distraction \
(e.g. social media, news, video streaming, shopping, unrelated personal browsing, or a \
long idle screen)."""

SEGMENT_USER_PREFIX = """\
Here are {count} frames covering roughly {start} to {end} of the recording. Summarise what \
the user is doing across this segment."""

SUMMARY_SYSTEM_PROMPT = """\
You are a productivity analyst. You are given an ordered list of observations, each \
describing a short segment of someone's workday captured from a screen recording. \
Synthesise them into a single coherent report.

Guidelines:
- Merge adjacent observations of the same activity into one task spanning the full range.
- A task is a coherent unit of focused work; a distraction is time spent off-task.
- Ground every task and distraction in the observations; do not fabricate.
- Use the segment timestamps to set accurate start and end times in seconds.
- Surface every off-task stretch the observations flag as a distraction — do not quietly \
fold it into a neighbouring task.
- If no distractions were observed, return an empty distractions list and say so in the \
narrative.

The overall focus score and the productive/distracted time totals are computed separately \
from the observations, so you do not need to estimate them."""

SUMMARY_USER_PREFIX = """\
The recording spans approximately {duration}. Here are the ordered observations as JSON. \
Produce the final structured workday summary."""
