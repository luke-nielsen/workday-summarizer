from __future__ import annotations

import base64
from pathlib import Path

import pytest

from workday_summarizer.errors import FrameExtractionError
from workday_summarizer.frames import ExtractedFrame, FfmpegFrameExtractor


def test_as_data_url_round_trips() -> None:
    frame = ExtractedFrame(index=0, offset_seconds=0.0, jpeg_bytes=b"hello")
    url = frame.as_data_url()
    assert url.startswith("data:image/jpeg;base64,")
    payload = url.split(",", 1)[1]
    assert base64.b64decode(payload) == b"hello"


def test_extract_rejects_missing_file(tmp_path: Path) -> None:
    extractor = FfmpegFrameExtractor()
    with pytest.raises(FrameExtractionError, match="not found"):
        extractor.extract(tmp_path / "does-not-exist.mp4")


def test_invalid_interval_rejected() -> None:
    with pytest.raises(ValueError):
        FfmpegFrameExtractor(interval_seconds=0)


def test_collect_frames_assigns_offsets(tmp_path: Path) -> None:
    for n in range(1, 4):
        (tmp_path / f"frame_{n:06d}.jpg").write_bytes(b"x")
    extractor = FfmpegFrameExtractor(interval_seconds=30.0)
    frames = list(extractor._collect_frames(tmp_path))
    assert [f.offset_seconds for f in frames] == [0.0, 30.0, 60.0]
    assert [f.index for f in frames] == [0, 1, 2]


def test_collect_frames_requires_output(tmp_path: Path) -> None:
    extractor = FfmpegFrameExtractor()
    with pytest.raises(FrameExtractionError, match="no frames"):
        list(extractor._collect_frames(tmp_path))
