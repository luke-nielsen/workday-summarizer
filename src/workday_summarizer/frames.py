"""Frame extraction from a screen recording.

The public surface is the :class:`FrameExtractor` protocol so the analysis pipeline can
be unit-tested with a trivial in-memory fake. The production implementation,
:class:`FfmpegFrameExtractor`, shells out to ``ffmpeg`` — by far the most reliable way to
decode the wide range of codecs a screen recorder might produce.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, runtime_checkable

from workday_summarizer.errors import FfmpegNotFoundError, FrameExtractionError


@dataclass(frozen=True, slots=True)
class ExtractedFrame:
    """A single sampled frame plus the moment in the recording it was taken from."""

    index: int
    offset_seconds: float
    jpeg_bytes: bytes

    def as_data_url(self) -> str:
        """Encode the frame as a ``data:`` URL suitable for the OpenAI vision API."""
        encoded = base64.b64encode(self.jpeg_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


@runtime_checkable
class FrameExtractor(Protocol):
    """Samples frames from a video at a fixed cadence."""

    def extract(self, video_path: Path) -> list[ExtractedFrame]:
        """Return the sampled frames in chronological order."""
        ...


class FfmpegFrameExtractor:
    """Extract one downscaled JPEG frame every ``interval_seconds`` using ffmpeg.

    A single ffmpeg invocation samples and rescales every frame in one decode pass, which
    is dramatically faster than seeking to each timestamp individually.
    """

    def __init__(
        self,
        *,
        interval_seconds: float = 30.0,
        max_dimension: int = 768,
        jpeg_quality: int = 4,
        ffmpeg_path: str = "ffmpeg",
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if max_dimension <= 0:
            raise ValueError("max_dimension must be positive")
        self._interval = interval_seconds
        self._max_dimension = max_dimension
        self._jpeg_quality = jpeg_quality
        self._ffmpeg_path = ffmpeg_path

    def extract(self, video_path: Path) -> list[ExtractedFrame]:
        if not video_path.is_file():
            raise FrameExtractionError(f"Video file not found: {video_path}")
        if shutil.which(self._ffmpeg_path) is None:
            raise FfmpegNotFoundError(
                f"'{self._ffmpeg_path}' is not on the PATH. Install ffmpeg to extract frames."
            )

        with TemporaryDirectory(prefix="wds-frames-") as tmp:
            tmp_dir = Path(tmp)
            self._run_ffmpeg(video_path, tmp_dir)
            return list(self._collect_frames(tmp_dir))

    def _run_ffmpeg(self, video_path: Path, out_dir: Path) -> None:
        # fps=1/interval samples one frame per interval; scale downsizes the longest edge
        # while preserving aspect ratio (-2 keeps the other dimension even for the encoder).
        fps = 1.0 / self._interval
        scale = (
            f"scale='if(gt(iw,ih),min({self._max_dimension},iw),-2)'"
            f":'if(gt(iw,ih),-2,min({self._max_dimension},ih))'"
        )
        command = [
            self._ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(video_path),
            "-vf", f"fps={fps:.10f},{scale}",
            "-q:v", str(self._jpeg_quality),
            str(out_dir / "frame_%06d.jpg"),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise FrameExtractionError(
                f"ffmpeg failed (exit {result.returncode}): {result.stderr.strip()}"
            )

    def _collect_frames(self, out_dir: Path) -> Iterator[ExtractedFrame]:
        frame_paths = sorted(out_dir.glob("frame_*.jpg"))
        if not frame_paths:
            raise FrameExtractionError(
                "ffmpeg produced no frames; the recording may be empty or "
                "shorter than one interval."
            )
        for index, path in enumerate(frame_paths):
            # ffmpeg numbers output 1..N; frame N covers the window starting at N*interval.
            offset = index * self._interval
            yield ExtractedFrame(index=index, offset_seconds=offset, jpeg_bytes=path.read_bytes())


__all__ = ["ExtractedFrame", "FfmpegFrameExtractor", "FrameExtractor"]
