# workday-summarizer

Turn a screen recording of your workday into a structured summary of **what you got done**
and **where you got distracted** — powered by an OpenAI vision model.

The tool samples one frame every 30 seconds, asks the model to describe each segment, and
synthesises those observations into a single JSON report: tasks, distractions, a focus
score, accomplishments, and recommendations.

## How it works

```
recording.mp4
     │
     ▼  ffmpeg samples 1 frame / 30s, downscaled to ≤768px
  [frames]
     │
     ▼  map: batches of frames → SegmentObservation   (OpenAI, concurrent, structured output)
[observations]
     │
     ▼  reduce: all observations → WorkdaySummaryDraft     (OpenAI, structured output)
     ▼  + focus metrics computed deterministically from the observations
 WorkdaySummary  → JSON + a Rich report
```

Batching keeps every request inside the model's context window and bounds cost on long
recordings; the map/reduce split lets the final summary reason over the whole day at once.
Batches are analyzed concurrently (`WDS_MAX_CONCURRENCY`, default 4), and rate-limit
responses are honoured via the server's `Retry-After` header.

The focus score and the productive/distracted second totals are **not** guessed by the
model — they are computed directly from the per-segment observations, so the headline
numbers can never contradict the segments they summarise. Each run also reports its token
usage and an estimated cost (when the model's pricing is known).

## Requirements

- Python 3.10+
- [`ffmpeg`](https://ffmpeg.org/) on your `PATH`
- An OpenAI API key

## Install

```bash
# with uv (recommended)
uv sync --extra dev

# or with pip
pip install -e ".[dev]"
```

Then provide your key:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

## Usage

```bash
# Render a report to the terminal
workday-summarizer recording.mp4

# Save the structured JSON
workday-summarizer recording.mp4 --output report.json

# Sample more often, pick a different model, show progress
workday-summarizer recording.mp4 --interval 15 --model gpt-4o --verbose

# Pipe machine-readable JSON to another tool
workday-summarizer recording.mp4 --quiet | jq '.focus_score'
```

Run `workday-summarizer --help` for all options. You can also invoke it as
`python -m workday_summarizer`.

### Example output (`report.json`)

```json
{
  "headline": "A focused morning of API work with a brief afternoon dip.",
  "narrative": "...",
  "tasks": [
    {
      "title": "Implement the login endpoint",
      "description": "Wrote and tested the /auth/login route in FastAPI.",
      "category": "coding",
      "start_seconds": 0.0,
      "end_seconds": 3600.0,
      "applications": ["VS Code", "Terminal"],
      "evidence": ["auth.py open in the editor", "pytest output in the terminal"]
    }
  ],
  "distractions": [
    {
      "description": "Scrolled a news site.",
      "category": "news",
      "start_seconds": 5400.0,
      "end_seconds": 5700.0,
      "applications": ["Safari"]
    }
  ],
  "key_accomplishments": ["Shipped the login endpoint"],
  "recommendations": ["Silence notifications during deep-work blocks"],
  "focus_score": 84,
  "productive_seconds": 5100.0,
  "distracted_seconds": 600.0
}
```

## Configuration

All settings can be set via environment variables (or `.env`). See
[`.env.example`](.env.example) for the full list and defaults. Anything in the config can
also be overridden per run with CLI flags.

## Development

```bash
uv run pytest        # tests (no network or ffmpeg needed — collaborators are faked)
uv run ruff check .  # lint
uv run mypy src      # type-check
```

### Design notes

- **Dependency-injected pipeline.** `WorkdayPipeline` takes a `FrameExtractor` and a
  `WorkdayAnalyzer`, both defined as `Protocol`s, so the whole flow is testable without
  ffmpeg or a network connection. `WorkdayPipeline.from_settings` wires the real ones.
- **Structured outputs over parsing.** Both model calls use OpenAI structured outputs
  bound to Pydantic schemas, so there is no brittle JSON post-processing.
- **Trustworthy timestamps.** Frame offsets come from our own sampling, not the model's
  arithmetic — segment times are overwritten with the true offsets before synthesis.
- **Resilient calls.** Transient failures (network, rate limits) are retried with
  exponential backoff; other API errors surface as a clear `AnalysisError`.

## Privacy

Frames are sent to the OpenAI API for analysis. Don't record screens containing secrets
you aren't comfortable transmitting, and review your OpenAI data-retention settings.

## License

MIT — see [LICENSE](LICENSE).
