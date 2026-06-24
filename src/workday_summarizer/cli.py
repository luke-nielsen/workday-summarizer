"""Command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console

from workday_summarizer.config import Settings
from workday_summarizer.errors import WorkdaySummarizerError
from workday_summarizer.pipeline import WorkdayPipeline
from workday_summarizer.reporting import render_summary

app = typer.Typer(
    add_completion=False,
    help="Summarize a screen recording of your workday into tasks and distractions.",
)

console = Console()
err_console = Console(stderr=True)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def summarize(
    video: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the screen recording to analyze.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the summary as JSON to this file."
    ),
    interval: float | None = typer.Option(
        None, "--interval", "-i", min=1.0, help="Seconds between sampled frames (default 30)."
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Override the OpenAI model."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress the rendered report."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log progress to stderr."),
) -> None:
    """Extract frames from VIDEO, analyze them, and report the day's tasks and distractions."""
    _configure_logging(verbose)

    try:
        settings = Settings()  # values come from the environment / .env
    except ValidationError:
        err_console.print(
            "[red]Missing configuration.[/red] Set OPENAI_API_KEY (e.g. in a .env file)."
        )
        raise typer.Exit(code=2) from None

    if interval is not None:
        settings.frame_interval_seconds = interval
    if model is not None:
        settings.model = model

    pipeline = WorkdayPipeline.from_settings(settings)

    try:
        with err_console.status("Extracting frames and analyzing your workday…"):
            result = pipeline.run(video)
    except WorkdaySummarizerError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if output is not None:
        output.write_text(result.summary.model_dump_json(indent=2), encoding="utf-8")
        err_console.print(f"[green]Wrote summary to[/green] {output}")

    if not quiet:
        render_summary(result.summary, console)
    elif output is None:
        # Nothing else would be shown, so emit JSON to stdout to stay scriptable.
        console.print_json(result.summary.model_dump_json())


def main() -> None:  # pragma: no cover - thin entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
