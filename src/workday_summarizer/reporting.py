"""Render a :class:`WorkdaySummary` to a Rich console."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from workday_summarizer.models import WorkdaySummary, _format_clock


def _ratio_line(summary: WorkdaySummary) -> str:
    total = summary.productive_seconds + summary.distracted_seconds
    if total <= 0:
        return "No activity measured."
    productive_pct = 100 * summary.productive_seconds / total
    return (
        f"Productive: {_format_clock(summary.productive_seconds)} ({productive_pct:.0f}%)  •  "
        f"Distracted: {_format_clock(summary.distracted_seconds)} ({100 - productive_pct:.0f}%)"
    )


def render_summary(summary: WorkdaySummary, console: Console | None = None) -> None:
    """Print a human-friendly view of the summary."""
    console = console or Console()

    console.print(
        Panel(
            f"[bold]{summary.headline}[/bold]\n\n{summary.narrative}",
            title=f"Workday Summary  •  Focus score {summary.focus_score}/100",
            border_style="cyan",
        )
    )
    console.print(_ratio_line(summary), style="dim")

    if summary.tasks:
        tasks = Table(title="Tasks", title_style="bold green", expand=True)
        tasks.add_column("Time", style="green", no_wrap=True)
        tasks.add_column("Task", style="bold")
        tasks.add_column("Category", style="cyan", no_wrap=True)
        tasks.add_column("Details")
        for task in summary.tasks:
            tasks.add_row(task.clock_range, task.title, task.category, task.description)
        console.print(tasks)

    if summary.distractions:
        distractions = Table(title="Distractions", title_style="bold yellow", expand=True)
        distractions.add_column("Time", style="yellow", no_wrap=True)
        distractions.add_column("Category", style="magenta", no_wrap=True)
        distractions.add_column("What happened")
        for distraction in summary.distractions:
            distractions.add_row(
                distraction.clock_range, distraction.category, distraction.description
            )
        console.print(distractions)
    else:
        console.print("No distractions detected. 🎯", style="green")

    if summary.key_accomplishments:
        console.print("\n[bold]Key accomplishments[/bold]")
        for accomplishment in summary.key_accomplishments:
            console.print(f"  • {accomplishment}")

    if summary.recommendations:
        console.print("\n[bold]Recommendations[/bold]")
        for recommendation in summary.recommendations:
            console.print(f"  → {recommendation}", style="cyan")


__all__ = ["render_summary"]
