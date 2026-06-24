from __future__ import annotations

from dataclasses import dataclass

from workday_summarizer.usage import TokenUsage


def test_addition_accumulates() -> None:
    total = TokenUsage(10, 5, 1) + TokenUsage(3, 2, 1)
    assert total == TokenUsage(13, 7, 2)
    assert total.total_tokens == 20


def test_from_completion_usage_reads_fields() -> None:
    @dataclass
    class _Usage:
        prompt_tokens: int = 7
        completion_tokens: int = 4

    usage = TokenUsage.from_completion_usage(_Usage())
    assert usage == TokenUsage(prompt_tokens=7, completion_tokens=4, requests=1)


def test_from_completion_usage_handles_missing() -> None:
    # Some OpenAI-compatible endpoints omit usage; the request is still counted.
    usage = TokenUsage.from_completion_usage(None)
    assert usage == TokenUsage(prompt_tokens=0, completion_tokens=0, requests=1)
