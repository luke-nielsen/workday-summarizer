from __future__ import annotations

from workday_summarizer.pricing import estimate_cost
from workday_summarizer.usage import TokenUsage


def test_known_model_uses_table() -> None:
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, requests=1)
    # gpt-4o: $2.50 in + $10.00 out per million tokens.
    assert estimate_cost("gpt-4o-2024-08-06", usage) == 12.5


def test_longest_prefix_wins() -> None:
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=0, requests=1)
    # "gpt-4o-mini" must beat "gpt-4o": $0.15 not $2.50.
    assert estimate_cost("gpt-4o-mini", usage) == 0.15


def test_unknown_model_returns_none() -> None:
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000, requests=1)
    assert estimate_cost("some-private-model", usage) is None


def test_overrides_take_precedence() -> None:
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, requests=1)
    cost = estimate_cost(
        "some-private-model", usage, input_per_mtok=1.0, output_per_mtok=2.0
    )
    assert cost == 3.0
