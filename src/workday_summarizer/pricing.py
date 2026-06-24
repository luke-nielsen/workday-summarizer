"""Best-effort cost estimation for a run.

Prices change and vary by endpoint, so this is deliberately a *best effort*: a small,
explicit table of published per-million-token rates keyed by model, with callers able to
override the rates (for Azure, proxies, or negotiated pricing). When the model is unknown
and no override is supplied, :func:`estimate_cost` returns ``None`` and the UI shows token
counts only — an honest "unknown" beats a confidently wrong dollar figure.
"""

from __future__ import annotations

from workday_summarizer.usage import TokenUsage

_MILLION = 1_000_000

# (input, output) US dollars per million tokens. Longest matching prefix wins, so
# "gpt-4o-mini" is matched before "gpt-4o".
_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


def _lookup_rates(model: str) -> tuple[float, float] | None:
    matches = [prefix for prefix in _PRICE_PER_MTOK if model.startswith(prefix)]
    if not matches:
        return None
    return _PRICE_PER_MTOK[max(matches, key=len)]


def estimate_cost(
    model: str,
    usage: TokenUsage,
    *,
    input_per_mtok: float | None = None,
    output_per_mtok: float | None = None,
) -> float | None:
    """Estimate the USD cost of ``usage`` for ``model``.

    Explicit ``*_per_mtok`` overrides take precedence over the built-in table. Returns
    ``None`` when no rate is known, signalling the caller to omit a dollar figure.
    """
    rates = _lookup_rates(model)
    input_rate = input_per_mtok if input_per_mtok is not None else (rates[0] if rates else None)
    output_rate = output_per_mtok if output_per_mtok is not None else (rates[1] if rates else None)
    if input_rate is None or output_rate is None:
        return None
    return (
        usage.prompt_tokens / _MILLION * input_rate
        + usage.completion_tokens / _MILLION * output_rate
    )


__all__ = ["estimate_cost"]
