"""Token-usage accounting for OpenAI calls.

Every request reports how many tokens it consumed; :class:`TokenUsage` accumulates those
figures across the whole map/reduce run so the cost of a summarization is observable
rather than a mystery. The type is immutable and composes with ``+``, which keeps the
analyzer's accounting free of shared mutable state — each request produces a usage value
and the totals are summed at the end, even when requests run concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Tokens consumed across one or more model requests."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            requests=self.requests + other.requests,
        )

    @classmethod
    def from_completion_usage(cls, usage: Any) -> TokenUsage:
        """Build a usage record for a single request from an OpenAI ``usage`` object.

        Some OpenAI-compatible endpoints omit usage entirely; in that case the request is
        still counted, but with zero tokens so the totals never silently overstate cost.
        """
        if usage is None:
            return cls(requests=1)
        return cls(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            requests=1,
        )


__all__ = ["TokenUsage"]
