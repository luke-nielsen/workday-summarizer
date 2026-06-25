"""Runtime configuration, loaded from the environment or an ``.env`` file."""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Values are read from environment variables (optionally prefixed ``WDS_``) or a local
    ``.env`` file. The OpenAI API key is the only required value; everything else has a
    sensible default that the CLI can still override per-invocation.
    """

    model_config = SettingsConfigDict(
        env_prefix="WDS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # The OpenAI SDK also reads OPENAI_API_KEY directly; we accept it here so the value
    # can be validated up-front and surfaced with a clear error if it is missing.
    openai_api_key: SecretStr = Field(
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key used for all requests.",
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias="OPENAI_BASE_URL",
        description="Optional override for the OpenAI-compatible API endpoint.",
    )

    model: str = Field(
        default="gpt-4o-2024-08-06",
        description="Vision-capable model that supports structured outputs.",
    )

    frame_interval_seconds: float = Field(
        default=30.0, gt=0, description="Seconds between sampled frames."
    )
    max_frame_dimension: int = Field(
        default=768, gt=0, description="Longest edge a frame is downscaled to before upload."
    )
    image_detail: str = Field(
        default="auto", description="Vision detail hint sent to the model: low, high, or auto."
    )
    batch_size: int = Field(
        default=12, gt=0, description="Number of frames sent per analysis request."
    )
    max_concurrency: int = Field(
        default=4, gt=0, description="Maximum number of analysis requests in flight at once."
    )

    request_timeout_seconds: float = Field(
        default=120.0, gt=0, description="Per-request timeout for OpenAI calls."
    )
    max_retries: int = Field(
        default=4, ge=0, description="Retry attempts for transient API failures."
    )

    # Cost-estimate overrides for non-standard pricing (Azure, proxies, negotiated rates).
    # Left unset, the built-in price table is used; unknown models simply report no cost.
    input_cost_per_mtok: float | None = Field(
        default=None, gt=0, description="USD per million input tokens, overriding the table."
    )
    output_cost_per_mtok: float | None = Field(
        default=None, gt=0, description="USD per million output tokens, overriding the table."
    )


__all__ = ["Settings"]
