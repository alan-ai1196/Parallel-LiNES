"""Configuration loading utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from .env and environment variables."""

    openai_api_key: str
    openai_base_url: str | None
    openai_org_id: str | None
    openai_project_id: str | None
    planner_model: str
    worker_model: str
    planner_temperature: float
    worker_temperature: float
    max_concurrency: int
    max_retries: int
    retry_base_delay_ms: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _pick_env_file(explicit_env_file: Path | str | None) -> Path | None:
    if explicit_env_file is not None:
        path = Path(explicit_env_file).expanduser().resolve()
        return path if path.exists() else None

    package_root = Path(__file__).resolve().parents[1]
    candidates = [
        Path.cwd() / ".env",
        package_root / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _get_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def _get_optional(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_int(name: str, raw: str, *, minimum: int = 0) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got: {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got: {value}")
    return value


def _parse_float(name: str, raw: str, *, minimum: float = 0.0, maximum: float = 2.0) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a float, got: {raw!r}") from exc
    if value < minimum or value > maximum:
        raise ConfigError(f"{name} must be in [{minimum}, {maximum}], got: {value}")
    return value


def load_settings(env_file: Path | str | None = None) -> Settings:
    """Load runtime configuration.

    Precedence: existing process environment overrides values loaded from .env.
    """

    resolved_env = _pick_env_file(env_file)
    if resolved_env is not None:
        load_dotenv(resolved_env, override=False)

    return Settings(
        openai_api_key=_get_required("OPENAI_API_KEY"),
        openai_base_url=_get_optional("OPENAI_BASE_URL"),
        openai_org_id=_get_optional("OPENAI_ORG_ID"),
        openai_project_id=_get_optional("OPENAI_PROJECT_ID"),
        planner_model=_get_required("PLANNER_MODEL"),
        worker_model=_get_required("WORKER_MODEL"),
        planner_temperature=_parse_float(
            "PLANNER_TEMPERATURE",
            os.getenv("PLANNER_TEMPERATURE", "0.2"),
            minimum=0.0,
            maximum=2.0,
        ),
        worker_temperature=_parse_float(
            "WORKER_TEMPERATURE",
            os.getenv("WORKER_TEMPERATURE", "0.2"),
            minimum=0.0,
            maximum=2.0,
        ),
        max_concurrency=_parse_int("MAX_CONCURRENCY", os.getenv("MAX_CONCURRENCY", "4"), minimum=1),
        max_retries=_parse_int("MAX_RETRIES", os.getenv("MAX_RETRIES", "3"), minimum=0),
        retry_base_delay_ms=_parse_int(
            "RETRY_BASE_DELAY_MS", os.getenv("RETRY_BASE_DELAY_MS", "500"), minimum=1
        ),
    )
