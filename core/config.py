import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEZONE = "UTC"
DEFAULT_DB_PATH = "praxis.db"


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    admin_telegram_id: int
    gemini_api_key: str
    llm_model: str = DEFAULT_LLM_MODEL
    timezone: ZoneInfo = ZoneInfo(DEFAULT_TIMEZONE)
    db_path: Path = Path(DEFAULT_DB_PATH)
    env: str = "development"


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"CRITICAL: {name} not found in environment.")
    return value


def _parse_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise RuntimeError(
            f"CRITICAL: TIMEZONE '{name}' is not a valid IANA timezone, e.g. 'Asia/Amman'."
        ) from exc


def load_settings() -> Settings:
    admin_telegram_id = _required_env("ADMIN_TELEGRAM_ID")

    try:
        parsed_admin_id = int(admin_telegram_id)
    except ValueError as exc:
        raise RuntimeError("CRITICAL: ADMIN_TELEGRAM_ID must be a valid number.") from exc

    telegram_bot_token = _required_env("TELEGRAM_BOT_TOKEN")
    if telegram_bot_token == "paste_your_botfather_token_here":
        raise RuntimeError("CRITICAL: TELEGRAM_BOT_TOKEN is still using the placeholder value.")

    return Settings(
        telegram_bot_token=telegram_bot_token,
        admin_telegram_id=parsed_admin_id,
        gemini_api_key=_required_env("GEMINI_API_KEY"),
        llm_model=os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL,
        timezone=_parse_timezone(os.getenv("TIMEZONE") or DEFAULT_TIMEZONE),
        db_path=Path(os.getenv("DB_PATH") or DEFAULT_DB_PATH),
        env=os.getenv("ENV", "development"),
    )
