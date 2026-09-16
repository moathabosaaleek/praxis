import logging
import tomllib
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"
_FALLBACK_VERSION = "0.0.0-unknown"


@lru_cache(maxsize=1)
def get_version() -> str:
    try:
        data = tomllib.loads(_PYPROJECT_PATH.read_text())
        return data["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Could not read version from pyproject.toml: %s", exc)
        return _FALLBACK_VERSION
