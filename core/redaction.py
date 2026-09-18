import re

REDACTED = "[REDACTED]"

# Ordered most specific first; each pattern replaces the whole match.
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----.*?-----END[A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b"),  # Telegram bot token
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),  # Google API key (older format)
    re.compile(r"\bAQ\.[A-Za-z0-9_-]{20,}\b"),  # Google API key (newer AI Studio format)
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),  # GitHub token
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),  # OpenAI-style key
    re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),  # card-like number
)

_LABELLED_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|token|password|passwd|secret)\b(\s*[:=]\s*)(\S+)"
)


def redact(text: str) -> str:
    """Remove secret-looking values before storing text or sending it to a provider."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return _LABELLED_SECRET.sub(rf"\1\2{REDACTED}", text)
