import re
from html import escape

# Leaves headroom under Telegram's 4096-character limit for HTML escaping and tags.
CHUNK_LIMIT = 3500

_CODE_BLOCK = re.compile(r"```[\w+-]*\n?(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_PLACEHOLDER = re.compile(r"\x00(\d+)\x00")


def markdown_to_telegram_html(text: str) -> str:
    stashed: list[str] = []

    def stash(html: str) -> str:
        stashed.append(html)
        return f"\x00{len(stashed) - 1}\x00"

    text = text.replace("\x00", "")
    text = _CODE_BLOCK.sub(lambda m: stash(f"<pre>{escape(m.group(1), quote=False)}</pre>"), text)
    text = _INLINE_CODE.sub(
        lambda m: stash(f"<code>{escape(m.group(1), quote=False)}</code>"), text
    )
    text = escape(text, quote=False)
    text = _HEADING.sub(lambda m: f"<b>{m.group(1).replace('**', '')}</b>", text)
    text = _BOLD.sub(r"<b>\1</b>", text)
    return _PLACEHOLDER.sub(lambda m: stashed[int(m.group(1))], text)


def split_message(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()

    while len(remaining) > limit:
        cut = remaining.rfind("\n\n", 0, limit)
        if cut <= 0:
            cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            cut = remaining.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks
