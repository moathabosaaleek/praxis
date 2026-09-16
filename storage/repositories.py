import json
from dataclasses import dataclass
from typing import Any

from core.messages import ConversationTurn
from storage.db import Database, utc_now_iso


class MessageRepository:
    def __init__(self, database: Database):
        self._db = database

    async def add(
        self,
        *,
        chat_id: int,
        role: str,
        content: str,
        user_id: int | None = None,
        sensitivity: str = "low",
    ) -> None:
        await self._db.connection.execute(
            "INSERT INTO messages (chat_id, user_id, role, content, sensitivity, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, user_id, role, content, sensitivity, utc_now_iso()),
        )
        await self._db.connection.commit()

    async def recent(self, chat_id: int, limit: int) -> tuple[ConversationTurn, ...]:
        async with self._db.connection.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()

        return tuple(ConversationTurn(row["role"], row["content"]) for row in reversed(rows))


@dataclass(frozen=True)
class Session:
    plugin: str
    state: dict[str, Any]


class SessionRepository:
    """One active session per chat, so a plugin's multi-turn flow survives restarts."""

    def __init__(self, database: Database):
        self._db = database

    async def get(self, chat_id: int) -> Session | None:
        async with self._db.connection.execute(
            "SELECT plugin, state FROM sessions WHERE chat_id = ?", (chat_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None
        return Session(plugin=row["plugin"], state=json.loads(row["state"]))

    async def set(self, chat_id: int, plugin: str, state: dict[str, Any]) -> None:
        await self._db.connection.execute(
            "INSERT INTO sessions (chat_id, plugin, state, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(chat_id) DO UPDATE SET "
            "plugin = excluded.plugin, state = excluded.state, updated_at = excluded.updated_at",
            (chat_id, plugin, json.dumps(state), utc_now_iso()),
        )
        await self._db.connection.commit()

    async def clear(self, chat_id: int) -> None:
        await self._db.connection.execute("DELETE FROM sessions WHERE chat_id = ?", (chat_id,))
        await self._db.connection.commit()
