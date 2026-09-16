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
