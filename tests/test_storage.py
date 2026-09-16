import pytest

from storage.db import Database
from storage.repositories import MessageRepository, SessionRepository


@pytest.fixture
async def database(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    yield db
    await db.close()


async def test_migrations_create_the_messages_table(database):
    async with database.connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ) as cursor:
        tables = {row["name"] for row in await cursor.fetchall()}

    assert "messages" in tables
    assert "sessions" in tables
    assert "schema_migrations" in tables


async def test_migrations_are_applied_only_once(database):
    async def count_applied():
        async with database.connection.execute(
            "SELECT COUNT(*) AS n FROM schema_migrations"
        ) as cur:
            return (await cur.fetchone())["n"]

    before = await count_applied()
    await database.migrate()
    after = await count_applied()

    assert before > 0
    assert after == before


async def test_recent_returns_turns_in_chronological_order(database):
    repo = MessageRepository(database)
    await repo.add(chat_id=1, role="user", content="first", user_id=42)
    await repo.add(chat_id=1, role="assistant", content="second")
    await repo.add(chat_id=1, role="user", content="third", user_id=42)

    turns = await repo.recent(chat_id=1, limit=10)

    assert [(t.role, t.text) for t in turns] == [
        ("user", "first"),
        ("assistant", "second"),
        ("user", "third"),
    ]


async def test_recent_keeps_the_newest_within_the_limit(database):
    repo = MessageRepository(database)
    for index in range(5):
        await repo.add(chat_id=1, role="user", content=f"m{index}")

    turns = await repo.recent(chat_id=1, limit=2)

    assert [t.text for t in turns] == ["m3", "m4"]


async def test_history_is_separated_per_chat(database):
    repo = MessageRepository(database)
    await repo.add(chat_id=1, role="user", content="mine")
    await repo.add(chat_id=2, role="user", content="theirs")

    turns = await repo.recent(chat_id=1, limit=10)

    assert [t.text for t in turns] == ["mine"]


async def test_invalid_role_is_rejected_by_the_schema(database):
    with pytest.raises(Exception, match="CHECK constraint failed"):
        await MessageRepository(database).add(chat_id=1, role="robot", content="nope")


async def test_missing_session_returns_none(database):
    assert await SessionRepository(database).get(chat_id=1) is None


async def test_session_round_trips_plugin_and_state(database):
    repo = SessionRepository(database)
    await repo.set(chat_id=1, plugin="writer", state={"step": "await_stance", "topic": "AI"})

    session = await repo.get(chat_id=1)

    assert session.plugin == "writer"
    assert session.state == {"step": "await_stance", "topic": "AI"}


async def test_setting_a_session_again_overwrites_it(database):
    repo = SessionRepository(database)
    await repo.set(chat_id=1, plugin="writer", state={"step": "a"})
    await repo.set(chat_id=1, plugin="writer", state={"step": "b"})

    session = await repo.get(chat_id=1)

    assert session.state == {"step": "b"}


async def test_clear_removes_the_session(database):
    repo = SessionRepository(database)
    await repo.set(chat_id=1, plugin="writer", state={"step": "a"})

    await repo.clear(chat_id=1)

    assert await repo.get(chat_id=1) is None


async def test_clearing_a_missing_session_does_not_raise(database):
    await SessionRepository(database).clear(chat_id=999)


async def test_sessions_are_separated_per_chat(database):
    repo = SessionRepository(database)
    await repo.set(chat_id=1, plugin="writer", state={"step": "mine"})
    await repo.set(chat_id=2, plugin="chat", state={"step": "theirs"})

    session = await repo.get(chat_id=1)

    assert session.plugin == "writer"
    assert session.state == {"step": "mine"}
