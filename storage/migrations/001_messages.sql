CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    user_id INTEGER,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sensitivity TEXT NOT NULL DEFAULT 'low'
        CHECK (sensitivity IN ('low', 'medium', 'high', 'secret')),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_id_created
    ON messages (chat_id, id);
