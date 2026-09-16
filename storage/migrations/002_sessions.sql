CREATE TABLE IF NOT EXISTS sessions (
    chat_id INTEGER PRIMARY KEY,
    plugin TEXT NOT NULL,
    state TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
