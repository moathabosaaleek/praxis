# Praxis

Praxis is a private, personal assistant designed to run through a messaging interface.

The current prototype uses Telegram as the first interface and Gemini as the first LLM provider. The long-term goal is to keep the assistant core platform-agnostic so the messaging interface can be replaced or extended later.

## Current Features

- Private Telegram bot interface
- Admin-only access by Telegram user ID
- `/start` health message
- `/ping` responsiveness check
- `/ask` command backed by Gemini
- Basic current-time grounding for LLM responses
- Google Search enabled for Gemini responses

## Planned Direction

- Natural conversation instead of command-only usage
- Interface adapters for Telegram, Matrix, or other chat platforms
- Contextual memory with explicit user approval
- Sensitivity detection before storing memory or using cloud AI
- Encrypted local storage
- Private extension modules through a separate `praxis-ext` repo
- Collaborative drafting and research workflows

## Security Model

Praxis is intended for private use by a single user.

The current Telegram prototype is not suitable for highly sensitive conversations. Telegram bot chats are not the same as end-to-end encrypted secret chats. Sensitive data should not be sent through the Telegram interface until a stronger secure mode is implemented.

Planned security principles:

- Keep secrets out of Git
- Use `.env` for local credentials
- Commit only `.env.example`
- Store approved memory only
- Avoid logging sensitive message content
- Encrypt local storage before storing personal memory
- Use cloud AI only for low-risk data unless explicitly approved

## Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

Then fill in the required values in `.env`.

Run the bot:

```bash
python main.py
```

## Environment Variables

See `.env.example` for all variables.

| Variable | Required | Purpose |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | Yes | Bot token from BotFather |
| `ADMIN_TELEGRAM_ID` | Yes | The only Telegram user allowed to use the bot |
| `GEMINI_API_KEY` | Yes | Gemini API key |
| `LLM_MODEL` | No | Gemini model name, defaults to `gemini-2.5-flash` |
| `TIMEZONE` | No | IANA timezone such as `Asia/Amman`, defaults to `UTC` |
| `DB_PATH` | No | SQLite file for history and memory, defaults to `praxis.db` |
| `ENV` | No | `development` or `production` |

## Development

Install development tools:

```bash
pip install -r requirements-dev.txt
```

Run tests and checks before opening a pull request:

```bash
pytest
ruff check .
ruff format --check .
```

## Versioning and Releases

Praxis follows [Semantic Versioning](https://semver.org/) and uses [Conventional Commits](https://www.conventionalcommits.org/) for commit messages, most commonly:

- `feat:` a new capability
- `fix:` a bug fix
- `chore:` tooling, dependencies, or config with no behavior change
- `refactor:` internal restructuring with no behavior change
- `docs:` documentation only
- `test:` tests only

On every push to `main`, [release-please](https://github.com/googleapis/release-please) opens or updates a release pull request that accumulates the pending changes into `CHANGELOG.md`. Merging that pull request tags the release and bumps the version in `pyproject.toml`.

The bot reports its own running version with `/version` in Telegram, and logs it on startup. This is the fastest way to confirm which build is deployed when debugging.

While the project is at `0.x`, breaking changes can land without a major version bump.

## Repository Plan

This repository contains the public Praxis core.

Private personal workflows, prompts, ledgers, and integrations should live in a separate private repository named `praxis-ext`.
