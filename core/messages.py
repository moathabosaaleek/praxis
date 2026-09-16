from dataclasses import dataclass, field
from datetime import datetime

# Telegram limits callback payloads to 64 bytes.
MAX_CHOICE_VALUE_BYTES = 64


@dataclass(frozen=True)
class Choice:
    label: str
    value: str

    def __post_init__(self):
        if len(self.value.encode()) > MAX_CHOICE_VALUE_BYTES:
            raise ValueError(f"Choice value exceeds {MAX_CHOICE_VALUE_BYTES} bytes: {self.value!r}")


@dataclass(frozen=True)
class IncomingMessage:
    user_id: int
    chat_id: int
    text: str
    received_at: datetime
    choice: str | None = None

    @property
    def is_choice(self) -> bool:
        return self.choice is not None


@dataclass(frozen=True)
class AssistantResponse:
    text: str
    choices: tuple[Choice, ...] = field(default=())
