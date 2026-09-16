from types import SimpleNamespace

import pytest

from core.messages import MAX_CHOICE_VALUE_BYTES, AssistantResponse, Choice
from interfaces.telegram.bot import build_keyboard, to_incoming_message


def make_update(text=None, user_id=42, callback_data=None):
    message = SimpleNamespace(message_id=1, chat_id=7, text=text)
    callback_query = SimpleNamespace(data=callback_data) if callback_data else None
    return SimpleNamespace(
        effective_message=message,
        effective_user=SimpleNamespace(id=user_id),
        callback_query=callback_query,
    )


def test_converts_a_text_message():
    incoming = to_incoming_message(make_update(text="hello"))

    assert incoming.user_id == 42
    assert incoming.chat_id == 7
    assert incoming.text == "hello"
    assert incoming.choice is None
    assert incoming.is_choice is False
    assert incoming.received_at.tzinfo is not None


def test_converts_a_button_tap_into_a_choice():
    incoming = to_incoming_message(make_update(text="Pick one", callback_data="topic:1"))

    assert incoming.text == "topic:1"
    assert incoming.choice == "topic:1"
    assert incoming.is_choice is True


def test_ignores_updates_without_a_user():
    update = make_update(text="hello")
    update.effective_user = None

    assert to_incoming_message(update) is None


def test_ignores_non_text_messages():
    assert to_incoming_message(make_update(text=None)) is None


def test_no_keyboard_without_choices():
    assert build_keyboard(AssistantResponse("hi")) is None


def test_keyboard_has_one_button_per_choice():
    response = AssistantResponse(
        "Pick", choices=(Choice("Overhyped", "a"), Choice("Underrated", "b"))
    )

    keyboard = build_keyboard(response)
    buttons = [row[0] for row in keyboard.inline_keyboard]

    assert [b.text for b in buttons] == ["Overhyped", "Underrated"]
    assert [b.callback_data for b in buttons] == ["a", "b"]


def test_choice_value_over_telegram_limit_is_rejected():
    with pytest.raises(ValueError, match="64 bytes"):
        Choice("too long", "x" * (MAX_CHOICE_VALUE_BYTES + 1))
