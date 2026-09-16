from datetime import UTC, datetime

from core.config import Settings
from core.llm_router import LLMError
from core.messages import IncomingMessage
from core.plugin import PluginContext
from plugins.writer import (
    ADDENDUM_PROMPT,
    CUSTOM_TOPIC_PROMPT,
    NEED_ANSWER,
    NEED_TOPIC,
    TAP_STANCE,
    TAP_SUMMARY,
    WriterPlugin,
    _normalize_bullets,
    _strip_markdown_emphasis,
)
from storage.repositories import Session

SETTINGS = Settings(telegram_bot_token="123:abc", admin_telegram_id=42, gemini_api_key="key")


class FakeLLM:
    def __init__(self, answers=None, error=None):
        self.answers = list(answers) if answers else []
        self.error = error
        self.prompts = []

    async def generate_response(self, prompt, system_instruction=None, history=()):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.answers.pop(0) if self.answers else "generated text"


class FakeSessions:
    def __init__(self, session: Session | None = None):
        self._session = session
        self.saved = []
        self.cleared = False

    async def get(self, chat_id):
        return self._session

    async def set(self, chat_id, plugin, state):
        self._session = Session(plugin=plugin, state=state)
        self.saved.append(dict(state))

    async def clear(self, chat_id):
        self._session = None
        self.cleared = True


def make_message(text="", choice=None, chat_id=7):
    return IncomingMessage(
        user_id=42, chat_id=chat_id, text=text, received_at=datetime.now(UTC), choice=choice
    )


def make_ctx(llm=None, sessions=None):
    return PluginContext(settings=SETTINGS, llm=llm or FakeLLM(), messages=None, sessions=sessions)


async def test_fresh_start_offers_two_topics_and_a_custom_option():
    sessions = FakeSessions()

    response = await WriterPlugin().handle(make_message(), make_ctx(sessions=sessions))

    assert len(response.choices) == 3
    assert response.choices[-1].label == "Something else"
    assert sessions.saved[-1]["step"] == "await_topic_choice"
    assert len(sessions.saved[-1]["options"]) == 2


async def test_fresh_start_with_initial_text_skips_the_topic_menu():
    llm = FakeLLM(answers=["Is X actually dead?"])
    sessions = FakeSessions()

    response = await WriterPlugin().handle(
        make_message("ransomware trends"), make_ctx(llm, sessions)
    )

    assert response.text == "Is X actually dead?"
    assert {c.label for c in response.choices} == {"Agree", "Disagree", "It's complicated"}
    assert sessions.saved[-1]["topic"] == "ransomware trends"


async def test_topic_button_tap_generates_a_claim_and_stance_buttons():
    session = Session(plugin="writer", state={"step": "await_topic_choice", "options": ["A", "B"]})
    llm = FakeLLM(answers=["Claim about A"])

    response = await WriterPlugin().handle(
        make_message(choice="writer:topic:0"), make_ctx(llm, FakeSessions(session))
    )

    assert response.text == "Claim about A"
    assert {c.value for c in response.choices} == {
        "writer:stance:agree",
        "writer:stance:disagree",
        "writer:stance:complicated",
    }


async def test_custom_topic_button_asks_for_free_text():
    session = Session(plugin="writer", state={"step": "await_topic_choice", "options": ["A", "B"]})

    response = await WriterPlugin().handle(
        make_message(choice="writer:topic:custom"), make_ctx(sessions=FakeSessions(session))
    )

    assert response.text == CUSTOM_TOPIC_PROMPT


async def test_typing_instead_of_tapping_a_topic_is_treated_as_custom():
    session = Session(plugin="writer", state={"step": "await_topic_choice", "options": ["A", "B"]})
    llm = FakeLLM(answers=["Claim about phishing"])

    response = await WriterPlugin().handle(
        make_message("phishing kits"), make_ctx(llm, FakeSessions(session))
    )

    assert response.text == "Claim about phishing"


async def test_custom_topic_text_generates_a_claim():
    session = Session(plugin="writer", state={"step": "await_custom_topic"})
    llm = FakeLLM(answers=["Claim about B"])

    response = await WriterPlugin().handle(
        make_message("my custom topic"), make_ctx(llm, FakeSessions(session))
    )

    assert response.text == "Claim about B"


async def test_empty_custom_topic_asks_again():
    session = Session(plugin="writer", state={"step": "await_custom_topic"})

    response = await WriterPlugin().handle(
        make_message("   "), make_ctx(sessions=FakeSessions(session))
    )

    assert response.text == NEED_TOPIC


async def test_stance_button_asks_a_follow_up_question():
    session = Session(
        plugin="writer", state={"step": "await_stance", "topic": "X", "claim": "C", "answers": []}
    )
    llm = FakeLLM(answers=["Why do you think that?"])
    sessions = FakeSessions(session)

    response = await WriterPlugin().handle(
        make_message(choice="writer:stance:agree"), make_ctx(llm, sessions)
    )

    assert response.text == "Why do you think that?"
    assert sessions.saved[-1]["step"] == "await_answer_1"
    assert sessions.saved[-1]["stance"] == "agree"


async def test_typed_stance_word_also_works():
    session = Session(
        plugin="writer", state={"step": "await_stance", "topic": "X", "claim": "C", "answers": []}
    )
    llm = FakeLLM(answers=["Question?"])

    response = await WriterPlugin().handle(
        make_message("disagree"), make_ctx(llm, FakeSessions(session))
    )

    assert response.text == "Question?"


async def test_unrecognized_stance_reply_asks_to_tap_a_button():
    session = Session(
        plugin="writer", state={"step": "await_stance", "topic": "X", "claim": "C", "answers": []}
    )

    response = await WriterPlugin().handle(
        make_message("maybe?"), make_ctx(sessions=FakeSessions(session))
    )

    assert response.text == TAP_STANCE


async def test_first_answer_asks_the_second_follow_up():
    session = Session(
        plugin="writer",
        state={
            "step": "await_answer_1",
            "topic": "X",
            "claim": "C",
            "stance": "agree",
            "answers": [],
        },
    )
    llm = FakeLLM(answers=["Second question?"])
    sessions = FakeSessions(session)

    response = await WriterPlugin().handle(make_message("my first answer"), make_ctx(llm, sessions))

    assert response.text == "Second question?"
    assert sessions.saved[-1]["step"] == "await_answer_2"
    assert sessions.saved[-1]["answers"] == ["my first answer"]


async def test_empty_answer_is_rejected():
    session = Session(
        plugin="writer",
        state={
            "step": "await_answer_1",
            "topic": "X",
            "claim": "C",
            "stance": "agree",
            "answers": [],
        },
    )

    response = await WriterPlugin().handle(
        make_message("  "), make_ctx(sessions=FakeSessions(session))
    )

    assert response.text == NEED_ANSWER


async def test_second_answer_produces_a_summary_with_confirm_choices():
    session = Session(
        plugin="writer",
        state={
            "step": "await_answer_2",
            "topic": "X",
            "claim": "C",
            "stance": "agree",
            "answers": ["first"],
        },
    )
    llm = FakeLLM(answers=["- You think X\n- Because Y"])
    sessions = FakeSessions(session)

    response = await WriterPlugin().handle(make_message("second answer"), make_ctx(llm, sessions))

    assert "You think X" in response.text
    assert {c.value for c in response.choices} == {"writer:summary:confirm", "writer:summary:more"}
    assert sessions.saved[-1]["step"] == "await_summary_choice"


async def test_confirming_the_summary_produces_a_draft_and_clears_the_session():
    session = Session(
        plugin="writer",
        state={"step": "await_summary_choice", "topic": "X", "summary": "- point"},
    )
    llm = FakeLLM(answers=["Option 1:\ndraft one"])
    sessions = FakeSessions(session)

    response = await WriterPlugin().handle(
        make_message(choice="writer:summary:confirm"), make_ctx(llm, sessions)
    )

    assert response.text == "Option 1:\ndraft one"
    assert sessions.cleared is True


async def test_asking_to_add_something_requests_an_addendum():
    session = Session(
        plugin="writer",
        state={"step": "await_summary_choice", "topic": "X", "summary": "- point"},
    )
    sessions = FakeSessions(session)

    response = await WriterPlugin().handle(
        make_message(choice="writer:summary:more"), make_ctx(sessions=sessions)
    )

    assert response.text == ADDENDUM_PROMPT
    assert sessions.saved[-1]["step"] == "await_addendum"
    assert sessions.cleared is False


async def test_typing_instead_of_tapping_the_summary_choice_asks_to_tap():
    session = Session(
        plugin="writer",
        state={"step": "await_summary_choice", "topic": "X", "summary": "- point"},
    )

    response = await WriterPlugin().handle(
        make_message("looks good"), make_ctx(sessions=FakeSessions(session))
    )

    assert response.text == TAP_SUMMARY


async def test_addendum_text_produces_a_draft_including_the_note():
    session = Session(
        plugin="writer",
        state={"step": "await_addendum", "topic": "X", "summary": "- point"},
    )
    llm = FakeLLM(answers=["final draft"])
    sessions = FakeSessions(session)

    response = await WriterPlugin().handle(
        make_message("make it punchier"), make_ctx(llm, sessions)
    )

    assert response.text == "final draft"
    assert "make it punchier" in llm.prompts[0]
    assert sessions.cleared is True


async def test_llm_failure_during_claim_generation_falls_back_gracefully():
    sessions = FakeSessions()
    llm = FakeLLM(error=LLMError("boom"))

    response = await WriterPlugin().handle(make_message("some topic"), make_ctx(llm, sessions))

    assert "some topic" in response.text
    assert len(response.choices) == 3


async def test_unknown_session_step_restarts_the_flow():
    session = Session(plugin="writer", state={"step": "not-a-real-step"})
    sessions = FakeSessions(session)

    response = await WriterPlugin().handle(make_message(), make_ctx(sessions=sessions))

    assert len(response.choices) == 3
    assert sessions.saved[-1]["step"] == "await_topic_choice"


async def test_full_topic_text_is_listed_in_the_message_body():
    llm = FakeLLM()
    sessions = FakeSessions()

    response = await WriterPlugin().handle(make_message(), make_ctx(llm, sessions))

    for index, topic in enumerate(sessions.saved[-1]["options"]):
        assert f"{index + 1}. {topic}" in response.text


async def test_topic_buttons_stay_short_so_telegram_cannot_elide_them():
    llm = FakeLLM()
    sessions = FakeSessions()

    response = await WriterPlugin().handle(make_message(), make_ctx(llm, sessions))

    assert [c.label for c in response.choices] == ["Topic 1", "Topic 2", "Something else"]
    assert [c.value for c in response.choices] == [
        "writer:topic:0",
        "writer:topic:1",
        "writer:topic:custom",
    ]


async def test_draft_strips_stray_markdown_emphasis_since_it_is_copied_verbatim():
    session = Session(
        plugin="writer",
        state={"step": "await_summary_choice", "topic": "X", "summary": "- point"},
    )
    llm = FakeLLM(answers=["Option 1:\nthey *definitely* broke something\n\nHashtags: none"])

    response = await WriterPlugin().handle(
        make_message(choice="writer:summary:confirm"), make_ctx(llm, FakeSessions(session))
    )

    assert "*" not in response.text
    assert "they definitely broke something" in response.text


def test_strip_markdown_emphasis_leaves_plain_text_untouched():
    assert _strip_markdown_emphasis("plain text, no markup") == "plain text, no markup"


def test_normalize_bullets_uses_one_consistent_symbol_regardless_of_input():
    assert _normalize_bullets("* one\n- two\nthree") == "• one\n• two\n• three"


async def test_summary_uses_a_consistent_bullet_regardless_of_model_style():
    session = Session(
        plugin="writer",
        state={
            "step": "await_answer_2",
            "topic": "X",
            "claim": "C",
            "stance": "agree",
            "answers": ["first"],
        },
    )
    llm = FakeLLM(answers=["* Point one\n* Point two"])

    response = await WriterPlugin().handle(
        make_message("second answer"), make_ctx(llm, FakeSessions(session))
    )

    assert "• Point one" in response.text
    assert "* Point one" not in response.text
