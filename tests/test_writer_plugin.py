from datetime import UTC, datetime

import pytest

from core.config import Settings
from core.llm_router import LLMError, LLMQuotaError, LLMRateLimitError
from core.messages import IncomingMessage
from core.plugin import PluginContext
from plugins.writer import (
    ADDENDUM_PROMPT,
    CUSTOM_TOPIC_PROMPT,
    JUST_WRITE_IT,
    MAX_QUESTIONS,
    MAX_TURNS,
    NEED_ANSWER,
    NEED_TOPIC,
    NO_OPINION_YET,
    SIMPLE_ENGLISH,
    STANCE_CHOICES,
    SUMMARY_CHOICES,
    TAP_STANCE,
    TAP_SUMMARY,
    WriterPlugin,
    _normalize_bullets,
    _strip_markdown_emphasis,
)
from storage.repositories import Session

SETTINGS = Settings(telegram_bot_token="123:abc", admin_telegram_id=42, gemini_api_key="key")


class FakeLLM:
    def __init__(self, answers=None, decisions=None, error=None, json_error=None):
        self.answers = list(answers) if answers else []
        self.decisions = list(decisions) if decisions else []
        self.error = error
        self.json_error = json_error
        self.prompts = []
        self.json_prompts = []
        self.searches = []

    async def generate_response(self, prompt, system_instruction=None, history=(), search=False):
        self.prompts.append(prompt)
        self.searches.append(search)
        if self.error:
            raise self.error
        return self.answers.pop(0) if self.answers else "generated text"

    async def generate_json(self, prompt, spec, system_instruction=None):
        self.json_prompts.append(prompt)
        if self.json_error:
            raise self.json_error
        return self.decisions.pop(0) if self.decisions else {"action": "ready", "message": ""}


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


def interview_session(questions_asked=1, turns=1, transcript=None):
    return Session(
        plugin="writer",
        state={
            "step": "interview",
            "topic": "X",
            "claim": "C",
            "stance": "agree",
            "transcript": transcript or [{"role": "assistant", "text": "First question?"}],
            "questions_asked": questions_asked,
            "turns": turns,
        },
    )


# --- topic selection -------------------------------------------------------


async def test_fresh_start_offers_two_topics_and_a_custom_option():
    sessions = FakeSessions()

    response = await WriterPlugin().handle(make_message(), make_ctx(sessions=sessions))

    assert len(response.choices) == 3
    assert response.choices[-1].label == "Something else"
    assert sessions.saved[-1]["step"] == "await_topic_choice"
    assert len(sessions.saved[-1]["options"]) == 2


async def test_full_topic_text_is_listed_in_the_message_body():
    sessions = FakeSessions()

    response = await WriterPlugin().handle(make_message(), make_ctx(sessions=sessions))

    for index, topic in enumerate(sessions.saved[-1]["options"]):
        assert f"{index + 1}. {topic}" in response.text


async def test_topic_buttons_stay_short_so_telegram_cannot_elide_them():
    sessions = FakeSessions()

    response = await WriterPlugin().handle(make_message(), make_ctx(sessions=sessions))

    assert [c.label for c in response.choices] == ["Topic 1", "Topic 2", "Something else"]


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
        JUST_WRITE_IT.value,
    }


async def test_custom_topic_button_asks_for_free_text():
    session = Session(plugin="writer", state={"step": "await_topic_choice", "options": ["A", "B"]})

    response = await WriterPlugin().handle(
        make_message(choice="writer:topic:custom"), make_ctx(sessions=FakeSessions(session))
    )

    assert response.text == CUSTOM_TOPIC_PROMPT


async def test_a_question_at_the_topic_menu_is_answered_instead_of_becoming_the_topic():
    session = Session(plugin="writer", state={"step": "await_topic_choice", "options": ["A", "B"]})
    llm = FakeLLM(decisions=[{"action": "question", "message": "It means coordinated disclosure."}])
    sessions = FakeSessions(session)

    response = await WriterPlugin().handle(make_message("explain it"), make_ctx(llm, sessions))

    assert response.text == "It means coordinated disclosure."
    assert [c.label for c in response.choices] == ["Topic 1", "Topic 2", "Something else"]
    assert llm.prompts == []  # no claim was generated for "explain it"


async def test_a_real_topic_typed_at_the_menu_starts_the_interview():
    session = Session(plugin="writer", state={"step": "await_topic_choice", "options": ["A", "B"]})
    llm = FakeLLM(
        answers=["Claim about phishing"],
        decisions=[{"action": "topic", "message": "phishing kits"}],
    )
    sessions = FakeSessions(session)

    response = await WriterPlugin().handle(make_message("phishing kits"), make_ctx(llm, sessions))

    assert response.text == "Claim about phishing"
    assert sessions.saved[-1]["topic"] == "phishing kits"


async def test_topic_decision_failure_falls_back_to_treating_text_as_a_topic():
    session = Session(plugin="writer", state={"step": "await_topic_choice", "options": ["A", "B"]})
    llm = FakeLLM(answers=["A claim"], json_error=LLMError("boom"))

    response = await WriterPlugin().handle(
        make_message("my topic"), make_ctx(llm, FakeSessions(session))
    )

    assert response.text == "A claim"


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


# --- stance ----------------------------------------------------------------


async def test_stance_starts_the_interview_with_the_first_question():
    session = Session(
        plugin="writer",
        state={"step": "await_stance", "topic": "X", "claim": "C", "transcript": []},
    )
    llm = FakeLLM(answers=["Why do you think that?"])
    sessions = FakeSessions(session)

    response = await WriterPlugin().handle(
        make_message(choice="writer:stance:agree"), make_ctx(llm, sessions)
    )

    assert response.text == "Why do you think that?"
    assert response.choices == (JUST_WRITE_IT,)
    saved = sessions.saved[-1]
    assert saved["step"] == "interview"
    assert saved["stance"] == "agree"
    assert saved["questions_asked"] == 1
    assert saved["transcript"] == [{"role": "assistant", "text": "Why do you think that?"}]


async def test_typed_stance_word_also_works():
    session = Session(
        plugin="writer",
        state={"step": "await_stance", "topic": "X", "claim": "C", "transcript": []},
    )
    llm = FakeLLM(answers=["Question?"])

    response = await WriterPlugin().handle(
        make_message("disagree"), make_ctx(llm, FakeSessions(session))
    )

    assert response.text == "Question?"


async def test_unrecognized_stance_reply_asks_to_tap_a_button():
    session = Session(
        plugin="writer",
        state={"step": "await_stance", "topic": "X", "claim": "C", "transcript": []},
    )
    llm = FakeLLM(decisions=[{"action": "nonsense", "message": ""}])

    response = await WriterPlugin().handle(
        make_message("maybe?"), make_ctx(llm, FakeSessions(session))
    )

    assert response.text == TAP_STANCE
    assert response.choices == STANCE_CHOICES


def stance_session():
    return Session(
        plugin="writer",
        state={"step": "await_stance", "topic": "X", "claim": "C", "transcript": []},
    )


async def test_just_write_it_typed_at_the_stance_step_writes_a_general_post():
    llm = FakeLLM(answers=["the draft"], decisions=[{"action": "skip", "message": ""}])
    sessions = FakeSessions(stance_session())

    response = await WriterPlugin().handle(make_message("just write it"), make_ctx(llm, sessions))

    assert NO_OPINION_YET in response.text
    assert sessions.cleared is True


async def test_just_write_it_tapped_at_the_stance_step_writes_a_general_post():
    llm = FakeLLM(answers=["the draft"])
    sessions = FakeSessions(stance_session())

    response = await WriterPlugin().handle(
        make_message(choice=JUST_WRITE_IT.value), make_ctx(llm, sessions)
    )

    assert NO_OPINION_YET in response.text
    assert llm.json_prompts == []  # a tap needs no interpretation


async def test_saying_nothing_never_puts_words_in_the_users_mouth():
    """Regression: an empty interview used to be summarized as 'You think <our own claim>'."""
    llm = FakeLLM(answers=["the draft"])
    sessions = FakeSessions(stance_session())

    response = await WriterPlugin().handle(
        make_message(choice=JUST_WRITE_IT.value), make_ctx(llm, sessions)
    )

    assert "Here's what I heard" not in response.text
    assert "You think" not in response.text
    assert "not invent" in llm.prompts[0] or "not given their opinion" in llm.prompts[0]


async def test_the_writer_never_asks_for_web_grounding():
    """Grounding is metered separately, and drafting the user's own opinion never needs it."""
    llm = FakeLLM(answers=["a claim"])

    await WriterPlugin().handle(make_message("some topic"), make_ctx(llm, FakeSessions()))

    assert llm.searches == [False]


async def test_quota_errors_are_not_swallowed_by_the_writer():
    llm = FakeLLM(error=LLMQuotaError("429"))

    with pytest.raises(LLMQuotaError):
        await WriterPlugin().handle(make_message("a topic"), make_ctx(llm, FakeSessions()))


async def test_rate_limit_errors_are_not_swallowed_by_the_writer():
    llm = FakeLLM(error=LLMRateLimitError("429", retry_after=1.0))

    with pytest.raises(LLMRateLimitError):
        await WriterPlugin().handle(make_message("a topic"), make_ctx(llm, FakeSessions()))


async def test_quota_errors_during_interpretation_are_not_swallowed():
    session = Session(plugin="writer", state={"step": "await_topic_choice", "options": ["A", "B"]})
    llm = FakeLLM(json_error=LLMQuotaError("429"))

    with pytest.raises(LLMQuotaError):
        await WriterPlugin().handle(
            make_message("explain it"), make_ctx(llm, FakeSessions(session))
        )


async def test_a_question_at_the_stance_step_is_answered_and_keeps_the_buttons():
    llm = FakeLLM(decisions=[{"action": "question", "message": "It means the vault leaked."}])
    sessions = FakeSessions(stance_session())

    response = await WriterPlugin().handle(
        make_message("what do you mean by that?"), make_ctx(llm, sessions)
    )

    assert response.text == "It means the vault leaked."
    assert response.choices == STANCE_CHOICES
    assert llm.prompts == []  # no interview was started


async def test_a_stance_in_the_users_own_words_starts_the_interview():
    llm = FakeLLM(
        answers=["What changed your mind?"], decisions=[{"action": "disagree", "message": ""}]
    )
    sessions = FakeSessions(stance_session())

    response = await WriterPlugin().handle(
        make_message("honestly i think that's overblown"), make_ctx(llm, sessions)
    )

    assert response.text == "What changed your mind?"
    assert sessions.saved[-1]["stance"] == "disagree"
    assert sessions.saved[-1]["step"] == "interview"


# --- the interview loop ----------------------------------------------------


async def test_the_interview_keeps_asking_while_the_model_wants_more():
    llm = FakeLLM(decisions=[{"action": "ask", "message": "What broke for you?"}])
    sessions = FakeSessions(interview_session())

    response = await WriterPlugin().handle(make_message("my answer"), make_ctx(llm, sessions))

    assert response.text == "What broke for you?"
    assert response.choices == (JUST_WRITE_IT,)
    saved = sessions.saved[-1]
    assert saved["step"] == "interview"
    assert saved["questions_asked"] == 2
    assert [t["text"] for t in saved["transcript"]] == [
        "First question?",
        "my answer",
        "What broke for you?",
    ]


async def test_the_interview_is_not_capped_at_two_questions():
    llm = FakeLLM(decisions=[{"action": "ask", "message": "Question four?"}])
    sessions = FakeSessions(interview_session(questions_asked=3, turns=5))

    response = await WriterPlugin().handle(make_message("another answer"), make_ctx(llm, sessions))

    assert response.text == "Question four?"
    assert sessions.saved[-1]["questions_asked"] == 4


async def test_a_question_mid_interview_is_explained_without_using_up_a_question():
    llm = FakeLLM(decisions=[{"action": "explain", "message": "It means the vendor gets 90 days."}])
    sessions = FakeSessions(interview_session(questions_asked=1))

    response = await WriterPlugin().handle(
        make_message("what do you mean?"), make_ctx(llm, sessions)
    )

    assert response.text == "It means the vendor gets 90 days."
    saved = sessions.saved[-1]
    assert saved["step"] == "interview"
    assert saved["questions_asked"] == 1
    assert saved["transcript"][-1] == {
        "role": "assistant",
        "text": "It means the vendor gets 90 days.",
    }


async def test_ready_moves_on_to_the_summary():
    llm = FakeLLM(
        answers=["You think X\nBecause Y"], decisions=[{"action": "ready", "message": ""}]
    )
    sessions = FakeSessions(interview_session())

    response = await WriterPlugin().handle(make_message("final answer"), make_ctx(llm, sessions))

    assert "Here's what I heard:" in response.text
    assert {c.value for c in response.choices} == {"writer:summary:confirm", "writer:summary:more"}
    assert sessions.saved[-1]["step"] == "await_summary_choice"


async def test_just_write_it_skips_straight_to_the_summary():
    llm = FakeLLM(answers=["You think X"])
    sessions = FakeSessions(
        interview_session(
            transcript=[
                {"role": "assistant", "text": "First question?"},
                {"role": "user", "text": "my answer"},
            ]
        )
    )

    response = await WriterPlugin().handle(
        make_message(choice=JUST_WRITE_IT.value), make_ctx(llm, sessions)
    )

    assert "Here's what I heard:" in response.text
    assert sessions.saved[-1]["step"] == "await_summary_choice"
    assert llm.json_prompts == []  # no decision call needed


async def test_the_question_cap_forces_the_summary():
    llm = FakeLLM(answers=["You think X"])
    sessions = FakeSessions(interview_session(questions_asked=MAX_QUESTIONS))

    response = await WriterPlugin().handle(make_message("one more"), make_ctx(llm, sessions))

    assert "Here's what I heard:" in response.text
    assert llm.json_prompts == []


async def test_the_turn_cap_forces_the_summary():
    llm = FakeLLM(answers=["You think X"])
    sessions = FakeSessions(interview_session(questions_asked=1, turns=MAX_TURNS))

    response = await WriterPlugin().handle(make_message("one more"), make_ctx(llm, sessions))

    assert "Here's what I heard:" in response.text
    assert llm.json_prompts == []


async def test_a_failed_decision_moves_on_when_an_answer_already_exists():
    llm = FakeLLM(answers=["You think X"], json_error=LLMError("boom"))
    sessions = FakeSessions(interview_session())

    response = await WriterPlugin().handle(make_message("an answer"), make_ctx(llm, sessions))

    assert "Here's what I heard:" in response.text


async def test_a_malformed_decision_is_not_sent_to_the_user():
    llm = FakeLLM(decisions=[{"action": "nonsense", "message": ""}], answers=["You think X"])
    sessions = FakeSessions(interview_session())

    response = await WriterPlugin().handle(make_message("an answer"), make_ctx(llm, sessions))

    assert "nonsense" not in response.text


async def test_empty_answer_is_rejected():
    sessions = FakeSessions(interview_session())

    response = await WriterPlugin().handle(make_message("  "), make_ctx(sessions=sessions))

    assert response.text == NEED_ANSWER
    assert response.choices == (JUST_WRITE_IT,)


# --- summary and draft -----------------------------------------------------


async def test_confirming_the_summary_produces_a_draft_and_clears_the_session():
    session = Session(
        plugin="writer",
        state={"step": "await_summary_choice", "topic": "X", "summary": "• point"},
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
        state={"step": "await_summary_choice", "topic": "X", "summary": "• point"},
    )
    sessions = FakeSessions(session)

    response = await WriterPlugin().handle(
        make_message(choice="writer:summary:more"), make_ctx(sessions=sessions)
    )

    assert response.text == ADDENDUM_PROMPT
    assert sessions.saved[-1]["step"] == "await_addendum"
    assert sessions.cleared is False


def summary_session():
    return Session(
        plugin="writer",
        state={"step": "await_summary_choice", "topic": "X", "summary": "• point"},
    )


async def test_an_uninterpretable_summary_reply_asks_to_tap():
    llm = FakeLLM(decisions=[{"action": "nonsense", "message": ""}])

    response = await WriterPlugin().handle(
        make_message("hmm"), make_ctx(llm, FakeSessions(summary_session()))
    )

    assert response.text == TAP_SUMMARY
    assert response.choices == SUMMARY_CHOICES


async def test_typed_approval_of_the_summary_produces_the_draft():
    llm = FakeLLM(answers=["the draft"], decisions=[{"action": "confirm", "message": ""}])
    sessions = FakeSessions(summary_session())

    response = await WriterPlugin().handle(
        make_message("yes that's right"), make_ctx(llm, sessions)
    )

    assert response.text == "the draft"
    assert sessions.cleared is True


async def test_a_typed_correction_goes_straight_into_the_draft():
    llm = FakeLLM(answers=["the draft"], decisions=[{"action": "more", "message": ""}])
    sessions = FakeSessions(summary_session())

    response = await WriterPlugin().handle(
        make_message("add that it only matters for small teams"), make_ctx(llm, sessions)
    )

    assert response.text == "the draft"
    assert "only matters for small teams" in llm.prompts[0]
    assert sessions.cleared is True


async def test_a_question_about_the_summary_is_answered_and_keeps_the_buttons():
    llm = FakeLLM(decisions=[{"action": "question", "message": "It means your own vault."}])

    response = await WriterPlugin().handle(
        make_message("what do you mean by vault?"), make_ctx(llm, FakeSessions(summary_session()))
    )

    assert response.text == "It means your own vault."
    assert response.choices == SUMMARY_CHOICES


async def test_addendum_text_produces_a_draft_including_the_note():
    session = Session(
        plugin="writer",
        state={"step": "await_addendum", "topic": "X", "summary": "• point"},
    )
    llm = FakeLLM(answers=["final draft"])
    sessions = FakeSessions(session)

    response = await WriterPlugin().handle(
        make_message("make it punchier"), make_ctx(llm, sessions)
    )

    assert response.text == "final draft"
    assert "make it punchier" in llm.prompts[0]
    assert sessions.cleared is True


async def test_the_draft_is_given_the_users_own_words_not_just_the_summary():
    """Drafting from the paraphrase alone is what made earlier posts sound generic."""
    session = Session(
        plugin="writer",
        state={
            "step": "await_summary_choice",
            "topic": "password managers",
            "summary": "• You think offline managers are safer",
            "transcript": [
                {"role": "assistant", "text": "How do you keep passwords safe?"},
                {"role": "user", "text": "offline like keepassxc is a real improvment"},
            ],
        },
    )
    llm = FakeLLM(answers=["the draft"])

    await WriterPlugin().handle(
        make_message(choice="writer:summary:confirm"), make_ctx(llm, FakeSessions(session))
    )

    prompt = llm.prompts[0]
    assert "offline like keepassxc is a real improvment" in prompt
    assert "You think offline managers are safer" in prompt


async def test_the_draft_is_told_to_fix_mistakes_but_not_to_guess():
    session = Session(
        plugin="writer",
        state={"step": "await_summary_choice", "topic": "X", "summary": "• point"},
    )
    llm = FakeLLM(answers=["the draft"])

    await WriterPlugin().handle(
        make_message(choice="writer:summary:confirm"), make_ctx(llm, FakeSessions(session))
    )

    prompt = llm.prompts[0]
    assert "Correct any mistakes" in prompt
    assert "not certain" in prompt
    assert "fix every grammar mistake" in prompt


async def test_the_draft_does_not_get_the_simple_english_rule():
    """Plain English helps the questions, but it flattens the user's voice in a post."""
    session = Session(
        plugin="writer",
        state={"step": "await_summary_choice", "topic": "X", "summary": "• point"},
    )
    llm = FakeLLM(answers=["the draft"])

    await WriterPlugin().handle(
        make_message(choice="writer:summary:confirm"), make_ctx(llm, FakeSessions(session))
    )

    assert SIMPLE_ENGLISH not in llm.prompts[0]


async def test_the_questions_still_use_simple_english():
    llm = FakeLLM(answers=["a claim"])

    await WriterPlugin().handle(make_message("some topic"), make_ctx(llm, FakeSessions()))

    assert SIMPLE_ENGLISH in llm.prompts[0]


async def test_the_two_options_are_asked_to_differ():
    session = Session(
        plugin="writer",
        state={"step": "await_summary_choice", "topic": "X", "summary": "• point"},
    )
    llm = FakeLLM(answers=["the draft"])

    await WriterPlugin().handle(
        make_message(choice="writer:summary:confirm"), make_ctx(llm, FakeSessions(session))
    )

    prompt = llm.prompts[0]
    assert "must make different points" in prompt
    assert "Option 1 is the sharp opinion" in prompt
    assert "Option 2 makes it concrete" in prompt


async def test_the_draft_may_not_invent_personal_experience():
    """Regression: 'they spend 90 days' was turned into 'I've spent the 90-day window'."""
    session = Session(
        plugin="writer",
        state={"step": "await_summary_choice", "topic": "X", "summary": "• point"},
    )
    llm = FakeLLM(answers=["the draft"])

    await WriterPlugin().handle(
        make_message(choice="writer:summary:confirm"), make_ctx(llm, FakeSessions(session))
    )

    prompt = llm.prompts[0]
    assert "Only write it as their personal experience" in prompt
    assert "made-up personal story is a false claim" in prompt


async def test_the_draft_is_told_not_to_parrot_the_users_sentences():
    """Showing their words is for tone, not for copy-paste."""
    session = Session(
        plugin="writer",
        state={
            "step": "await_summary_choice",
            "topic": "X",
            "summary": "• point",
            "transcript": [{"role": "user", "text": "a real problems happen"}],
        },
    )
    llm = FakeLLM(answers=["the draft"])

    await WriterPlugin().handle(
        make_message(choice="writer:summary:confirm"), make_ctx(llm, FakeSessions(session))
    )

    prompt = llm.prompts[0]
    assert "Do NOT reuse their sentences" in prompt
    assert "fix every grammar mistake" in prompt


async def test_draft_strips_stray_markdown_emphasis_since_it_is_copied_verbatim():
    session = Session(
        plugin="writer",
        state={"step": "await_summary_choice", "topic": "X", "summary": "• point"},
    )
    llm = FakeLLM(answers=["Option 1:\nthey *definitely* broke something\n\nHashtags: none"])

    response = await WriterPlugin().handle(
        make_message(choice="writer:summary:confirm"), make_ctx(llm, FakeSessions(session))
    )

    assert "*" not in response.text
    assert "they definitely broke something" in response.text


async def test_summary_uses_a_consistent_bullet_regardless_of_model_style():
    llm = FakeLLM(answers=["* Point one\n* Point two"], decisions=[{"action": "ready"}])
    sessions = FakeSessions(interview_session())

    response = await WriterPlugin().handle(make_message("answer"), make_ctx(llm, sessions))

    assert "• Point one" in response.text
    assert "* Point one" not in response.text


# --- helpers ---------------------------------------------------------------


def test_strip_markdown_emphasis_leaves_plain_text_untouched():
    assert _strip_markdown_emphasis("plain text, no markup") == "plain text, no markup"


def test_normalize_bullets_uses_one_consistent_symbol_regardless_of_input():
    assert _normalize_bullets("* one\n- two\nthree") == "• one\n• two\n• three"


async def test_llm_failure_during_claim_generation_falls_back_gracefully():
    llm = FakeLLM(error=LLMError("boom"))

    response = await WriterPlugin().handle(
        make_message("some topic"), make_ctx(llm, FakeSessions())
    )

    assert "some topic" in response.text
    assert response.choices == STANCE_CHOICES


async def test_unknown_session_step_restarts_the_flow():
    session = Session(plugin="writer", state={"step": "not-a-real-step"})
    sessions = FakeSessions(session)

    response = await WriterPlugin().handle(make_message(), make_ctx(sessions=sessions))

    assert len(response.choices) == 3
    assert sessions.saved[-1]["step"] == "await_topic_choice"
