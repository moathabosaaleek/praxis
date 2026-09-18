import logging
import random
import re
from typing import Any

from core.llm_router import LLMError, LLMQuotaError, LLMRateLimitError
from core.messages import AssistantResponse, Choice, IncomingMessage
from core.plugin import PluginContext

logger = logging.getLogger(__name__)

# Hardcoded for now; a later milestone replaces this with live feed-driven suggestions.
FIXED_TOPICS = [
    "AI-generated code showing up in security audits with nobody reviewing it line by line",
    "Password managers getting breached and what that means for 'best practices' advice",
    "Coordinated vs full disclosure timelines for zero-days",
    "Junior developers leaning on LLMs to explain security concepts they haven't learned yet",
]

STANCES = {"agree", "disagree", "complicated"}

# The interview runs until the model says it understands, bounded so it can never
# loop forever. MAX_TURNS also counts explanations, which do not consume questions.
MAX_QUESTIONS = 6
MAX_TURNS = 14

_EMPHASIS_PAIR = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*")
_LEADING_BULLET = re.compile(r"^[\s*•\-]+")

TOPIC_PROMPT = "What do you want to write about?"
CUSTOM_TOPIC_PROMPT = "What's the topic? Type it in your own words."
NEED_TOPIC = "I need a topic to work with. What do you want to write about?"
NEED_ANSWER = "Go ahead, type your answer whenever you're ready."
TAP_STANCE = "Tap one of the buttons: Agree, Disagree, or It's complicated."
TAP_SUMMARY = "Tap 'Looks right' or 'Let me add something'."
ADDENDUM_PROMPT = "What should I add or change?"
CLAIM_FALLBACK_TEMPLATE = "Is {topic} actually a big deal?"
FOLLOW_UP_FALLBACK = "Can you say a bit more about that?"
DRAFT_FALLBACK = "I couldn't put a draft together this time. Want to try again?"

JUST_WRITE_IT = Choice("Just write it", "writer:interview:done")
STANCE_CHOICES = (
    Choice("Agree", "writer:stance:agree"),
    Choice("Disagree", "writer:stance:disagree"),
    Choice("It's complicated", "writer:stance:complicated"),
    JUST_WRITE_IT,
)
SUMMARY_CHOICES = (
    Choice("Looks right", "writer:summary:confirm"),
    Choice("Let me add something", "writer:summary:more"),
)
INTERVIEW_SPEC = {"action": ["ask", "explain", "ready"], "message": None}

ANSWER_QUESTION_HINT = (
    'For "question", answer their question in under 60 words as the message. '
    "For every other action, leave the message empty."
)

# The reader is a developer whose first language is not English.
SIMPLE_ENGLISH = (
    "Write in simple, plain English. Use short sentences and everyday words. "
    "Avoid academic or complicated phrasing. Do not repeat the question back to them."
)

NO_OPINION_YET = (
    "You haven't told me your opinion yet, so this is a general post about the topic "
    "rather than your own view.\n\n"
)


def _state(step: str, **extra: Any) -> dict[str, Any]:
    return {"step": step, **extra}


def _strip_markdown_emphasis(text: str) -> str:
    """Remove **bold**/*italic* markers the model adds despite being told not to.

    The draft is meant to be copied verbatim into a tweet, where a literal
    asterisk would show up in the posted text.
    """
    return _EMPHASIS_PAIR.sub(lambda m: m.group(1) or m.group(2), text)


def _normalize_bullets(text: str) -> str:
    """Render each line with one consistent bullet, regardless of what the model used."""
    lines = [_LEADING_BULLET.sub("", line).strip() for line in text.splitlines()]
    return "\n".join(f"• {line}" for line in lines if line)


def _render(transcript: list[dict[str, str]]) -> str:
    speaker = {"assistant": "You asked", "user": "They said"}
    return "\n".join(f"{speaker[t['role']]}: {t['text']}" for t in transcript)


def _answers(transcript: list[dict[str, str]]) -> list[str]:
    return [turn["text"] for turn in transcript if turn["role"] == "user"]


def _topic_choices(count: int) -> tuple[Choice, ...]:
    return tuple(Choice(f"Topic {i + 1}", f"writer:topic:{i}") for i in range(count)) + (
        Choice("Something else", "writer:topic:custom"),
    )


class WriterPlugin:
    name = "writer"
    description = (
        "Starts a guided interview to help draft a technical or cybersecurity opinion post "
        "for X (Twitter), then produces a draft and hashtags. Use this when the user wants to "
        "write, post, tweet, or share an opinion publicly, or asks for post ideas."
    )

    async def handle(self, message: IncomingMessage, ctx: PluginContext) -> AssistantResponse:
        session = await ctx.sessions.get(message.chat_id) if ctx.sessions else None
        state = session.state if session else None
        step = state.get("step") if state else None

        if step is None:
            text = message.text.strip()
            if text:
                return await self._claim_for_topic(text, message, ctx)
            return await self._offer_topics(message, ctx)
        if step == "await_custom_topic":
            return await self._claim_for_topic(message.text.strip(), message, ctx)
        if step == "await_topic_choice":
            return await self._handle_topic_choice(message, ctx, state)
        if step == "await_stance":
            return await self._handle_stance(message, ctx, state)
        if step == "interview":
            return await self._interview(message, ctx, state)
        if step == "await_summary_choice":
            return await self._handle_summary_choice(message, ctx, state)
        if step == "await_addendum":
            return await self._draft(message, ctx, state, addendum=message.text.strip())

        logger.warning(
            "Unknown writer session step %r for chat %s, restarting", step, message.chat_id
        )
        return await self._offer_topics(message, ctx)

    async def _interpret(
        self,
        ctx: PluginContext,
        text: str,
        *,
        situation: str,
        actions: dict[str, str],
    ) -> tuple[str, str]:
        """Work out what typed text means at a step that expects a button tap.

        Returns an empty action when the model fails or answers with something
        unexpected, so every caller can fall back to its own safe default.
        """
        described = "\n".join(f'- "{name}": {desc}' for name, desc in actions.items())
        prompt = (
            f"{situation}\n\n"
            f'The user typed: "{text}"\n\n'
            f"Choose the action that matches what they meant:\n{described}\n\n"
            f"{ANSWER_QUESTION_HINT}\n{SIMPLE_ENGLISH}"
        )

        try:
            decision = await ctx.llm.generate_json(
                prompt, {"action": list(actions), "message": None}
            )
        except (LLMQuotaError, LLMRateLimitError):
            raise
        except LLMError:
            logger.warning("Could not interpret typed text, falling back")
            return "", ""

        action = decision.get("action", "")
        if action not in actions:
            return "", ""
        return action, (decision.get("message") or "").strip()

    async def _offer_topics(
        self, message: IncomingMessage, ctx: PluginContext
    ) -> AssistantResponse:
        options = random.sample(FIXED_TOPICS, k=min(2, len(FIXED_TOPICS)))
        await self._save(ctx, message.chat_id, _state("await_topic_choice", options=options))

        # Telegram renders button labels on a single line and elides the middle when they
        # are too wide, so the full topic goes in the message body and buttons just select.
        listing = "\n\n".join(f"{index + 1}. {topic}" for index, topic in enumerate(options))
        return AssistantResponse(
            f"{TOPIC_PROMPT}\n\n{listing}", choices=_topic_choices(len(options))
        )

    async def _handle_topic_choice(
        self, message: IncomingMessage, ctx: PluginContext, state: dict
    ) -> AssistantResponse:
        choice = message.choice
        if choice is None:
            return await self._free_text_at_topic_menu(message, ctx, state)

        if choice == "writer:topic:custom":
            await self._save(ctx, message.chat_id, _state("await_custom_topic"))
            return AssistantResponse(CUSTOM_TOPIC_PROMPT)

        try:
            index = int(choice.rsplit(":", 1)[-1])
            topic = state["options"][index]
        except (ValueError, IndexError, KeyError, TypeError):
            return await self._offer_topics(message, ctx)

        return await self._claim_for_topic(topic, message, ctx)

    async def _free_text_at_topic_menu(
        self, message: IncomingMessage, ctx: PluginContext, state: dict
    ) -> AssistantResponse:
        text = message.text.strip()
        options = state.get("options", [])
        listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(options))

        action, reply = await self._interpret(
            ctx,
            text,
            situation=f"These topics were offered to the user for a post:\n{listing}",
            actions={
                "topic": "they are naming their own topic to write about instead",
                "question": "they are asking about the offered topics, not choosing one",
            },
        )

        if action == "question":
            return AssistantResponse(
                reply or FOLLOW_UP_FALLBACK, choices=_topic_choices(len(options))
            )
        return await self._claim_for_topic(text, message, ctx)

    async def _claim_for_topic(
        self, topic: str, message: IncomingMessage, ctx: PluginContext
    ) -> AssistantResponse:
        if not topic:
            return AssistantResponse(NEED_TOPIC)

        prompt = (
            "Write one short, specific, mildly debatable claim (max 25 words, no hashtags, "
            f"no quotation marks) about this topic that a technically-minded reader could agree "
            f"or disagree with: {topic}\n\nOutput only the claim.\n{SIMPLE_ENGLISH}"
        )
        claim = await self._ask(ctx, prompt, CLAIM_FALLBACK_TEMPLATE.format(topic=topic))

        await self._save(
            ctx, message.chat_id, _state("await_stance", topic=topic, claim=claim, transcript=[])
        )
        return AssistantResponse(claim, choices=STANCE_CHOICES)

    async def _handle_stance(
        self, message: IncomingMessage, ctx: PluginContext, state: dict
    ) -> AssistantResponse:
        if message.choice == JUST_WRITE_IT.value:
            return await self._to_summary(message, ctx, {**state, "stance": "unstated"})

        if message.choice:
            stance = message.choice.rsplit(":", 1)[-1]
            if stance in STANCES:
                return await self._start_interview(stance, message, ctx, state)
            return AssistantResponse(TAP_STANCE, choices=STANCE_CHOICES)

        text = message.text.strip()
        if text.lower() in STANCES:
            return await self._start_interview(text.lower(), message, ctx, state)

        action, reply = await self._interpret(
            ctx,
            text,
            situation=(
                f"The user was shown this claim about {state.get('topic')} and asked whether "
                f"they agree with it:\n{state.get('claim')}"
            ),
            actions={
                "agree": "they agree with the claim",
                "disagree": "they disagree with the claim",
                "complicated": "they partly agree, or it depends",
                "question": "they are asking a question instead of giving a view",
                "skip": "they want to stop being asked and just get the post written",
            },
        )

        if action in STANCES:
            return await self._start_interview(action, message, ctx, state)
        if action == "question":
            return AssistantResponse(reply or FOLLOW_UP_FALLBACK, choices=STANCE_CHOICES)
        if action == "skip":
            return await self._to_summary(message, ctx, {**state, "stance": "unstated"})
        return AssistantResponse(TAP_STANCE, choices=STANCE_CHOICES)

    async def _start_interview(
        self, stance: str, message: IncomingMessage, ctx: PluginContext, state: dict
    ) -> AssistantResponse:
        prompt = (
            f"The user's stance on this claim ('{state.get('claim')}') about "
            f"{state.get('topic')} is: {stance}. Ask ONE short, specific question (under 20 "
            "words) to learn their real opinion or a concrete experience. Do not repeat the "
            f"claim. Output only the question.\n{SIMPLE_ENGLISH}"
        )
        question = await self._ask(ctx, prompt, FOLLOW_UP_FALLBACK)

        state = {
            **state,
            "stance": stance,
            "step": "interview",
            "transcript": [{"role": "assistant", "text": question}],
            "questions_asked": 1,
            "turns": 1,
        }
        await self._save(ctx, message.chat_id, state)
        return AssistantResponse(question, choices=(JUST_WRITE_IT,))

    async def _interview(
        self, message: IncomingMessage, ctx: PluginContext, state: dict
    ) -> AssistantResponse:
        if message.choice == JUST_WRITE_IT.value:
            return await self._to_summary(message, ctx, state)

        text = message.text.strip()
        if not text:
            return AssistantResponse(NEED_ANSWER, choices=(JUST_WRITE_IT,))

        transcript = [*state.get("transcript", []), {"role": "user", "text": text}]
        turns = state.get("turns", 0) + 1
        questions_asked = state.get("questions_asked", 0)
        state = {**state, "transcript": transcript, "turns": turns}

        if turns >= MAX_TURNS or questions_asked >= MAX_QUESTIONS:
            return await self._to_summary(message, ctx, state)

        action, reply = await self._decide(ctx, state)

        if action == "ready":
            return await self._to_summary(message, ctx, state)

        state = {
            **state,
            "transcript": [*transcript, {"role": "assistant", "text": reply}],
            "questions_asked": questions_asked + (1 if action == "ask" else 0),
        }
        await self._save(ctx, message.chat_id, state)
        return AssistantResponse(reply, choices=(JUST_WRITE_IT,))

    async def _decide(self, ctx: PluginContext, state: dict) -> tuple[str, str]:
        prompt = (
            "You are interviewing someone to understand their genuine opinion, so you can "
            "draft a short post for X (Twitter) that sounds like them.\n\n"
            f"Topic: {state.get('topic')}\n"
            f"Claim they reacted to: {state.get('claim')}\n"
            f"Their stance: {state.get('stance')}\n\n"
            f"Conversation so far:\n{_render(state.get('transcript', []))}\n\n"
            "Choose the next action:\n"
            '- "ask": you need more of their opinion. Give ONE short, specific question under '
            "20 words. Never repeat a question you already asked.\n"
            '- "explain": their last message was a question or a request for clarification '
            "rather than an opinion. Answer it in under 60 words, in plain language.\n"
            '- "ready": you understand their opinion well enough to write the post.\n\n'
            f"You have asked {state.get('questions_asked', 0)} of at most {MAX_QUESTIONS} "
            'questions. Choose "ready" once you have a clear stance plus at least one '
            "concrete reason or example. If they say they don't know, ask to stop, or seem "
            'done, choose "ready" rather than pressing them. If they contradict something '
            "they said earlier, ask about that contradiction instead of ignoring it.\n"
            f"{SIMPLE_ENGLISH}"
        )

        try:
            decision = await ctx.llm.generate_json(prompt, INTERVIEW_SPEC)
        except LLMError:
            logger.warning("Interview decision failed, falling back")
            return self._fallback_decision(state)

        action = decision.get("action")
        reply = (decision.get("message") or "").strip()
        if action not in {"ask", "explain", "ready"} or (action != "ready" and not reply):
            return self._fallback_decision(state)
        return action, reply

    def _fallback_decision(self, state: dict) -> tuple[str, str]:
        """Never strand the user: move on if we have something, otherwise ask once more."""
        if _answers(state.get("transcript", [])):
            return "ready", ""
        return "ask", FOLLOW_UP_FALLBACK

    async def _to_summary(
        self, message: IncomingMessage, ctx: PluginContext, state: dict
    ) -> AssistantResponse:
        # With nothing from the user, a summary could only parrot our own claim back
        # as if it were their opinion, so write a general post and say so instead.
        if not _answers(state.get("transcript", [])):
            return await self._draft(message, ctx, state, general=True)

        summary = await self._summarize(ctx, state)
        state = {**state, "step": "await_summary_choice", "summary": summary}
        await self._save(ctx, message.chat_id, state)
        return AssistantResponse(f"Here's what I heard:\n\n{summary}", choices=SUMMARY_CHOICES)

    async def _handle_summary_choice(
        self, message: IncomingMessage, ctx: PluginContext, state: dict
    ) -> AssistantResponse:
        if message.choice:
            action = message.choice.rsplit(":", 1)[-1]
            if action == "more":
                await self._save(ctx, message.chat_id, {**state, "step": "await_addendum"})
                return AssistantResponse(ADDENDUM_PROMPT)
            if action == "confirm":
                return await self._draft(message, ctx, state)
            return AssistantResponse(TAP_SUMMARY, choices=SUMMARY_CHOICES)

        action, reply = await self._interpret(
            ctx,
            message.text.strip(),
            situation=(
                "The user was shown this summary of their own opinion and asked to confirm "
                f"it:\n{state.get('summary')}"
            ),
            actions={
                "confirm": "the summary is right, go ahead and write the post",
                "more": "they want to add or change something in the summary",
                "question": "they are asking a question about the summary",
            },
        )

        if action == "confirm":
            return await self._draft(message, ctx, state)
        if action == "more":
            # They already said what to change, so use it instead of asking again.
            return await self._draft(message, ctx, state, addendum=message.text.strip())
        if action == "question":
            return AssistantResponse(reply or FOLLOW_UP_FALLBACK, choices=SUMMARY_CHOICES)
        return AssistantResponse(TAP_SUMMARY, choices=SUMMARY_CHOICES)

    async def _draft(
        self,
        message: IncomingMessage,
        ctx: PluginContext,
        state: dict,
        addendum: str | None = None,
        general: bool = False,
    ) -> AssistantResponse:
        summary = state.get("summary", "")
        extra = f"\nAdditional note from the user: {addendum}" if addendum else ""
        if general:
            view = (
                "The user has not given their opinion, so write a general post about the "
                "topic. Do not invent or claim a personal opinion for them."
            )
        else:
            view = f"Summary of the user's view:\n{summary}{extra}"

        prompt = (
            "Write two alternative short posts for X (Twitter), each under 280 characters, "
            "in a direct, first-person, non-corporate tone. Write them "
            "as plain text with no markdown formatting at all (no asterisks, no bold, no "
            "italics) since this will be copied directly into a tweet:\n\n"
            f"Topic: {state.get('topic')}\n{view}\n\n"
            "Only suggest hashtags if they would genuinely help (many technical audiences see "
            "hashtags as spam, so it is fine to suggest none). Format exactly as:\n\n"
            "Option 1:\n<text>\n\nOption 2:\n<text>\n\nHashtags: <comma-separated or 'none'>\n\n"
            f"{SIMPLE_ENGLISH}"
        )
        draft = await self._ask(ctx, prompt, DRAFT_FALLBACK)
        draft = _strip_markdown_emphasis(draft)

        if ctx.sessions:
            await ctx.sessions.clear(message.chat_id)
        return AssistantResponse(f"{NO_OPINION_YET}{draft}" if general else draft)

    async def _summarize(self, ctx: PluginContext, state: dict) -> str:
        transcript = state.get("transcript", [])
        prompt = (
            f"Summarize the user's opinion on {state.get('topic')} as 2-4 short points "
            'in second person ("You think..."), based on:\n'
            f"- Stance: {state.get('stance')}\n- Claim reacted to: {state.get('claim')}\n"
            f"- Conversation:\n{_render(transcript)}\n\nOutput only the points, one per line, "
            "with no bullet symbols or numbering. Only include opinions they actually stated. "
            "Never turn a question they asked into an opinion they hold. If they said very "
            f"little, say only that.\n{SIMPLE_ENGLISH}"
        )
        fallback = "\n".join(_answers(transcript)) or state.get("claim", "")
        summary = await self._ask(ctx, prompt, fallback)
        return _normalize_bullets(summary)

    async def _ask(self, ctx: PluginContext, prompt: str, fallback: str) -> str:
        try:
            return await ctx.llm.generate_response(prompt)
        except (LLMQuotaError, LLMRateLimitError):
            raise
        except LLMError:
            logger.warning("Writer plugin LLM call failed, using fallback text")
            return fallback

    async def _save(self, ctx: PluginContext, chat_id: int, state: dict) -> None:
        if ctx.sessions:
            await ctx.sessions.set(chat_id, self.name, state)
