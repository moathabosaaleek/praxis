import logging
import random
import re
from typing import Any

from core.llm_router import LLMError
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

FOLLOW_UP_COUNT = 2
STANCES = {"agree", "disagree", "complicated"}

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
        if step == "await_answer_1":
            return await self._handle_answer(message, ctx, state, index=1)
        if step == "await_answer_2":
            return await self._handle_answer(message, ctx, state, index=2)
        if step == "await_summary_choice":
            return await self._handle_summary_choice(message, ctx, state)
        if step == "await_addendum":
            return await self._draft(message, ctx, state, addendum=message.text.strip())

        logger.warning(
            "Unknown writer session step %r for chat %s, restarting", step, message.chat_id
        )
        return await self._offer_topics(message, ctx)

    async def _offer_topics(
        self, message: IncomingMessage, ctx: PluginContext
    ) -> AssistantResponse:
        options = random.sample(FIXED_TOPICS, k=min(2, len(FIXED_TOPICS)))
        await self._save(ctx, message.chat_id, _state("await_topic_choice", options=options))

        # Telegram renders button labels on a single line and elides the middle when they
        # are too wide, so the full topic goes in the message body and buttons just select.
        listing = "\n\n".join(f"{index + 1}. {topic}" for index, topic in enumerate(options))
        choices = tuple(
            Choice(f"Topic {index + 1}", f"writer:topic:{index}") for index in range(len(options))
        ) + (Choice("Something else", "writer:topic:custom"),)
        return AssistantResponse(f"{TOPIC_PROMPT}\n\n{listing}", choices=choices)

    async def _handle_topic_choice(
        self, message: IncomingMessage, ctx: PluginContext, state: dict
    ) -> AssistantResponse:
        choice = message.choice
        if choice is None:
            # The user typed instead of tapping a button; treat their text as the topic.
            return await self._claim_for_topic(message.text.strip(), message, ctx)

        if choice == "writer:topic:custom":
            await self._save(ctx, message.chat_id, _state("await_custom_topic"))
            return AssistantResponse(CUSTOM_TOPIC_PROMPT)

        try:
            index = int(choice.rsplit(":", 1)[-1])
            topic = state["options"][index]
        except (ValueError, IndexError, KeyError, TypeError):
            return await self._offer_topics(message, ctx)

        return await self._claim_for_topic(topic, message, ctx)

    async def _claim_for_topic(
        self, topic: str, message: IncomingMessage, ctx: PluginContext
    ) -> AssistantResponse:
        if not topic:
            return AssistantResponse(NEED_TOPIC)

        prompt = (
            "Write one short, specific, mildly debatable claim (max 25 words, no hashtags, "
            f"no quotation marks) about this topic that a technically-minded reader could agree "
            f"or disagree with: {topic}\n\nOutput only the claim."
        )
        claim = await self._ask(ctx, prompt, CLAIM_FALLBACK_TEMPLATE.format(topic=topic))

        await self._save(
            ctx, message.chat_id, _state("await_stance", topic=topic, claim=claim, answers=[])
        )
        choices = (
            Choice("Agree", "writer:stance:agree"),
            Choice("Disagree", "writer:stance:disagree"),
            Choice("It's complicated", "writer:stance:complicated"),
        )
        return AssistantResponse(claim, choices=choices)

    async def _handle_stance(
        self, message: IncomingMessage, ctx: PluginContext, state: dict
    ) -> AssistantResponse:
        raw = message.choice or message.text.strip().lower()
        stance = raw.rsplit(":", 1)[-1] if ":" in raw else raw
        if stance not in STANCES:
            return AssistantResponse(TAP_STANCE)

        state = {**state, "stance": stance, "step": "await_answer_1"}
        question = await self._follow_up_question(ctx, state, first=True)
        await self._save(ctx, message.chat_id, state)
        return AssistantResponse(question)

    async def _handle_answer(
        self, message: IncomingMessage, ctx: PluginContext, state: dict, index: int
    ) -> AssistantResponse:
        answer = message.text.strip()
        if not answer:
            return AssistantResponse(NEED_ANSWER)

        answers = [*state.get("answers", []), answer]

        if index < FOLLOW_UP_COUNT:
            state = {**state, "answers": answers, "step": f"await_answer_{index + 1}"}
            question = await self._follow_up_question(ctx, state, first=False)
            await self._save(ctx, message.chat_id, state)
            return AssistantResponse(question)

        state = {**state, "answers": answers, "step": "await_summary_choice"}
        summary = await self._summarize(ctx, state)
        state = {**state, "summary": summary}
        await self._save(ctx, message.chat_id, state)

        choices = (
            Choice("Looks right", "writer:summary:confirm"),
            Choice("Let me add something", "writer:summary:more"),
        )
        return AssistantResponse(f"Here's what I heard:\n\n{summary}", choices=choices)

    async def _handle_summary_choice(
        self, message: IncomingMessage, ctx: PluginContext, state: dict
    ) -> AssistantResponse:
        choice = message.choice
        if choice is None:
            return AssistantResponse(TAP_SUMMARY)

        action = choice.rsplit(":", 1)[-1]
        if action == "more":
            await self._save(ctx, message.chat_id, {**state, "step": "await_addendum"})
            return AssistantResponse(ADDENDUM_PROMPT)
        if action == "confirm":
            return await self._draft(message, ctx, state)
        return AssistantResponse(TAP_SUMMARY)

    async def _draft(
        self,
        message: IncomingMessage,
        ctx: PluginContext,
        state: dict,
        addendum: str | None = None,
    ) -> AssistantResponse:
        summary = state.get("summary", "")
        extra = f"\nAdditional note from the user: {addendum}" if addendum else ""
        prompt = (
            "Write two alternative short posts for X (Twitter), each under 280 characters, "
            "in a direct, first-person, non-corporate tone, expressing this opinion. Write them "
            "as plain text with no markdown formatting at all (no asterisks, no bold, no "
            "italics) since this will be copied directly into a tweet:\n\n"
            f"Topic: {state.get('topic')}\nSummary of the user's view:\n{summary}{extra}\n\n"
            "Only suggest hashtags if they would genuinely help (many technical audiences see "
            "hashtags as spam, so it is fine to suggest none). Format exactly as:\n\n"
            "Option 1:\n<text>\n\nOption 2:\n<text>\n\nHashtags: <comma-separated or 'none'>"
        )
        draft = await self._ask(ctx, prompt, DRAFT_FALLBACK)
        draft = _strip_markdown_emphasis(draft)

        if ctx.sessions:
            await ctx.sessions.clear(message.chat_id)
        return AssistantResponse(draft)

    async def _follow_up_question(self, ctx: PluginContext, state: dict, first: bool) -> str:
        topic, claim, stance = state.get("topic"), state.get("claim"), state.get("stance")
        if first:
            prompt = (
                f"The user's stance on this claim ('{claim}') about {topic} is: {stance}. "
                "Ask ONE short, specific follow-up question (under 20 words) to learn their real "
                "opinion or a concrete experience related to this. Do not repeat the claim. "
                "Output only the question."
            )
        else:
            prior = state.get("answers", [""])[-1]
            prompt = (
                f'About {topic}, the user previously said: "{prior}". Ask ONE more short, '
                "different follow-up question (under 20 words) that digs deeper or asks for a "
                "concrete example. Output only the question."
            )
        return await self._ask(ctx, prompt, FOLLOW_UP_FALLBACK)

    async def _summarize(self, ctx: PluginContext, state: dict) -> str:
        prompt = (
            f"Summarize the user's opinion on {state.get('topic')} as 2-4 short points "
            'in second person ("You think..."), based on:\n'
            f"- Stance: {state.get('stance')}\n- Claim reacted to: {state.get('claim')}\n"
            f"- Answers: {state.get('answers')}\n\nOutput only the points, one per line, with "
            "no bullet symbols or numbering."
        )
        fallback = "\n".join(state.get("answers", []))
        summary = await self._ask(ctx, prompt, fallback)
        return _normalize_bullets(summary)

    async def _ask(self, ctx: PluginContext, prompt: str, fallback: str) -> str:
        try:
            return await ctx.llm.generate_response(prompt)
        except LLMError:
            logger.warning("Writer plugin LLM call failed, using fallback text")
            return fallback

    async def _save(self, ctx: PluginContext, chat_id: int, state: dict) -> None:
        if ctx.sessions:
            await ctx.sessions.set(chat_id, self.name, state)
