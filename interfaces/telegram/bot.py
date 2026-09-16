import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from core.assistant import Assistant
from core.config import Settings
from core.llm_router import PraxisLLM
from core.messages import AssistantResponse, IncomingMessage
from interfaces.telegram.formatting import markdown_to_telegram_html, split_message
from storage.db import Database
from storage.repositories import MessageRepository

logger = logging.getLogger(__name__)

THINKING = "Thinking..."


def to_incoming_message(update: Update) -> IncomingMessage | None:
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return None

    received_at = datetime.now(UTC)

    if update.callback_query is not None:
        choice = update.callback_query.data or ""
        return IncomingMessage(
            user_id=user.id,
            chat_id=message.chat_id,
            text=choice,
            received_at=received_at,
            choice=choice,
        )

    if not message.text:
        return None

    return IncomingMessage(
        user_id=user.id,
        chat_id=message.chat_id,
        text=message.text,
        received_at=received_at,
    )


def build_keyboard(response: AssistantResponse) -> InlineKeyboardMarkup | None:
    if not response.choices:
        return None
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(choice.label, callback_data=choice.value)]
            for choice in response.choices
        ]
    )


async def send_formatted(
    send: Callable[..., Awaitable],
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    try:
        await send(
            markdown_to_telegram_html(text), parse_mode=ParseMode.HTML, reply_markup=reply_markup
        )
    except BadRequest:
        logger.warning("Telegram rejected formatted message; falling back to plain text")
        await send(text, reply_markup=reply_markup)


async def handle_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    incoming = to_incoming_message(update)
    if incoming is None:
        return

    assistant: Assistant = context.bot_data["assistant"]
    if not assistant.is_authorized(incoming.user_id):
        logger.warning("Unauthorized access attempt blocked from User ID: %s", incoming.user_id)
        return

    if update.callback_query is not None:
        await update.callback_query.answer()

    placeholder = await context.bot.send_message(incoming.chat_id, THINKING)
    response = await assistant.handle(incoming)
    if response is None:
        await placeholder.delete()
        return

    async def reply(text: str, **kwargs) -> None:
        await context.bot.send_message(incoming.chat_id, text, **kwargs)

    chunks = split_message(response.text)
    keyboard = build_keyboard(response)

    for index, chunk in enumerate(chunks):
        is_last = index == len(chunks) - 1
        markup = keyboard if is_last else None
        send = placeholder.edit_text if index == 0 else reply
        await send_formatted(send, chunk, markup)


async def on_startup(app: Application) -> None:
    settings: Settings = app.bot_data["settings"]
    database = Database(settings.db_path)
    await database.connect()
    app.bot_data["database"] = database
    app.bot_data["assistant"] = Assistant(
        settings, PraxisLLM(settings), MessageRepository(database)
    )
    logger.info("Storage ready at %s", settings.db_path)


async def on_shutdown(app: Application) -> None:
    database: Database | None = app.bot_data.get("database")
    if database is not None:
        await database.close()
        logger.info("Storage closed")


def build_application(settings: Settings) -> Application:
    app = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .post_init(on_startup)
        .post_shutdown(on_shutdown)
        .build()
    )
    app.bot_data["settings"] = settings

    # UpdateType.MESSAGE excludes edited messages, which would otherwise re-trigger handlers.
    app.add_handler(MessageHandler(filters.UpdateType.MESSAGE & filters.TEXT, handle_update))
    app.add_handler(CallbackQueryHandler(handle_update))
    return app


def run_telegram_bot(settings: Settings):
    app = build_application(settings)
    logger.info(
        "Praxis is listening on Telegram (model=%s, env=%s)", settings.llm_model, settings.env
    )
    # Ignore anything sent while the bot was offline instead of replying to a backlog.
    app.run_polling(drop_pending_updates=True)
