import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from functools import wraps

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

from core.config import Settings
from core.llm_router import LLMError, PraxisLLM
from interfaces.telegram.formatting import markdown_to_telegram_html, split_message

logger = logging.getLogger(__name__)


def require_admin(settings: Settings):
    def decorator(func):
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            if not update.effective_user:
                return

            user_id = update.effective_user.id
            if user_id != settings.admin_telegram_id:
                logger.warning("Unauthorized access attempt blocked from User ID: %s", user_id)
                return

            return await func(update, context, *args, **kwargs)

        return wrapper

    return decorator


def build_system_prompt(settings: Settings) -> str:
    now = datetime.now(settings.timezone)
    return (
        "You are Praxis, a concise technical assistant for a single user. "
        f"The user's current local time is {now:%A, %B %d, %Y %H:%M} ({settings.timezone.key}). "
        "Use it for any question about today, tomorrow, or other dates. "
        "Keep answers concise and accurate, and avoid corporate jargon. "
        "Format only with **bold**, `inline code`, and fenced code blocks. Do not use tables."
    )


async def send_formatted(send: Callable[..., Awaitable], text: str) -> None:
    try:
        await send(markdown_to_telegram_html(text), parse_mode=ParseMode.HTML)
    except BadRequest:
        logger.warning("Telegram rejected formatted message; falling back to plain text")
        await send(text)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Praxis is online.")


async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Pong! The core engine is responsive.")


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    settings: Settings = context.bot_data["settings"]
    llm: PraxisLLM = context.bot_data["llm"]

    user_query = " ".join(context.args or [])
    if not user_query:
        await message.reply_text(
            "Please provide a question. Example: /ask What is a Git submodule?"
        )
        return

    placeholder = await message.reply_text("Thinking...")

    try:
        answer = await llm.generate_response(
            user_query, system_instruction=build_system_prompt(settings)
        )
    except LLMError:
        await placeholder.edit_text("Sorry, I couldn't get an answer right now. Please try again.")
        return

    chunks = split_message(answer)
    await send_formatted(placeholder.edit_text, chunks[0])
    for chunk in chunks[1:]:
        await send_formatted(message.reply_text, chunk)


def build_application(settings: Settings) -> Application:
    app = ApplicationBuilder().token(settings.telegram_bot_token).build()
    app.bot_data["settings"] = settings
    app.bot_data["llm"] = PraxisLLM(settings)

    admin_only = require_admin(settings)
    app.add_handler(CommandHandler("start", admin_only(start_command)))
    app.add_handler(CommandHandler("ping", admin_only(ping_command)))
    app.add_handler(CommandHandler("ask", admin_only(ask_command)))
    return app


def run_telegram_bot(settings: Settings):
    app = build_application(settings)
    logger.info(
        "Praxis is listening on Telegram (model=%s, env=%s)", settings.llm_model, settings.env
    )
    app.run_polling()
