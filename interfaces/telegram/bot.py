import logging
import os
from datetime import datetime
from functools import wraps

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from core.llm_router import PraxisLLM

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def get_admin_id() -> int:
    admin_id = os.getenv("ADMIN_TELEGRAM_ID")
    if not admin_id:
        raise RuntimeError("CRITICAL: ADMIN_TELEGRAM_ID not found in .env file.")

    try:
        return int(admin_id)
    except ValueError as exc:
        raise RuntimeError("CRITICAL: ADMIN_TELEGRAM_ID in .env is not a valid number.") from exc


def require_admin(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        if not update.effective_user:
            return

        user_id = update.effective_user.id
        if user_id != get_admin_id():
            logging.warning("Unauthorized access attempt blocked from User ID: %s", user_id)
            return

        return await func(update, context, *args, **kwargs)

    return wrapper


@require_admin
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Praxis System Online. Secure connection established.")


@require_admin
async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Pong! The core engine is responsive.")


@require_admin
async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = " ".join(context.args)

    if not user_query:
        await update.message.reply_text(
            "Please provide a question. Example: `/ask What is a Git Submodule?`",
            parse_mode="Markdown",
        )
        return

    processing_msg = await update.message.reply_text("Processing...")

    try:
        llm = PraxisLLM()
        now = datetime.now().strftime("%A, %B %d, %Y - %H:%M:%S")

        sys_prompt = (
            "You are Praxis, a highly efficient, technical AI assistant. "
            f"The current system time is {now}. "
            "Always rely on this system time for questions about 'today', 'tomorrow', etc. "
            "Keep responses concise, accurate, and format them cleanly using Markdown. Do not use corporate jargon."
        )

        response = await llm.generate_response(user_query, system_instruction=sys_prompt)
        await processing_msg.edit_text(response, parse_mode="Markdown")

    except Exception as e:
        await processing_msg.edit_text(f"Error: {str(e)}")


def run_telegram_bot():
    get_admin_id()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "paste_your_botfather_token_here":
        print("CRITICAL: Valid TELEGRAM_BOT_TOKEN not found in .env file.")
        exit(1)

    print("Booting Praxis Core...")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("ping", ping_command))
    app.add_handler(CommandHandler("ask", ask_command))

    print("Praxis is now listening for your commands on Telegram...")
    print("(Press Ctrl+C to shut down the server)")

    app.run_polling()