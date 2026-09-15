import logging

from dotenv import load_dotenv

from core.config import load_settings
from interfaces.telegram.bot import run_telegram_bot


def configure_logging():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    # httpx logs full request URLs at INFO, and Telegram API URLs contain the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    configure_logging()
    load_dotenv()
    settings = load_settings()
    run_telegram_bot(settings)


if __name__ == "__main__":
    main()
