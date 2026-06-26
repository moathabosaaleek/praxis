from dotenv import load_dotenv

from core.config import load_settings
from interfaces.telegram.bot import run_telegram_bot


def main():
    load_dotenv()
    settings = load_settings()
    run_telegram_bot(settings)


if __name__ == "__main__":
    main()