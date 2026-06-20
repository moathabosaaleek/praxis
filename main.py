from dotenv import load_dotenv
from interfaces.telegram.bot import run_telegram_bot


def main():
    load_dotenv()
    run_telegram_bot()


if __name__ == "__main__":
    main()