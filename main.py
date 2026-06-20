import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from functools import wraps
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from core.llm_router import PraxisLLM

# Load environment variables from .env
load_dotenv()

# Setup basic logging to see errors and activity in your terminal
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Fetch your Telegram ID safely
try:
    ADMIN_ID = int(os.getenv("ADMIN_TELEGRAM_ID", 0))
except ValueError:
    print("CRITICAL: ADMIN_TELEGRAM_ID in .env is not a valid number.")
    exit(1)

# --- SECURITY GATEKEEPER ---
def require_admin(func):
    """
    Middleware decorator: Only allows execution if the user's Telegram ID 
    matches your ADMIN_ID. Otherwise, it drops the message silently.
    """
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        if not update.effective_user:
            return
            
        user_id = update.effective_user.id
        if user_id != ADMIN_ID:
            logging.warning(f"Unauthorized access attempt blocked from User ID: {user_id}")
            return  # Silently drop the request
            
        return await func(update, context, *args, **kwargs)
    return wrapper

# --- COMMAND HANDLERS ---
@require_admin
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /start command"""
    await update.message.reply_text("Praxis System Online. Secure connection established.")

@require_admin
async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /ping command"""
    await update.message.reply_text("Pong! The core engine is responsive.")

@require_admin
async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /ask command using the LLM Router"""
    # Extract the user's question from the command arguments
    user_query = " ".join(context.args)
    
    if not user_query:
        await update.message.reply_text("Please provide a question. Example: `/ask What is a Git Submodule?`", parse_mode="Markdown")
        return
        
    # Send a temporary loading message
    processing_msg = await update.message.reply_text("🧠 Processing...")
    
    try:
        # Initialize our brain
        llm = PraxisLLM()
        
        # --- NEW: Temporal Grounding ---
        # Get the exact current day, date, and time
        now = datetime.now().strftime("%A, %B %d, %Y - %H:%M:%S")
        
        # The base personality for this command, now with a live clock!
        sys_prompt = (
            f"You are Praxis, a highly efficient, technical AI assistant. "
            f"The current system time is {now}. "
            f"Always rely on this system time for questions about 'today', 'tomorrow', etc. "
            f"Keep responses concise, accurate, and format them cleanly using Markdown. Do not use corporate jargon."
        )
        
        # Get the answer
        response = await llm.generate_response(user_query, system_instruction=sys_prompt)
        
        # Edit the temporary message with the final answer
        await processing_msg.edit_text(response, parse_mode='Markdown')
        
    except Exception as e:
        await processing_msg.edit_text(f"⚠️ Error: {str(e)}")

# --- MAIN SYSTEM ENGINE ---
if __name__ == '__main__':
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "paste_your_botfather_token_here":
        print("CRITICAL: Valid TELEGRAM_BOT_TOKEN not found in .env file.")
        exit(1)
        
    print("Booting Praxis Core...")
    
    # Initialize the Telegram Application
    app = ApplicationBuilder().token(token).build()
    
    # Register our commands to the application
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('ping', ping_command))
    app.add_handler(CommandHandler('ask', ask_command))
    
    print("Praxis is now listening for your commands on Telegram...")
    print("(Press Ctrl+C to shut down the server)")
    
    # Start checking Telegram for messages (Polling)
    app.run_polling()