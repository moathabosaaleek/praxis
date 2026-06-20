import os
import asyncio
from google import genai
from google.genai import types

class PraxisLLM:
    def __init__(self):
        # Initialize the Gemini client using the key from .env
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.dev_mode = os.getenv("ENV") == "development"

    async def generate_response(self, prompt: str, system_instruction: str = None) -> str:
        """
        Generates a response using Gemini as the primary engine.
        Uses asyncio.to_thread to prevent the synchronous Google SDK 
        from blocking your asynchronous Telegram bot.
        """
        try:
            print("[ROUTER] Generating response via Gemini...")
            
            # Configure the system instructions and ENABLE GOOGLE SEARCH
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[{"google_search": {}}],
            )
            
            # Run the synchronous API call in a background thread
            response = await asyncio.to_thread(
                self.gemini_client.models.generate_content,
                model='gemini-2.5-flash',
                contents=prompt,
                config=config
            )
            
            return response.text
            
        except Exception as e:
            print(f"[ROUTER] Gemini failed: {e}")
            return f"⚠️ Core Failure: Unable to reach Gemini API. Error: {str(e)}"