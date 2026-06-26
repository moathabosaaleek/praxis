import asyncio

from google import genai
from google.genai import types

from core.config import Settings


class PraxisLLM:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gemini_client = genai.Client(api_key=settings.gemini_api_key)

    async def generate_response(self, prompt: str, system_instruction: str = None) -> str:
        """
        Generates a response using Gemini as the primary engine.
        Uses asyncio.to_thread to prevent the synchronous Google SDK
        from blocking the asynchronous messaging interface.
        """
        try:
            print("[ROUTER] Generating response via Gemini...")

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[{"google_search": {}}],
            )

            response = await asyncio.to_thread(
                self.gemini_client.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt,
                config=config,
            )

            return response.text

        except Exception as e:
            print(f"[ROUTER] Gemini failed: {e}")
            return f"Core Failure: Unable to reach Gemini API. Error: {str(e)}"
