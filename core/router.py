import logging
from collections.abc import Sequence

from core.llm_router import LLMError
from core.messages import IncomingMessage
from core.plugin import Plugin, PluginContext

logger = logging.getLogger(__name__)

ROUTING_PROMPT = (
    "Pick exactly one plugin name from the list below to handle the user's message. "
    "Reply with only the plugin name and nothing else.\n\n{options}\n\nUser message: {text}"
)


class Router:
    def __init__(self, plugins: Sequence[Plugin], default_name: str):
        self._plugins = {plugin.name: plugin for plugin in plugins}
        if default_name not in self._plugins:
            raise ValueError(f"default_name {default_name!r} is not a registered plugin")
        self._default_name = default_name

    @property
    def default(self) -> Plugin:
        return self._plugins[self._default_name]

    async def choose(self, message: IncomingMessage, ctx: PluginContext) -> Plugin:
        if len(self._plugins) == 1:
            return self.default

        options = "\n".join(f"- {p.name}: {p.description}" for p in self._plugins.values())
        prompt = ROUTING_PROMPT.format(options=options, text=message.text)

        try:
            answer = await ctx.llm.generate_response(prompt)
        except LLMError:
            logger.warning("Plugin routing call failed, falling back to default plugin")
            return self.default

        return self._plugins.get(answer.strip().lower(), self.default)
