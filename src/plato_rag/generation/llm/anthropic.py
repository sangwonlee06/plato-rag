"""Anthropic Claude LLM implementation."""

from __future__ import annotations

from anthropic import AsyncAnthropic

from plato_rag.protocols.generation import LLMMessage


class AnthropicLLM:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048,
    ) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def generate(self, messages: list[LLMMessage]) -> str:
        # Separate system message from conversation
        system_msg = ""
        conversation = []
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                conversation.append({"role": msg.role, "content": msg.content})

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_msg,
            messages=conversation,
        )
        return response.content[0].text

    def model_name(self) -> str:
        return self._model
