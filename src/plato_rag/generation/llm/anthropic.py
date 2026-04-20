"""Anthropic Claude LLM implementation."""

from __future__ import annotations

from typing import cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, TextBlock

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
        # Separate system message from conversation turns
        system_msg = ""
        conversation: list[MessageParam] = []
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                # role is "user" or "assistant" by protocol contract
                conversation.append(cast(MessageParam, {"role": msg.role, "content": msg.content}))

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_msg,
            messages=conversation,
        )
        block = response.content[0]
        if not isinstance(block, TextBlock):
            raise TypeError(f"Expected TextBlock in response, got {type(block).__name__}")
        return block.text

    def model_name(self) -> str:
        return self._model
