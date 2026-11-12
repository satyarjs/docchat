"""Step 4 of querying: send the prompt to an LLM and get the answer back.

Groq, Ollama, OpenRouter, LM Studio and OpenAI all speak the same
"OpenAI-compatible" Chat Completions API. So we use the official `openai`
package and just change `base_url`. Under the hood it is one HTTP request:

  POST {base_url}/chat/completions
  Authorization: Bearer {api_key}
  {"model": "...", "messages": [...], "temperature": 0.2, "max_tokens": 2048}

The answer comes back in response.choices[0].message.content.
"""

from __future__ import annotations

import openai
from openai import OpenAI


class LLMError(Exception):
    """A clear, user-facing error about the LLM connection or configuration."""


class LLMClient:
    """Minimal chat client for any OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.temperature = temperature  # low = focused, repeatable answers
        self.max_tokens = max_tokens  # upper limit on answer length
        # The SDK refuses an empty key; Ollama ignores the key, so any text works.
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def chat(self, messages: list[dict]) -> str:
        """Send messages and return the model's reply text."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except openai.APIConnectionError as exc:
            raise LLMError(
                f"Could not reach the LLM at {self.base_url}. "
                "If you use Ollama, run `ollama serve`. Check LLM_BASE_URL."
            ) from exc
        except openai.AuthenticationError as exc:
            raise LLMError(
                "The LLM rejected your API key. Check LLM_API_KEY in .env."
            ) from exc
        except openai.NotFoundError as exc:
            raise LLMError(
                f"Model '{self.model}' was not found. Check LLM_MODEL against your "
                f"provider's model list (Ollama: `ollama pull {self.model}`)."
            ) from exc
        except openai.RateLimitError as exc:
            raise LLMError("Rate limit hit. Wait a few seconds and try again.") from exc
        except openai.APIError as exc:
            raise LLMError(f"The LLM returned an error: {exc}") from exc

        content = response.choices[0].message.content or ""
        return content.strip()
