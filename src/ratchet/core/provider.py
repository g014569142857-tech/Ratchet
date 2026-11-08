"""Provider abstraction: the single seam that makes "any LLM" true.

Everything else in Ratchet is model-agnostic because it only ever talks to this
protocol. Adding support for a new model backend means implementing one method.

The base package ships two providers that need no network and no API key:

  * ``EchoProvider`` - offline default, so a fresh clone works immediately
  * ``OpenAICompatProvider`` - any OpenAI-compatible endpoint (OpenAI, Azure,
    Ollama, vLLM, LM Studio, DeepSeek, Qwen/DashScope compatible mode...)

Closed-API providers belong behind the same interface. Note that for those,
route A (weight updates) is impossible - only route B (context-level
self-improvement) applies. That is a property of the provider, not of Ratchet.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass


class ProviderError(RuntimeError):
    """Raised when a model backend cannot produce a completion."""


class Provider(ABC):
    """Minimal completion interface every model backend must satisfy."""

    name: str = "provider"

    @abstractmethod
    def complete(self, prompt: str, **kwargs: object) -> str:
        """Return the model's text completion for ``prompt``."""


class EchoProvider(Provider):
    """Offline provider that reflects its prompt back.

    Makes the whole loop runnable with no credentials, which is what lets CI and
    a first-time user both get a working system immediately.
    """

    name = "echo"

    def complete(self, prompt: str, **kwargs: object) -> str:
        return f"[echo] {prompt}"


@dataclass
class OpenAICompatProvider(Provider):
    """Any endpoint speaking the OpenAI chat-completions protocol.

    Uses only the standard library so the core package keeps zero third-party
    runtime dependencies. Pass ``api_key=None`` for local servers such as Ollama
    or LM Studio that do not require authentication.
    """

    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    timeout: float = 60.0
    name: str = "openai_compat"

    def complete(self, prompt: str, **kwargs: object) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
        }
        if kwargs.get("max_tokens") is not None:
            payload["max_tokens"] = kwargs["max_tokens"]

        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ProviderError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise ProviderError(f"could not reach {self.base_url}: {exc.reason}") from exc

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover
            raise ProviderError(f"unexpected response shape: {body!r}") from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
