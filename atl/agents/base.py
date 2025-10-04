"""Base utilities for prompt-driven agents."""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


class SupportsLLM(Protocol):
    """Minimal protocol for language models compatible with LangChain."""

    def invoke(self, input: str) -> Any: ...

    async def ainvoke(self, input: str) -> Any: ...


def _coerce_response_content(response: Any) -> str:
    if response is None:
        return ""
    if hasattr(response, "content"):
        return str(response.content)
    if isinstance(response, (str, bytes)):
        return response.decode() if isinstance(response, bytes) else response
    if isinstance(response, dict):
        return json.dumps(response)
    return str(response)


@dataclass
class PromptConfig:
    """Configuration of a prompt file and default variables."""

    path: Path
    variables: Dict[str, Any]


class BasePromptAgent:
    """Utility for loading markdown prompts and invoking LLMs."""

    def __init__(
        self,
        prompt_path: Path,
        *,
        llm: Optional[SupportsLLM] = None,
        default_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not prompt_path.exists():
            raise FileNotFoundError(prompt_path)
        self.prompt_path = prompt_path
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        self.llm = llm
        self.default_context = default_context or {}

    def render_prompt(self, **context: Any) -> str:
        payload = {**self.default_context, **context}
        try:
            return self.prompt_template.format(**payload)
        except KeyError as exc:  # pragma: no cover - developer mistake
            missing = str(exc).strip("'")
            raise ValueError(f"Missing prompt variables: {missing}") from exc

    def _invoke(self, rendered_prompt: str) -> str:
        if self.llm is None:
            raise RuntimeError("LLM is not configured for this agent")
        response = self.llm.invoke(rendered_prompt)
        return _coerce_response_content(response)

    async def _ainvoke(self, rendered_prompt: str) -> str:
        if self.llm is None:
            raise RuntimeError("LLM is not configured for this agent")
        response = await self.llm.ainvoke(rendered_prompt)
        return _coerce_response_content(response)

    def invoke(self, **context: Any) -> str:
        rendered = self.render_prompt(**context)
        return self._invoke(rendered)

    async def ainvoke(self, **context: Any) -> str:
        rendered = self.render_prompt(**context)
        return await self._ainvoke(rendered)


def load_prompt_config(name: str, **variables: Any) -> PromptConfig:
    """Load a prompt file located in ``atl/prompts``."""

    base_dir = Path(__file__).resolve().parent.parent / "prompts"
    path = base_dir / name
    return PromptConfig(path=path, variables=variables)


def create_ollama_llm(model: str = "gpt-oss:120b-cloud", **kwargs: Any) -> SupportsLLM:
    """Instantiate a ``ChatOllama`` model on demand."""

    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, **kwargs)


__all__ = [
    "SupportsLLM",
    "BasePromptAgent",
    "PromptConfig",
    "load_prompt_config",
    "create_ollama_llm",
]
