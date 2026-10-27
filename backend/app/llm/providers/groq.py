from __future__ import annotations
from typing import Any
from langchain_openai import ChatOpenAI
from app.llm.config import llm_settings
from app.llm.exceptions import LLMConfigurationError


def _validate_groq_key() -> str:
    token = llm_settings.GROQ_API_KEY.strip()
    if not token:
        raise LLMConfigurationError('groq', 'GROQ_API_KEY is empty.')
    if llm_settings._is_placeholder(token):
        raise LLMConfigurationError('groq', 'GROQ_API_KEY is still a placeholder.')
    return token

def get_groq_chat_model(
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> ChatOpenAI:
    config = llm_settings.provider_config('groq')
    return ChatOpenAI(
        model=model or config['model'],
        temperature=temperature,
        max_tokens=kwargs.pop('max_tokens', 1024),
        timeout=kwargs.pop('timeout', llm_settings.LLM_TIMEOUT_SECONDS),
        api_key=_validate_groq_key(),
        base_url=config['base_url'],
        max_retries=kwargs.pop('max_retries', max(0, llm_settings.LLM_MAX_RETRIES)),
        **kwargs,
    )
