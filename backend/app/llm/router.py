from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from typing import Any
import httpx
from openai import APIError
from app.llm.config import llm_settings
from app.llm.exceptions import LLMUnavailableError, LLMProviderError, normalize_provider_exception, safe_error_category
from app.llm.providers.groq import get_groq_chat_model
from app.llm.providers.openrouter import get_openrouter_chat_model
from app.observability.timing import emit_timing, next_llm_call_number, next_llm_invocation_number

logger = logging.getLogger(__name__)
_PROVIDER_ORDER = ('groq', 'openrouter')


class LLMProviderRouter:
    def __init__(self, *, settings: Any | None = None) -> None:
        self.settings = settings or llm_settings
        self.provider_order = tuple(
            provider for provider in _PROVIDER_ORDER
            if self.settings.is_configured(provider)
        )

    def configured_providers(self) -> list[str]:
        return [provider for provider in self.provider_order if self.settings.is_configured(provider)]

    def get_model(self, provider: str, *, model: str | None = None, temperature: float = 0.0, tools: Any | None = None, **kwargs: Any):
        provider_name = provider.lower()
        if provider_name == 'groq':
            built_model = get_groq_chat_model(
                model=model or self.settings.provider_config(provider)['model'],
                temperature=temperature,
                **kwargs,
            )
        elif provider_name == 'openrouter':
            built_model = get_openrouter_chat_model(
                model=model or self.settings.provider_config(provider)['model'],
                temperature=temperature,
                **kwargs,
            )
        else:
            raise ValueError(f'Unsupported provider: {provider}')
        if tools is not None and hasattr(built_model, 'bind_tools'):
            built_model = built_model.bind_tools(tools)
        return built_model

    def invoke(
        self,
        messages: Any,
        *,
        model_overrides: Mapping[str, str] | None = None,
        temperature: float = 0.0,
        tools: Any | None = None,
        operation: str = 'chat',
        **kwargs: Any,
    ):
        invocation_number = next_llm_invocation_number()
        router_started = time.perf_counter()
        attempt_count = 0
        last_error: Exception | None = None
        request_timeout = kwargs.pop('timeout', self.settings.LLM_TIMEOUT_SECONDS)
        total_budget_seconds = self.settings.LLM_TOTAL_TIMEOUT_SECONDS
        deadline = time.monotonic() + total_budget_seconds
        message_chars = sum(
            len(str(getattr(message, 'content', message)))
            for message in messages
        ) if isinstance(messages, (list, tuple)) else len(str(messages))
        tool_chars = sum(
            len(str(getattr(tool, 'name', '')))
            + len(str(getattr(tool, 'description', '') or ''))
            + len(str(getattr(tool, 'args', '')))
            for tool in (tools or [])
        )
        emit_timing(
            'llm.prompt_profile',
            0,
            operation=operation,
            invocation_number=invocation_number,
            message_count=len(messages) if isinstance(messages, (list, tuple)) else 1,
            message_chars=message_chars,
            tool_count=len(tools or []),
            tool_schema_chars=tool_chars,
            estimated_input_tokens=math.ceil((message_chars + tool_chars) / 4),
            token_estimate='characters_divided_by_4',
        )
        candidate_count = 0
        for provider_index, provider in enumerate(self.provider_order):
            if not self.settings.is_configured(provider):
                continue
            primary_model = (model_overrides or {}).get(provider) or self.settings.provider_config(provider)['model']
            fallback_models = self.settings.openrouter_fallback_models if provider == 'openrouter' else ()
            models = (primary_model, *fallback_models)
            candidate_count += len(models)
            for index, model_name in enumerate(models):
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    emit_timing(
                        'llm.fallback_skipped_deadline',
                        0,
                        provider=provider,
                        model=model_name,
                        attempt=index + 1,
                    )
                    break
                call_number = next_llm_call_number()
                attempt_count += 1
                attempt_timeout = min(request_timeout, remaining_seconds)
                try:
                    init_started = time.perf_counter()
                    provider_model = self.get_model(
                        provider,
                        model=model_name,
                        temperature=temperature,
                        tools=tools,
                        timeout=attempt_timeout,
                        **kwargs,
                    )
                    emit_timing(
                        'llm.client_init',
                        (time.perf_counter() - init_started) * 1000,
                        provider=provider,
                        model=model_name,
                        call_number=call_number,
                        attempt=index + 1,
                        operation=operation,
                        invocation_number=invocation_number,
                        timeout_seconds=attempt_timeout,
                    )
                    invoke_started = time.perf_counter()
                    try:
                        response = provider_model.invoke(messages)
                    except (LLMProviderError, APIError, httpx.HTTPError, TimeoutError, ConnectionError) as exc:
                        emit_timing(
                            'llm.response',
                            (time.perf_counter() - invoke_started) * 1000,
                            provider=provider,
                            model=model_name,
                            call_number=call_number,
                            attempt=index + 1,
                            operation=operation,
                            invocation_number=invocation_number,
                            timeout_seconds=attempt_timeout,
                            outcome='error',
                            error_category=safe_error_category(
                                normalize_provider_exception(exc, provider=provider)
                            ),
                        )
                        raise
                    emit_timing(
                        'llm.response',
                        (time.perf_counter() - invoke_started) * 1000,
                        provider=provider,
                        model=model_name,
                        call_number=call_number,
                        attempt=index + 1,
                        operation=operation,
                        invocation_number=invocation_number,
                        timeout_seconds=attempt_timeout,
                        outcome='success',
                    )
                    emit_timing(
                        'llm.router_invocation',
                        (time.perf_counter() - router_started) * 1000,
                        outcome='success',
                        operation=operation,
                        invocation_number=invocation_number,
                        attempt_count=attempt_count,
                        candidate_count=candidate_count,
                        total_budget_seconds=total_budget_seconds,
                    )
                    return response
                except (LLMProviderError, APIError, httpx.HTTPError, TimeoutError, ConnectionError) as exc:
                    normalized = normalize_provider_exception(exc, provider=provider)
                    last_error = normalized
                    category = safe_error_category(normalized)
                    logger.warning(
                        'LLM provider %s model %s failed (status=%s, category=%s)',
                        provider,
                        model_name,
                        normalized.status_code if normalized.status_code is not None else 'unavailable',
                        category,
                    )
                    has_fallback = index + 1 < len(models)
                    model_specific_bad_request = normalized.status_code == 400 and has_fallback
                    if category == 'authentication_failed':
                        if provider_index + 1 < len(self.provider_order):
                            next_provider = self.provider_order[provider_index + 1]
                            logger.info(
                                'Falling back from LLM provider %s to %s after authentication failure',
                                provider,
                                next_provider,
                            )
                        break
                    if not normalized.retryable and not model_specific_bad_request:
                        if provider_index + 1 < len(self.provider_order):
                            next_provider = self.provider_order[provider_index + 1]
                            logger.info(
                                'Falling back from LLM provider %s to %s after provider error (category=%s)',
                                provider,
                                next_provider,
                                category,
                            )
                            break
                        raise LLMProviderError(
                            provider=provider,
                            message='Provider request failed.',
                            status_code=normalized.status_code,
                            retryable=False,
                        ) from None

        if last_error is not None:
            emit_timing(
                'llm.router_invocation',
                (time.perf_counter() - router_started) * 1000,
                outcome='failed',
                operation=operation,
                invocation_number=invocation_number,
                attempt_count=attempt_count,
                candidate_count=candidate_count,
                total_budget_seconds=total_budget_seconds,
            )
            raise LLMUnavailableError(
                provider='multi-provider',
                message='All configured LLM providers failed. The service is unavailable right now.',
                retryable=True,
                cause=last_error,
            )

        emit_timing(
            'llm.router_invocation',
            (time.perf_counter() - router_started) * 1000,
            outcome='not_configured',
            operation=operation,
            invocation_number=invocation_number,
            attempt_count=attempt_count,
            candidate_count=candidate_count,
            total_budget_seconds=total_budget_seconds,
        )
        raise LLMUnavailableError('multi-provider', 'No configured LLM providers are available for this request.')

    route = invoke
    __call__ = invoke
