from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator
from uuid import UUID

logger = logging.getLogger('app.chat_timing')
_run_id: ContextVar[str | None] = ContextVar('chat_timing_run_id', default=None)
_llm_call_count: ContextVar[int] = ContextVar('chat_timing_llm_call_count', default=0)
_llm_invocation_count: ContextVar[int] = ContextVar('chat_timing_llm_invocation_count', default=0)


@contextmanager
def bind_chat_run(run_id: UUID | str) -> Iterator[None]:
    run_token = _run_id.set(str(run_id))
    count_token = _llm_call_count.set(0)
    invocation_token = _llm_invocation_count.set(0)
    try:
        yield
    finally:
        _llm_invocation_count.reset(invocation_token)
        _llm_call_count.reset(count_token)
        _run_id.reset(run_token)


def next_llm_call_number() -> int:
    count = _llm_call_count.get() + 1
    _llm_call_count.set(count)
    return count


def current_llm_call_count() -> int:
    return _llm_call_count.get()

def next_llm_invocation_number() -> int:
    count = _llm_invocation_count.get() + 1
    _llm_invocation_count.set(count)
    return count

def current_llm_invocation_count() -> int:
    return _llm_invocation_count.get()


def is_chat_run_active() -> bool:
    return _run_id.get() is not None


def emit_timing(stage: str, duration_ms: float, **fields: object) -> None:
    record = {
        'event': 'chat_timing',
        'stage': stage,
        'duration_ms': round(duration_ms, 2),
    }
    active_run_id = _run_id.get()
    if active_run_id:
        record['run_id'] = active_run_id
    record.update(fields)
    logger.info('%s', json.dumps(record, sort_keys=True, separators=(',', ':')))


@contextmanager
def timed_stage(stage: str, **fields: object) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        emit_timing(stage, (time.perf_counter() - started) * 1000, **fields)
