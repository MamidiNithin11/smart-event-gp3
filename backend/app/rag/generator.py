from __future__ import annotations
import logging
import math
import re
import time
from dataclasses import dataclass, field
from app.core.ai_config import ai_settings
from app.llm.exceptions import LLMProviderError
from app.llm.router import LLMProviderRouter
from app.rag.search import RetrievedChunk
from app.observability.timing import emit_timing
logger = logging.getLogger(__name__)
NO_ANSWER = 'I could not find that in the event policy documents. Please check with an event administrator.'
_ANSWER_PROMPT = 'You answer questions about event-management policies using ONLY the context below.\n\nRules:\n- Use only what the context states. Do not add general knowledge.\n- If the context does not contain the answer, reply exactly: {no_answer}\n- Cite the source in brackets after each claim, e.g. [Cancellation Policy].\n- Be concise. Two or three sentences unless the question needs a list.\n{extra_rules}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:'
_GROUNDING_PROMPT = 'Check whether the ANSWER is fully supported by the CONTEXT.\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nAn answer is supported only if every factual claim in it appears in the\ncontext. Added details, softened conditions, or invented numbers make it\nunsupported.\n\nReply with exactly one line:\nGROUNDED: yes\nor\nGROUNDED: no | <what is unsupported, in under 15 words>\n'

@dataclass
class GenerationResult:
    answer: str
    grounded: bool
    retry_count: int = 0
    sources: list[str] = field(default_factory=list)
    check_note: str = ''

def build_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return ''
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        summary = chunk.metadata.get('summary', '')
        header = f'[{i}] Source: {chunk.citation()}'
        if summary:
            header += f'\nSummary: {summary}'
        blocks.append(f'{header}\n{chunk.content}')
    return '\n\n---\n\n'.join(blocks)

def generate_answer(question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
    if not chunks:
        return GenerationResult(answer=NO_ANSWER, grounded=True, sources=[])
    started = time.perf_counter()
    context_started = time.perf_counter()
    context = build_context(chunks)
    emit_timing(
        'rag.context_prepare',
        (time.perf_counter() - context_started) * 1000,
        chunk_count=len(chunks),
        context_chars=len(context),
        estimated_context_tokens=math.ceil(len(context) / 4),
        token_estimate='characters_divided_by_4',
    )
    sources = _unique_sources(chunks)
    extra_rules = ''
    retry_count = 0
    answer = ''
    note = ''
    max_attempts = 1 + (ai_settings.MAX_GENERATION_RETRIES if ai_settings.ENABLE_HALLUCINATION_CHECK else 0)
    for attempt in range(max_attempts):
        response = LLMProviderRouter().invoke(
            _ANSWER_PROMPT.format(no_answer=NO_ANSWER, context=context, question=question, extra_rules=extra_rules),
            temperature=0.0,
            operation='rag_answer',
        )
        answer = str(response.content).strip()
        if not answer:
            raise RuntimeError('RAG chat model returned an empty answer.')
        if not ai_settings.ENABLE_HALLUCINATION_CHECK:
            emit_timing('rag.answer_generation', (time.perf_counter() - started) * 1000, grounding_enabled=False)
            return GenerationResult(answer=answer, grounded=True, sources=sources)
        if NO_ANSWER.lower() in answer.lower():
            emit_timing('rag.answer_generation', (time.perf_counter() - started) * 1000, grounding_enabled=True)
            return GenerationResult(answer=NO_ANSWER, grounded=True, retry_count=retry_count, sources=sources)
        grounded, note = check_grounding(answer, context)
        if grounded:
            emit_timing('rag.answer_generation', (time.perf_counter() - started) * 1000, grounding_enabled=True)
            return GenerationResult(answer=answer, grounded=True, retry_count=retry_count, sources=sources, check_note=note)
        logger.warning('Ungrounded answer on attempt %d.', attempt + 1)
        retry_count += 1
        extra_rules = f'- A previous attempt was rejected as unsupported ({note}). Quote the context closely and omit anything it does not state.'
    emit_timing('rag.answer_generation', (time.perf_counter() - started) * 1000, grounding_enabled=True)
    return GenerationResult(answer=NO_ANSWER, grounded=False, retry_count=retry_count, sources=sources, check_note=note)

def check_grounding(answer: str, context: str) -> tuple[bool, str]:
    try:
        response = LLMProviderRouter().invoke(
            _GROUNDING_PROMPT.format(context=context, answer=answer),
            temperature=0.0,
            operation='rag_grounding',
        )
        raw = str(response.content).strip()
    except LLMProviderError:
        logger.warning('Grounding check unavailable; rejecting the unverified answer.')
        return (False, 'grounding checker unavailable')
    match = re.search('GROUNDED\\s*:\\s*(yes|no)', raw, re.IGNORECASE)
    if not match:
        logger.warning('Grounding checker returned an unparseable verdict; rejecting the answer.')
        return (False, 'unparseable grounding verdict')
    if match.group(1).lower() == 'yes':
        return (True, '')
    note = ''
    if '|' in raw:
        note = raw.split('|', 1)[1].strip()
    return (False, note or 'unsupported claim')

def _unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    seen: list[str] = []
    for chunk in chunks:
        label = chunk.citation()
        if label not in seen:
            seen.append(label)
    return seen