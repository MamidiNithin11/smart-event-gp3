import logging
import time
from dataclasses import dataclass, field
from sqlalchemy.orm import Session
from app.core.ai_config import ai_settings
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.generator import GenerationResult, generate_answer
from app.rag.reranker import rerank
from app.rag.search import RetrievedChunk, dense_search, sparse_search
from app.observability.timing import emit_timing
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    question: str
    answer: str
    grounded: bool
    sources: list[str] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    retry_count: int = 0
    latency_ms: float = 0.0
    stage_counts: dict = field(default_factory=dict)

def retrieve(db: Session, query: str, doc_type: str | None=None, final_k: int | None=None) -> tuple[list[RetrievedChunk], dict]:
    total_started = time.perf_counter()
    final_k = final_k or ai_settings.FINAL_TOP_K
    failures = []
    dense_started = time.perf_counter()
    try:
        dense = dense_search(db, query, doc_type=doc_type)
    except Exception as exc:
        logger.warning('Dense search failed: %s', exc)
        dense = []
        failures.append(('dense', exc))
    emit_timing(
        'rag.dense_search',
        (time.perf_counter() - dense_started) * 1000,
        result_count=len(dense),
        failed=bool(failures and failures[-1][0] == 'dense'),
    )
    sparse_started = time.perf_counter()
    try:
        sparse = sparse_search(db, query, doc_type=doc_type)
    except Exception as exc:
        logger.warning('Sparse search failed: %s', exc)
        sparse = []
        failures.append(('sparse', exc))
    emit_timing(
        'rag.sparse_stage',
        (time.perf_counter() - sparse_started) * 1000,
        result_count=len(sparse),
        failed=bool(failures and failures[-1][0] == 'sparse'),
    )
    if not dense and not sparse and failures:
        emit_timing(
            'rag.retrieve_total',
            (time.perf_counter() - total_started) * 1000,
            dense=0,
            sparse=0,
            failed=True,
        )
        failed_stages = ', '.join(stage for stage, _ in failures)
        raise RuntimeError(f'Policy retrieval failed for all available search stages: {failed_stages}.') from failures[0][1]
    if not dense and not sparse:
        emit_timing('rag.retrieve_total', (time.perf_counter() - total_started) * 1000, dense=0, sparse=0, fused=0, final=0)
        return ([], {'dense': 0, 'sparse': 0, 'fused': 0, 'final': 0})
    if failures:
        logger.warning(
            'Policy retrieval is using partial results; failed stages: %s',
            ', '.join(stage for stage, _ in failures),
        )
    fusion_started = time.perf_counter()
    fused = reciprocal_rank_fusion([dense, sparse])
    emit_timing('rag.fusion', (time.perf_counter() - fusion_started) * 1000, result_count=len(fused))
    rerank_started = time.perf_counter()
    final = rerank(query, fused, top_k=final_k)
    emit_timing(
        'rag.rerank',
        (time.perf_counter() - rerank_started) * 1000,
        enabled=ai_settings.ENABLE_RERANKER,
        candidate_count=len(fused),
        result_count=len(final),
    )
    counts = {'dense': len(dense), 'sparse': len(sparse), 'fused': len(fused), 'final': len(final)}
    logger.info('Retrieval for %r: %s', query[:60], counts)
    emit_timing('rag.retrieve_total', (time.perf_counter() - total_started) * 1000, **counts)
    return (final, counts)

def answer(db: Session, query: str, doc_type: str | None=None) -> RAGResponse:
    started = time.perf_counter()
    chunks, counts = retrieve(db, query, doc_type=doc_type)
    generation_started = time.perf_counter()
    result: GenerationResult = generate_answer(query, chunks)
    emit_timing('rag.generation_total', (time.perf_counter() - generation_started) * 1000, chunk_count=len(chunks), retry_count=result.retry_count)
    latency_ms = (time.perf_counter() - started) * 1000
    emit_timing('rag.answer_total', latency_ms, chunk_count=len(chunks))
    return RAGResponse(question=query, answer=result.answer, grounded=result.grounded, sources=result.sources, chunks=chunks, retry_count=result.retry_count, latency_ms=latency_ms, stage_counts=counts)