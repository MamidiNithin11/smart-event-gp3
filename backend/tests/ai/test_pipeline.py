from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.rag import pipeline


def test_retrieve_reports_failure_when_both_search_stages_fail():
    with patch('app.rag.pipeline.dense_search', side_effect=RuntimeError('embedding provider down')), patch(
        'app.rag.pipeline.sparse_search',
        side_effect=RuntimeError('database unavailable'),
    ):
        with pytest.raises(RuntimeError, match='all available search stages: dense, sparse'):
            pipeline.retrieve(MagicMock(), 'refund policy')


def test_retrieve_keeps_partial_results_when_one_search_stage_fails():
    chunk = MagicMock(chunk_id='chunk-1', dense_score=0.9, sparse_score=None)
    with patch('app.rag.pipeline.dense_search', return_value=[chunk]), patch(
        'app.rag.pipeline.sparse_search',
        side_effect=RuntimeError('sparse index unavailable'),
    ), patch('app.rag.pipeline.reciprocal_rank_fusion', return_value=[chunk]), patch(
        'app.rag.pipeline.rerank',
        return_value=[chunk],
    ):
        chunks, counts = pipeline.retrieve(MagicMock(), 'refund policy')

    assert chunks == [chunk]
    assert counts == {'dense': 1, 'sparse': 0, 'fused': 1, 'final': 1}

def test_retrieve_logs_dense_sparse_fusion_and_rerank_stages(caplog):
    import json

    chunk = MagicMock(chunk_id='chunk-1', dense_score=0.9, sparse_score=0.8)
    with patch('app.rag.pipeline.dense_search', return_value=[chunk]), patch(
        'app.rag.pipeline.sparse_search',
        return_value=[chunk],
    ), patch('app.rag.pipeline.reciprocal_rank_fusion', return_value=[chunk]), patch(
        'app.rag.pipeline.rerank',
        return_value=[chunk],
    ), caplog.at_level('INFO', logger='app.chat_timing'):
        pipeline.retrieve(MagicMock(), 'policy question')

    stages = {
        json.loads(record.message)['stage']
        for record in caplog.records
        if record.name == 'app.chat_timing'
    }
    assert {
        'rag.dense_search',
        'rag.sparse_stage',
        'rag.fusion',
        'rag.rerank',
        'rag.retrieve_total',
    }.issubset(stages)
