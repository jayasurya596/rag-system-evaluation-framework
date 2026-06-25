import os
import sys
import pytest
from pathlib import Path

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.chunking import DocumentChunker
from backend.feedback_db import FeedbackDatabase
from backend.monitoring import RAGMonitor

def test_document_chunker():
    # Test text clean and chunk split
    chunker = DocumentChunker(chunk_size=100, overlap=20)
    
    doc = {
        "content": "This is a simple financial text statement. " * 30, # Long enough to chunk
        "company_name": "Test Company",
        "filing_type": "10-K",
        "filing_date": "2024-01-01",
        "source_url": "http://example.com/test",
        "document_id": "TEST_DOC_001"
    }
    
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    assert chunks[0]["company_name"] == "Test Company"
    assert chunks[0]["filing_type"] == "10-K"
    assert chunks[0]["document_id"].startswith("TEST_DOC_001_c")
    assert "token_count" in chunks[0]

def test_feedback_loop_trigger(tmp_path):
    # Setup temporary feedback JSON file
    db_file = tmp_path / "feedback_test.json"
    db = FeedbackDatabase(json_path=str(db_file), threshold=3)
    
    # Verify initial normal state
    params = db.get_rag_parameters()
    assert params["retrieval_depth"] == 15
    assert params["reranker_top_k"] == 5
    
    # Log 2 bad answers (under threshold of 3)
    res = db.log_feedback("Query 1", "Answer 1", "bad")
    assert res["adjustment_triggered"] == False
    
    res = db.log_feedback("Query 2", "Answer 2", "bad")
    assert res["adjustment_triggered"] == False
    
    # 3rd bad answer should trigger adjustment
    res = db.log_feedback("Query 3", "Answer 3", "bad")
    assert res["adjustment_triggered"] == True
    assert "adapted" in res["message"]
    
    # Verify adapted parameters are active
    adapted_params = db.get_rag_parameters()
    assert adapted_params["retrieval_depth"] == 30
    assert adapted_params["reranker_top_k"] == 8
    
    # Log good answer, parameters should remain in adapted until reset
    db.log_feedback("Query 4", "Answer 4", "good")
    assert db.get_rag_parameters()["retrieval_depth"] == 30
    
    # Reset
    db.reset_adjustments()
    assert db.get_rag_parameters()["retrieval_depth"] == 15

def test_monitoring_logger(tmp_path):
    mon_file = tmp_path / "monitor_test.json"
    monitor = RAGMonitor(json_path=str(mon_file))
    
    # Initial state
    summary = monitor.get_summary_metrics()
    assert summary["total_queries"] == 0
    
    # Log metrics
    metrics = {
        "embedding_time": 0.05,
        "retrieval_time": 0.20,
        "reranking_time": 0.15,
        "llm_time": 0.80,
        "total_latency": 1.20,
        "input_tokens": 1200,
        "output_tokens": 300,
        "cost": 0.00018
    }
    monitor.log_metrics("Test Query", metrics)
    
    # Check aggregation
    summary = monitor.get_summary_metrics()
    assert summary["total_queries"] == 1
    assert summary["avg_latency"] == 1.20
    assert summary["total_cost"] == 0.00018
    assert summary["total_tokens"] == 1500
