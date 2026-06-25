import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app
from backend.feedback_db import FeedbackDatabase

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_query_rag_endpoint():
    # Test query response structure
    # Use a dummy query
    response = client.post("/query", json={"query": "Test query for AMD"})
    assert response.status_code == 200
    res_data = response.json()
    assert "query" in res_data
    assert "answer" in res_data
    assert "citations" in res_data
    assert "metrics" in res_data
    assert "potential_hallucination" in res_data

def test_feedback_submission_endpoint():
    # Submit bad feedback
    # We first reset to normal state
    db = FeedbackDatabase()
    db.reset_adjustments()
    
    response = client.post("/feedback", json={
        "query": "Test query feedback",
        "answer": "Test answer",
        "rating": "bad"
    })
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["feedback_logged"] == "bad"

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    res_data = response.json()
    assert "total_queries" in res_data
    assert "avg_latency" in res_data
    assert "total_cost" in res_data
