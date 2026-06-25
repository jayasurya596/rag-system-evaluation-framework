import time
import logging
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json
from pathlib import Path

from backend.config import DATA_DIR, logger
from backend.rag_pipeline import RAGPipeline
from backend.feedback_db import FeedbackDatabase
from backend.monitoring import RAGMonitor

app = FastAPI(
    title="Production-Grade Financial RAG Service",
    description="FastAPI service for financial filings analysis with hybrid search, reranking, and self-adaptive feedback loop.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy initialization of RAG pipeline
rag_pipeline = None

def get_rag_pipeline() -> RAGPipeline:
    global rag_pipeline
    if rag_pipeline is None:
        logger.info("Initializing core RAG Pipeline...")
        rag_pipeline = RAGPipeline()
    return rag_pipeline

# Token-Bucket Rate Limiter
class TokenBucketRateLimiter:
    def __init__(self, capacity: int, fill_rate: float):
        self.capacity = capacity
        self.fill_rate = fill_rate  # tokens per second
        self.buckets: Dict[str, Dict[str, float]] = {}

    def _allow_request(self, client_ip: str) -> bool:
        now = time.time()
        if client_ip not in self.buckets:
            self.buckets[client_ip] = {
                "tokens": self.capacity - 1.0,
                "last_updated": now
            }
            return True
            
        bucket = self.buckets[client_ip]
        elapsed = now - bucket["last_updated"]
        
        # Add new tokens
        new_tokens = elapsed * self.fill_rate
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + new_tokens)
        bucket["last_updated"] = now
        
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False

# Limit to 60 requests per minute per IP (capacity=60, fill=1.0 req/sec)
rate_limiter = TokenBucketRateLimiter(capacity=60, fill_rate=1.0)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    
    # Bypass local loops if needed, but apply globally for safety
    if not rate_limiter._allow_request(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Too many requests. Rate limit is 60 requests per minute."}
        )
    return await call_next(request)

# Performance/Access Logging Middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start_time = time.time()
    method = request.method
    path = request.url.path
    client_ip = request.client.host if request.client else "unknown"
    
    response = await call_next(request)
    
    latency = time.time() - start_time
    logger.info(
        f"IP: {client_ip} | Method: {method} | Path: {path} | "
        f"Status: {response.status_code} | Latency: {latency:.3f}s"
    )
    return response

# Pydantic schemas
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User query about company financials.")

class CitationSchema(BaseModel):
    document_id: str
    company_name: str
    filing_date: str

class ChunkSchema(BaseModel):
    document_id: str
    content: str
    company_name: str
    filing_type: str
    filing_date: str
    source_url: str

class QueryResponse(BaseModel):
    query: str
    rewritten_query: str
    answer: str
    citations: List[CitationSchema]
    retrieved_chunks: List[ChunkSchema]
    metrics: Dict[str, Any]
    potential_hallucination: bool

class FeedbackRequest(BaseModel):
    query: str = Field(...)
    answer: str = Field(...)
    rating: str = Field(..., description="Must be 'good' or 'bad'")

# Endpoints
@app.get("/")
async def get_root():
    # Fetch database count and freshness
    db = FeedbackDatabase()
    metrics_mon = RAGMonitor()
    pipeline = get_rag_pipeline()
    
    doc_count = pipeline.db_client.count_documents()
    feedback_stats = db.get_feedback_summary()
    monitor_stats = metrics_mon.get_summary_metrics()
    
    # Check freshness
    freshness_status = "unknown"
    newest_date = "N/A"
    warning = False
    
    all_docs = pipeline.db_client.get_all_documents()
    if all_docs:
        dates = [d.get("filing_date", "1970-01-01") for d in all_docs if d.get("filing_date") != "Unknown"]
        if dates:
            newest_date = max(dates)
            try:
                dt = datetime.strptime(newest_date, "%Y-%m-%d")
                delta_days = (datetime.now() - dt).days
                if delta_days > 30:
                    freshness_status = f"Outdated ({delta_days} days old)"
                    warning = True
                else:
                    freshness_status = "Fresh"
            except Exception:
                pass

    return {
        "service": "Production-Grade Financial RAG API",
        "status": "active",
        "version": "1.0.0",
        "database": {
            "collection": "financial_documents",
            "document_count": doc_count,
            "freshness": {
                "newest_filing_date": newest_date,
                "status": freshness_status,
                "warning": warning
            }
        },
        "statistics": {
            "feedback": {
                "good_answers": feedback_stats["good"],
                "bad_answers": feedback_stats["bad"],
                "mode": feedback_stats["current_mode"]
            },
            "latency": monitor_stats
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(body: QueryRequest):
    try:
        pipeline = get_rag_pipeline()
        result = pipeline.query(body.query)
        return QueryResponse(
            query=result["query"],
            rewritten_query=result["rewritten_query"],
            answer=result["answer"],
            citations=result["citations"],
            retrieved_chunks=result["retrieved_chunks"],
            metrics=result["metrics"],
            potential_hallucination=result["potential_hallucination"]
        )
    except Exception as e:
        logger.error(f"Error querying RAG system: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the RAG pipeline: {str(e)}"
        )

@app.post("/feedback")
async def submit_feedback(body: FeedbackRequest):
    if body.rating not in ["good", "bad"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rating must be 'good' or 'bad'"
        )
    try:
        db = FeedbackDatabase()
        res = db.log_feedback(body.query, body.answer, body.rating)
        return {
            "status": "success",
            "feedback_logged": body.rating,
            "self_adjustment_triggered": res["adjustment_triggered"],
            "current_bad_feedback_count": res["bad_count"],
            "message": res["message"] or "Feedback logged successfully."
        }
    except Exception as e:
        logger.error(f"Error logging feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save feedback to the database."
        )

@app.get("/metrics")
async def get_metrics():
    try:
        monitor = RAGMonitor()
        return monitor.get_summary_metrics()
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read metrics database."
        )

@app.get("/evaluation")
async def get_evaluation_results():
    # Search for compiled evaluation report in DATA_DIR / evaluation_report.json
    report_path = DATA_DIR / "evaluation_report.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation metrics report not found. Run evaluation runner first."
        )
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error reading evaluation file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error reading evaluation report."
        )
