import os
import time
import logging
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import json
from pathlib import Path

from config import RATE_LIMIT_TOKENS, RATE_LIMIT_WINDOW, DATA_DIR
from rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag-eval-api")

app = FastAPI(
    title="Production-Grade RAG API with Evaluation Harness",
    description="FastAPI service for a Retrieval-Augmented Generation (RAG) system with hybrid search and pipeline evaluation.",
    version="1.0.0"
)

# Initialize pipelines lazily to avoid heavy loading on server start
pipelines = {}

def get_pipeline(mode: str) -> RAGPipeline:
    if mode not in pipelines:
        logger.info(f"Initializing RAG pipeline in '{mode}' mode...")
        pipelines[mode] = RAGPipeline(mode=mode)
    return pipelines[mode]

# Custom Token-Bucket Rate Limiter Middleware
class TokenBucketRateLimiter:
    def __init__(self, capacity: int, fill_rate: float):
        self.capacity = capacity
        self.fill_rate = fill_rate  # tokens per second
        # Storage of bucket state per IP: {ip: {"tokens": float, "last_updated": float}}
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
        
        # Add new tokens based on elapsed time
        new_tokens = elapsed * self.fill_rate
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + new_tokens)
        bucket["last_updated"] = now
        
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False

# Initialize rate limiter: capacity of 30 requests, refills 0.5 request/second (30 req/min)
rate_limiter = TokenBucketRateLimiter(capacity=30, fill_rate=0.5)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Retrieve client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Bypass local testing endpoints or health checks if preferred, but we apply to all
    if not rate_limiter._allow_request(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Too many requests. Please try again later."}
        )
        
    response = await call_next(request)
    return response

# Structured Request Logging Middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    method = request.method
    path = request.url.path
    client_ip = request.client.host if request.client else "unknown"
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    status_code = response.status_code
    
    logger.info(
        f"Client IP: {client_ip} | Method: {method} | Path: {path} | "
        f"Status: {status_code} | Latency: {duration:.3f}s"
    )
    return response

# Pydantic schemas
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The query string to process through RAG.")
    mode: Optional[str] = Field("improved", description="Mode must be 'baseline' or 'improved'")

class ChunkResponse(BaseModel):
    id: str
    text: str
    source: str
    score: Optional[float] = None
    rrf_score: Optional[float] = None
    rank: Optional[int] = None

class QueryResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    answer: str
    citations: List[str]
    retrieved_chunks: List[ChunkResponse]
    latency_seconds: float
    estimated_cost_usd: float

@app.get("/health")
async def health_check():
    return {"status": "healthy", "time": time.time()}

@app.post("/query", response_model=QueryResponse)
async def query_rag(body: QueryRequest):
    if body.mode not in ["baseline", "improved"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mode must be 'baseline' or 'improved'"
        )
        
    try:
        pipeline = get_pipeline(body.mode)
        result = pipeline.query(body.query)
        
        # Clean up retrieved chunks structures
        chunks_resp = []
        for c in result["retrieved_chunks"]:
            chunks_resp.append(ChunkResponse(
                id=c["id"],
                text=c["text"],
                source=c["source"],
                score=c.get("score"),
                rrf_score=c.get("rrf_score"),
                rank=c.get("rank")
            ))
            
        return QueryResponse(
            query=result["query"],
            rewritten_query=result.get("rewritten_query"),
            answer=result["answer"],
            citations=result["citations"],
            retrieved_chunks=chunks_resp,
            latency_seconds=result["metrics"]["total_latency"],
            estimated_cost_usd=result["metrics"].get("total_cost", result["metrics"]["cost"])
        )
    except Exception as e:
        logger.error(f"Error querying RAG system: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the RAG pipeline: {str(e)}"
        )

@app.get("/metrics")
async def get_evaluation_metrics():
    report_path = DATA_DIR / "evaluation_report.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation metrics report not found. Please trigger the evaluation runner first."
        )
        
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
            
        # Return compiled summaries only to avoid massive JSON dump
        return {
            "judge_validation": report_data["judge_validation"],
            "baseline_summary": report_data["baseline_summary"],
            "improved_summary": report_data["improved_summary"]
        }
    except Exception as e:
        logger.error(f"Error loading evaluation report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error reading evaluation report."
        )
