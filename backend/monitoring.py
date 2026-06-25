import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from backend.config import WORKSPACE_DIR, logger

class RAGMonitor:
    def __init__(self, json_path: str = None):
        if json_path is None:
            json_path = str(WORKSPACE_DIR / "monitoring_db.json")
        self.json_path = json_path
        self._init_db()

    def _init_db(self):
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load_data(self) -> List[Dict[str, Any]]:
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_data(self, data: List[Dict[str, Any]]):
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save monitoring metrics to JSON: {e}")

    def log_metrics(self, query: str, metrics: Dict[str, Any]) -> int:
        """Log latency and cost details for a single query. Returns index position."""
        data = self._load_data()
        now = time.time()
        
        entry = {
            "id": len(data) + 1,
            "timestamp": now,
            "query": query,
            "embedding_time": metrics.get("embedding_time", 0.0),
            "retrieval_time": metrics.get("retrieval_time", 0.0),
            "reranking_time": metrics.get("reranking_time", 0.0),
            "llm_time": metrics.get("llm_time", 0.0),
            "total_latency": metrics.get("total_latency", 0.0),
            "input_tokens": metrics.get("input_tokens", 0),
            "output_tokens": metrics.get("output_tokens", 0),
            "cost": metrics.get("cost", 0.0)
        }
        
        data.append(entry)
        self._save_data(data)
        
        logger.info(
            f"[MONITOR] Logged Query ID {entry['id']} | Total Latency: {entry['total_latency']:.3f}s | Cost: ${entry['cost']:.6f}"
        )
        return entry["id"]

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Aggregate performance and cost stats for display."""
        data = self._load_data()
        
        if not data:
            return {
                "total_queries": 0,
                "avg_latency": 0.0,
                "avg_embedding": 0.0,
                "avg_retrieval": 0.0,
                "avg_reranking": 0.0,
                "avg_llm": 0.0,
                "total_cost": 0.0,
                "total_tokens": 0
            }
            
        total_queries = len(data)
        avg_latency = sum(item["total_latency"] for item in data) / total_queries
        avg_embedding = sum(item["embedding_time"] for item in data) / total_queries
        avg_retrieval = sum(item["retrieval_time"] for item in data) / total_queries
        avg_reranking = sum(item["reranking_time"] for item in data) / total_queries
        avg_llm = sum(item["llm_time"] for item in data) / total_queries
        total_cost = sum(item["cost"] for item in data)
        total_tokens = sum(item["input_tokens"] + item["output_tokens"] for item in data)
        
        return {
            "total_queries": total_queries,
            "avg_latency": round(avg_latency, 3),
            "avg_embedding": round(avg_embedding, 3),
            "avg_retrieval": round(avg_retrieval, 3),
            "avg_reranking": round(avg_reranking, 3),
            "avg_llm": round(avg_llm, 3),
            "total_cost": round(total_cost, 6),
            "total_tokens": total_tokens
        }

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Retrieve all metrics records sorted by timestamp."""
        data = self._load_data()
        sorted_data = sorted(data, key=lambda x: x["timestamp"])
        
        return [{
            "time": datetime.fromtimestamp(item["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "query": item["query"],
            "embedding": item["embedding_time"],
            "retrieval": item["retrieval_time"],
            "reranking": item["reranking_time"],
            "llm": item["llm_time"],
            "total": item["total_latency"],
            "input_tokens": item["input_tokens"],
            "output_tokens": item["output_tokens"],
            "cost": item["cost"]
        } for item in sorted_data]
