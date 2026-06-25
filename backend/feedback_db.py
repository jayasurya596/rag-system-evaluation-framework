import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from backend.config import WORKSPACE_DIR, ADAPTIVE_CONFIG, logger

class FeedbackDatabase:
    def __init__(self, json_path: str = None, threshold: int = 5):
        if json_path is None:
            json_path = str(WORKSPACE_DIR / "feedback_db.json")
        self.json_path = json_path
        self.threshold = threshold
        self._init_db()

    def _init_db(self):
        if not os.path.exists(self.json_path):
            initial_state = {
                "feedback": [],
                "adjustments_log": []
            }
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(initial_state, f, indent=2)

    def _load_data(self) -> Dict[str, Any]:
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"feedback": [], "adjustments_log": []}

    def _save_data(self, data: Dict[str, Any]):
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write feedback data to JSON: {e}")

    def log_feedback(self, query: str, answer: str, rating: str) -> Dict[str, Any]:
        """Insert user feedback. Returns if adjustment was triggered."""
        if rating not in ["good", "bad"]:
            raise ValueError("Rating must be 'good' or 'bad'")
            
        data = self._load_data()
        now = time.time()
        
        # Add feedback item
        feedback_id = len(data["feedback"]) + 1
        data["feedback"].append({
            "id": feedback_id,
            "query": query,
            "answer": answer,
            "rating": rating,
            "timestamp": now,
            "processed": 0
        })
        
        # Count active bad feedback
        bad_count = sum(1 for item in data["feedback"] if item["rating"] == "bad" and item["processed"] == 0)
        
        adjustment_triggered = False
        log_message = ""
        
        if bad_count >= self.threshold:
            # Mark active bad feedback as processed
            for item in data["feedback"]:
                if item["rating"] == "bad" and item["processed"] == 0:
                    item["processed"] = 1
                    
            reason = f"Bad feedback count reached threshold of {self.threshold} (actual: {bad_count})"
            data["adjustments_log"].append({
                "id": len(data["adjustments_log"]) + 1,
                "timestamp": now,
                "reason": reason,
                "previous_mode": "normal",
                "new_mode": "adapted"
            })
            
            adjustment_triggered = True
            log_message = f"RAG System Self-Adjustment Triggered: {reason}. Switching to 'adapted' pipeline parameters."
            logger.info(log_message)
            
        self._save_data(data)
        return {
            "adjustment_triggered": adjustment_triggered,
            "bad_count": bad_count,
            "message": log_message
        }

    def get_rag_parameters(self) -> Dict[str, Any]:
        """Return the current active RAG search parameters based on feedback state."""
        data = self._load_data()
        logs = data.get("adjustments_log", [])
        if logs and logs[-1]["new_mode"] == "adapted":
            return ADAPTIVE_CONFIG["adapted"]
        return ADAPTIVE_CONFIG["normal"]

    def reset_adjustments(self):
        """Reset the system parameters back to normal configuration."""
        data = self._load_data()
        data["adjustments_log"] = []
        for item in data["feedback"]:
            item["processed"] = 1
        self._save_data(data)
        logger.info("Feedback adjustments reset. Reverted RAG to 'normal' parameters.")

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Return aggregate statistics and recent history logs."""
        data = self._load_data()
        
        good_count = sum(1 for item in data["feedback"] if item["rating"] == "good")
        bad_count = sum(1 for item in data["feedback"] if item["rating"] == "bad")
        
        # Sort logs by timestamp descending
        sorted_feedback = sorted(data["feedback"], key=lambda x: x["timestamp"], reverse=True)[:20]
        logs = []
        for item in sorted_feedback:
            # Format time
            t_str = datetime.fromtimestamp(item["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            logs.append({
                "query": item["query"],
                "answer": item["answer"],
                "rating": item["rating"],
                "time": t_str
            })
            
        adjustments = []
        sorted_adjust = sorted(data["adjustments_log"], key=lambda x: x["timestamp"], reverse=True)
        for item in sorted_adjust:
            t_str = datetime.fromtimestamp(item["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            adjustments.append({
                "time": t_str,
                "reason": item["reason"],
                "previous_mode": item["previous_mode"],
                "new_mode": item["new_mode"]
            })
            
        current_mode = "adapted" if data.get("adjustments_log") else "normal"
        
        return {
            "good": good_count,
            "bad": bad_count,
            "logs": logs,
            "adjustments": adjustments,
            "current_mode": current_mode
        }
