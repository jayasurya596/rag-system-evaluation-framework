import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from google import genai
from google.genai import types

from backend.config import LLM_MODEL, logger, get_api_key, retry_api_call

class RAGEvaluator:
    def __init__(self):
        try:
            api_key = get_api_key()
            self.genai_client = genai.Client(api_key=api_key)
            self.has_api_key = True
        except Exception as e:
            logger.warning(f"Gemini client not initialized in evaluator (using mock scores): {e}")
            self.genai_client = None
            self.has_api_key = False

    def evaluate_retrieval(self, retrieved_chunks: List[Dict[str, Any]], relevant_doc_ids: List[str]) -> Dict[str, float]:
        """Compute Precision@5, Recall@5, MRR, and NDCG."""
        if not relevant_doc_ids:
            return {"precision_at_5": 1.0, "recall_at_5": 1.0, "mrr": 1.0, "ndcg": 1.0}
            
        retrieved_ids = [re.sub(r'_c\d+$', '', c.get("document_id", "")) for c in retrieved_chunks[:5]]
        
        # 1. Precision@5
        hits_5 = sum(1 for doc_id in retrieved_ids if doc_id in relevant_doc_ids)
        precision_at_5 = hits_5 / 5.0 if retrieved_ids else 0.0
        
        # 2. Recall@5
        unique_hits_5 = len(set(doc_id for doc_id in retrieved_ids if doc_id in relevant_doc_ids))
        recall_at_5 = unique_hits_5 / len(relevant_doc_ids)
        
        # 3. MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_doc_ids:
                mrr = 1.0 / (rank + 1)
                break
                
        # 4. NDCG (Normalized Discounted Cumulative Gain)
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_doc_ids:
                dcg += 1.0 / np.log2(rank + 2)
                
        idcg = 0.0
        for rank in range(min(len(relevant_doc_ids), 5)):
            idcg += 1.0 / np.log2(rank + 2)
            
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return {
            "precision_at_5": float(precision_at_5),
            "recall_at_5": float(recall_at_5),
            "mrr": float(mrr),
            "ndcg": float(ndcg)
        }

    def evaluate_faithfulness(self, context: str, answer: str, query: str) -> float:
        """Use LLM-as-a-judge to measure faithfulness (groundedness: fraction of supported claims)."""
        if answer == "Insufficient evidence found.":
            return 1.0  # Refusal is 100% faithful
            
        if not self.has_api_key or not self.genai_client:
            # Fallback mock score
            return 1.0 if "[0]" in answer else 0.5
            
        prompt = (
            "You are a strict Q&A judge checking for answer groundedness (faithfulness).\n"
            "Your task is to analyze if the generated answer is fully supported by the retrieved context.\n"
            "Do not use outside knowledge.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            "Instructions:\n"
            "1. Extract all claims/facts in the Generated Answer.\n"
            "2. For each claim, check if it is directly and explicitly supported by the Context. If it requires extrapolation, mark it as supported = false.\n"
            "3. Output a JSON object with two fields:\n"
            "   - 'claims': a list of objects, each with 'claim' (string), 'supported' (boolean), and 'reason' (string).\n"
            "   - 'faithfulness_score': a float between 0.0 and 1.0 (fraction of supported claims).\n\n"
            "Return ONLY the raw JSON object. Do not enclose in markdown blocks."
        )
        
        try:
            response = retry_api_call(
                self.genai_client.models.generate_content,
                max_retries=3,
                initial_delay=1.0,
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            text = response.text.strip()
            # Parse json
            # Handle potential code block backticks
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            data = json.loads(text)
            return float(data.get("faithfulness_score", 1.0))
        except Exception as e:
            logger.error(f"Failed to judge faithfulness: {e}")
            return 1.0 if "[0]" in answer else 0.5

    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Use LLM-as-a-judge to measure how relevant the answer is to the query (1-5 scale)."""
        if not self.has_api_key or not self.genai_client:
            return 4.0 if answer != "Insufficient evidence found." else 3.0
            
        prompt = (
            "You are an AI judge evaluating answer relevance on a 1.0 to 5.0 scale.\n"
            "Assess how well the generated answer addresses the question. Ignore factuality or correctness; focus ONLY on completeness and direct relevance.\n"
            "A rating of 5.0 means the answer completely and directly answers the question.\n"
            "A rating of 1.0 means the answer is completely off-topic.\n"
            "Output a JSON object with two fields:\n"
            "   - 'score': float between 1.0 and 5.0.\n"
            "   - 'reason': string explaining the score.\n\n"
            f"Question:\n{query}\n\n"
            f"Answer:\n{answer}\n\n"
            "Return ONLY the raw JSON object. Do not use code blocks."
        )
        
        try:
            response = retry_api_call(
                self.genai_client.models.generate_content,
                max_retries=3,
                initial_delay=1.0,
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            text = response.text.strip()
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            data = json.loads(text)
            return float(data.get("score", 4.0))
        except Exception as e:
            logger.error(f"Failed to judge answer relevance: {e}")
            return 4.0

    def evaluate_context_relevance(self, query: str, context: str) -> float:
        """Use LLM-as-a-judge to measure context relevance (1-5 scale)."""
        if not self.has_api_key or not self.genai_client:
            return 4.0
            
        prompt = (
            "You are an AI judge evaluating context relevance on a 1.0 to 5.0 scale.\n"
            "Assess if the retrieved context is relevant, clean, and contains the required facts to answer the question.\n"
            "A score of 5.0 means the context contains all required facts without irrelevant noise.\n"
            "A score of 1.0 means the context is completely irrelevant to the question.\n"
            "Output a JSON object with two fields:\n"
            "   - 'score': float between 1.0 and 5.0.\n"
            "   - 'reason': string explaining the score.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            "Return ONLY the raw JSON object. Do not use code blocks."
        )
        
        try:
            response = retry_api_call(
                self.genai_client.models.generate_content,
                max_retries=3,
                initial_delay=1.0,
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            text = response.text.strip()
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            data = json.loads(text)
            return float(data.get("score", 4.0))
        except Exception as e:
            logger.error(f"Failed to judge context relevance: {e}")
            return 4.0
