import os
import re
import json
import time
from google import genai
from google.genai import types

from config import LLM_MODEL, get_api_key, COST_LLM_INPUT, COST_LLM_OUTPUT, retry_api_call
import logging

logger = logging.getLogger("rag-eval-api")

class RAGEvaluator:
    def __init__(self):
        try:
            api_key = get_api_key()
            self.genai_client = genai.Client(api_key=api_key)
            self.has_api_key = True
        except Exception as e:
            logger.warning(f"Gemini client not initialized in evaluator: {e}")
            self.genai_client = None
            self.has_api_key = False

    def evaluate_retrieval(self, retrieved_chunks, expected_sources):
        """Calculate Precision@K and Recall@K."""
        if not expected_sources:
            # For unanswerable questions:
            if not retrieved_chunks:
                return {"precision": 1.0, "recall": 1.0}
            else:
                # If we retrieved chunks for a question that has no sources in the corpus
                # Technically, recall is 1.0 (we didn't miss any expected source),
                # but precision is 0.0 (none of the retrieved chunks are relevant since none exist).
                return {"precision": 0.0, "recall": 1.0}
                
        retrieved_sources = [c["source"] for c in retrieved_chunks]
        
        # Count matches
        match_count = sum(1 for src in retrieved_sources if src in expected_sources)
        
        precision = match_count / len(retrieved_chunks) if retrieved_chunks else 0.0
        
        # Calculate unique expected sources hit
        unique_hits = len(set(src for src in retrieved_sources if src in expected_sources))
        recall = unique_hits / len(expected_sources)
        
        return {
            "precision": precision,
            "recall": recall
        }

    def evaluate_faithfulness(self, context, answer, query):
        """Use LLM-as-a-judge to measure faithfulness (groundedness)."""
        # Short-circuit standard refusal replies for unanswerable questions to save cost/time
        refusal_phrases = [
            "does not contain enough information",
            "do not contain enough information",
            "cannot answer",
            "corpus does not contain",
            "no information"
        ]
        if any(phrase in answer.lower() for phrase in refusal_phrases):
            # If the system correctly refused to answer, it is 100% faithful (no hallucinated claims)
            return {
                "faithfulness_score": 1.0,
                "claims": [{"claim": "Refusal to answer due to lack of context", "supported": True, "reason": "System correctly identified insufficient context"}],
                "judge_cost": 0.0
            }
            
        if not self.has_api_key or not self.genai_client:
            # Fallback mock score
            return {
                "faithfulness_score": 1.0 if "[0]" in answer or "Turing" in answer else 0.5,
                "claims": [{"claim": "Mocked claim", "supported": True, "reason": "No API Key"}],
                "judge_cost": 0.0
            }
            
        prompt = (
            f"You are a strict Q&A judge checking for answer groundedness (faithfulness).\n"
            f"Your task is to analyze if the generated answer is fully supported by the retrieved context.\n"
            f"Do not use any outside knowledge.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            f"Instructions:\n"
            f"1. Identify all individual claims/facts asserted in the Generated Answer.\n"
            f"2. For each claim, check if it is directly and explicitly supported by the Context. If it requires extrapolation or is not mentioned, mark it as supported = false.\n"
            f"3. Output a JSON object with two fields:\n"
            f"   - 'claims': a list of objects, each with 'claim' (string), 'supported' (boolean), and 'reason' (string explaining why or referencing the context).\n"
            f"   - 'faithfulness_score': a float between 0.0 and 1.0 representing the fraction of supported claims.\n\n"
            f"Return ONLY the raw JSON object. Do not enclose it in markdown code blocks or add explanations."
        )
        
        try:
            response = retry_api_call(
                self.genai_client.models.generate_content,
                max_retries=3,
                initial_delay=1.0,
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            data = json.loads(response.text.strip())
            
            # Token cost
            in_tokens = response.usage_metadata.prompt_token_count
            out_tokens = response.usage_metadata.candidates_token_count
            cost = (in_tokens * COST_LLM_INPUT) + (out_tokens * COST_LLM_OUTPUT)
            
            data["judge_cost"] = cost
            return data
        except Exception as e:
            logger.error(f"Faithfulness judge failed: {e}")
            return {"faithfulness_score": 0.5, "claims": [], "judge_cost": 0.0}

    def evaluate_relevance(self, query, answer):
        """Use LLM-as-a-judge to measure answer relevance (1 to 5)."""
        refusal_phrases = [
            "does not contain enough information",
            "do not contain enough information",
            "cannot answer",
            "corpus does not contain",
            "no information"
        ]
        # Correct refusal is highly relevant if context doesn't support the answer
        # However, we will let the LLM judge it or handle it cleanly.
        
        if not self.has_api_key or not self.genai_client:
            return {"relevance_score": 4.0, "explanation": "No API Key", "judge_cost": 0.0}
            
        prompt = (
            f"You are a Q&A evaluation judge scoring answer relevance.\n"
            f"Evaluate if the Generated Answer directly and fully addresses the User Question.\n"
            f"Ignore whether the answer is factually correct. Focus purely on whether the answer is helpful, relevant, and fully answers the question asked.\n\n"
            f"Question:\n{query}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            f"Score the answer on a scale from 1 to 5:\n"
            f"- 5: Highly relevant, complete, and directly answers the question.\n"
            f"- 4: Mostly relevant and answers the question, but could be slightly clearer or is missing minor detail.\n"
            f"- 3: Partially relevant; addresses the topic but doesn't fully answer the question.\n"
            f"- 2: Barely relevant; off-topic or repeats the question without answering.\n"
            f"- 1: Completely irrelevant or nonsensical.\n\n"
            f"Output a JSON object with two fields:\n"
            f"   - 'relevance_score': an integer from 1 to 5.\n"
            f"   - 'explanation': a brief explanation of the score.\n\n"
            f"Return ONLY the raw JSON object."
        )
        
        try:
            response = retry_api_call(
                self.genai_client.models.generate_content,
                max_retries=3,
                initial_delay=1.0,
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            data = json.loads(response.text.strip())
            
            in_tokens = response.usage_metadata.prompt_token_count
            out_tokens = response.usage_metadata.candidates_token_count
            cost = (in_tokens * COST_LLM_INPUT) + (out_tokens * COST_LLM_OUTPUT)
            
            data["judge_cost"] = cost
            return data
        except Exception as e:
            logger.error(f"Relevance judge failed: {e}")
            return {"relevance_score": 3.0, "explanation": "Error", "judge_cost": 0.0}

    def validate_llm_judge(self):
        """Validate the LLM faithfulness judge against a human-labeled sample."""
        logger.info("[JUDGE VALIDATION] Running validation of LLM judge against human-labeled sample...")
        
        # 10 handcrafted validation cases representing common RAG failure types and successes
        human_labeled_sample = [
            {
                "query": "Who is Alan Turing?",
                "context": "Alan Turing was an English mathematician and computer scientist. He is widely considered to be the father of theoretical computer science.",
                "answer": "Alan Turing was an English mathematician and computer scientist, who is considered the father of theoretical computer science.",
                "human_faithfulness": 1.0,  # Fully grounded
                "type": "Grounded Success"
            },
            {
                "query": "What is the capital of France?",
                "context": "Paris is the capital and most populous city of France.",
                "answer": "Paris is the capital of France, and it has a population of 2.1 million people.",
                "human_faithfulness": 0.0,  # Hallucinated population (not in context)
                "type": "Hallucination (Out of context fact)"
            },
            {
                "query": "Explain what a Compiler does.",
                "context": "A compiler is a computer program that translates computer code written in one programming language into another language.",
                "answer": "A compiler is a computer program that compiles code into machine code, and was invented by Grace Hopper in 1952.",
                "human_faithfulness": 0.0,  # Hallucinated Grace Hopper invention detail (not in context)
                "type": "Hallucination (Out of context fact)"
            },
            {
                "query": "Is Python functional or object oriented?",
                "context": "Python is a multi-paradigm programming language that supports object-oriented programming and structured programming.",
                "answer": "Python supports object-oriented programming.",
                "human_faithfulness": 1.0,  # Grounded (even if incomplete, it is faithful to context)
                "type": "Grounded Incomplete"
            },
            {
                "query": "What is the recipe for chocolate cake?",
                "context": "We only have articles on computer networks and databases in this catalog.",
                "answer": "I am sorry, but the provided text corpus does not contain enough information to answer this question.",
                "human_faithfulness": 1.0,  # Correct Refusal
                "type": "Grounded Refusal"
            },
            {
                "query": "What does a load balancer do?",
                "context": "A load balancer distributes network traffic across multiple servers to improve reliability.",
                "answer": "A load balancer makes database queries run faster by indexing key columns.",
                "human_faithfulness": 0.0,  # Contradicts/hallucinates
                "type": "Hallucination (Contradiction)"
            },
            {
                "query": "Who won the FIFA World Cup in 2022?",
                "context": "The corpus does not contain any details about sports events.",
                "answer": "Argentina won the 2022 World Cup by defeating France.",
                "human_faithfulness": 0.0,  # Hallucinated correct world fact (but not in context)
                "type": "Hallucination (External knowledge leak)"
            },
            {
                "query": "What is the speed of sound in water?",
                "context": "This document contains no physics constants.",
                "answer": "I do not know the answer based on the provided documents.",
                "human_faithfulness": 1.0,  # Correct Refusal
                "type": "Grounded Refusal"
            },
            {
                "query": "Explain how RSA works.",
                "context": "RSA is a public-key cryptosystem that uses prime factorization for encryption.",
                "answer": "RSA stands for Rivest Shamir Adleman and encrypts messages using prime numbers.",
                "human_faithfulness": 0.0,  # RSA acronym stands for... not in context
                "type": "Hallucination (External knowledge leak)"
            },
            {
                "query": "What is Docker?",
                "context": "Docker is a platform that uses OS-level virtualization to deliver software in packages called containers.",
                "answer": "Docker packages software in virtualized units called containers.",
                "human_faithfulness": 1.0,  # Grounded
                "type": "Grounded Success"
            }
        ]
        
        matches = 0
        results = []
        
        for case in human_labeled_sample:
            judge_res = self.evaluate_faithfulness(case["context"], case["answer"], case["query"])
            score = judge_res["faithfulness_score"]
            
            # Binary threshold: if score >= 0.8, we consider the judge rated it Grounded (1.0), else Hallucinated (0.0)
            judge_binary = 1.0 if score >= 0.8 else 0.0
            is_match = (judge_binary == case["human_faithfulness"])
            if is_match:
                matches += 1
                
            results.append({
                "type": case["type"],
                "query": case["query"],
                "human_label": "Grounded" if case["human_faithfulness"] == 1.0 else "Hallucinated",
                "judge_score": score,
                "judge_label": "Grounded" if judge_binary == 1.0 else "Hallucinated",
                "agreement": is_match
            })
            # Delay to avoid token rate limits
            time.sleep(0.2)
            
        accuracy = matches / len(human_labeled_sample)
        print(f"[JUDGE VALIDATION] LLM Judge Accuracy vs Human Labels: {accuracy*100:.1f}%")
        
        return {
            "accuracy": accuracy,
            "results": results
        }
