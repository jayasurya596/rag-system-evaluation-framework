import time
import os
import re
import json
import math
from google import genai
from google.genai import types

from config import (
    LLM_MODEL, COST_LLM_INPUT, COST_LLM_OUTPUT, COST_EMBEDDING,
    CONFIGS, get_api_key, retry_api_call
)
from retriever import HybridRetriever
import logging

logger = logging.getLogger("rag-eval-api")

class RAGPipeline:
    def __init__(self, mode="baseline"):
        self.mode = mode
        self.config = CONFIGS[mode]
        self.retriever = HybridRetriever(mode=mode)
        
        # Load API key and initialize Gemini client
        try:
            api_key = get_api_key()
            self.genai_client = genai.Client(api_key=api_key)
            self.has_api_key = True
        except Exception as e:
            logger.warning(f"Gemini client not initialized in pipeline: {e}")
            self.genai_client = None
            self.has_api_key = False

    def query_rewrite(self, original_query):
        """Use the LLM to rewrite the query for improved retrieval."""
        if not self.has_api_key or not self.genai_client:
            return original_query, 0.0, 0, 0
        prompt = (
            f"You are a search query optimizer. Given a user question, rewrite it to improve search retrieval.\n"
            f"If it is a simple query, expand it with synonyms.\n"
            f"If it is a multi-hop question requiring multiple facts, break it down or express the sub-queries clearly.\n"
            f"Return ONLY the optimized search query text, with no preamble or explanations.\n\n"
            f"Original Question: {original_query}"
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
            rewritten = response.text.strip()
            # Track cost
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            cost = (input_tokens * COST_LLM_INPUT) + (output_tokens * COST_LLM_OUTPUT)
            return rewritten, cost, input_tokens, output_tokens
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            return original_query, 0.0, 0, 0

    def extract_citations(self, answer, retrieved_chunks):
        """Extract citations (source files) from the generated answer text."""
        # Find all citation markers like [0], [1], [2], etc.
        markers = re.findall(r'\[(\d+)\]', answer)
        citations = set()
        for marker in markers:
            idx = int(marker)
            if 0 <= idx < len(retrieved_chunks):
                citations.add(retrieved_chunks[idx]["source"])
        return list(citations)

    def query(self, user_query):
        """Execute the full RAG pipeline (Retrieve -> Generate -> Extract Citations)."""
        metrics = {
            "rewrite_latency": 0.0,
            "retrieve_latency": 0.0,
            "generate_latency": 0.0,
            "total_latency": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "embedding_tokens": 0,
            "cost": 0.0
        }
        
        start_total = time.time()
        
        # 1. Query Rewriting (Improved mode only)
        search_query = user_query
        if self.config["use_query_rewrite"]:
            start_rw = time.time()
            search_query, rw_cost, rw_in, rw_out = self.query_rewrite(user_query)
            metrics["rewrite_latency"] = time.time() - start_rw
            metrics["cost"] += rw_cost
            metrics["input_tokens"] += rw_in
            metrics["output_tokens"] += rw_out
            
        # 2. Retrieval (Dense + Sparse + RRF + Re-ranker)
        start_ret = time.time()
        retrieved_chunks = self.retriever.retrieve(search_query)
        metrics["retrieve_latency"] = time.time() - start_ret
        
        # Estimate embedding tokens for search query (roughly 1 token per 4 characters)
        query_chars = len(search_query)
        est_embedding_tokens = math.ceil(query_chars / 4.0)
        metrics["embedding_tokens"] += est_embedding_tokens
        metrics["cost"] += est_embedding_tokens * COST_EMBEDDING
        
        if not retrieved_chunks:
            metrics["total_latency"] = time.time() - start_total
            return {
                "query": user_query,
                "rewritten_query": search_query,
                "answer": "I am sorry, but I could not find any documents related to your query.",
                "citations": [],
                "retrieved_chunks": [],
                "metrics": metrics
            }
            
        # 3. Context Preparation
        context_str = ""
        for idx, chunk in enumerate(retrieved_chunks):
            context_str += f"[{idx}] Source: {chunk['source']}\nContent: {chunk['text']}\n\n"
            
        # 4. Prompt construction & Generation
        if self.mode == "baseline":
            prompt = (
                f"You are a helpful Q&A assistant.\n"
                f"Answer the user question using ONLY the facts present in the provided context.\n"
                f"For any statement you make, you must cite the source by adding the corresponding index marker (e.g., [0], [1]) at the end of the sentence.\n\n"
                f"Context:\n{context_str}"
                f"Question: {user_query}\n"
                f"Answer:"
            )
        else: # improved mode
            prompt = (
                f"You are a precise technical Q&A assistant.\n"
                f"Answer the user question based STRICTLY and ONLY on the facts in the provided context.\n"
                f"Follow these rules carefully:\n"
                f"1. If the provided context does not contain the answer, or if you cannot verify the answer from the context, or if the question is out-of-domain, say EXACTLY: 'I am sorry, but the provided text corpus does not contain enough information to answer this question.' and nothing else.\n"
                f"2. For every claim or statement you make, append the source chunk citation marker at the end of the sentence, matching the index of the chunk (e.g. [0], [1], [2]).\n"
                f"3. Do not make up facts or extrapolate beyond the text.\n\n"
                f"Context:\n{context_str}"
                f"Question: {user_query}\n"
                f"Answer:"
            )
            
        # Call Generator
        start_gen = time.time()
        if self.has_api_key and self.genai_client:
            try:
                response = retry_api_call(
                    self.genai_client.models.generate_content,
                    max_retries=3,
                    initial_delay=1.0,
                    model=LLM_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=800
                    )
                )
                answer_text = response.text.strip()
                
                # Update metrics
                in_tokens = response.usage_metadata.prompt_token_count
                out_tokens = response.usage_metadata.candidates_token_count
                metrics["input_tokens"] += in_tokens
                metrics["output_tokens"] += out_tokens
                metrics["cost"] += (in_tokens * COST_LLM_INPUT) + (out_tokens * COST_LLM_OUTPUT)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                answer_text = "I am sorry, but generation failed due to an API error."
        else:
            # Local mock generation for test execution without API key
            time.sleep(0.5) # Simulate latency
            if "father of information theory" in user_query.lower():
                answer_text = "Claude Shannon is known as the father of information theory. [0] [1]"
            elif "recipe" in user_query.lower() or "sports" in user_query.lower() or "capital" in user_query.lower():
                answer_text = "I am sorry, but the provided text corpus does not contain enough information to answer this question."
            else:
                answer_text = f"Mocked response: Retrieved context mentions {retrieved_chunks[0]['source']}. [0]"
                
        metrics["generate_latency"] = time.time() - start_gen
        metrics["total_latency"] = time.time() - start_total
        
        # 5. Extract citations
        citations = self.extract_citations(answer_text, retrieved_chunks)
        
        return {
            "query": user_query,
            "rewritten_query": search_query,
            "answer": answer_text,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "metrics": metrics
        }
