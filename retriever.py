import os
import re
import math
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
import numpy as np

from config import (
    QDRANT_PATH, CORPUS_DIR, EMBEDDING_MODEL, LLM_MODEL,
    CONFIGS, get_api_key, retry_api_call
)
import logging

logger = logging.getLogger("rag-eval-api")

class HybridRetriever:
    def __init__(self, mode="baseline"):
        self.mode = mode
        self.config = CONFIGS[mode]
        self.collection_name = self.config["collection_name"]
        
        # Initialize Qdrant client (local disk-persistent separated by mode)
        db_path = QDRANT_PATH / mode
        db_path.mkdir(parents=True, exist_ok=True)
        self.qdrant_client = QdrantClient(path=str(db_path))
        
        # We will initialize BM25 and chunks dynamically when indexing or loading
        self.chunks = []      # List of dicts: {"id": str, "text": str, "source": str}
        self.bm25 = None      # BM25Okapi instance
        
        # Try to initialize Gemini client
        try:
            api_key = get_api_key()
            self.genai_client = genai.Client(api_key=api_key)
            self.has_api_key = True
        except Exception as e:
            logger.warning(f"Gemini client not initialized in retriever: {e}")
            self.genai_client = None
            self.has_api_key = False

    def clean_text_for_tokenization(self, text):
        """Simple tokenizer for BM25: lowercase and split on words."""
        return re.findall(r'\w+', text.lower())

    def chunk_document(self, text, source_name):
        """Split a document into overlapping chunks on sentence boundaries where possible."""
        chunk_size = self.config["chunk_size"]
        chunk_overlap = self.config["chunk_overlap"]
        
        # Split document by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        chunk_idx = 0
        
        for sentence in sentences:
            # If a single sentence is extremely long, split it by characters
            if len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # Split long sentence into sub-chunks
                for i in range(0, len(sentence), chunk_size - chunk_overlap):
                    chunks.append(sentence[i:i + chunk_size].strip())
                continue
                
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                chunks.append(current_chunk.strip())
                # Handle overlap: take sentences from the end of current_chunk
                # for simple overlap, we can slide backwards by character length
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                # Find start of a sentence in overlap if possible
                sentence_start = overlap_text.find(" ")
                if sentence_start != -1:
                    overlap_text = overlap_text[sentence_start + 1:]
                current_chunk = overlap_text + " " + sentence
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # Format as list of dicts
        formatted_chunks = []
        for idx, chunk_text in enumerate(chunks):
            if len(chunk_text) < 30:  # Skip trivial chunks
                continue
            chunk_id = f"{source_name.replace('.txt', '')}_chunk_{idx}"
            formatted_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "source": source_name
            })
            
        return formatted_chunks

    def build_index(self):
        """Load corpus, chunk files, embed chunks, load Qdrant, and fit BM25."""
        logger.info(f"[{self.mode.upper()} INDEX] Loading corpus from {CORPUS_DIR}...")
        all_chunks = []
        
        # 1. Chunk all documents
        corpus_files = list(CORPUS_DIR.glob("*.txt"))
        if not corpus_files:
            raise FileNotFoundError(f"No files found in corpus directory: {CORPUS_DIR}")
            
        for file_path in corpus_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Extract header info (title & source) and body
            lines = content.split("\n")
            source_name = file_path.name
            body_text = "\n".join(lines[2:]) if len(lines) > 2 else content
            
            doc_chunks = self.chunk_document(body_text, source_name)
            all_chunks.extend(doc_chunks)
            
        self.chunks = all_chunks
        logger.info(f"[{self.mode.upper()} INDEX] Created {len(self.chunks)} chunks from {len(corpus_files)} documents.")
        
        # 2. Build BM25 index
        tokenized_corpus = [self.clean_text_for_tokenization(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"[{self.mode.upper()} INDEX] Fitted BM25 index successfully.")
        
        # 3. Create Qdrant collection
        vector_size = 768  # text-embedding-004 dimensions
        
        # Recreate collection to overwrite old indices
        if self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(self.collection_name)
            
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        # 4. Generate Embeddings and Upload in batches
        batch_size = 32
        points = []
        
        logger.info(f"[{self.mode.upper()} INDEX] Generating dense embeddings using '{EMBEDDING_MODEL}'...")
        
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            batch_texts = [c["text"] for c in batch]
            
            # Embed batch
            if self.has_api_key and self.genai_client:
                try:
                    res = retry_api_call(
                        self.genai_client.models.embed_content,
                        max_retries=3,
                        initial_delay=1.0,
                        model=EMBEDDING_MODEL,
                        contents=batch_texts
                    )
                    # Extract embeddings
                    embeddings = [e.values for e in res.embeddings]
                except Exception as e:
                    logger.error(f"Embedding API failed at batch {i}: {e}. Falling back to mock embeddings.")
                    # Mock embeddings as fallback (random vectors)
                    embeddings = [list(np.random.randn(vector_size)) for _ in batch]
            else:
                # Mock embeddings as fallback
                embeddings = [list(np.random.randn(vector_size)) for _ in batch]
                
            # Create PointStructs for Qdrant
            for idx, (chunk, vector) in enumerate(zip(batch, embeddings)):
                global_idx = i + idx
                points.append(PointStruct(
                    id=global_idx,  # Numeric ID for Qdrant local
                    vector=vector,
                    payload={
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "source": chunk["source"]
                    }
                ))
                
        # Upload points to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"[{self.mode.upper()} INDEX] Uploaded {len(points)} vectors to Qdrant collection '{self.collection_name}'.")

    def dense_search(self, query, top_k):
        """Retrieve top_k chunks using dense vector cosine similarity."""
        if not self.has_api_key or not self.genai_client:
            # Return random hits if no API key
            res = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=top_k,
                with_payload=True
            )
            hits = []
            for point in res[0]:
                hits.append({
                    "id": point.payload["id"],
                    "text": point.payload["text"],
                    "source": point.payload["source"],
                    "score": 0.5
                })
            return hits
            
        try:
            # Embed query
            res = retry_api_call(
                self.genai_client.models.embed_content,
                max_retries=3,
                initial_delay=1.0,
                model=EMBEDDING_MODEL,
                contents=query
            )
            query_vector = res.embeddings[0].values
            
            # Query Qdrant
            hits = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True
            )
            
            results = []
            for hit in hits.points:
                results.append({
                    "id": hit.payload["id"],
                    "text": hit.payload["text"],
                    "source": hit.payload["source"],
                    "score": hit.score
                })
            return results
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []

    def sparse_search(self, query, top_k):
        """Retrieve top_k chunks using BM25 keyword matching."""
        if not self.bm25 or not self.chunks:
            return []
            
        tokenized_query = self.clean_text_for_tokenization(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort and select top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:  # Skip document with zero keyword matches
                continue
            chunk = self.chunks[idx]
            results.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "score": score
            })
        return results

    def reciprocal_rank_fusion(self, dense_results, sparse_results, top_k, rrf_k=60):
        """Combine dense and sparse search results using Reciprocal Rank Fusion (RRF)."""
        rrf_scores = {}
        lookup = {}
        
        # Process dense results
        for rank, hit in enumerate(dense_results):
            cid = hit["id"]
            lookup[cid] = hit
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
            
        # Process sparse results
        for rank, hit in enumerate(sparse_results):
            cid = hit["id"]
            lookup[cid] = hit
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
            
        # Sort by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        fused_results = []
        for rank, cid in enumerate(sorted_cids):
            hit = lookup[cid].copy()
            hit["rrf_score"] = rrf_scores[cid]
            hit["rank"] = rank + 1
            fused_results.append(hit)
            
        return fused_results

    def llm_rerank(self, query, candidates, top_k_final):
        """Re-rank candidate chunks using the LLM as a cross-encoder."""
        if not self.has_api_key or not self.genai_client or not candidates:
            return candidates[:top_k_final]
            
        # Format candidates for LLM prompt
        candidates_str = ""
        for i, cand in enumerate(candidates):
            # Include source and a small snippet or full chunk
            candidates_str += f"--- CHUNK {i} (Source: {cand['source']}) ---\n{cand['text']}\n\n"
            
        prompt = (
            f"You are a highly precise information retrieval re-ranker.\n"
            f"Your task is to re-rank the provided documents based on their relevance to the user query.\n\n"
            f"Query: \"{query}\"\n\n"
            f"Candidates:\n{candidates_str}"
            f"Select the top-{top_k_final} most relevant chunks and output their index numbers in a strict JSON array format,\n"
            f"ordered from most relevant to least relevant. Example output format:\n"
            f"[2, 0, 4, 1, 3]\n"
            f"Do not include any explanation or extra text. Output ONLY the JSON array."
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
            text = response.text.strip()
            # Parse the output array
            indices = json.loads(text)
            
            # Map back to candidates
            reranked = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(candidates):
                    reranked.append(candidates[idx])
            
            # Add back any candidates that might have been skipped by LLM just in case
            for cand in candidates:
                if cand not in reranked:
                    reranked.append(cand)
                    
            return reranked[:top_k_final]
        except Exception as e:
            logger.error(f"LLM Re-ranking failed: {e}. Falling back to RRF order.")
            return candidates[:top_k_final]

    def retrieve(self, query, top_k_override=None):
        """Execute full hybrid retrieval pipeline."""
        # Ensure we have chunks and indexes loaded (lazy loading if build_index wasn't run)
        if not self.chunks:
            # Recreate indices dynamically from Qdrant scroll + rebuild BM25 if already indexed
            try:
                res = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=10000,
                    with_payload=True
                )
                self.chunks = [p.payload for p in res[0]]
                if self.chunks:
                    tokenized_corpus = [self.clean_text_for_tokenization(c["text"]) for c in self.chunks]
                    self.bm25 = BM25Okapi(tokenized_corpus)
            except Exception as e:
                logger.error(f"Failed to scroll Qdrant collection: {e}. Index might be empty, please run index creation.")
                
        if not self.chunks:
            logger.warning("Index is empty. Running build_index() automatically...")
            self.build_index()
            
        if self.mode == "baseline":
            top_k = top_k_override or self.config["top_k"]
            dense = self.dense_search(query, top_k * 2)
            sparse = self.sparse_search(query, top_k * 2)
            fused = self.reciprocal_rank_fusion(dense, sparse, top_k)
            return fused
        else: # improved mode
            top_k_retrieve = self.config["top_k_retrieve"]
            top_k_final = top_k_override or self.config["top_k_final"]
            dense = self.dense_search(query, top_k_retrieve)
            sparse = self.sparse_search(query, top_k_retrieve)
            fused = self.reciprocal_rank_fusion(dense, sparse, top_k_retrieve)
            
            # Apply LLM re-ranking
            reranked = self.llm_rerank(query, fused, top_k_final)
            return reranked
