import torch
import hashlib
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from backend.config import EMBEDDING_MODEL_NAME, logger

class EmbeddingGenerator:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingGenerator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        if self._initialized:
            return
            
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # We wrap loading in a try-catch with a small timeout or check
        logger.info(f"Attempting to load embedding model '{self.model_name}' on device: {self.device}...")
        try:
            # We set a flags to check if we should bypass loading heavy models
            if os.getenv("BYPASS_HF_MODELS") == "true":
                raise RuntimeError("Bypassing Hugging Face model loading due to environment variables.")
                
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model.max_seq_length = 512
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load sentence-transformers model ({e}). Operating in deterministic Mock Embedding mode (1024-dim).")
            self.model = None
            
        self._initialized = True

    def generate_embeddings(self, texts: Union[str, List[str]], batch_size: int = 32) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return []

        if self.model:
            try:
                embeddings = self.model.encode(
                    texts, 
                    batch_size=batch_size, 
                    show_progress_bar=False, 
                    normalize_embeddings=True
                )
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Error generating embeddings with SentenceTransformer: {e}. Falling back to Mock.")
                
        # Pure-python deterministic mock embeddings
        results = []
        for text in texts:
            # Generate seed based on text MD5 hash to make embeddings deterministic
            h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
            # Clip seed to fit in 32-bit unsigned integer for numpy compatibility
            np.random.seed(h % 4294967295)
            vec = np.random.randn(1024).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec.tolist())
        return results

    def get_embedding_dimension(self) -> int:
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 1024

import os
