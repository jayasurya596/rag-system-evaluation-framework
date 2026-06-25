import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Workspace paths
WORKSPACE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = WORKSPACE_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_DATASET_PATH = DATA_DIR / "eval_dataset.json"
QDRANT_PATH = WORKSPACE_DIR / "qdrant_db"

# Create directories if they don't exist
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_PATH.mkdir(parents=True, exist_ok=True)

# Model configuration
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"

# Pricing rates per token (USD)
# Gemini 2.5 Flash pricing:
# Input: $0.075 per 1 million tokens ($0.000000075 per token)
# Output: $0.30 per 1 million tokens ($0.000000300 per token)
# Text Embedding 004 pricing:
# $0.025 per 1 million tokens ($0.000000025 per token)
COST_LLM_INPUT = 0.075 / 1_000_000
COST_LLM_OUTPUT = 0.30 / 1_000_000
COST_EMBEDDING = 0.025 / 1_000_000

# Pipeline configurations
CONFIGS = {
    "baseline": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "collection_name": "rag_baseline",
        "top_k": 5,
        "use_query_rewrite": False,
        "use_reranker": False,
    },
    "improved": {
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "collection_name": "rag_improved",
        "top_k_retrieve": 15,  # Number of chunks retrieved before re-ranking
        "top_k_final": 5,      # Number of chunks passed to generator after re-ranking
        "use_query_rewrite": True,
        "use_reranker": True,
    }
}

# API Configuration
RATE_LIMIT_TOKENS = 60  # Maximum requests
RATE_LIMIT_WINDOW = 60  # Time window in seconds

import logging
import time

def get_api_key():
    """Retrieve Gemini API Key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set in the environment or in a .env file.\n"
            "Please create a .env file in the workspace directory with:\n"
            "GEMINI_API_KEY=your_gemini_api_key_here"
        )
    return api_key

logger = logging.getLogger("rag-eval-api")

def retry_api_call(func, max_retries=3, initial_delay=1.0, backoff_factor=2.0, *args, **kwargs):
    """Executes a function and retries on failure with exponential backoff."""
    delay = initial_delay
    last_error = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt == max_retries - 1:
                break
            logger.warning(
                f"API call to {func.__name__ if hasattr(func, '__name__') else str(func)} "
                f"failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s..."
            )
            time.sleep(delay)
            delay *= backoff_factor
    
    logger.error(
        f"API call to {func.__name__ if hasattr(func, '__name__') else str(func)} "
        f"failed after {max_retries} attempts."
    )
    raise last_error
