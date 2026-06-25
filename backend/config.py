import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
WORKSPACE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = WORKSPACE_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_DIR = WORKSPACE_DIR / "evaluation"
EVAL_DATASET_PATH = EVAL_DIR / "evaluation_dataset.csv"
FEEDBACK_DB_PATH = WORKSPACE_DIR / "feedback.db"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Weaviate Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_COLLECTION = "financial_documents"

# Model configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"  # 1024-dim
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "gemini-2.5-flash"

# Pricing rates per 1M tokens (USD)
# Gemini 2.5 Flash:
# Input: $0.075 / 1M tokens
# Output: $0.30 / 1M tokens
# Local embeddings & reranker: $0.0 (since run locally)
COST_LLM_INPUT = 0.075 / 1_000_000
COST_LLM_OUTPUT = 0.30 / 1_000_000

# Base parameters
DEFAULT_CHUNK_SIZE = 1000  # tokens (approx characters/words mapping)
DEFAULT_CHUNK_OVERLAP = 200

# Retrieval self-adaptation limits (modified by feedback loop)
ADAPTIVE_CONFIG = {
    "normal": {
        "rewrite_query": True,
        "retrieval_depth": 15,    # K chunks for dense & sparse search
        "reranker_top_k": 5,     # K final chunks to pass to LLM
        "reranker_threshold": 0.0, # Cross-encoder threshold
    },
    "adapted": {
        "rewrite_query": True,
        "retrieval_depth": 30,    # Increased depth
        "reranker_top_k": 8,     # Increased context
        "reranker_threshold": -1.0, # Lower threshold to allow more recall
    }
}

import logging
import time

logger = logging.getLogger("financial-rag")

def get_api_key():
    """Retrieve Gemini API Key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add GEMINI_API_KEY to your environment or .env file."
        )
    return api_key

def retry_api_call(func, max_retries=3, initial_delay=1.0, backoff_factor=2.0, *args, **kwargs):
    """Executes an API function and retries on failure with exponential backoff."""
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
                f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s..."
            )
            time.sleep(delay)
            delay *= backoff_factor
    logger.error(f"API call failed after {max_retries} attempts.")
    raise last_error
