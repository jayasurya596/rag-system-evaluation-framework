import re
import hashlib
from typing import List, Dict, Any
from transformers import AutoTokenizer
from backend.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, EMBEDDING_MODEL_NAME, logger

class DocumentChunker:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.warning(f"Could not load Hugging Face tokenizer: {e}. Falling back to word-split approximation.")
            self.tokenizer = None

    def clean_text(self, text: str) -> str:
        """Sanitize raw text from SEC filings and reports."""
        if not text:
            return ""
        # Remove SEC HTML markup if present
        text = re.sub(r'<[^>]*>', ' ', text)
        # Normalize whitespace (replace multiple newlines and spaces with single space/newline)
        text = re.sub(r'\s+', ' ', text)
        # Remove typical SEC page markers and lines of hyphens
        text = re.sub(r'-----+\s*PAGE\s+\d+\s*-----+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'-{3,}', ' ', text)
        return text.strip()

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a document into chunks of self.chunk_size tokens with self.overlap tokens of overlap."""
        content = self.clean_text(doc.get("content", ""))
        if not content:
            return []

        company_name = doc.get("company_name", "Unknown")
        filing_type = doc.get("filing_type", "Unknown")
        filing_date = doc.get("filing_date", "Unknown")
        source_url = doc.get("source_url", "Unknown")
        base_doc_id = doc.get("document_id", "doc")

        chunks = []
        
        if self.tokenizer:
            # Tokenize and chunk using Hugging Face AutoTokenizer
            try:
                tokens = self.tokenizer.encode(content, add_special_tokens=False)
                num_tokens = len(tokens)
                
                if num_tokens <= self.chunk_size:
                    chunk_text = self.tokenizer.decode(tokens)
                    chunk_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()[:12]
                    chunks.append({
                        "document_id": f"{base_doc_id}_c0",
                        "content": chunk_text,
                        "company_name": company_name,
                        "filing_type": filing_type,
                        "filing_date": filing_date,
                        "source_url": source_url,
                        "chunk_index": 0,
                        "token_count": num_tokens
                    })
                else:
                    step = self.chunk_size - self.overlap
                    chunk_idx = 0
                    for i in range(0, num_tokens, step):
                        # Exit if we are near the end and have already covered the text
                        if i > 0 and i + self.overlap >= num_tokens:
                            break
                        
                        chunk_tokens = tokens[i:i + self.chunk_size]
                        chunk_text = self.tokenizer.decode(chunk_tokens)
                        
                        # Only add if we have sufficient content
                        if len(chunk_text.strip()) > 50:
                            chunks.append({
                                "document_id": f"{base_doc_id}_c{chunk_idx}",
                                "content": chunk_text,
                                "company_name": company_name,
                                "filing_type": filing_type,
                                "filing_date": filing_date,
                                "source_url": source_url,
                                "chunk_index": chunk_idx,
                                "token_count": len(chunk_tokens)
                            })
                            chunk_idx += 1
            except Exception as e:
                logger.error(f"HF tokenization failed, falling back to word estimation: {e}")
                self.tokenizer = None

        # Fallback to word split approximation if tokenizer fails or is not loaded
        if not self.tokenizer:
            words = content.split(" ")
            num_words = len(words)
            # Estimate 1 token = 0.75 words, so 1000 tokens ≈ 750 words, 200 tokens ≈ 150 words
            word_chunk_size = int(self.chunk_size * 0.75)
            word_overlap = int(self.overlap * 0.75)
            word_step = word_chunk_size - word_overlap
            
            if num_words <= word_chunk_size:
                chunk_text = " ".join(words)
                chunks.append({
                    "document_id": f"{base_doc_id}_c0",
                    "content": chunk_text,
                    "company_name": company_name,
                    "filing_type": filing_type,
                    "filing_date": filing_date,
                    "source_url": source_url,
                    "chunk_index": 0,
                    "token_count": int(num_words / 0.75)
                })
            else:
                chunk_idx = 0
                for i in range(0, num_words, word_step):
                    if i > 0 and i + word_overlap >= num_words:
                        break
                    
                    chunk_words = words[i:i + word_chunk_size]
                    chunk_text = " ".join(chunk_words)
                    
                    if len(chunk_text.strip()) > 50:
                        chunks.append({
                            "document_id": f"{base_doc_id}_c{chunk_idx}",
                            "content": chunk_text,
                            "company_name": company_name,
                            "filing_type": filing_type,
                            "filing_date": filing_date,
                            "source_url": source_url,
                            "chunk_index": chunk_idx,
                            "token_count": int(len(chunk_words) / 0.75)
                        })
                        chunk_idx += 1
                        
        return chunks
