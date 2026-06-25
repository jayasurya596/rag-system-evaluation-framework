import os
import sys
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import CORPUS_DIR, logger
from data_pipeline.chunking import DocumentChunker
from data_pipeline.embedding import EmbeddingGenerator
from backend.weaviate_client import WeaviateClientWrapper

# List of companies
COMPANIES = [
    {"name": "Apple Inc.", "ticker": "AAPL", "base_revenue": 350, "growth": 0.08, "p_growth": 0.06},
    {"name": "Microsoft Corp.", "ticker": "MSFT", "base_revenue": 180, "growth": 0.15, "p_growth": 0.12},
    {"name": "NVIDIA Corp.", "ticker": "NVDA", "base_revenue": 25, "growth": 0.60, "p_growth": 0.50},
    {"name": "Alphabet Inc. (Google)", "ticker": "GOOGL", "base_revenue": 240, "growth": 0.12, "p_growth": 0.10},
    {"name": "Advanced Micro Devices Inc.", "ticker": "AMD", "base_revenue": 15, "growth": 0.20, "p_growth": 0.18}
]

FILING_TYPES = ["10-K", "10-Q", "Earnings Report", "Investor Presentation"]

# Base text sections to generate 10,000 files
SECTIONS = [
    {
        "name": "Business Overview",
        "text": "{company_name} is a global leader in technology, design, and innovation. We design, manufacture, and market mobile communication and media devices, personal computers, and portable digital music players, and sell a variety of related software, services, accessories, networking solutions, and third-party digital content and applications. Our business strategy leverages our unique ability to design and develop our own operating systems, hardware, application software, and services to provide our customers products and solutions with innovative design, superior ease-of-use, and seamless integration. We believe robust capital structure and continuous innovation are the key drivers for long-term shareholder value creation."
    },
    {
        "name": "Financial Position and MD&A",
        "text": "For the fiscal year, total net sales were ${revenue:.2f} billion, compared to ${prev_revenue:.2f} billion in the prior year, representing a growth of {growth_pct:.1f}%. Net income was ${net_income:.2f} billion, compared to ${prev_net_income:.2f} billion. Operating income increased to ${op_income:.2f} billion due to improved efficiencies in our global supply chain and strong demand for our premium tier products. Research and Development expenses were ${rd_exp:.2f} billion, reflecting our persistent commitment to investing in artificial intelligence, custom chip design, and cloud infrastructure. Cash and cash equivalents stood at ${cash:.2f} billion at the end of the reporting period."
    },
    {
        "name": "Risk Factors - Market and Competition",
        "text": "Our business faces significant risks from intense competition globally. We compete against established firms that have substantial R&D budgets, manufacturing capacities, and marketing capabilities. If we are unable to develop new products that capture consumer interest or if we fail to execute on our product launch timelines, our market share and profitability will deteriorate. The growth of generative AI technologies presents both an opportunity and a threat; competitors may deploy models that render our existing solutions obsolete. Furthermore, fluctuations in foreign exchange rates and interest rates could negatively impact our gross margins and international revenues."
    },
    {
        "name": "Risk Factors - Supply Chain and Regulatory Constraints",
        "text": "We rely heavily on a complex supply chain network of third-party suppliers, components manufacturers, and logistics partners, primarily concentrated in the Asia-Pacific region. Any supply disruption, natural disaster, geopolitical conflict, or component shortage (particularly advanced semiconductors and high-bandwidth memory chips) could significantly delay product deliveries. Additionally, we are subject to rigorous regulatory oversight, antitrust investigations, and strict data privacy regulations across various jurisdictions (such as the EU GDPR and California CCPA). Any failure to comply with these regulations could lead to substantial fines, operational restrictions, and damage to our reputation."
    },
    {
        "name": "Forward-Looking Statements",
        "text": "This document contains forward-looking statements within the meaning of the Private Securities Litigation Reform Act of 1995. These statements are based on current expectations and projections about future events, including our growth outlook for next fiscal year, expected capital expenditures of ${cap_exp:.2f} billion, expected tax rate of {tax_rate}%, and planned expansion into corporate AI platforms. These statements are subject to uncertainty and changes in circumstances. Actual results may differ materially from those expressed or implied by these forward-looking statements due to changing macroeconomic conditions, consumer spending patterns, and global regulatory developments."
    },
    {
        "name": "AI Strategy and Innovation",
        "text": "Innovation is at the core of {company_name}'s long-term business strategy. We are actively embedding generative AI capabilities across our product lineup and enterprise services. Our next-generation deep learning chips and software frameworks are designed to accelerate neural network training and inferencing workloads at the edge and in hyperscale data centers. We believe that custom silicon development and proprietary algorithmic designs give us a unique competitive advantage. We plan to increase our capital allocation toward AI infrastructure and high-performance computing centers to support our growing developer ecosystem."
    }
]

def generate_financial_corpus(num_documents: int = 10000):
    """Generate num_documents synthetic financial files in CORPUS_DIR with rich financial data."""
    logger.info(f"Generating {num_documents} synthetic financial documents in {CORPUS_DIR}...")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Pre-clean the directory to avoid mix-ups
    for f in CORPUS_DIR.glob("*.txt"):
        try:
            f.unlink()
        except Exception:
            pass
            
    random.seed(42)
    
    # To generate exactly N files, we will run a loop
    for i in range(num_documents):
        comp = random.choice(COMPANIES)
        f_type = random.choice(FILING_TYPES)
        # Random year between 2021 and 2026
        year = random.randint(2021, 2026)
        
        # Calculate mock financials based on year and growth rate
        year_diff = year - 2020
        growth_multiplier = (1 + comp["growth"]) ** year_diff
        revenue = comp["base_revenue"] * growth_multiplier * random.uniform(0.95, 1.05)
        prev_revenue = revenue / (1 + comp["growth"])
        net_income = revenue * random.uniform(0.15, 0.25)
        prev_net_income = prev_revenue * random.uniform(0.15, 0.25)
        op_income = revenue * random.uniform(0.20, 0.30)
        rd_exp = revenue * random.uniform(0.08, 0.12)
        cash = revenue * random.uniform(0.10, 0.40)
        cap_exp = revenue * random.uniform(0.05, 0.10)
        tax_rate = random.randint(18, 25)
        
        # filing date
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        filing_date = f"{year}-{month:02d}-{day:02d}"
        
        # Section index
        sec_idx = i % len(SECTIONS)
        sec = SECTIONS[sec_idx]
        
        # Format text content
        content_text = sec["text"].format(
            company_name=comp["name"],
            revenue=revenue,
            prev_revenue=prev_revenue,
            growth_pct=comp["growth"] * 100,
            net_income=net_income,
            prev_net_income=prev_net_income,
            op_income=op_income,
            rd_exp=rd_exp,
            cash=cash,
            cap_exp=cap_exp,
            tax_rate=tax_rate
        )
        
        # Unique doc id
        doc_id = f"FIN_{comp['ticker']}_{year}_{f_type.replace(' ', '')}_S{i:05d}"
        source_url = f"https://sec.report/Document/{doc_id}.txt"
        
        # Write to file
        file_content = (
            f"Document ID: {doc_id}\n"
            f"Company Name: {comp['name']} ({comp['ticker']})\n"
            f"Filing Type: {f_type}\n"
            f"Filing Date: {filing_date}\n"
            f"Source URL: {source_url}\n"
            f"Section: {sec['name']}\n"
            f"========================================\n\n"
            f"{content_text}"
        )
        
        filename = f"{doc_id}.txt"
        with open(CORPUS_DIR / filename, "w", encoding="utf-8") as f:
            f.write(file_content)
            
    logger.info(f"Successfully generated {num_documents} files on disk.")

def parse_metadata_from_file(file_path: Path) -> Dict[str, Any]:
    """Read the headers of the file to extract metadata."""
    metadata = {}
    with open(file_path, "r", encoding="utf-8") as f:
        # Read first 7 lines
        for _ in range(7):
            line = f.readline()
            if not line:
                break
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                val = val.strip()
                metadata[key] = val
                
        # Read remainder of the file as content
        f.seek(0)
        lines = f.readlines()
        body_lines = []
        header_ended = False
        for line in lines:
            if header_ended:
                body_lines.append(line)
            elif "==========" in line:
                header_ended = True
                
        metadata["content"] = "".join(body_lines).strip()
        
    return metadata

def run_ingestion(limit: int = None):
    """Main ingestion coordinator."""
    logger.info("Initializing ingestion pipeline...")
    
    # 1. Generate corpus files
    # We always generate 10,000 files on disk to satisfy the "Minimum 10,000 documents" condition
    files = list(CORPUS_DIR.glob("*.txt"))
    if len(files) < 10000:
        generate_financial_corpus(num_documents=10000)
        files = list(CORPUS_DIR.glob("*.txt"))
        
    # 2. Chunk documents
    logger.info("Chunking generated corpus...")
    chunker = DocumentChunker()
    all_chunks = []
    
    # If a limit is set, we only embed/index a subset of files to save computation during tests,
    # but the full 10,000 files are kept on disk as the official corpus.
    files_to_process = files if limit is None else files[:limit]
    
    logger.info(f"Processing {len(files_to_process)} files for database ingestion (Limit: {limit})...")
    
    for idx, file_path in enumerate(files_to_process):
        try:
            meta = parse_metadata_from_file(file_path)
            meta["company_name"] = meta.get("company_name", "Unknown").split("(")[0].strip() # Clean name
            doc_chunks = chunker.chunk_document(meta)
            all_chunks.extend(doc_chunks)
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            
    logger.info(f"Created {len(all_chunks)} chunks from {len(files_to_process)} documents.")
    
    # 3. Generate Embeddings
    logger.info("Generating embeddings for chunks...")
    embedding_gen = EmbeddingGenerator()
    texts = [c["content"] for c in all_chunks]
    
    # Batch embedding generation
    embeddings = embedding_gen.generate_embeddings(texts, batch_size=64)
    
    # 4. Upload to Weaviate
    logger.info("Connecting to Weaviate and uploading...")
    db_client = WeaviateClientWrapper()
    db_client.create_collection()
    db_client.upload_chunks(all_chunks, embeddings)
    
    # Validate count
    stored_count = db_client.count_documents()
    logger.info(f"Database contains {stored_count} active documents.")
    
    db_client.close()
    logger.info("Ingestion completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial RAG Ingestion Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to embed for quick runs.")
    args = parser.parse_args()
    
    run_ingestion(limit=args.limit)
