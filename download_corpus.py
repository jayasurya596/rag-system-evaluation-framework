import os
import re
import time
import requests
from urllib.parse import quote
from config import CORPUS_DIR

# 110 high-quality, highly technical Computer Science and Machine Learning Wikipedia articles
ARTICLES = [
    # Machine Learning & AI
    "Artificial intelligence", "Machine learning", "Deep learning", "Neural network", 
    "Natural language processing", "Transformer (deep learning architecture)", "Information retrieval", 
    "Vector database", "Support vector machine", "Decision tree", "Random forest", 
    "Gradient boosting", "Linear regression", "Logistic regression", "K-means clustering", 
    "Principal component analysis", "Reinforcement learning", "Q-learning", "Markov decision process", 
    "Generative adversarial network", "Large language model", "Prompt engineering", "Semantic search", 
    "Cosine similarity", "Tf-idf", "Word embedding", "Attention (machine learning)", "Recurrent neural network",
    "Convolutional neural network", "Overfitting", "Regularization (mathematics)", "Supervised learning",
    "Unsupervised learning", "Semi-supervised learning", "Active learning (machine learning)",
    
    # Foundational Theory
    "Information theory", "Claude Shannon", "Alan Turing", "Turing machine", "Universal Turing machine", 
    "Computability theory", "Complexity theory", "P versus NP problem", "Halting problem", 
    "Automata theory", "Formal language", "Chomsky hierarchy", "Recursion (computer science)",
    "Algorithm", "Big O notation", "Sorting algorithm", "Search algorithm", "Graph theory",
    
    # Systems, Programming & Languages
    "Compiler", "Interpreter (computing)", "Operating system", "Linux", "Unix", "Virtual machine", 
    "Docker (software)", "Kubernetes", "Microservices", "Object-oriented programming", 
    "Functional programming", "Procedural programming", "Software engineering", "Agile software development", 
    "Scrum (software development)", "Git", "GitHub", "Continuous integration", "Unit testing", 
    "Software design pattern", "Model-view-controller", "Singleton pattern", "Observer pattern", 
    "Factory method pattern", "Clean code", "Refactoring", "Code smell", "Technical debt", 
    "Application programming interface", "Web scraping", "Garbage collection (computer science)",
    
    # Databases & Networking
    "Database", "Relational database", "SQL", "NoSQL", "Graph database", "Data warehouse", 
    "Data lake", "Data pipeline", "Extract, transform, load", "Big data", "Apache Spark", 
    "Apache Hadoop", "MapReduce", "Caching", "Redis", "Memcached", "Computer network", 
    "Internet protocol suite", "Transmission Control Protocol", "User Datagram Protocol", 
    "Hypertext Transfer Protocol", "Domain Name System", "Network socket", "Load balancing (computing)",
    
    # Cryptography & Security
    "Cryptography", "Public-key cryptography", "Symmetric-key algorithm", "Advanced Encryption Standard", 
    "RSA (cryptosystem)", "Diffie-Hellman key exchange", "Blockchain", "Bitcoin", "Ethereum", 
    "Smart contract", "Computer security", "Transport Layer Security"
]

def sanitize_filename(name):
    """Sanitize the article title to create a valid filename."""
    return re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")

def clean_wiki_text(text):
    """Remove references, further reading, and external links from Wikipedia text."""
    # Find and truncate standard trailing sections
    patterns = [
        r"\n==\s*See also\s*==.*",
        r"\n==\s*References\s*==.*",
        r"\n==\s*Further reading\s*==.*",
        r"\n==\s*External links\s*==.*"
    ]
    for pattern in patterns:
        text = re.split(pattern, text, flags=re.DOTALL | re.IGNORECASE)[0]
    
    # Clean multiple consecutive newlines and leading/trailing whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def download_article(title):
    """Fetch text of a Wikipedia article via API."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
        "redirects": 1
    }
    
    headers = {
        "User-Agent": "RAG-Eval-System-Agent/1.0 (jayashurya.chpj@gmail.com)"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                print(f"[-] Article not found: '{title}'")
                return False
            
            raw_text = page_data.get("extract", "")
            if not raw_text or len(raw_text) < 100:
                print(f"[-] Empty or too short extract for: '{title}'")
                return False
            
            cleaned_text = clean_wiki_text(raw_text)
            
            # Format text file content with title on first line
            content = f"Title: {title}\nSource: https://en.wikipedia.org/wiki/{quote(title)}\n\n{cleaned_text}"
            
            filename = f"{sanitize_filename(title)}.txt"
            filepath = CORPUS_DIR / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"[+] Downloaded and saved: '{title}' ({len(cleaned_text)} chars)")
            return True
            
    except Exception as e:
        print(f"[-] Error downloading '{title}': {e}")
        return False

def main():
    print(f"Starting corpus download of {len(ARTICLES)} articles to {CORPUS_DIR}...")
    success_count = 0
    
    for i, title in enumerate(ARTICLES):
        print(f"[{i+1}/{len(ARTICLES)}] Fetching '{title}'...")
        success = download_article(title)
        if success:
            success_count += 1
        # Be gentle to Wikipedia servers
        time.sleep(0.2)
        
    print(f"\nDownload summary: Successfully downloaded {success_count} / {len(ARTICLES)} articles.")
    
    # Double check actual file count in directory
    files = list(CORPUS_DIR.glob("*.txt"))
    print(f"Total text files in corpus directory: {len(files)}")
    if len(files) < 100:
        print("[WARNING] Sizable document corpus requires at least 100+ documents. Current: ", len(files))

if __name__ == "__main__":
    main()
