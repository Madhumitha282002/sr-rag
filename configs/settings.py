from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_PAPERS_DIR = DATA_DIR / "raw_papers"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
LOGS_DIR = BASE_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Chunking defaults (tune in Week 1 Day 4)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Retrieval defaults
DEFAULT_TOP_K = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "sr_papers"
PAPER_METADATA_CSV = str(PROCESSED_DIR / "paper_metadata.csv")
