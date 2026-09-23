import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Google GenAI Model Names
# Using current official model names (gemini-3.6-flash is the latest recommended model)
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
FALLBACK_GEMINI_MODELS = ["gemini-3.6-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
EMBEDDING_DIMENSION = 768


# Chunking Configuration
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval Configuration
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))

# Relevance Threshold:
# We use Cosine Similarity computed via Inner Product on L2-normalized vectors.
# Range is [-1.0, 1.0]. A score >= 0.38 indicates strong semantic alignment for HR policy chunks.
# Queries scoring below this threshold trigger the grounding fallback mechanism.
DEFAULT_RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.38"))

# Grounding exact fallback message as required by specification
GROUNDING_FALLBACK_MESSAGE = (
    "I couldn't find enough information in the provided company documents to answer this question."
)

# File Paths
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_documents"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
DOCS_DIR = BASE_DIR / "docs"

# Persistence paths
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_STORE_PATH = VECTOR_STORE_DIR / "metadata.json"

# Ensure essential directories exist
for directory in [DATA_DIR, SAMPLE_DOCS_DIR, UPLOADS_DIR, VECTOR_STORE_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
