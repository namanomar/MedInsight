
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# --- Ollama LLM ---
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL_NAME: str = os.environ.get("LLM_MODEL_NAME", "qwen2.5:4b")
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.5"))
LLM_MAX_NEW_TOKENS: int = int(os.environ.get("LLM_MAX_NEW_TOKENS", "512"))

# --- Embeddings ---
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Vector Store ---
DB_FAISS_PATH: str = os.environ.get("DB_FAISS_PATH", "vectorstore/db_faiss")
RETRIEVER_TOP_K: int = int(os.environ.get("RETRIEVER_TOP_K", "3"))

# --- Data Ingestion ---
DATA_PATH: str = os.environ.get("DATA_PATH", "data/")
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "50"))

# --- Prompt Settings ---
USE_PROMPT_EXAMPLES: bool = os.environ.get("USE_PROMPT_EXAMPLES", "true").lower() == "true"
