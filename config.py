"""
MedInsight - Centralized Configuration
"""
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# --- HuggingFace ---
HF_TOKEN: str = os.environ.get("HUGGING_FACE_TOKEN", "")

# --- Model ---
LLM_REPO_ID: str = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_TEMPERATURE: float = 0.5
LLM_MAX_NEW_TOKENS: int = 512

# --- Embeddings ---
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# --- Vector Store ---
DB_FAISS_PATH: str = "vectorstore/db_faiss"
RETRIEVER_TOP_K: int = 3

# --- Data Ingestion ---
DATA_PATH: str = "data/"
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# --- Prompt ---
PROMPT_TEMPLATE: str = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Only provide information from the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""
