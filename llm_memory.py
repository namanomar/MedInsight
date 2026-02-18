"""
MedInsight - Data Ingestion Pipeline

Loads PDF documents from the data/ directory, splits them into chunks,
embeds them, and stores the resulting FAISS index in vectorstore/db_faiss/.

Usage:
    python llm_memory.py
    python llm_memory.py --data path/to/pdfs --output vectorstore/db_faiss
"""
import argparse
import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import config


def load_pdf_files(data_path: str) -> list:
    path = Path(data_path)
    if not path.exists():
        print(f"[ERROR] Data directory '{data_path}' does not exist.")
        sys.exit(1)

    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        print(f"[ERROR] No PDF files found in '{data_path}'.")
        sys.exit(1)

    print(f"[INFO] Loaded {len(documents)} page(s) from PDFs in '{data_path}'.")
    return documents


def create_chunks(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Created {len(chunks)} text chunk(s).")
    return chunks


def build_vectorstore(chunks: list, output_path: str) -> None:
    print(f"[INFO] Loading embedding model '{config.EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    print("[INFO] Building FAISS index (this may take a while)...")
    db = FAISS.from_documents(chunks, embeddings)

    Path(output_path).mkdir(parents=True, exist_ok=True)
    db.save_local(output_path)
    print(f"[INFO] Vector store saved to '{output_path}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="MedInsight ingestion pipeline")
    parser.add_argument("--data", default=config.DATA_PATH, help="Directory containing PDF files")
    parser.add_argument("--output", default=config.DB_FAISS_PATH, help="Output path for FAISS index")
    args = parser.parse_args()

    documents = load_pdf_files(args.data)
    chunks = create_chunks(documents)
    build_vectorstore(chunks, args.output)
    print("[DONE] Ingestion complete. You can now run the chatbot.")


if __name__ == "__main__":
    main()
