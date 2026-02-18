
import argparse
import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from memory.vector_store import VectorStore
import config


def load_pdf_files(data_path: str) -> list:
    """Load PDF files from directory"""
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
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Created {len(chunks)} text chunk(s).")
    return chunks


def build_vectorstore(chunks: list, output_path: str) -> None:
    """Build and save FAISS vector store"""
    print(f"[INFO] Loading embedding model '{config.EMBEDDING_MODEL}'...")
    vector_store = VectorStore(db_path=output_path)
    
    print("[INFO] Building FAISS index (this may take a while)...")
    vector_store.save(chunks, output_path)
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
