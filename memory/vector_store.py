
import os
from pathlib import Path
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Import config from parent directory
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import config


class VectorStore:
    """Wrapper for FAISS vector store operations"""
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize Vector Store
        
        Args:
            db_path: Path to FAISS index directory
            embedding_model: Name of embedding model
        """
        self.db_path = db_path or config.DB_FAISS_PATH
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._db: Optional[FAISS] = None
    
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Get or create embeddings instance"""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model
            )
        return self._embeddings
    
    def load(self) -> FAISS:
        """Load FAISS vector store from disk"""
        if self._db is None:
            embeddings = self.get_embeddings()
            db_path = Path(self.db_path)
            
            if not db_path.exists():
                raise FileNotFoundError(
                    f"Vector store not found at '{self.db_path}'. "
                    "Please run the ingestion pipeline first."
                )
            
            self._db = FAISS.load_local(
                str(db_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        return self._db
    
    def get_retriever(self, top_k: Optional[int] = None) -> FAISS:
        """Get retriever with specified top_k"""
        db = self.load()
        top_k = top_k or config.RETRIEVER_TOP_K
        return db.as_retriever(search_kwargs={"k": top_k})
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Search for similar documents"""
        db = self.load()
        top_k = top_k or config.RETRIEVER_TOP_K
        return db.similarity_search(query, k=top_k)
    
    def save(self, documents: List[Document], output_path: Optional[str] = None) -> None:
        """Save documents to FAISS vector store"""
        embeddings = self.get_embeddings()
        output_path = output_path or self.db_path
        
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        if self._db is None:
            self._db = FAISS.from_documents(documents, embeddings)
        else:
            # Add to existing store
            self._db.add_documents(documents)
        
        self._db.save_local(output_path)


def get_vector_store(
    db_path: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> VectorStore:
    """
    Factory function to get Vector Store instance
    
    Args:
        db_path: Path to FAISS index
        embedding_model: Embedding model name
    
    Returns:
        VectorStore instance
    """
    return VectorStore(db_path=db_path, embedding_model=embedding_model)

