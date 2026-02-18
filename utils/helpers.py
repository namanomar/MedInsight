
import logging
import os
from typing import List, Optional
from langchain_core.documents import Document


def format_sources(source_docs: List[Document], max_sources: int = 10) -> str:
    """
    Format source documents into a readable string
    
    Args:
        source_docs: List of source documents
        max_sources: Maximum number of sources to display
    
    Returns:
        Formatted string with sources
    """
    if not source_docs:
        return ""
    
    seen = set()
    lines = []
    
    for doc in source_docs[:max_sources]:
        src = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "")
        
        if page != "":
            label = f"{src} — p.{page + 1}"
        else:
            label = src
        
        if label not in seen:
            seen.add(label)
            lines.append(f"- {label}")
    
    if len(source_docs) > max_sources:
        lines.append(f"\n... and {len(source_docs) - max_sources} more sources")
    
    return "\n\n**Sources:**\n" + "\n".join(lines)


def check_ollama_connection(base_url: Optional[str] = None) -> bool:
    """
    Check if Ollama is running and accessible
    
    Args:
        base_url: Ollama base URL (default: http://localhost:11434)
    
    Returns:
        True if Ollama is accessible, False otherwise
    """
    try:
        import requests
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def setup_logging(level: int = logging.INFO) -> None:
    """
    Setup logging configuration
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

