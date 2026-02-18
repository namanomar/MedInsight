
import os
from typing import Optional
from langchain_ollama import OllamaLLM as LangChainOllamaLLM
from langchain_core.language_models import BaseLanguageModel

# Import config from parent directory
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import config


class OllamaLLM:
    """Wrapper for Ollama LLM with Qwen 3 4B model"""
    
    def __init__(
        self,
        model_name: str = "qwen2.5:4b",
        base_url: Optional[str] = None,
        temperature: float = 0.5,
        num_predict: int = 512,
    ):
        """
        Initialize Ollama LLM client
        
        Args:
            model_name: Name of the Ollama model (default: qwen2.5:4b)
            base_url: Base URL for Ollama API (default: http://localhost:11434)
            temperature: Sampling temperature (0.0-1.0)
            num_predict: Maximum number of tokens to generate
        """
        self.model_name = model_name
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.temperature = temperature
        self.num_predict = num_predict
        self._llm: Optional[BaseLanguageModel] = None
    
    def get_llm(self) -> BaseLanguageModel:
        """Get or create LangChain Ollama LLM instance"""
        if self._llm is None:
            self._llm = LangChainOllamaLLM(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.num_predict,
            )
        return self._llm
    
    def check_connection(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                return any(self.model_name in name for name in model_names)
            return False
        except Exception:
            return False


def get_ollama_llm(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
) -> BaseLanguageModel:
    """
    Factory function to get Ollama LLM instance
    
    Args:
        model_name: Model name (defaults to config)
        base_url: Ollama base URL (defaults to config)
        temperature: Temperature (defaults to config)
        num_predict: Max tokens (defaults to config)
    
    Returns:
        LangChain Ollama LLM instance
    """
    client = OllamaLLM(
        model_name=model_name or config.LLM_MODEL_NAME,
        base_url=base_url or config.OLLAMA_BASE_URL,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
        num_predict=num_predict or config.LLM_MAX_NEW_TOKENS,
    )
    
    # Check connection
    if not client.check_connection():
        raise ConnectionError(
            f"Ollama is not running or model '{client.model_name}' is not available. "
            f"Please ensure Ollama is running and pull the model: "
            f"ollama pull {client.model_name}"
        )
    
    return client.get_llm()

