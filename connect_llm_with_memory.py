
import sys

from langchain.chains import RetrievalQA

from llm.ollama_client import get_ollama_llm
from memory.vector_store import get_vector_store
from prompts.rag_prompt import get_rag_prompt, validate_prompt
from utils.helpers import format_sources, check_ollama_connection, setup_logging
import config

# Setup logging
setup_logging()


def build_qa_chain() -> RetrievalQA:
    """Build RAG QA chain with Ollama and vector store"""
    print("[INFO] Checking Ollama connection...")
    if not check_ollama_connection(config.OLLAMA_BASE_URL):
        print(f"[ERROR] Cannot connect to Ollama at {config.OLLAMA_BASE_URL}")
        print("[INFO] Please ensure Ollama is running and the model is pulled:")
        print(f"      ollama pull {config.LLM_MODEL_NAME}")
        sys.exit(1)
    
    print("[INFO] Loading embeddings...")
    vector_store = get_vector_store()
    
    print("[INFO] Loading FAISS vector store...")
    retriever = vector_store.get_retriever(top_k=config.RETRIEVER_TOP_K)
    
    print(f"[INFO] Loading LLM ({config.LLM_MODEL_NAME})...")
    llm = get_ollama_llm()
    
    print("[INFO] Loading prompt template...")
    prompt = get_rag_prompt(use_examples=config.USE_PROMPT_EXAMPLES)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def main() -> None:
    print("=" * 60)
    print("  MedInsight - Medical Knowledge Chatbot (CLI)")
    print(f"  Model: {config.LLM_MODEL_NAME} via Ollama")
    print("  Type 'exit' or press Ctrl+C to quit.")
    print("=" * 60)

    try:
        qa_chain = build_qa_chain()
    except Exception as exc:
        print(f"[ERROR] Failed to initialise: {exc}")
        sys.exit(1)

    print("\n[READY] Ask your medical questions below.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:
            # Validate query
            if len(query) > 1000:
                print("[WARNING] Query is too long. Please keep it under 1000 characters.\n")
                continue
            
            response = qa_chain.invoke({"query": query})
            answer = response["result"]
            
            print(f"\nAssistant: {answer}")
            
            sources = response.get("source_documents", [])
            if sources:
                sources_text = format_sources(sources)
                print(sources_text)
            print()

        except Exception as exc:
            print(f"[ERROR] {exc}\n")


if __name__ == "__main__":
    main()
