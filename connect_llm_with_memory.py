"""
MedInsight - Interactive CLI Chatbot

A minimal command-line interface for querying the medical knowledge base.

Usage:
    python connect_llm_with_memory.py
"""
import sys

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

import config


def build_qa_chain() -> RetrievalQA:
    print("[INFO] Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    print("[INFO] Loading FAISS vector store...")
    db = FAISS.load_local(
        config.DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print(f"[INFO] Loading LLM ({config.LLM_REPO_ID})...")
    llm = HuggingFaceEndpoint(
        repo_id=config.LLM_REPO_ID,
        temperature=config.LLM_TEMPERATURE,
        model_kwargs={
            "token": config.HF_TOKEN,
            "max_new_tokens": config.LLM_MAX_NEW_TOKENS,
        },
    )

    prompt = PromptTemplate(
        template=config.PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def main() -> None:
    print("=" * 60)
    print("  MedInsight - Medical Knowledge Chatbot (CLI)")
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
            response = qa_chain.invoke({"query": query})
            print(f"\nAssistant: {response['result']}")

            sources = response.get("source_documents", [])
            if sources:
                seen = set()
                print("\nSources:")
                for doc in sources:
                    label = doc.metadata.get("source", "Unknown")
                    if label not in seen:
                        seen.add(label)
                        print(f"  - {label}")
            print()

        except Exception as exc:
            print(f"[ERROR] {exc}\n")


if __name__ == "__main__":
    main()
