
from llm.ollama_client import get_ollama_llm
from memory.vector_store import get_vector_store
from prompts.rag_prompt import get_rag_prompt
from eval.rag_evaluator import RAGEvaluator
from langchain.chains import RetrievalQA
import config


def run_evaluation():
    """Run evaluation on sample questions"""
    
    # Initialize components
    print("[INFO] Initializing RAG system...")
    vector_store = get_vector_store()
    retriever = vector_store.get_retriever(top_k=config.RETRIEVER_TOP_K)
    llm = get_ollama_llm()
    prompt = get_rag_prompt(use_examples=config.USE_PROMPT_EXAMPLES)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    
    # Sample questions for evaluation
    test_questions = [
        "What is hypertension?",
        "What are the symptoms of diabetes?",
        "How is blood pressure measured?",
    ]
    
    print(f"\n[INFO] Running evaluation on {len(test_questions)} questions...\n")
    
    evaluator = RAGEvaluator()
    results = []
    
    for question in test_questions:
        print(f"Question: {question}")
        try:
            response = qa_chain.invoke({"query": question})
            answer = response["result"]
            context = response.get("source_documents", [])
            
            result = evaluator.evaluate_answer(
                question=question,
                answer=answer,
                context=context,
            )
            results.append(result)
            
            print(f"Answer: {answer[:100]}...")
            print(f"Quality: {result.answer_quality}, Relevance: {result.relevance_score:.2f}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Generate report
    print(evaluator.generate_report(results))


if __name__ == "__main__":
    run_evaluation()

