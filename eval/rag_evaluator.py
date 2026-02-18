
from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain_core.documents import Document


@dataclass
class EvaluationResult:
    """Result of a single evaluation"""
    question: str
    expected_answer: Optional[str]
    actual_answer: str
    context_used: List[Document]
    relevance_score: float
    answer_quality: str  # "good", "partial", "poor"
    notes: Optional[str] = None


class RAGEvaluator:
    """Evaluator for RAG system performance"""
    
    def __init__(self):
        """Initialize evaluator"""
        pass
    
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        context: List[Document],
        expected_answer: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single Q&A pair
        
        Args:
            question: User question
            answer: Generated answer
            context: Context documents used
            expected_answer: Expected answer (optional)
        
        Returns:
            EvaluationResult object
        """
        # Simple relevance scoring (can be enhanced with semantic similarity)
        relevance_score = self._calculate_relevance(answer, context)
        
        # Answer quality assessment
        answer_quality = self._assess_answer_quality(answer, context, expected_answer)
        
        return EvaluationResult(
            question=question,
            expected_answer=expected_answer,
            actual_answer=answer,
            context_used=context,
            relevance_score=relevance_score,
            answer_quality=answer_quality,
        )
    
    def _calculate_relevance(self, answer: str, context: List[Document]) -> float:
        """Calculate relevance score between answer and context"""
        if not answer or not context:
            return 0.0
        
        # Simple keyword overlap (can be enhanced with embeddings)
        answer_lower = answer.lower()
        context_text = " ".join([doc.page_content.lower() for doc in context])
        
        answer_words = set(answer_lower.split())
        context_words = set(context_text.split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words & context_words)
        total = len(answer_words)
        
        return min(overlap / total, 1.0) if total > 0 else 0.0
    
    def _assess_answer_quality(
        self,
        answer: str,
        context: List[Document],
        expected: Optional[str] = None,
    ) -> str:
        """Assess answer quality"""
        if not answer or answer.strip() == "":
            return "poor"
        
        # Check for "don't know" responses
        dont_know_phrases = [
            "don't know",
            "don't have information",
            "not in the context",
            "cannot answer",
        ]
        
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in dont_know_phrases):
            if not context:
                return "good"  # Correctly says don't know when no context
            else:
                return "partial"  # Says don't know but has context
        
        # Check if answer seems to use context
        if context and len(answer) > 20:
            return "good"
        
        return "partial"
    
    def generate_report(self, results: List[EvaluationResult]) -> str:
        """
        Generate evaluation report
        
        Args:
            results: List of evaluation results
        
        Returns:
            Formatted report string
        """
        if not results:
            return "No evaluation results available."
        
        total = len(results)
        good = sum(1 for r in results if r.answer_quality == "good")
        partial = sum(1 for r in results if r.answer_quality == "partial")
        poor = sum(1 for r in results if r.answer_quality == "poor")
        
        avg_relevance = sum(r.relevance_score for r in results) / total
        
        report = f"""
RAG System Evaluation Report
{'=' * 50}
Total Questions: {total}
Answer Quality:
  - Good: {good} ({good/total*100:.1f}%)
  - Partial: {partial} ({partial/total*100:.1f}%)
  - Poor: {poor} ({poor/total*100:.1f}%)
Average Relevance Score: {avg_relevance:.2f}
"""
        return report


def evaluate_rag_system(
    questions: List[str],
    answers: List[str],
    contexts: List[List[Document]],
    expected_answers: Optional[List[str]] = None,
) -> str:
    """
    Evaluate RAG system on multiple questions
    
    Args:
        questions: List of questions
        answers: List of generated answers
        contexts: List of context documents for each question
        expected_answers: Optional list of expected answers
    
    Returns:
        Evaluation report string
    """
    evaluator = RAGEvaluator()
    results = []
    
    for i, question in enumerate(questions):
        result = evaluator.evaluate_answer(
            question=question,
            answer=answers[i] if i < len(answers) else "",
            context=contexts[i] if i < len(contexts) else [],
            expected_answer=expected_answers[i] if expected_answers and i < len(expected_answers) else None,
        )
        results.append(result)
    
    return evaluator.generate_report(results)

