"""
RAG Prompt Templates with Examples and Validation
Provides detailed prompt templates with few-shot examples
"""
from typing import Dict, List, Optional, Tuple
from langchain_core.prompts import PromptTemplate
import re

# Import config from parent directory
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import config


class RAGPrompt:
    """RAG Prompt Template with validation and examples"""
    
    # Few-shot examples for better prompt understanding
    EXAMPLES = [
        {
            "context": "Hypertension, also known as high blood pressure, is a condition where the force of blood against artery walls is too high. Normal blood pressure is below 120/80 mmHg.",
            "question": "What is hypertension?",
            "answer": "Hypertension, also known as high blood pressure, is a condition where the force of blood against artery walls is too high. Normal blood pressure is below 120/80 mmHg."
        },
        {
            "context": "Diabetes is a chronic condition that affects how your body processes blood sugar. Type 1 diabetes occurs when the pancreas produces little or no insulin.",
            "question": "What causes Type 1 diabetes?",
            "answer": "Type 1 diabetes occurs when the pancreas produces little or no insulin."
        },
        {
            "context": "The information provided does not contain details about this specific condition.",
            "question": "What is the treatment for rare disease X?",
            "answer": "I don't have information about the treatment for rare disease X in the provided context. Please consult a qualified healthcare professional for accurate medical advice."
        }
    ]
    
    BASE_TEMPLATE = """You are a medical knowledge assistant. Your role is to provide accurate, evidence-based answers using only the information provided in the context.

Guidelines:
1. Use ONLY the information from the provided context to answer the question
2. If the context doesn't contain enough information, clearly state that you don't know
3. Do NOT make up or infer information not present in the context
4. Be concise but complete in your answers
5. Always prioritize patient safety - recommend consulting healthcare professionals for serious concerns

Context: {context}

Question: {question}

Answer:"""
    
    TEMPLATE_WITH_EXAMPLES = """You are a medical knowledge assistant. Your role is to provide accurate, evidence-based answers using only the information provided in the context.

Guidelines:
1. Use ONLY the information from the provided context to answer the question
2. If the context doesn't contain enough information, clearly state that you don't know
3. Do NOT make up or infer information not present in the context
4. Be concise but complete in your answers
5. Always prioritize patient safety - recommend consulting healthcare professionals for serious concerns

Examples:

Example 1:
Context: {example1_context}
Question: {example1_question}
Answer: {example1_answer}

Example 2:
Context: {example2_context}
Question: {example2_question}
Answer: {example2_answer}

Example 3:
Context: {example3_context}
Question: {example3_question}
Answer: {example3_answer}

Now answer the following question:

Context: {context}
Question: {question}
Answer:"""
    
    def __init__(self, use_examples: bool = True):
        """
        Initialize RAG Prompt
        
        Args:
            use_examples: Whether to include few-shot examples in prompt
        """
        self.use_examples = use_examples
        self._template: Optional[PromptTemplate] = None
    
    def get_template(self) -> PromptTemplate:
        """Get prompt template with or without examples"""
        if self._template is None:
            if self.use_examples:
                template_str = self.TEMPLATE_WITH_EXAMPLES
                input_variables = [
                    "context", "question",
                    "example1_context", "example1_question", "example1_answer",
                    "example2_context", "example2_question", "example2_answer",
                    "example3_context", "example3_question", "example3_answer",
                ]
            else:
                template_str = self.BASE_TEMPLATE
                input_variables = ["context", "question"]
            
            self._template = PromptTemplate(
                template=template_str,
                input_variables=input_variables,
            )
        return self._template
    
    def format_prompt(self, context: str, question: str) -> str:
        """Format prompt with context and question"""
        if self.use_examples:
            return self.get_template().format(
                context=context,
                question=question,
                example1_context=self.EXAMPLES[0]["context"],
                example1_question=self.EXAMPLES[0]["question"],
                example1_answer=self.EXAMPLES[0]["answer"],
                example2_context=self.EXAMPLES[1]["context"],
                example2_question=self.EXAMPLES[1]["question"],
                example2_answer=self.EXAMPLES[1]["answer"],
                example3_context=self.EXAMPLES[2]["context"],
                example3_question=self.EXAMPLES[2]["question"],
                example3_answer=self.EXAMPLES[2]["answer"],
            )
        else:
            return self.get_template().format(
                context=context,
                question=question,
            )
    
    def validate_inputs(self, context: str, question: str) -> Tuple[bool, Optional[str]]:
        """
        Validate prompt inputs
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not question or not question.strip():
            return False, "Question cannot be empty"
        
        if not context or not context.strip():
            return False, "Context cannot be empty"
        
        if len(question) > 1000:
            return False, "Question is too long (max 1000 characters)"
        
        if len(context) > 10000:
            return False, "Context is too long (max 10000 characters)"
        
        return True, None


def get_rag_prompt(use_examples: bool = True) -> PromptTemplate:
    """
    Factory function to get RAG prompt template
    
    Args:
        use_examples: Whether to include few-shot examples
    
    Returns:
        PromptTemplate instance
    """
    prompt = RAGPrompt(use_examples=use_examples)
    return prompt.get_template()


def validate_prompt(context: str, question: str) -> Tuple[bool, Optional[str]]:
    """
    Validate prompt inputs
    
    Args:
        context: Context string
        question: Question string
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    prompt = RAGPrompt()
    return prompt.validate_inputs(context, question)

