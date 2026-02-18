# MedInsight 

> AI-powered medical knowledge assistant built on Retrieval-Augmented Generation (RAG).

## 🏗️ Project Structure

```
MedInsight/
├── llm/                    # LLM module
│   ├── __init__.py
│   └── ollama_client.py    # Ollama integration for Qwen 3 4B
│
├── memory/                 # Vector store module
│   ├── __init__.py
│   └── vector_store.py     # FAISS vector store operations
│
├── prompts/                # Prompt templates
│   ├── __init__.py
│   └── rag_prompt.py       # RAG prompts with few-shot examples
│
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── helpers.py          # Helper functions
│
├── eval/                   # Evaluation module
│   ├── __init__.py
│   └── rag_evaluator.py    # RAG system evaluation
│
├── config.py               # Centralized configuration
├── main.py                 # Streamlit web application
├── connect_llm_with_memory.py  # CLI chatbot
├── llm_memory.py          # Data ingestion pipeline
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Ollama

Install Ollama from https://ollama.ai and pull the Qwen 3 4B model:

```bash
ollama pull qwen2.5:4b
```

Ensure Ollama is running:

```bash
ollama serve
```

### 3. Ingest Documents

Place your PDF files in the `data/` directory and run:

```bash
python llm_memory.py
```

### 4. Run the Application

**Web UI:**

```bash
streamlit run main.py
```

**CLI:**

```bash
python connect_llm_with_memory.py
```

## 📋 Configuration

Edit `config.py` or set environment variables:

- `OLLAMA_BASE_URL`: Ollama API URL (default: http://localhost:11434)
- `LLM_MODEL_NAME`: Model name (default: qwen2.5:4b)
- `LLM_TEMPERATURE`: Sampling temperature (default: 0.5)
- `LLM_MAX_NEW_TOKENS`: Max tokens to generate (default: 512)
- `EMBEDDING_MODEL`: Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)
- `RETRIEVER_TOP_K`: Number of documents to retrieve (default: 3)
- `USE_PROMPT_EXAMPLES`: Enable few-shot examples (default: true)

## 🔧 Module Details

### LLM Module (`llm/`)

- **Ollama Integration**: Connects to Ollama API
- **Model**: Qwen 3 4B (qwen2.5:4b)
- **Connection Checking**: Validates Ollama availability

### Memory Module (`memory/`)

- **Vector Store**: FAISS-based document storage
- **Embeddings**: HuggingFace sentence transformers
- **Retrieval**: Top-K similarity search

### Prompts Module (`prompts/`)

- **Templates**: Detailed RAG prompt templates
- **Few-shot Examples**: Includes examples for better understanding
- **Validation**: Input validation for prompts

### Utils Module (`utils/`)

- **Source Formatting**: Formats source citations
- **Connection Checking**: Ollama connection validation
- **Logging**: Setup logging configuration

### Eval Module (`eval/`)

- **Evaluation Metrics**: Relevance scoring, answer quality assessment
- **Reports**: Generate evaluation reports

## 🎯 Features

- ✅ Modern modular architecture
- ✅ Ollama integration with Qwen 3 4B
- ✅ Detailed prompt templates with examples
- ✅ Input validation and error handling
- ✅ Evaluation framework
- ✅ Both web UI and CLI interfaces

## 📝 Notes

- Ensure Ollama is running before starting the application
- The model name should match what you've pulled in Ollama (e.g., `qwen2.5:4b`)
- Prompt examples can be disabled by setting `USE_PROMPT_EXAMPLES=false`
