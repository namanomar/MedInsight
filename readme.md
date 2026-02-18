# MedInsight

> AI-powered medical knowledge assistant built on Retrieval-Augmented Generation (RAG).

MedInsight combines a **FAISS vector store** with **Mistral-7B** to deliver grounded,
source-cited answers to medical queries — no hallucination, no guessing.

---

## Architecture

```
User Query
    |
    v
Streamlit Web UI  (main.py)
    |
    v
RetrievalQA Chain  (LangChain)
   /          \
  v            v
FAISS         Mistral-7B
Vector Store  LLM Endpoint
(top-k docs)  (HuggingFace)
    \          /
     v        v
  Formatted Answer + Sources

── Ingestion Pipeline (llm_memory.py) ──────────────────
PDF files → PyPDFLoader → TextSplitter → HF Embeddings
                               → FAISS.save_local()
```

| Layer | Technology |
|---|---|
| Web UI | Streamlit |
| LLM | Mistral-7B-Instruct-v0.3 (HuggingFace Inference) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (CPU) |
| Orchestration | LangChain |
| Config | config.py + .env |

---

## Project Structure

```
MedInsight/
├── config.py                  # All settings in one place
├── main.py                    # Streamlit web application
├── connect_llm_with_memory.py # Interactive CLI chatbot
├── llm_memory.py              # Data ingestion pipeline
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore
├── data/                      # Drop your PDF files here
├── vectorstore/
│   └── db_faiss/              # FAISS index (auto-generated)
└── images/                    # Architecture diagrams
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/namanomar/MedInsight.git
cd MedInsight
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your HuggingFace token

```bash
cp .env.example .env
```

Edit `.env` and paste your token:

```env
HUGGING_FACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get a free token at https://huggingface.co/settings/tokens
Mistral-7B requires a **read** access token and acceptance of the model license.

### 5. Ingest your medical PDFs

Place PDF files into the `data/` directory, then run:

```bash
python llm_memory.py
```

Optional flags:

```bash
python llm_memory.py --data path/to/pdfs --output vectorstore/db_faiss
```

> A pre-built index is already included in `vectorstore/db_faiss/`.
> Skip this step if you want to use the existing index.

### 6. Launch the web app

```bash
streamlit run main.py
```

Open http://localhost:8501 in your browser.

### Alternative: CLI chatbot

```bash
python connect_llm_with_memory.py
```

---

## Configuration

All tuneable settings live in `config.py`:

| Variable | Default | Description |
|---|---|---|
| `LLM_REPO_ID` | mistralai/Mistral-7B-Instruct-v0.3 | HuggingFace model ID |
| `LLM_TEMPERATURE` | 0.5 | Sampling temperature |
| `LLM_MAX_NEW_TOKENS` | 512 | Max tokens to generate |
| `EMBEDDING_MODEL` | sentence-transformers/all-MiniLM-L6-v2 | Embedding model |
| `RETRIEVER_TOP_K` | 3 | Documents retrieved per query |
| `CHUNK_SIZE` | 500 | Characters per text chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `DATA_PATH` | data/ | Directory with source PDFs |
| `DB_FAISS_PATH` | vectorstore/db_faiss | FAISS index location |

---

## Use Cases

- **Medical Q&A** — ask symptom, diagnosis, or treatment questions grounded in your PDFs
- **Clinical Decision Support** — retrieve relevant guidelines and literature fast
- **Research Assistance** — search across large collections of medical papers
- **Document Summarization** — get structured answers from lengthy clinical documents

---

## What Changed (vs. Original)

| Area | Before | After |
|---|---|---|
| Configuration | Hard-coded constants spread across files | Single config.py |
| Dependencies | 4 packages, unpinned | Fully listed with minimum versions |
| Env vars | python-dotenv missing from requirements | Added; .env.example provided |
| LLM caching | Re-initialised on every query | @st.cache_resource for vectorstore and LLM |
| Source display | Same file shown multiple times | Deduplicated with page numbers |
| Ingestion script | Ran on import, no error handling | Proper main(), argparse, path validation |
| CLI | Single-shot, exits after one query | Interactive loop, Ctrl+C / exit to quit |

---

## Roadmap

- [ ] GPU support (faiss-gpu) for faster retrieval at scale
- [ ] Support DOCX, plain text, and web URL ingestion
- [ ] Streaming responses in the Streamlit UI
- [ ] Multi-turn conversation memory (ConversationBufferMemory)
- [ ] Docker image for one-command deployment
- [ ] Automated evaluation with RAGAS

---

## Disclaimer

MedInsight is a research and educational tool.
It is **not a substitute for professional medical advice, diagnosis, or treatment**.
Always consult a qualified healthcare professional.

---

## License

[MIT](LICENSE)
