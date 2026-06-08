# SR-RAG — Super-Resolution Research Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** system for searching, comparing, and citing image super-resolution research papers using natural language.

[![CI](https://github.com/YOUR_USERNAME/sr-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/sr-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Demo

![SR-RAG Streamlit UI](docs/figures/app_screenshot_v1.png)

**Ask natural language questions across 10 landmark SR papers and get grounded, cited answers:**

> *"What loss function does SRGAN use?"*
> → SRGAN uses a perceptual loss function combining adversarial loss and content loss [1]. The adversarial component pushes outputs toward the natural image manifold using a discriminator [1].

---

## What it does

- **Natural language search** across SRCNN, VDSR, SRGAN, EDSR, RCAN, ESRGAN, RDN, RealESRGAN, SwinIR, and HAT
- **Grounded answers** — every claim is cited with paper name, page number, and similarity score
- **Refuses to answer** when the corpus lacks sufficient evidence
- **Reranking** — optional cross-encoder reranker improves precision
- **REST API** (FastAPI) + **interactive UI** (Streamlit) + **Docker** deployment

---

## Architecture

```
PDF corpus
    │
    ▼
PDF Extraction (PyMuPDF)
    │
    ▼
Chunking (LangChain RecursiveCharacterTextSplitter)
    │
    ▼
Embeddings (sentence-transformers/all-MiniLM-L6-v2)
    │
    ▼
Vector Store (ChromaDB — cosine similarity)
    │
    ▼
Query ──► Dense Retrieval ──► [Reranker] ──► Prompt Builder ──► LLM ──► Cited Answer
```

**Stack:** Python 3.11 · sentence-transformers · ChromaDB · LangChain · FastAPI · Streamlit · Docker

---

## Corpus

| Method | Year | Venue | Key Contribution |
|--------|------|-------|-----------------|
| SRCNN | 2014 | ECCV | First end-to-end CNN for SR |
| VDSR | 2015 | CVPR | Very deep residual network |
| SRGAN / SRResNet | 2016 | CVPR | GAN + perceptual loss for photo-realistic SR |
| EDSR | 2017 | CVPRW | Removes batch norm, NTIRE 2017 winner |
| RCAN | 2018 | ECCV | Residual channel attention |
| ESRGAN | 2018 | ECCVW | RRDB + relativistic discriminator |
| RDN | 2018 | CVPR | Residual dense connections |
| RealESRGAN | 2021 | ICCVW | Practical blind SR with synthetic degradation |
| SwinIR | 2021 | ICCVW | First transformer backbone for SR |
| HAT | 2022 | CVPR | Hybrid attention transformer, SOTA at release |

---

## Evaluation

Evaluated on a hand-labeled set of 30 stratified questions (10 easy, 10 medium, 5 hard, 5 unanswerable).

| Metric | Score |
|--------|-------|
| Recall@5 | see `data/processed/eval_report.json` |
| MRR | see `data/processed/eval_report.json` |
| Hit@1 | see `data/processed/eval_report.json` |
| Refusal accuracy | see `data/processed/eval_report.json` |

Run the evaluation yourself:
```bash
python scripts/run_evaluation.py
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- Docker (for containerised deployment)

### Local setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/sr-rag.git
cd sr-rag

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env — set LLM_PROVIDER=mock for testing (no API key needed)
# Set LLM_PROVIDER=openai and OPENAI_API_KEY=sk-... for real answers

# 5. Download papers and build the index
python scripts/download_papers.py
python scripts/build_metadata.py
```

Then open and run these notebooks in order:
```
notebooks/01_pdf_extraction.ipynb   → extracts text from PDFs
notebooks/02_chunking.ipynb          → chunks text, saves chunks.pkl
notebooks/03_embeddings.ipynb        → embeds chunks, builds ChromaDB index
```

### Run the apps

```bash
# Streamlit UI
streamlit run app/streamlit_app.py

# FastAPI (Swagger at http://localhost:8000/docs)
uvicorn api.main:app --reload --port 8000
```

### Docker

```bash
# Build
docker build -t sr-rag .

# Run API
docker run -p 8000:8000 \
  -v $(pwd)/vector_store:/app/vector_store \
  -v $(pwd)/data/processed:/app/data/processed:ro \
  -v $(pwd)/logs:/app/logs \
  sr-rag api

# Run both services
docker compose up --build
# API: http://localhost:8000/docs
# UI:  http://localhost:8501
```

---

## API Reference

### `POST /query`
```json
{
  "question": "What loss function does SRGAN use?",
  "top_k": 5,
  "use_reranker": false,
  "prompt_template": "v2",
  "method_filter": null,
  "year_from": null,
  "year_to": null
}
```

**Response:**
```json
{
  "question": "What loss function does SRGAN use?",
  "answer": "SRGAN uses a perceptual loss [1] combining adversarial loss...",
  "sources": [{"method": "SRGAN", "page_number": 4, "score": 0.82, ...}],
  "citations_valid": true,
  "retrieval_ms": 45.2,
  "total_ms": 312.8
}
```

### Other endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/info` | GET | Pipeline config and collection stats |
| `/corpus` | GET | List all indexed papers |
| `/feedback` | POST | Log thumbs up/down |

---

## Project Structure

```
sr-rag/
├── src/
│   ├── ingestion/          # PDF extraction and chunking
│   ├── indexing/           # Embeddings and ChromaDB
│   ├── retrieval/          # Retriever and reranker
│   ├── generation/         # Prompts, LLM calls, citations
│   ├── evaluation/         # Retrieval and answer metrics
│   ├── monitoring/         # Logging and query tracking
│   └── pipeline.py         # End-to-end orchestration
├── api/
│   └── main.py             # FastAPI application
├── app/
│   └── streamlit_app.py    # Streamlit UI
├── tests/                  # pytest test suite
├── scripts/                # Data download, eval, log analysis
├── notebooks/              # Exploration and development notebooks
├── data/
│   ├── raw_papers/         # PDF files (gitignored)
│   └── processed/          # Chunks, metadata, eval results
├── vector_store/           # ChromaDB index (gitignored)
├── logs/                   # Query logs (gitignored)
├── Dockerfile
└── docker-compose.yml
```

---

## LLM Providers

Set `LLM_PROVIDER` in `.env`:

| Provider | Key needed | Model |
|----------|-----------|-------|
| `mock` | None | Built-in (for development) |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` (default) |
| `gemini` | `GEMINI_API_KEY` | `gemini-1.5-flash` |
| `ollama` | None (local) | `llama3` |

---

## Future Improvements

- **Hybrid search** — BM25 + dense retrieval to improve exact keyword matching for technical terms (RCAN, PSNR, etc.)
- **Embedding model comparison** — benchmark `BAAI/bge-small-en-v1.5` vs `all-MiniLM-L6-v2` on this corpus
- **LLM-based faithfulness scoring** — use GPT to rate answer grounding (currently heuristic)
- **Larger corpus** — expand to 50+ papers covering diffusion-based SR methods

---

## Development

```bash
# Run all quality checks before pushing
bash scripts/run_all_checks.sh

# Run tests
pytest tests/ -v

# Lint and format
ruff check src/ api/ tests/ --fix
ruff format src/ api/ tests/
```

---

## Author

**Madhumitha Katam** — Graduate Student, Arizona State University
Applied ML · Computer Vision · RAG Systems

[LinkedIn](https://www.linkedin.com/in/madhumithakatam/) · [GitHub](https://github.com/KMadhumitha282002)
