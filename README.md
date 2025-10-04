# 🛡️ AI-Driven SOC — Explainable Retrieval-Augmented Triage

<div align="center">

![Search](https://img.shields.io/badge/Vector%20Search-FAISS-blue?style=for-the-badge&logo=apache)
![Embeddings](https://img.shields.io/badge/Embeddings-SentenceTransformers-green?style=for-the-badge&logo=huggingface)
![LLM](https://img.shields.io/badge/Summarization-Transformers-orange?style=for-the-badge&logo=transformer)
![API](https://img.shields.io/badge/API-FastAPI-teal?style=for-the-badge&logo=fastapi)
![Finetune](https://img.shields.io/badge/LoRA-Finetuning-purple?style=for-the-badge&logo=huggingface)

**Enterprise-ready RAG for security alerts** — index EDR logs with embeddings, retrieve high-signal context using FAISS, and generate analyst-ready briefs with a lightweight summarizer. Includes LoRA finetuning utilities for threat-intel classification and a FastAPI service.

</div>

---

## 🌟 Overview

This project provides an end-to-end workflow for Security Operations Centers:

- Build a vector index over JSONL EDR logs
- Search relevant context for alerts
- Summarize findings into an investigation brief
- Expose search and contextualization via an API
- Optionally finetune an LLM for classification with LoRA

### ✨ Key Features

- 🔎 **High-quality retrieval**: Sentence-Transformers + FAISS (inner-product, normalized)
- 🧠 **LLM contextualization**: Transformers summarization pipeline with SOC-specific prompt
- ⚙️ **Configurable**: All knobs via `app/config.py` or environment variables
- 🧪 **Finetune-ready**: LoRA training pipeline for LLaMA 3 SFT classification
- 📈 **Reporting**: Weekly HTML KPI report from triage logs

## 🏗️ Architecture

```
┌──────────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
│  JSONL EDR Logs      │ → │  Embeddings +      │ → │  FAISS Vector Store   │
│  (ingest.py)         │   │  Document Builder  │   │  (vector_store.py)    │
└──────────────────────┘   └────────────────────┘   └──────────────────────┘
             │                          │                         │
             └─────────────── search ───┼─────────────────────────┘
                                        │
                              ┌──────────────────────┐
                              │  LLM Contextualizer  │
                              │  (contextualize.py)  │
                              └─────────┬────────────┘
                                        │
                              ┌──────────────────────┐
                              │  FastAPI Endpoints   │
                              │   /search, /context  │
                              └──────────────────────┘
```

## 🛠️ Tech Stack

- **Vector Search**: FAISS (`IndexFlatIP` with normalized embeddings)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (configurable)
- **Summarization**: `google/flan-t5-base` by default (configurable)
- **API**: FastAPI + Uvicorn
- **Finetuning**: LoRA with PEFT + TRL SFTTrainer
- **Reports**: Jinja2 HTML report generation

## 🚀 Quick Start

### 1) Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Build index from JSONL EDR logs

```bash
PYTHONPATH=$(pwd) python scripts/build_index.py /path/to/*.jsonl --index-dir indexes/default
```

Optional flags:
- `--grouping actorname` (default)
- `--embedding-model sentence-transformers/all-MiniLM-L6-v2`

### 3) Search the index

```bash
PYTHONPATH=$(pwd) python scripts/search.py "svchost suspicious access" --top-k 10
```

### 4) Contextualize an alert

```bash
PYTHONPATH=$(pwd) python scripts/contextualize.py "svchost contacted external IP 10.193.66.115"
```

### 5) Run the API

```bash
chmod +x scripts/serve_api.sh
./scripts/serve_api.sh
```

Then visit `http://localhost:8000/docs`.

## 🔌 API Endpoints

- `GET /healthz` — health check
- `POST /search` → `[ { score, doc_id, text, metadata } ]`
- `POST /contextualize` → `{ brief, num_context }`
- `POST /triage` → `{ alert, brief, search_results }`

## ⚙️ Configuration (`app/config.py`)

Override via env vars or `.env` file.

- **Paths**: `data_dir`, `index_dir`
- **Embeddings**: `embedding_model_name`, `search_top_k`
- **API**: `api_host`, `api_port`
- **Contextualizer**: `summarizer_model_name`
- **Finetune**: `llama_base_model`, `lora_output_dir`, `hf_token`

## 🧪 LoRA Finetuning (optional)

Prepare a JSON/JSONL/CSV with `text` and `label` fields.

```bash
PYTHONPATH=$(pwd) python -m app.finetune.train_lora /path/to/data.jsonl \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --output artifacts/finetune_lora \
  --epochs 2 --batch 2 --lr 2e-4
```

The loader converts data to SFT format and trains a LoRA adapter. Artifacts are saved under `lora_output_dir`.

## 📊 Weekly Report (from triage JSONL)

Generate an HTML summary of KPIs (alerts triaged, median MTTI, relevant match rate):

```bash
PYTHONPATH=$(pwd) python scripts/generate_weekly_report.py triage_results.jsonl --out reports/weekly_report.html
```

## 📁 Project Layout

```
app/
  api/main.py            # FastAPI service
  config.py              # Central settings
  indexing/
    ingest.py            # Parse JSONL logs -> documents
    vector_store.py      # FAISS vector store
  llm/
    contextualize.py     # Summarization for alert context
  finetune/
    dataset.py           # Threat dataset loader -> SFT mapping
    train_lora.py        # LoRA finetuning CLI
  reports/
    weekly.py            # KPI report generator
scripts/
  build_index.py         # Build the FAISS index
  search.py              # Query the index
  contextualize.py       # Retrieval + summarization CLI
  generate_weekly_report.py  # Render weekly HTML report
  serve_api.sh           # Run API server
```