# AI-Driven SOC

Modern, showcase-ready Automated Security Operations Center that demonstrates:

- Fine-tuning LLaMA 3 with LoRA on curated threat intelligence for better classification
- Vector-search pipeline over large log volumes for high-quality alert context
- AI-driven alert contextualization to reduce Mean Time to Investigation (MTTI)
- FastAPI service exposing search and contextualization endpoints
- Weekly executive reporting with metrics and risk summaries

 
## Quickstart

1) Install dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Build an index from JSONL EDR logs

```bash
PYTHONPATH=$(pwd) python scripts/build_index.py /path/to/*.jsonl --index-dir indexes/default
```

3) Query the index

```bash
PYTHONPATH=$(pwd) python scripts/search.py "lsass suspicious access" --top-k 10
```

4) Contextualize an alert (retrieval + LLM brief)

```bash
PYTHONPATH=$(pwd) python scripts/contextualize.py "svchost contacted external IP 10.193.66.115"
```

5) Run the API

```bash
chmod +x scripts/serve_api.sh
./scripts/serve_api.sh
```

Then visit `http://localhost:8000/docs`.

## Project Layout

```
app/
  api/main.py            # FastAPI service
  config.py              # Central settings
  indexing/
    ingest.py            # Parse JSONL logs -> documents
    vector_store.py      # FAISS vector store
  llm/
    contextualize.py     # Summarization for alert context
scripts/
  build_index.py         # Build the FAISS index
  search.py              # Query the index
  contextualize.py       # Retrieval + summarization CLI
  serve_api.sh           # Run API server
```
