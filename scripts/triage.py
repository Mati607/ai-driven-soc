#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.indexing.vector_store import VectorStore
from app.llm.contextualize import AlertContextualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end triage: search + brief, append to triage JSONL")
    parser.add_argument("alert", type=str, help="Alert text")
    parser.add_argument("--triage-file", type=Path, default=Path("data/triage_results.jsonl"))
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    store = VectorStore.load(settings.embedding_model_name, args.index_dir or settings.index_dir)
    contextualizer = AlertContextualizer(settings.summarizer_model_name)

    results = store.search(args.alert, top_k=args.top_k or settings.search_top_k)
    passages = [doc.text for _, doc in results]
    brief = contextualizer.summarize(args.alert, passages)

    now = datetime.now(timezone.utc).isoformat()
    record = {
        "alert": args.alert,
        "created_at": now,
        "triaged_at": now,
        "search_results": [{"score": float(score), "doc_id": doc.doc_id} for score, doc in results],
        "brief": brief,
        "classification": None,
    }
    args.triage_file.parent.mkdir(parents=True, exist_ok=True)
    with args.triage_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print("Appended triage result to", args.triage_file)


if __name__ == "__main__":
    main()


