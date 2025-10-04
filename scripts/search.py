#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.indexing.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Query FAISS index for relevant context")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=None, help="Number of results")
    parser.add_argument("--index-dir", type=Path, default=None, help="Index directory")
    parser.add_argument("--embedding-model", type=str, default=None, help="Model name override")
    args = parser.parse_args()

    settings = get_settings()
    top_k = args.top_k or settings.search_top_k
    index_dir = args.index_dir or settings.index_dir
    model_name = args.embedding_model or settings.embedding_model_name

    store = VectorStore.load(model_name=model_name, index_dir=index_dir)
    results = store.search(args.query, top_k=top_k)
    for rank, (score, doc) in enumerate(results, start=1):
        print(f"[{rank}] score={score:.4f} id={doc.doc_id} len={len(doc.text)} meta={doc.metadata}")


if __name__ == "__main__":
    main()


