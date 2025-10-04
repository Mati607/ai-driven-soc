#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.indexing.vector_store import VectorStore
from app.llm.contextualize import AlertContextualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Contextualize an alert using vector search + summarization")
    parser.add_argument("alert", type=str, help="Alert text or description")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k search results to include")
    parser.add_argument("--index-dir", type=Path, default=None, help="Index directory")
    parser.add_argument("--embedding-model", type=str, default=None, help="Model name override")
    parser.add_argument("--summarizer-model", type=str, default=None, help="Summarizer model override")
    args = parser.parse_args()

    settings = get_settings()
    top_k = args.top_k or settings.search_top_k
    index_dir = args.index_dir or settings.index_dir
    embed_name = args.embedding_model or settings.embedding_model_name
    sum_name = args.summarizer_model or settings.summarizer_model_name

    store = VectorStore.load(model_name=embed_name, index_dir=index_dir)
    results = store.search(args.alert, top_k=top_k)
    passages = [doc.text for _, doc in results]

    contextualizer = AlertContextualizer(model_name=sum_name)
    brief = contextualizer.summarize(args.alert, passages)
    print(brief)


if __name__ == "__main__":
    main()


