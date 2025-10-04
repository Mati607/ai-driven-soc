#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from app.config import get_settings
from app.indexing.ingest import ingest_jsonl_logs
from app.indexing.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from JSONL logs")
    parser.add_argument("inputs", nargs="+", help="Paths to JSONL log files")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Directory to write the FAISS index and docs.jsonl",
    )
    parser.add_argument(
        "--grouping",
        type=str,
        default="actorname",
        help="Field to group events by into documents (default: actorname)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Sentence-Transformers model name (overrides config)",
    )
    args = parser.parse_args()

    settings = get_settings()
    index_dir = args.index_dir if args.index_dir else settings.index_dir
    model_name = args.embedding_model or settings.embedding_model_name

    docs = ingest_jsonl_logs([Path(p) for p in args.inputs], grouping=args.grouping)

    store = VectorStore(model_name=model_name, index_dir=index_dir)
    store.build_from_documents(docs)
    store.save()
    print(f"Indexed {len(docs)} documents to {index_dir}")


if __name__ == "__main__":
    main()


