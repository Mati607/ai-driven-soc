from __future__ import annotations

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import get_settings
from app.indexing.vector_store import VectorStore
from app.llm.contextualize import AlertContextualizer


app = FastAPI(title="AI-Driven SOC API")
settings = get_settings()
_store = VectorStore.load(model_name=settings.embedding_model_name, index_dir=settings.index_dir)
_contextualizer = AlertContextualizer(model_name=settings.summarizer_model_name)


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None


class SearchResult(BaseModel):
    score: float
    doc_id: str
    text: str
    metadata: dict


class ContextRequest(BaseModel):
    alert: str
    top_k: int | None = None


class TriageRequest(BaseModel):
    alert: str
    top_k: int | None = None


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search", response_model=List[SearchResult])
def search(body: SearchRequest):
    top_k = body.top_k or settings.search_top_k
    results = _store.search(body.query, top_k=top_k)
    payload: List[SearchResult] = []
    for score, doc in results:
        payload.append(
            SearchResult(score=score, doc_id=doc.doc_id, text=doc.text, metadata=doc.metadata)
        )
    return payload


@app.post("/contextualize")
def contextualize(body: ContextRequest) -> dict:
    top_k = body.top_k or settings.search_top_k
    results = _store.search(body.alert, top_k=top_k)
    passages = [doc.text for _, doc in results]
    brief = _contextualizer.summarize(body.alert, passages)
    return {"brief": brief, "num_context": len(passages)}


@app.post("/triage")
def triage(body: TriageRequest) -> dict:
    top_k = body.top_k or settings.search_top_k
    results = _store.search(body.alert, top_k=top_k)
    passages = [doc.text for _, doc in results]
    brief = _contextualizer.summarize(body.alert, passages)
    payload = [
        {"score": float(score), "doc_id": doc.doc_id, "metadata": doc.metadata}
        for score, doc in results
    ]
    return {"alert": body.alert, "brief": brief, "search_results": payload}


