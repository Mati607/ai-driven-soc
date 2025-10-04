from __future__ import annotations

from typing import List, Sequence

from transformers import pipeline


class AlertContextualizer:
    """Summarize top-k search hits into an analyst-ready brief."""

    def __init__(self, model_name: str) -> None:
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, query: str, passages: Sequence[str], max_len: int = 256) -> str:
        joined = "\n".join(passages)
        prompt = (
            "You are a security analyst. Given the alert and related context, "
            "produce a concise investigation brief with likely root cause, impacted assets, "
            "and recommended next steps.\n\n"
            f"Alert: {query}\n\nContext:\n{joined}\n\nBrief:"
        )
        output = self.summarizer(prompt, max_length=max_len, min_length=max(64, max_len // 4), do_sample=False)
        return output[0]["summary_text"].strip()


