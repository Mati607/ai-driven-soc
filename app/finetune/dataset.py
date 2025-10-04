from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from datasets import Dataset, load_dataset


@dataclass
class ThreatRecord:
    text: str
    label: str


def load_threat_dataset(
    data_path: Path,
    text_field: str = "text",
    label_field: str = "label",
    split_ratio: float = 0.9,
) -> Dict[str, Dataset]:
    """Load threat intel documents for instruction-style classification.

    Supports: JSON/JSONL/CSV via HuggingFace datasets 'load_dataset'.
    Returns dict with 'train' and 'eval' splits.
    """
    data_path = Path(data_path)
    if data_path.suffix.lower() in {".json", ".jsonl"}:
        ds = load_dataset("json", data_files=str(data_path))
    elif data_path.suffix.lower() == ".csv":
        ds = load_dataset("csv", data_files=str(data_path))
    else:
        raise ValueError(f"Unsupported dataset format: {data_path}")

    full = ds["train"]
    full = full.remove_columns([c for c in full.column_names if c not in {text_field, label_field}])

    def to_sft(example: Dict) -> Dict:
        text = str(example[text_field])
        label = str(example[label_field])
        instruction = (
            "Classify the following threat intelligence excerpt as one of the known signatures "
            "or 'unknown-novel'. Reply with only the class label."
        )
        return {
            "prompt": f"<s>[INST] {instruction}\n\n{text} [/INST]",
            "response": label,
        }

    mapped = full.map(to_sft, remove_columns=full.column_names)
    train_test = mapped.train_test_split(test_size=1 - split_ratio, seed=42)
    return {"train": train_test["train"], "eval": train_test["test"]}


