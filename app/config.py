from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the AI-driven SOC project.

    Values can be overridden via environment variables. For local development,
    create a .env file in the project root.
    """

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    index_dir: Path = Field(default_factory=lambda: Path("indexes/default"))

    # Embeddings / Vector search
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-Transformers model for embeddings",
    )
    search_top_k: int = 20

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Finetuning
    hf_token: Optional[str] = Field(default=None, description="HuggingFace token, if needed")
    llama_base_model: str = Field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        description="Base model id for LoRA finetuning",
    )
    lora_output_dir: Path = Field(default_factory=lambda: Path("artifacts/finetune_lora"))

    # Summarization / contextualization
    summarizer_model_name: str = Field(
        default="google/flan-t5-base",
        description="Lightweight default model for on-CPU summarization",
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


def get_settings() -> Settings:
    """Helper to get cached settings instance."""
    return Settings()


