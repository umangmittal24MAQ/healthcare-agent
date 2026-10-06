from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]


class LLMConfig(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    url: HttpUrl
    toolCalling: bool
    vision: bool
    maxInputTokens: int = Field(gt=0)
    maxOutputTokens: int = Field(gt=0)


class Settings(BaseSettings):
    indiaai_api_key: str | None = None
    database_path: str = "./healthcare_agent.db"
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_enable_thinking: bool = True
    retrieval_candidates: int = Field(default=10, ge=1, le=50)
    retrieval_top_k: int = Field(default=6, ge=1, le=20)
    llm_config_path: str = str(BASE_DIR / "config" / "llm.json")
    clinical_corpus_path: str = str(BASE_DIR / "data" / "clinical_corpus.json")
    safety_rules_path: str = str(BASE_DIR / "config" / "safety_rules.json")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_llm_config() -> LLMConfig:
    settings = get_settings()
    payload = json.loads(Path(settings.llm_config_path).read_text(encoding="utf-8"))
    return LLMConfig.model_validate(payload)


@lru_cache
def load_json_file(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
