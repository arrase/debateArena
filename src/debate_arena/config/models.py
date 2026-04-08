from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class RuntimeConfig:
    ollama_base_url: str
    request_timeout_seconds: float


@dataclass(frozen=True)
class DebateConfig:
    language: str
    max_rounds: int
    referee_interval: int
    max_response_words: int


@dataclass(frozen=True)
class PromptRepositoryConfig:
    directory: Path
    opening_prompt_file: str


@dataclass(frozen=True)
class ModelRoleConfig:
    role: str
    name: str
    temperature: float
    keep_alive: str
    prompt_file: str


@dataclass(frozen=True)
class ContextPolicyConfig:
    context_window: int
    usage_threshold_ratio: float
    response_buffer_tokens: int
    preserve_recent_messages: int
    compact_summary_max_chars: int


@dataclass(frozen=True)
class AppConfig:
    runtime: RuntimeConfig
    debate: DebateConfig
    prompts: PromptRepositoryConfig
    context_policy: ContextPolicyConfig
    models: Dict[str, ModelRoleConfig]
    config_path: Path

    def model_for(self, role: str) -> ModelRoleConfig:
        try:
            return self.models[role]
        except KeyError as exc:
            raise KeyError(f"Unknown role '{role}' in configuration.") from exc
