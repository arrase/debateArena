from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
import yaml


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    timeout_seconds: int
    keep_alive: str


@dataclass(frozen=True)
class ModelsConfig:
    language_model: str
    decision_model: str
    debater_temperature: float
    compactor_temperature: float


@dataclass(frozen=True)
class DebateConfig:
    language: str
    max_rounds: int
    max_response_words: int
    max_turn_attempts: int


@dataclass(frozen=True)
class ContextConfig:
    num_ctx: int
    compaction_threshold_ratio: float
    compaction_interval_rounds: int
    retain_recent_rounds: int


@dataclass(frozen=True)
class JudgeConfig:
    min_stance_probability: float
    max_repetition_probability: float
    end_debate_probability: float


@dataclass(frozen=True)
class AppConfig:
    ollama: OllamaConfig
    models: ModelsConfig
    debate: DebateConfig
    context: ContextConfig
    judge: JudgeConfig
    prompts_env: Environment = field(repr=False)

    def render_prompt(self, template_name: str, **kwargs: Any) -> str:
        return self.prompts_env.get_template(template_name).render(**kwargs).strip()


def load_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prompts_env = Environment(
        loader=FileSystemLoader(config_path.parent / "prompts"),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return AppConfig(
        ollama=OllamaConfig(**raw["ollama"]),
        models=ModelsConfig(**raw["models"]),
        debate=DebateConfig(**raw["debate"]),
        context=ContextConfig(**raw["context"]),
        judge=JudgeConfig(**raw["judge"]),
        prompts_env=prompts_env,
    )
