from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from debate_arena.config.models import (
    AppConfig,
    ContextPolicyConfig,
    DebateConfig,
    ModelRoleConfig,
    PromptRepositoryConfig,
    RuntimeConfig,
)


def load_config(config_path: Path) -> AppConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with config_path.open("r", encoding="utf-8") as file_handle:
        raw_data = yaml.safe_load(file_handle) or {}

    if not isinstance(raw_data, Mapping):
        raise RuntimeError("Configuration root must be a mapping.")

    runtime_raw = _mapping(raw_data, "runtime")
    debate_raw = _mapping(raw_data, "debate")
    prompts_raw = _mapping(raw_data, "prompt_repository")
    context_raw = _mapping(raw_data, "context_policy")
    models_raw = _mapping(raw_data, "models")

    prompt_directory = _resolve_path(config_path.parent, _string(prompts_raw, "directory"))
    runtime = RuntimeConfig(
        ollama_base_url=_string(runtime_raw, "ollama_base_url"),
        request_timeout_seconds=_float(runtime_raw, "request_timeout_seconds"),
    )
    debate = DebateConfig(
        language=_string(debate_raw, "language"),
        max_rounds=_positive_int(debate_raw, "max_rounds"),
        referee_interval=_positive_int(debate_raw, "referee_interval"),
        max_response_words=_positive_int(debate_raw, "max_response_words"),
    )
    prompts = PromptRepositoryConfig(
        directory=prompt_directory,
        opening_prompt_file=_string(prompts_raw, "opening_prompt_file"),
    )
    if not (prompt_directory / prompts.opening_prompt_file).exists():
        raise RuntimeError(
            f"Opening prompt file '{prompts.opening_prompt_file}' does not exist in {prompt_directory}."
        )
    if not (prompt_directory / "turn_guard.j2").exists():
        raise RuntimeError(f"Missing turn_guard.j2 prompt template required for role-coherence validation in {prompt_directory}.")
    context_policy = ContextPolicyConfig(
        context_window=_positive_int(context_raw, "context_window"),
        usage_threshold_ratio=_ratio(context_raw, "usage_threshold_ratio"),
        response_buffer_tokens=_positive_int(context_raw, "response_buffer_tokens"),
        preserve_recent_messages=_positive_int(context_raw, "preserve_recent_messages"),
        compact_summary_max_chars=_positive_int(context_raw, "compact_summary_max_chars"),
    )

    models = {
        role: _load_model(role, model_raw)
        for role, model_raw in models_raw.items()
    }
    _validate_models(models, prompt_directory)
    if "compactor" not in models:
        referee = models["referee"]
        models["compactor"] = ModelRoleConfig(
            role="compactor",
            name=referee.name,
            temperature=0.1,
            keep_alive=referee.keep_alive,
            prompt_file="compactor.j2",
        )
        if not (prompt_directory / "compactor.j2").exists():
            raise RuntimeError("Missing compactor.j2 prompt template required for the default compactor.")

    return AppConfig(
        runtime=runtime,
        debate=debate,
        prompts=prompts,
        context_policy=context_policy,
        models=models,
        config_path=config_path.resolve(),
    )


def _load_model(role: str, model_raw: Any) -> ModelRoleConfig:
    if not isinstance(model_raw, Mapping):
        raise RuntimeError(f"Model configuration for role '{role}' must be a mapping.")
    return ModelRoleConfig(
        role=role,
        name=_string(model_raw, "name"),
        temperature=_float(model_raw, "temperature"),
        keep_alive=_string(model_raw, "keep_alive"),
        prompt_file=_string(model_raw, "prompt_file"),
    )


def _validate_models(models: Dict[str, ModelRoleConfig], prompt_directory: Path) -> None:
    required_roles = {"debater_a", "debater_b", "referee"}
    missing = sorted(required_roles.difference(models.keys()))
    if missing:
        raise RuntimeError(f"Missing required model roles: {', '.join(missing)}")

    for role, model_config in models.items():
        if not (prompt_directory / model_config.prompt_file).exists():
            raise RuntimeError(
                f"Prompt file '{model_config.prompt_file}' for role '{role}' does not exist in {prompt_directory}."
            )


def _mapping(raw_data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw_data.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Configuration section '{key}' must be a mapping.")
    return value


def _string(raw_data: Mapping[str, Any], key: str) -> str:
    value = raw_data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Configuration value '{key}' must be a non-empty string.")
    return value.strip()


def _float(raw_data: Mapping[str, Any], key: str) -> float:
    value = raw_data.get(key)
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"Configuration value '{key}' must be numeric.")
    return float(value)


def _positive_int(raw_data: Mapping[str, Any], key: str) -> int:
    value = raw_data.get(key)
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"Configuration value '{key}' must be a positive integer.")
    return value


def _ratio(raw_data: Mapping[str, Any], key: str) -> float:
    value = _float(raw_data, key)
    if value <= 0 or value >= 1:
        raise RuntimeError(f"Configuration value '{key}' must be between 0 and 1.")
    return value


def _resolve_path(base_directory: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (base_directory / candidate).resolve()
    if not candidate.exists():
        raise RuntimeError(f"Configured path does not exist: {candidate}")
    return candidate
