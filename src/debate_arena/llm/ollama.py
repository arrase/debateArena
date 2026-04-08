from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from debate_arena.config.models import AppConfig, ModelRoleConfig
from debate_arena.llm.interfaces import LLMCallResult, ModelFactory, RoleModel


class OllamaRoleModel(RoleModel):
    def __init__(self, runtime_base_url: str, model_config: ModelRoleConfig, context_window: int):
        self._role = model_config.role
        self._model = ChatOllama(
            model=model_config.name,
            base_url=runtime_base_url,
            temperature=model_config.temperature,
            num_ctx=context_window,
            keep_alive=model_config.keep_alive,
            reasoning=False,
            format=_format_for_role(model_config.role),
        )

    def invoke(self, prompt: str) -> LLMCallResult:
        response = self._model.invoke([HumanMessage(content=prompt)])
        metadata = getattr(response, "response_metadata", {}) or {}
        content = response.content if isinstance(response.content, str) else str(response.content)
        return LLMCallResult(
            content=content,
            prompt_tokens=int(metadata.get("prompt_eval_count", 0)),
            completion_tokens=int(metadata.get("eval_count", 0)),
            total_duration_ns=int(metadata.get("total_duration", 0)),
        )


class OllamaChatFactory(ModelFactory):
    def __init__(self, config: AppConfig):
        self._config = config
        self._cache: Dict[str, RoleModel] = {}

    def get(self, role: str) -> RoleModel:
        if role not in self._cache:
            model_config = self._config.model_for(role)
            context_window = self._config.context_policy.context_window
            self._cache[role] = OllamaRoleModel(self._config.runtime.ollama_base_url, model_config, context_window)
        return self._cache[role]


def _format_for_role(role: str) -> Optional[dict[str, Any] | str]:
    if role != "referee":
        return None
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["continue", "end"]},
            "reason": {"type": "string"},
            "winner": {"type": ["string", "null"], "enum": ["debater_a", "debater_b", "draw", None]},
            "loop_detected": {"type": "boolean"},
            "no_new_arguments": {"type": "boolean"},
            "exhausted_argument_lines": {
                "type": "array",
                "items": {"type": "string"},
            },
            "strongest_point_a": {"type": "string"},
            "strongest_point_b": {"type": "string"},
            "claims_refuted": {
                "type": "array",
                "items": {"type": "string"},
            },
            "claims_unanswered": {
                "type": "array",
                "items": {"type": "string"},
            },
            "actionable_restrictions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "required_next_move": {"type": "string"},
            "decisive_line": {"type": "string"},
            "concessions_observed": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "decision",
            "reason",
            "winner",
            "loop_detected",
            "no_new_arguments",
            "exhausted_argument_lines",
            "strongest_point_a",
            "strongest_point_b",
            "claims_refuted",
            "claims_unanswered",
            "actionable_restrictions",
            "required_next_move",
            "decisive_line",
            "concessions_observed",
        ],
    }
