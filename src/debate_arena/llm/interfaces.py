from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMCallResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0


class RoleModel(Protocol):
    def invoke(self, prompt: str) -> LLMCallResult:
        ...


class ModelFactory(Protocol):
    def get(self, role: str) -> RoleModel:
        ...
