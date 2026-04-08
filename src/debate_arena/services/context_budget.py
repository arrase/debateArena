from __future__ import annotations

import math
from typing import Optional

from debate_arena.config.models import ContextPolicyConfig
from debate_arena.domain.models import UsageSnapshot

_DEFAULT_CHARS_PER_TOKEN = 4.0


class ContextBudgetService:
    def __init__(self, policy: ContextPolicyConfig):
        self._policy = policy
        self._chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN

    @property
    def context_window(self) -> int:
        return self._policy.context_window

    def should_compact(
        self,
        prompt_text: str,
        usage_snapshot: Optional[UsageSnapshot],
    ) -> bool:
        effective_budget = max(1, self._policy.context_window - self._policy.response_buffer_tokens)
        threshold = int(effective_budget * self._policy.usage_threshold_ratio)
        measured_tokens = usage_snapshot.prompt_tokens if usage_snapshot else 0
        if measured_tokens > 0:
            return measured_tokens >= threshold
        return self._estimate_tokens(prompt_text) >= threshold

    def calibrate(self, prompt_chars: int, actual_tokens: int) -> None:
        if actual_tokens > 0 and prompt_chars > 0:
            self._chars_per_token = prompt_chars / actual_tokens

    def _estimate_tokens(self, text: str) -> int:
        char_count = max(1, len(text))
        return int(math.ceil(char_count / self._chars_per_token))
