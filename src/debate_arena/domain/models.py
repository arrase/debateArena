from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


@dataclass(frozen=True)
class UsageSnapshot:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0


@dataclass(frozen=True)
class TranscriptEntry:
    role: str
    speaker: str
    content: str
    round_number: int


@dataclass(frozen=True)
class RoleGuidance:
    claims_refuted: List[str] = field(default_factory=list)
    claims_unanswered: List[str] = field(default_factory=list)
    actionable_restrictions: List[str] = field(default_factory=list)
    exhausted_argument_lines: List[str] = field(default_factory=list)
    required_next_move: str = ""


@dataclass(frozen=True)
class RefereeReview:
    decision: str
    reason: str
    winner: Optional[str] = None
    loop_detected: bool = False
    no_new_arguments: bool = False
    strongest_point_a: str = ""
    strongest_point_b: str = ""
    guidance_by_role: Dict[str, RoleGuidance] = field(default_factory=dict)

    def guidance_for(self, role: str) -> RoleGuidance:
        return self.guidance_by_role.get(role, RoleGuidance())

    @property
    def exhausted_argument_lines(self) -> List[str]:
        return self._flatten("exhausted_argument_lines")

    @property
    def claims_refuted(self) -> List[str]:
        return self._flatten("claims_refuted")

    @property
    def claims_unanswered(self) -> List[str]:
        return self._flatten("claims_unanswered")

    @property
    def actionable_restrictions(self) -> List[str]:
        return self._flatten("actionable_restrictions")

    @property
    def required_next_move(self) -> str:
        for role in ("debater_a", "debater_b"):
            value = self.guidance_for(role).required_next_move.strip()
            if value:
                return value
        return ""

    def _flatten(self, attribute: str) -> List[str]:
        values: List[str] = []
        for role in ("debater_a", "debater_b"):
            guidance = self.guidance_by_role.get(role)
            if not guidance:
                continue
            values.extend(getattr(guidance, attribute))
        return values


@dataclass(frozen=True)
class Verdict:
    winner: str
    reason: str
    decisive_line: str = ""
    concessions_observed: List[str] = field(default_factory=list)


class DebateObserver(Protocol):
    """Receives live events as a debate progresses."""

    def on_debate_start(self, topic: str) -> None: ...

    def on_turn(self, entry: TranscriptEntry) -> None: ...

    def on_review(self, round_number: int, decision: str, reason: str) -> None: ...

    def on_compaction(self, compactions: int) -> None: ...

    def on_final_verdict_start(self) -> None: ...


@dataclass(frozen=True)
class DebateResult:
    topic: str
    winner: str
    reason: str
    transcript: List[TranscriptEntry]
    restrictions: List[str]
    compact_summary: str
    compactions: int
    rounds_completed: int
    usage_by_role: Dict[str, UsageSnapshot]
    decisive_line: str = ""
    concessions_observed: List[str] = field(default_factory=list)
