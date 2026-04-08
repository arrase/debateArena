from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
class RefereeReview:
    decision: str
    reason: str
    winner: Optional[str] = None
    exhausted_argument_lines: List[str] = field(default_factory=list)
    loop_detected: bool = False
    no_new_arguments: bool = False
    strongest_point_a: str = ""
    strongest_point_b: str = ""
    claims_refuted: List[str] = field(default_factory=list)
    claims_unanswered: List[str] = field(default_factory=list)
    actionable_restrictions: List[str] = field(default_factory=list)
    required_next_move: str = ""


@dataclass(frozen=True)
class Verdict:
    winner: str
    reason: str
    decisive_line: str = ""
    concessions_observed: List[str] = field(default_factory=list)


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
