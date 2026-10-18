from dataclasses import dataclass
from typing import Literal

Role = Literal["pro", "con"]


@dataclass(frozen=True)
class Turn:
    speaker: Role
    speaker_name: str
    round_number: int
    content: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class TurnValidation:
    valid: bool
    keeps_stance_prob: float
    repeats_exhausted_prob: float
    reason: str


@dataclass(frozen=True)
class RoundEvaluation:
    round_number: int
    pro_score: float
    con_score: float
    concession: str
    stalemate_prob: float
    should_end_prob: float
    should_end: bool
    reason: str


@dataclass(frozen=True)
class Verdict:
    winner: str
    probabilities: dict[str, float]
    confidence: float
    decisive_factor: str
    explanation: str
    pro_avg_score: float
    con_avg_score: float


@dataclass(frozen=True)
class DebateResult:
    topic: str
    verdict: Verdict
    turns: list[Turn]
    round_evaluations: list[RoundEvaluation]
    compact_summary: str
    exhausted_arguments: list[str]
    compactions: int
