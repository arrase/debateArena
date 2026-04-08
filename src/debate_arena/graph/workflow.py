from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from debate_arena.config.models import AppConfig
from debate_arena.domain.models import DebateResult, RefereeReview, TranscriptEntry, UsageSnapshot, Verdict
from debate_arena.llm.interfaces import LLMCallResult, ModelFactory
from debate_arena.prompts.repository import PromptRepository
from debate_arena.services.context_budget import ContextBudgetService
from debate_arena.services.formatting import (
    render_restrictions,
    render_transcript,
    speaker_name_for_role,
    unique_lines,
)
from debate_arena.services.parsing import extract_json_object


class DebateState(TypedDict, total=False):
    topic: str
    language: str
    max_rounds: int
    round_number: int
    next_speaker: str
    transcript: List[Dict[str, Any]]
    compact_summary: str
    restrictions: List[str]
    compactions: int
    usage_by_role: Dict[str, Dict[str, int]]
    prepared_role: str
    prepared_prompt: str
    should_compact: bool
    needs_final_verdict: bool
    termination_reason: str
    winner: Optional[str]
    final_reason: str
    debate_phase: str
    strongest_point_a: str
    strongest_point_b: str
    claims_refuted: List[str]
    claims_unanswered: List[str]
    required_next_move: str
    decisive_line: str
    concessions_observed: List[str]


class DebateWorkflow:
    def __init__(self, config: AppConfig, prompt_repository: PromptRepository, model_factory: ModelFactory):
        self._config = config
        self._prompt_repository = prompt_repository
        self._model_factory = model_factory
        self._context_budget = ContextBudgetService(config.context_policy)
        self._graph = self._build_graph()

    def run(self, topic: str) -> DebateResult:
        initial_state = self._initial_state(topic)
        final_state = self._graph.invoke(initial_state)
        return self._build_result(final_state)

    def _build_graph(self):
        graph = StateGraph(DebateState)
        graph.add_node("prepare_turn", self._prepare_turn)
        graph.add_node("compact_context", self._compact_context)
        graph.add_node("speak_turn", self._speak_turn)
        graph.add_node("review_round", self._review_round)
        graph.add_node("final_verdict", self._final_verdict)

        graph.add_edge(START, "prepare_turn")
        graph.add_conditional_edges(
            "prepare_turn",
            self._route_after_prepare,
            {"compact_context": "compact_context", "speak_turn": "speak_turn"},
        )
        graph.add_edge("compact_context", "speak_turn")
        graph.add_conditional_edges(
            "speak_turn",
            self._route_after_speak,
            {"prepare_turn": "prepare_turn", "review_round": "review_round"},
        )
        graph.add_conditional_edges(
            "review_round",
            self._route_after_review,
            {"prepare_turn": "prepare_turn", "final_verdict": "final_verdict"},
        )
        graph.add_edge("final_verdict", END)
        return graph.compile()

    def _initial_state(self, topic: str) -> DebateState:
        return DebateState(
            topic=topic,
            language=self._config.debate.language,
            max_rounds=self._config.debate.max_rounds,
            round_number=1,
            next_speaker="debater_a",
            transcript=[],
            compact_summary="",
            restrictions=[],
            compactions=0,
            usage_by_role={},
            prepared_role="",
            prepared_prompt="",
            should_compact=False,
            needs_final_verdict=False,
            termination_reason="",
            winner=None,
            final_reason="",
            debate_phase="opening",
            strongest_point_a="",
            strongest_point_b="",
            claims_refuted=[],
            claims_unanswered=[],
            required_next_move="",
            decisive_line="",
            concessions_observed=[],
        )

    def _prepare_turn(self, state: DebateState) -> DebateState:
        role = state["next_speaker"]
        debate_phase = self._determine_phase(state)
        enriched_state = dict(state)
        enriched_state["debate_phase"] = debate_phase
        prompt = self._render_debater_prompt(role, enriched_state)
        usage = self._usage_snapshot(state, role)
        should_compact = self._context_budget.should_compact(
            prompt_text=prompt,
            usage_snapshot=usage,
        )
        return DebateState(
            prepared_role=role,
            prepared_prompt=prompt,
            should_compact=should_compact,
            debate_phase=debate_phase,
        )

    def _compact_context(self, state: DebateState) -> DebateState:
        keep_count = self._config.context_policy.preserve_recent_messages
        transcript = state["transcript"]
        if len(transcript) <= keep_count:
            return DebateState(should_compact=False)

        older_messages = transcript[:-keep_count]
        retained_messages = transcript[-keep_count:]
        compactor_prompt = self._prompt_repository.render(
            self._config.model_for("compactor").prompt_file,
            topic=state["topic"],
            language=state["language"],
            previous_summary=state["compact_summary"] or "[No prior summary]",
            transcript_text=render_transcript(older_messages),
            restrictions_text=render_restrictions(state["restrictions"]),
            target_chars=self._config.context_policy.compact_summary_max_chars,
        )
        compactor_result = self._model_factory.get("compactor").invoke(compactor_prompt)
        self._context_budget.calibrate(len(compactor_prompt), compactor_result.prompt_tokens)
        updated_usage = self._record_usage(state, "compactor", compactor_result)
        rebuilt_state = dict(state)
        rebuilt_state["compact_summary"] = compactor_result.content.strip()
        rebuilt_state["transcript"] = retained_messages
        rebuilt_state["usage_by_role"] = updated_usage
        rebuilt_state["compactions"] = state["compactions"] + 1
        rebuilt_prompt = self._render_debater_prompt(state["prepared_role"], rebuilt_state)
        return DebateState(
            compact_summary=rebuilt_state["compact_summary"],
            transcript=retained_messages,
            usage_by_role=updated_usage,
            compactions=rebuilt_state["compactions"],
            prepared_prompt=rebuilt_prompt,
            should_compact=False,
        )

    def _speak_turn(self, state: DebateState) -> DebateState:
        role = state["prepared_role"]
        prompt = state["prepared_prompt"]
        result = self._model_factory.get(role).invoke(prompt)
        self._context_budget.calibrate(len(prompt), result.prompt_tokens)
        entry = TranscriptEntry(
            role=role,
            speaker=speaker_name_for_role(role),
            content=result.content.strip(),
            round_number=state["round_number"],
        )
        updated_usage = self._record_usage(state, role, result)
        updated_transcript = list(state["transcript"])
        updated_transcript.append(asdict(entry))
        next_speaker = "debater_b" if role == "debater_a" else "debater_a"
        next_round = state["round_number"] if role == "debater_a" else state["round_number"] + 1
        return DebateState(
            transcript=updated_transcript,
            usage_by_role=updated_usage,
            next_speaker=next_speaker,
            round_number=next_round,
        )

    def _review_round(self, state: DebateState) -> DebateState:
        completed_rounds = state["round_number"] - 1
        if completed_rounds % self._config.debate.referee_interval != 0:
            if completed_rounds >= state["max_rounds"]:
                return DebateState(
                    needs_final_verdict=True,
                    termination_reason="Maximum configured rounds reached.",
                )
            return DebateState(needs_final_verdict=False, termination_reason="")

        review_prompt = self._prompt_repository.render(
            self._config.model_for("referee").prompt_file,
            topic=state["topic"],
            language=state["language"],
            completed_rounds=completed_rounds,
            compact_summary=state["compact_summary"] or "[No summary yet]",
            transcript_text=render_transcript(state["transcript"]),
            restrictions_text=render_restrictions(state["restrictions"]),
        )
        review_result = self._model_factory.get("referee").invoke(review_prompt)
        self._context_budget.calibrate(len(review_prompt), review_result.prompt_tokens)
        review = self._parse_review(review_result.content)
        updated_usage = self._record_usage(state, "referee", review_result)
        updated_restrictions = unique_lines(
            state["restrictions"] + review.actionable_restrictions + review.exhausted_argument_lines
        )
        needs_final_verdict = review.decision == "end" or completed_rounds >= state["max_rounds"]
        termination_reason = review.reason if review.decision == "end" else ""
        if completed_rounds >= state["max_rounds"] and not termination_reason:
            termination_reason = "Maximum configured rounds reached."
        return DebateState(
            usage_by_role=updated_usage,
            restrictions=updated_restrictions,
            needs_final_verdict=needs_final_verdict,
            termination_reason=termination_reason,
            winner=review.winner,
            strongest_point_a=review.strongest_point_a,
            strongest_point_b=review.strongest_point_b,
            claims_refuted=review.claims_refuted,
            claims_unanswered=review.claims_unanswered,
            required_next_move=review.required_next_move,
        )

    def _final_verdict(self, state: DebateState) -> DebateState:
        verdict_prompt = self._prompt_repository.render(
            self._config.model_for("referee").prompt_file.replace("review", "final"),
            topic=state["topic"],
            language=state["language"],
            completed_rounds=state["round_number"] - 1,
            compact_summary=state["compact_summary"] or "[No summary yet]",
            transcript_text=render_transcript(state["transcript"]),
            restrictions_text=render_restrictions(state["restrictions"]),
            termination_reason=state["termination_reason"] or "Arbiter must issue final verdict.",
            provisional_winner=state.get("winner") or "draw",
        )
        verdict_result = self._model_factory.get("referee").invoke(verdict_prompt)
        self._context_budget.calibrate(len(verdict_prompt), verdict_result.prompt_tokens)
        verdict = self._parse_verdict(verdict_result.content)
        updated_usage = self._record_usage(state, "referee", verdict_result)
        return DebateState(
            usage_by_role=updated_usage,
            winner=verdict.winner,
            final_reason=verdict.reason or state.get("termination_reason") or verdict.decisive_line or "No explicit verdict.",
            decisive_line=verdict.decisive_line,
            concessions_observed=verdict.concessions_observed,
        )

    @staticmethod
    def _route_after_prepare(state: DebateState) -> str:
        return "compact_context" if state.get("should_compact") else "speak_turn"

    @staticmethod
    def _route_after_speak(state: DebateState) -> str:
        return "review_round" if state.get("next_speaker") == "debater_a" else "prepare_turn"

    @staticmethod
    def _route_after_review(state: DebateState) -> str:
        return "final_verdict" if state.get("needs_final_verdict") else "prepare_turn"

    def _render_debater_prompt(self, role: str, state: Mapping[str, Any]) -> str:
        stance = "in favor" if role == "debater_a" else "against"
        strongest_opponent_point = state.get("strongest_point_b", "") if role == "debater_a" else state.get("strongest_point_a", "")
        strongest_own_point = state.get("strongest_point_a", "") if role == "debater_a" else state.get("strongest_point_b", "")
        opening_prompt = ""
        if not state.get("transcript"):
            opening_prompt = self._prompt_repository.render(
                self._config.prompts.opening_prompt_file,
                topic=state["topic"],
                language=state["language"],
                speaker_name=speaker_name_for_role(role),
                stance=stance,
            )
        return self._prompt_repository.render(
            self._config.model_for(role).prompt_file,
            topic=state["topic"],
            language=state["language"],
            speaker_name=speaker_name_for_role(role),
            stance=stance,
            debate_phase=state.get("debate_phase", "exchange"),
            max_response_words=self._config.debate.max_response_words,
            opening_prompt=opening_prompt,
            compact_summary=state.get("compact_summary") or "[No summary yet]",
            transcript_text=render_transcript(state.get("transcript", [])),
            restrictions_text=render_restrictions(state.get("restrictions", [])),
            strongest_opponent_point=strongest_opponent_point or "[Not available yet]",
            strongest_own_point=strongest_own_point or "[Not available yet]",
            claims_refuted_text=render_restrictions(state.get("claims_refuted", [])),
            claims_unanswered_text=render_restrictions(state.get("claims_unanswered", [])),
            required_next_move=state.get("required_next_move") or "Respond to the rival's best point and advance a new angle.",
            round_number=state["round_number"],
            max_rounds=state["max_rounds"],
        )

    def _usage_snapshot(self, state: Mapping[str, Any], role: str) -> Optional[UsageSnapshot]:
        raw_snapshot = state.get("usage_by_role", {}).get(role)
        if not raw_snapshot:
            return None
        return UsageSnapshot(
            prompt_tokens=int(raw_snapshot.get("prompt_tokens", 0)),
            completion_tokens=int(raw_snapshot.get("completion_tokens", 0)),
            total_duration_ns=int(raw_snapshot.get("total_duration_ns", 0)),
        )

    def _record_usage(
        self,
        state: Mapping[str, Any],
        role: str,
        result: LLMCallResult,
    ) -> Dict[str, Dict[str, int]]:
        usage_by_role = dict(state.get("usage_by_role", {}))
        usage_by_role[role] = {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_duration_ns": result.total_duration_ns,
        }
        return usage_by_role

    def _parse_review(self, content: str) -> RefereeReview:
        payload = extract_json_object(content)
        loop_detected = _coerce_bool(payload.get("loop_detected"))
        no_new_arguments = _coerce_bool(payload.get("no_new_arguments"))
        winner = self._normalize_winner(payload.get("winner"))
        decision = self._normalize_decision(
            payload.get("decision"),
            winner=winner,
            loop_detected=loop_detected,
            no_new_arguments=no_new_arguments,
        )
        return RefereeReview(
            decision=decision,
            reason=str(payload.get("reason", "")),
            winner=winner,
            exhausted_argument_lines=_coerce_string_list(payload.get("exhausted_argument_lines")),
            loop_detected=loop_detected,
            no_new_arguments=no_new_arguments,
            strongest_point_a=str(payload.get("strongest_point_a", "")),
            strongest_point_b=str(payload.get("strongest_point_b", "")),
            claims_refuted=_coerce_string_list(payload.get("claims_refuted")),
            claims_unanswered=_coerce_string_list(payload.get("claims_unanswered")),
            actionable_restrictions=_coerce_string_list(payload.get("actionable_restrictions")),
            required_next_move=str(payload.get("required_next_move", "")),
        )

    def _parse_verdict(self, content: str) -> Verdict:
        payload = extract_json_object(content)
        return Verdict(
            winner=self._normalize_winner(payload.get("winner")) or "draw",
            reason=str(payload.get("reason", "")),
            decisive_line=str(payload.get("decisive_line", "")),
            concessions_observed=_coerce_string_list(payload.get("concessions_observed")),
        )

    @staticmethod
    def _normalize_winner(value: Any) -> Optional[str]:
        normalized = str(value).strip().lower().replace(" ", "_")
        canonical = {"debater_a", "debater_b", "draw"}
        if normalized in canonical:
            return normalized
        # Handle single-letter abbreviations from schema drift
        if normalized == "a":
            return "debater_a"
        if normalized == "b":
            return "debater_b"
        if normalized in {"", "null", "none", "n/a"}:
            return None
        return None

    @staticmethod
    def _normalize_decision(
        value: Any,
        *,
        winner: Optional[str],
        loop_detected: bool,
        no_new_arguments: bool,
    ) -> str:
        normalized = str(value).strip().lower()
        if normalized == "continue":
            return "continue"
        if normalized == "end":
            return "end"
        # If the model returned something non-canonical, infer from signals
        if winner in {"debater_a", "debater_b"} and (loop_detected or no_new_arguments):
            return "end"
        return "continue"

    def _build_result(self, state: DebateState) -> DebateResult:
        transcript = [
            TranscriptEntry(
                role=str(item["role"]),
                speaker=str(item["speaker"]),
                content=str(item["content"]),
                round_number=int(item["round_number"]),
            )
            for item in state["transcript"]
        ]
        usage_by_role = {
            role: UsageSnapshot(
                prompt_tokens=int(snapshot.get("prompt_tokens", 0)),
                completion_tokens=int(snapshot.get("completion_tokens", 0)),
                total_duration_ns=int(snapshot.get("total_duration_ns", 0)),
            )
            for role, snapshot in state.get("usage_by_role", {}).items()
        }
        return DebateResult(
            topic=state["topic"],
            winner=state.get("winner") or "draw",
            reason=state.get("final_reason") or state.get("termination_reason") or "No explicit verdict.",
            transcript=transcript,
            restrictions=list(state.get("restrictions", [])),
            compact_summary=state.get("compact_summary", ""),
            compactions=int(state.get("compactions", 0)),
            rounds_completed=max(0, int(state.get("round_number", 1)) - 1),
            usage_by_role=usage_by_role,
            decisive_line=state.get("decisive_line", ""),
            concessions_observed=list(state.get("concessions_observed", [])),
        )

    def _determine_phase(self, state: Mapping[str, Any]) -> str:
        completed_rounds = int(state.get("round_number", 1)) - 1
        max_rounds = int(state.get("max_rounds", 1))
        if completed_rounds == 0:
            return "opening"
        if max_rounds - completed_rounds <= 2:
            return "closing"
        return "exchange"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0", "none", "null", ""}:
        return False
    return bool(normalized)


def _coerce_string_list(value: Any) -> List[str]:
    if value in (None, False):
        return []
    if isinstance(value, list):
        return unique_lines(value)
    if isinstance(value, str):
        return unique_lines([value])
    return []
