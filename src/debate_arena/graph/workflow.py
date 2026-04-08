from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from debate_arena.config.models import AppConfig
from debate_arena.domain.models import DebateObserver, DebateResult, RefereeReview, RoleGuidance, TranscriptEntry, UsageSnapshot, Verdict
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


DEBATER_ROLES = ("debater_a", "debater_b")
ROLE_SUFFIX = {"debater_a": "a", "debater_b": "b"}
MAX_ACTIVE_RESTRICTIONS_PER_ROLE = 6
MAX_TURN_ATTEMPTS = 2


class DebateState(TypedDict, total=False):
    topic: str
    language: str
    max_rounds: int
    round_number: int
    next_speaker: str
    transcript: List[Dict[str, Any]]
    compact_summary: str
    guidance_by_role: Dict[str, RoleGuidance]
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
    decisive_line: str
    concessions_observed: List[str]


class DebateWorkflow:
    def __init__(
        self,
        config: AppConfig,
        prompt_repository: PromptRepository,
        model_factory: ModelFactory,
        observer: Optional[DebateObserver] = None,
    ):
        self._config = config
        self._prompt_repository = prompt_repository
        self._model_factory = model_factory
        self._observer = observer
        self._context_budget = ContextBudgetService(config.context_policy)
        self._graph = self._build_graph()

    def run(self, topic: str) -> DebateResult:
        if self._observer:
            self._observer.on_debate_start(topic)
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
            {"prepare_turn": "prepare_turn", "review_round": "review_round", "final_verdict": "final_verdict"},
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
            guidance_by_role=_empty_guidance_by_role(),
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
            restrictions_text=self._render_all_restrictions_text(state),
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
        if self._observer:
            self._observer.on_compaction(rebuilt_state["compactions"])
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
        usage_state: DebateState = DebateState(usage_by_role=state.get("usage_by_role", {}))
        content = ""
        last_validation: Optional[RefereeReview] = None
        current_prompt = prompt

        for _ in range(MAX_TURN_ATTEMPTS):
            result = self._model_factory.get(role).invoke(current_prompt)
            self._context_budget.calibrate(len(current_prompt), result.prompt_tokens)
            usage_state = DebateState(usage_by_role=self._record_usage(usage_state, role, result))
            candidate = result.content.strip()
            validation, validation_result = self._validate_turn(state, role, candidate)
            usage_state = DebateState(usage_by_role=self._record_usage(usage_state, "referee", validation_result))
            if validation.decision == "continue":
                content = candidate
                last_validation = None
                break
            last_validation = validation
            current_prompt = self._build_retry_prompt(prompt, role, validation)

        if last_validation is not None:
            opponent = "debater_b" if role == "debater_a" else "debater_a"
            updated_guidance = self._merge_guidance_by_role(state.get("guidance_by_role"), last_validation)
            return DebateState(
                usage_by_role=usage_state["usage_by_role"],
                guidance_by_role=updated_guidance,
                restrictions=self._flatten_restrictions_by_role(updated_guidance),
                needs_final_verdict=True,
                termination_reason=self._build_invalid_turn_termination(role, last_validation),
                winner=opponent,
            )

        entry = TranscriptEntry(
            role=role,
            speaker=speaker_name_for_role(role),
            content=content,
            round_number=state["round_number"],
        )
        if self._observer:
            self._observer.on_turn(entry)
        updated_transcript = list(state["transcript"])
        updated_transcript.append(asdict(entry))
        next_speaker = "debater_b" if role == "debater_a" else "debater_a"
        next_round = state["round_number"] if role == "debater_a" else state["round_number"] + 1
        return DebateState(
            transcript=updated_transcript,
            usage_by_role=usage_state["usage_by_role"],
            next_speaker=next_speaker,
            round_number=next_round,
            needs_final_verdict=False,
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
            restrictions_text=self._render_all_restrictions_text(state),
        )
        review_result = self._model_factory.get("referee").invoke(review_prompt)
        self._context_budget.calibrate(len(review_prompt), review_result.prompt_tokens)
        review = self._parse_review(review_result.content)
        if self._observer:
            self._observer.on_review(completed_rounds, review.decision, review.reason)
        updated_usage = self._record_usage(state, "referee", review_result)
        updated_guidance = self._merge_guidance_by_role(state.get("guidance_by_role"), review)
        needs_final_verdict = review.decision == "end" or completed_rounds >= state["max_rounds"]
        termination_reason = review.reason if review.decision == "end" else ""
        if completed_rounds >= state["max_rounds"] and not termination_reason:
            termination_reason = "Maximum configured rounds reached."
        return DebateState(
            usage_by_role=updated_usage,
            guidance_by_role=updated_guidance,
            restrictions=self._flatten_restrictions_by_role(updated_guidance),
            needs_final_verdict=needs_final_verdict,
            termination_reason=termination_reason,
            winner=review.winner,
            strongest_point_a=review.strongest_point_a,
            strongest_point_b=review.strongest_point_b,
        )

    def _final_verdict(self, state: DebateState) -> DebateState:
        if self._observer:
            self._observer.on_final_verdict_start()
        verdict_prompt = self._prompt_repository.render(
            self._config.model_for("referee").prompt_file.replace("review", "final"),
            topic=state["topic"],
            language=state["language"],
            completed_rounds=state["round_number"] - 1,
            compact_summary=state["compact_summary"] or "[No summary yet]",
            transcript_text=render_transcript(state["transcript"]),
            restrictions_text=self._render_all_restrictions_text(state),
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
        if state.get("needs_final_verdict"):
            return "final_verdict"
        return "review_round" if state.get("next_speaker") == "debater_a" else "prepare_turn"

    @staticmethod
    def _route_after_review(state: DebateState) -> str:
        return "final_verdict" if state.get("needs_final_verdict") else "prepare_turn"

    def _render_debater_prompt(self, role: str, state: Mapping[str, Any]) -> str:
        stance = "in favor" if role == "debater_a" else "against"
        guidance = _guidance_for_role(state.get("guidance_by_role"), role)
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
            restrictions_text=render_restrictions(self._active_restrictions_for_role(guidance)),
            strongest_opponent_point=strongest_opponent_point or "[Not available yet]",
            strongest_own_point=strongest_own_point or "[Not available yet]",
            claims_refuted_text=render_restrictions(guidance.claims_refuted),
            claims_unanswered_text=render_restrictions(guidance.claims_unanswered),
            required_next_move=guidance.required_next_move or "Respond to the rival's best point and advance a new angle.",
            round_number=state["round_number"],
            max_rounds=state["max_rounds"],
        )

    def _validate_turn(
        self,
        state: Mapping[str, Any],
        role: str,
        candidate_turn: str,
    ) -> tuple[RefereeReview, LLMCallResult]:
        stance = "in favor" if role == "debater_a" else "against"
        opponent = "debater_b" if role == "debater_a" else "debater_a"
        validation_prompt = self._prompt_repository.render(
            "turn_guard.j2",
            topic=state["topic"],
            language=state["language"],
            role=role,
            speaker_name=speaker_name_for_role(role),
            rival_name=speaker_name_for_role(opponent),
            stance=stance,
            compact_summary=state.get("compact_summary") or "[No summary yet]",
            transcript_text=render_transcript(state.get("transcript", [])),
            restrictions_text=render_restrictions(
                self._active_restrictions_for_role(_guidance_for_role(state.get("guidance_by_role"), role))
            ),
            candidate_turn=candidate_turn or "[Empty turn]",
        )
        validation_result = self._model_factory.get("referee").invoke(validation_prompt)
        self._context_budget.calibrate(len(validation_prompt), validation_result.prompt_tokens)
        return self._parse_review(validation_result.content), validation_result

    def _build_retry_prompt(self, base_prompt: str, role: str, validation: RefereeReview) -> str:
        guidance = validation.guidance_for(role)
        issues = self._active_restrictions_for_role(guidance) or [validation.reason or "You broke role coherence."]
        required_next_move = guidance.required_next_move or "Defend your assigned stance directly and do not switch roles."
        return (
            f"{base_prompt}\n\n"
            "Arbiter correction on your previous draft:\n"
            f"Reason: {validation.reason or 'Role coherence failure.'}\n"
            f"Issues to fix:\n{render_restrictions(issues)}\n"
            f"Required rewrite:\n{required_next_move}\n"
            f"You are still {speaker_name_for_role(role)} and must keep the same stance.\n"
            "Rewrite the intervention now and return only the corrected intervention."
        )

    def _build_invalid_turn_termination(self, role: str, validation: RefereeReview) -> str:
        issues = self._active_restrictions_for_role(validation.guidance_for(role))
        issues_text = "; ".join(issues) if issues else "repeatedly failed the assigned-stance guard"
        return (
            f"{speaker_name_for_role(role)} failed the role-coherence guard twice and the debate was closed. "
            f"{validation.reason or 'The turn abandoned the assigned stance or speaker identity.'} "
            f"Detected issues: {issues_text}."
        )

    def _merge_guidance_by_role(
        self,
        current_guidance: Optional[Mapping[str, RoleGuidance]],
        review: RefereeReview,
    ) -> Dict[str, RoleGuidance]:
        merged: Dict[str, RoleGuidance] = {}
        for role in DEBATER_ROLES:
            previous = _guidance_for_role(current_guidance, role)
            latest = review.guidance_for(role)
            merged[role] = RoleGuidance(
                claims_refuted=latest.claims_refuted,
                claims_unanswered=latest.claims_unanswered,
                actionable_restrictions=_merge_active_lines(
                    previous.actionable_restrictions,
                    latest.actionable_restrictions,
                    limit=MAX_ACTIVE_RESTRICTIONS_PER_ROLE,
                ),
                exhausted_argument_lines=_merge_active_lines(
                    previous.exhausted_argument_lines,
                    latest.exhausted_argument_lines,
                    limit=MAX_ACTIVE_RESTRICTIONS_PER_ROLE,
                ),
                required_next_move=latest.required_next_move,
            )
        return merged

    def _render_all_restrictions_text(self, state: Mapping[str, Any]) -> str:
        return render_restrictions(self._flatten_restrictions_by_role(state.get("guidance_by_role")))

    def _flatten_restrictions_by_role(self, guidance_by_role: Optional[Mapping[str, RoleGuidance]]) -> List[str]:
        lines: List[str] = []
        for role in DEBATER_ROLES:
            guidance = _guidance_for_role(guidance_by_role, role)
            for restriction in self._active_restrictions_for_role(guidance):
                lines.append(f"{speaker_name_for_role(role)}: {restriction}")
        return lines

    @staticmethod
    def _active_restrictions_for_role(guidance: RoleGuidance) -> List[str]:
        return unique_lines(guidance.actionable_restrictions + guidance.exhausted_argument_lines)

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
            loop_detected=loop_detected,
            no_new_arguments=no_new_arguments,
            strongest_point_a=str(payload.get("strongest_point_a", "")),
            strongest_point_b=str(payload.get("strongest_point_b", "")),
            guidance_by_role=_parse_guidance_by_role(payload),
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
            restrictions=list(state.get("restrictions", self._flatten_restrictions_by_role(state.get("guidance_by_role")))),
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


def _empty_guidance_by_role() -> Dict[str, RoleGuidance]:
    return {role: RoleGuidance() for role in DEBATER_ROLES}


def _guidance_for_role(
    guidance_by_role: Optional[Mapping[str, RoleGuidance]],
    role: str,
) -> RoleGuidance:
    if not guidance_by_role:
        return RoleGuidance()
    guidance = guidance_by_role.get(role)
    if isinstance(guidance, RoleGuidance):
        return guidance
    return RoleGuidance()


def _merge_active_lines(existing: List[str], incoming: List[str], *, limit: int) -> List[str]:
    merged = unique_lines(list(existing) + list(incoming))
    if len(merged) <= limit:
        return merged
    return merged[-limit:]


def _parse_guidance_by_role(payload: Mapping[str, Any]) -> Dict[str, RoleGuidance]:
    legacy_mode = not any(
        f"{base}_{suffix}" in payload
        for base in (
            "claims_refuted",
            "claims_unanswered",
            "actionable_restrictions",
            "exhausted_argument_lines",
            "required_next_move",
        )
        for suffix in ROLE_SUFFIX.values()
    )
    guidance_by_role: Dict[str, RoleGuidance] = {}
    for role in DEBATER_ROLES:
        suffix = ROLE_SUFFIX[role]
        guidance_by_role[role] = RoleGuidance(
            claims_refuted=_coerce_role_list(payload, f"claims_refuted_{suffix}", "claims_refuted", legacy_mode),
            claims_unanswered=_coerce_role_list(payload, f"claims_unanswered_{suffix}", "claims_unanswered", legacy_mode),
            actionable_restrictions=_coerce_role_list(
                payload,
                f"actionable_restrictions_{suffix}",
                "actionable_restrictions",
                legacy_mode,
            ),
            exhausted_argument_lines=_coerce_role_list(
                payload,
                f"exhausted_argument_lines_{suffix}",
                "exhausted_argument_lines",
                legacy_mode,
            ),
            required_next_move=_coerce_role_string(
                payload,
                f"required_next_move_{suffix}",
                "required_next_move",
                legacy_mode,
            ),
        )
    return guidance_by_role


def _coerce_role_list(
    payload: Mapping[str, Any],
    role_key: str,
    legacy_key: str,
    legacy_mode: bool,
) -> List[str]:
    value = _coerce_string_list(payload.get(role_key))
    if value or not legacy_mode:
        return value
    return _coerce_string_list(payload.get(legacy_key))


def _coerce_role_string(
    payload: Mapping[str, Any],
    role_key: str,
    legacy_key: str,
    legacy_mode: bool,
) -> str:
    value = str(payload.get(role_key, "")).strip()
    if value or not legacy_mode:
        return value
    return str(payload.get(legacy_key, "")).strip()
