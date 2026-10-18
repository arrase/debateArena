import json
from typing import Any

from ollama import Client

from debate_arena.config import AppConfig
from debate_arena.context import SPEAKER_NAMES, STANCE_LABELS, DebateContext
from debate_arena.models import Role, RoundEvaluation, Turn, TurnValidation, Verdict

WINNER_LABELS: dict[str, str] = {
    "pro": SPEAKER_NAMES["pro"],
    "con": SPEAKER_NAMES["con"],
    "draw": "Empate técnico",
}

SCORE_CRITERIA: list[str] = [
    "Weak, evasive, or repetitive argument",
    "Basic defense with partial response to opponent",
    "Solid argument with direct refutation",
    "Decisive argument that dismantles opponent's core point",
]

COMPACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "compact_summary": {"type": "string"},
        "exhausted_arguments": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["compact_summary", "exhausted_arguments"],
}


class DebateJudge:
    def __init__(self, config: AppConfig, client: Client) -> None:
        self.config = config
        self.client = client

    def validate_turn(
        self, speaker: Role, candidate_turn: str, context: DebateContext
    ) -> TurnValidation:
        opponent_last_turn = context.turns[-1].content if context.turns else ""
        state = {
            "topic": context.topic,
            "speaker": SPEAKER_NAMES[speaker],
            "mandatory_stance": STANCE_LABELS[speaker],
            "exhausted_arguments": context.exhausted_arguments,
            "opponent_last_turn": opponent_last_turn,
            "candidate_turn": candidate_turn,
        }
        questions: dict[str, Any] = {
            "keeps_stance": {
                "type": "noul",
                "instructions": (
                    "Does `candidate_turn` defend `mandatory_stance` in first person "
                    "without switching roles or conceding the core thesis to the opponent?"
                ),
                "criteria": {
                    "false": "Switches roles, adopts the opponent stance, or concedes the core thesis",
                    "true": "Maintains its assigned stance and argues in favor of its side",
                },
            }
        }
        if context.exhausted_arguments:
            questions["repeats_exhausted"] = {
                "type": "noul",
                "instructions": (
                    "Does `candidate_turn` repeat any argument already listed in "
                    "`exhausted_arguments`?"
                ),
                "criteria": {
                    "false": "Introduces new angles or refutations not in exhausted_arguments",
                    "true": "Repeats an argument from exhausted_arguments",
                },
            }

        response = self.client.systemone(
            model=self.config.models.decision_model,
            state=state,
            questions=questions,
            keep_alive=self.config.ollama.keep_alive,
        )
        keeps_stance_prob = response.answers["keeps_stance"].noul
        repeats_exhausted_prob = (
            response.answers["repeats_exhausted"].noul
            if "repeats_exhausted" in response.answers
            else 0.0
        )

        reasons: list[str] = []
        if keeps_stance_prob < self.config.judge.min_stance_probability:
            reasons.append(
                f"No mantienes tu postura obligatoria ({STANCE_LABELS[speaker]}); "
                "debes defenderla en primera persona sin asumir el rol del oponente."
            )
        if repeats_exhausted_prob > self.config.judge.max_repetition_probability:
            reasons.append(
                "Estás repitiendo argumentos ya agotados; debes aportar un ángulo nuevo."
            )

        return TurnValidation(
            valid=not reasons,
            keeps_stance_prob=keeps_stance_prob,
            repeats_exhausted_prob=repeats_exhausted_prob,
            reason=" ".join(reasons),
        )

    def evaluate_round(
        self,
        round_number: int,
        pro_turn: Turn,
        con_turn: Turn,
        context: DebateContext,
    ) -> RoundEvaluation:
        state = {
            "topic": context.topic,
            "round_number": round_number,
            "compact_summary": context.compact_summary,
            "exhausted_arguments": context.exhausted_arguments,
            "pro_argument": pro_turn.content,
            "con_argument": con_turn.content,
        }
        questions: dict[str, Any] = {
            "pro_score": {
                "type": "score",
                "instructions": (
                    "Rate the argumentative rigor, clarity, and direct refutation "
                    "quality of `pro_argument` (IN FAVOR of `topic`)."
                ),
                "criteria": SCORE_CRITERIA,
            },
            "con_score": {
                "type": "score",
                "instructions": (
                    "Rate the argumentative rigor, clarity, and direct refutation "
                    "quality of `con_argument` (AGAINST `topic`)."
                ),
                "criteria": SCORE_CRITERIA,
            },
            "concession": {
                "type": "choice",
                "instructions": "Did either debater explicitly concede the core thesis in this round?",
                "criteria": {
                    "none": "Neither side conceded the core thesis",
                    "pro_conceded": "PRO conceded the core thesis to CON",
                    "con_conceded": "CON conceded the core thesis to PRO",
                },
            },
            "stalemate": {
                "type": "noul",
                "instructions": (
                    "Are both debaters stuck in a circular loop without advancing "
                    "new substantive arguments?"
                ),
            },
            "should_end": {
                "type": "noul",
                "instructions": (
                    "Should the debate conclude now because one side has decisively "
                    "refuted the other, a core concession occurred, or all productive "
                    "arguments are exhausted?"
                ),
            },
        }

        response = self.client.systemone(
            model=self.config.models.decision_model,
            state=state,
            questions=questions,
            keep_alive=self.config.ollama.keep_alive,
        )
        pro_score = response.answers["pro_score"].score
        con_score = response.answers["con_score"].score
        concession = response.answers["concession"].choice
        stalemate_prob = response.answers["stalemate"].noul
        should_end_prob = response.answers["should_end"].noul

        threshold = self.config.judge.end_debate_probability
        should_end = (
            concession != "none"
            or should_end_prob >= threshold
            or stalemate_prob >= threshold
        )

        if concession == "pro_conceded":
            reason = "El Debatiente A (A favor) concedió la tesis central."
        elif concession == "con_conceded":
            reason = "El Debatiente B (En contra) concedió la tesis central."
        elif stalemate_prob >= threshold:
            reason = "El juez detectó un bucle argumental sin nuevos avances."
        elif should_end_prob >= threshold:
            reason = "El juez determinó que el debate ha alcanzado una conclusión decisiva."
        else:
            reason = "El debate continúa."

        return RoundEvaluation(
            round_number=round_number,
            pro_score=pro_score,
            con_score=con_score,
            concession=concession,
            stalemate_prob=stalemate_prob,
            should_end_prob=should_end_prob,
            should_end=should_end,
            reason=reason,
        )

    def compact_context(
        self, turns_to_compact: list[Turn], context: DebateContext
    ) -> tuple[str, list[str]]:
        prompt = self.config.render_prompt(
            "compactor.j2",
            topic=context.topic,
            language=self.config.debate.language,
            compact_summary=context.compact_summary,
            exhausted_arguments=context.exhausted_arguments,
            turns_transcript=context.format_turns(turns_to_compact),
        )
        response = self.client.chat(
            model=self.config.models.language_model,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            format=COMPACTION_SCHEMA,
            options={
                "temperature": self.config.models.compactor_temperature,
                "num_ctx": self.config.context.num_ctx,
            },
            keep_alive=self.config.ollama.keep_alive,
        )
        data = json.loads(response.message.content)
        return data["compact_summary"], data["exhausted_arguments"]

    def issue_verdict(
        self,
        context: DebateContext,
        round_evaluations: list[RoundEvaluation],
        termination_reason: str,
        forced_winner: str = "",
    ) -> Verdict:
        if round_evaluations:
            pro_avg_score = sum(r.pro_score for r in round_evaluations) / len(
                round_evaluations
            )
            con_avg_score = sum(r.con_score for r in round_evaluations) / len(
                round_evaluations
            )
        else:
            pro_avg_score = 0.0
            con_avg_score = 0.0

        if forced_winner:
            winner = forced_winner
            probabilities = {
                "pro": 1.0 if forced_winner == "pro" else 0.0,
                "con": 1.0 if forced_winner == "con" else 0.0,
                "draw": 0.0,
            }
            confidence = 1.0
            decisive_factor = "disqualification_by_rule_violation"
        else:
            state = {
                "topic": context.topic,
                "compact_summary": context.compact_summary,
                "recent_turns": [
                    {
                        "round": t.round_number,
                        "speaker": t.speaker_name,
                        "content": t.content,
                    }
                    for t in context.active_turns()
                ],
                "round_scores": [
                    {
                        "round": r.round_number,
                        "pro_score": round(r.pro_score, 3),
                        "con_score": round(r.con_score, 3),
                        "concession": r.concession,
                    }
                    for r in round_evaluations
                ],
                "pro_avg_score": round(pro_avg_score, 3),
                "con_avg_score": round(con_avg_score, 3),
                "termination_reason": termination_reason,
            }
            questions: dict[str, Any] = {
                "winner": {
                    "type": "choice",
                    "instructions": (
                        "Evaluate the entire debate and determine which side prevailed "
                        "overall based on argumentative rigor, refutation quality, and "
                        "consistency."
                    ),
                    "criteria": {
                        "pro": "Debater A (IN FAVOR) presented stronger, more rigorous arguments and refutations",
                        "con": "Debater B (AGAINST) presented stronger, more rigorous arguments and refutations",
                        "draw": "Both debaters performed with equivalent strength and neither prevailed decisively",
                    },
                },
                "decisive_factor": {
                    "type": "choice",
                    "instructions": "Select the primary factor that determined the outcome of the debate.",
                    "criteria": {
                        "logical_refutation": "Superior logical refutation of the opponent's core premises",
                        "argumentative_depth": "Greater rigor, precision, and consistency across rounds",
                        "opponent_concession": "The opponent conceded key premises or contradicted their stance",
                        "opponent_repetition": "The opponent fell into repetition or evaded core counterarguments",
                        "balanced_stalemate": "Neither side achieved a decisive advantage",
                    },
                },
            }
            decision = self.client.systemone(
                model=self.config.models.decision_model,
                state=state,
                questions=questions,
                keep_alive=self.config.ollama.keep_alive,
            )
            winner_answer = decision.answers["winner"]
            winner = winner_answer.choice
            probabilities = dict(winner_answer.probabilities)
            confidence = winner_answer.confidence
            decisive_factor = decision.answers["decisive_factor"].choice

        summary_prompt = self.config.render_prompt(
            "verdict_summary.j2",
            language=self.config.debate.language,
            topic=context.topic,
            winner_label=WINNER_LABELS[winner],
            probabilities=probabilities,
            confidence=confidence,
            decisive_factor=decisive_factor,
            pro_avg_score=pro_avg_score,
            con_avg_score=con_avg_score,
            termination_reason=termination_reason,
            compact_summary=context.compact_summary,
            recent_transcript=context.format_turns(context.active_turns()),
        )
        explanation_resp = self.client.chat(
            model=self.config.models.language_model,
            messages=[{"role": "user", "content": summary_prompt}],
            think=False,
            options={
                "temperature": self.config.models.compactor_temperature,
                "num_ctx": self.config.context.num_ctx,
            },
            keep_alive=self.config.ollama.keep_alive,
        )

        return Verdict(
            winner=winner,
            probabilities=probabilities,
            confidence=confidence,
            decisive_factor=decisive_factor,
            explanation=explanation_resp.message.content.strip(),
            pro_avg_score=pro_avg_score,
            con_avg_score=con_avg_score,
        )
