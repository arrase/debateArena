from ollama import Client

from debate_arena.config import AppConfig
from debate_arena.context import SPEAKER_NAMES, DebateContext
from debate_arena.judge import DebateJudge
from debate_arena.models import DebateResult, Role, RoundEvaluation, Turn, TurnValidation
from debate_arena.presenter import ConsolePresenter


def _execute_turn(
    speaker: Role,
    round_number: int,
    context: DebateContext,
    judge: DebateJudge,
    config: AppConfig,
    client: Client,
    presenter: ConsolePresenter,
) -> tuple[Turn, TurnValidation]:
    retry_feedback = ""
    max_attempts = config.debate.max_turn_attempts

    for attempt in range(1, max_attempts + 1):
        messages = context.build_debater_messages(speaker, retry_feedback)
        response = client.chat(
            model=config.models.language_model,
            messages=messages,
            think=False,
            options={
                "temperature": config.models.debater_temperature,
                "num_ctx": config.context.num_ctx,
            },
            keep_alive=config.ollama.keep_alive,
        )
        content = response.message.content.strip()
        turn = Turn(
            speaker=speaker,
            speaker_name=SPEAKER_NAMES[speaker],
            round_number=round_number,
            content=content,
            prompt_tokens=int(response.prompt_eval_count),
            completion_tokens=int(response.eval_count),
        )
        validation = judge.validate_turn(speaker, content, context)
        if validation.valid or attempt == max_attempts:
            return turn, validation

        presenter.show_retry(
            SPEAKER_NAMES[speaker], attempt, max_attempts, validation
        )
        retry_feedback = validation.reason

    raise RuntimeError("Unreachable turn execution state")


def run_debate(
    topic: str,
    config: AppConfig,
    client: Client,
    presenter: ConsolePresenter,
) -> DebateResult:
    context = DebateContext(topic=topic, config=config)
    judge = DebateJudge(config=config, client=client)
    round_evaluations: list[RoundEvaluation] = []
    termination_reason = f"Se alcanzó el límite máximo de {config.debate.max_rounds} rondas."
    forced_winner = ""

    presenter.show_header(topic, config)

    for round_number in range(1, config.debate.max_rounds + 1):
        presenter.show_round_start(round_number, config.debate.max_rounds)

        pro_turn, pro_val = _execute_turn(
            "pro", round_number, context, judge, config, client, presenter
        )
        context.add_turn(pro_turn)
        presenter.show_turn(pro_turn, pro_val)
        if not pro_val.valid:
            forced_winner = "con"
            termination_reason = (
                f"Descalificación de {SPEAKER_NAMES['pro']}: {pro_val.reason}"
            )
            break

        con_turn, con_val = _execute_turn(
            "con", round_number, context, judge, config, client, presenter
        )
        context.add_turn(con_turn)
        presenter.show_turn(con_turn, con_val)
        if not con_val.valid:
            forced_winner = "pro"
            termination_reason = (
                f"Descalificación de {SPEAKER_NAMES['con']}: {con_val.reason}"
            )
            break

        evaluation = judge.evaluate_round(
            round_number, pro_turn, con_turn, context
        )
        round_evaluations.append(evaluation)
        presenter.show_round_evaluation(evaluation)

        if evaluation.should_end:
            termination_reason = evaluation.reason
            break

        if context.should_compact(round_number):
            to_compact = context.turns_to_compact()
            new_summary, new_exhausted = judge.compact_context(
                to_compact, context
            )
            context.apply_compaction(
                len(to_compact), new_summary, new_exhausted
            )
            presenter.show_compaction(
                len(to_compact),
                len(context.exhausted_arguments),
                context.compactions,
            )

    verdict = judge.issue_verdict(
        context, round_evaluations, termination_reason, forced_winner
    )
    result = DebateResult(
        topic=topic,
        verdict=verdict,
        turns=context.turns,
        round_evaluations=round_evaluations,
        compact_summary=context.compact_summary,
        exhausted_arguments=context.exhausted_arguments,
        compactions=context.compactions,
    )
    presenter.show_verdict(result)
    return result
