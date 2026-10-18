from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from debate_arena.config import AppConfig
from debate_arena.context import SPEAKER_NAMES
from debate_arena.judge import WINNER_LABELS
from debate_arena.models import (
    DebateResult,
    RoundEvaluation,
    Turn,
    TurnValidation,
)


class ConsolePresenter:
    def __init__(self) -> None:
        self.console = Console()

    def show_header(self, topic: str, config: AppConfig) -> None:
        header_text = (
            f"[bold]Tesis:[/bold] {topic}\n"
            f"[bold]Modelo de lenguaje:[/bold] {config.models.language_model}  |  "
            f"[bold]Juez (SystemOne):[/bold] {config.models.decision_model}\n"
            f"[bold]Rondas máximas:[/bold] {config.debate.max_rounds}  |  "
            f"[bold]Ventana (num_ctx):[/bold] {config.context.num_ctx}"
        )
        self.console.print(
            Panel(
                header_text,
                title="[bold cyan]DEBATE ARENA[/bold cyan]",
                border_style="cyan",
            )
        )

    def show_round_start(self, round_number: int, max_rounds: int) -> None:
        self.console.print()
        self.console.print(
            Rule(f"[bold yellow]Ronda {round_number} / {max_rounds}[/bold yellow]")
        )

    def show_turn(self, turn: Turn, validation: TurnValidation) -> None:
        border_style = "green" if turn.speaker == "pro" else "red"
        subtitle = (
            f"prompt: {turn.prompt_tokens} tok | "
            f"resp: {turn.completion_tokens} tok | "
            f"postura d1: {validation.keeps_stance_prob:.0%}"
        )
        self.console.print(
            Panel(
                Markdown(turn.content),
                title=f"[bold]{turn.speaker_name}[/bold]",
                subtitle=subtitle,
                border_style=border_style,
            )
        )

    def show_retry(
        self,
        speaker_name: str,
        attempt: int,
        max_attempts: int,
        validation: TurnValidation,
    ) -> None:
        self.console.print(
            f"[yellow]⚠ Reintento ({attempt}/{max_attempts}) para {speaker_name}: "
            f"{validation.reason} "
            f"(postura={validation.keeps_stance_prob:.0%}, "
            f"repetición={validation.repeats_exhausted_prob:.0%})[/yellow]"
        )

    def show_round_evaluation(self, evaluation: RoundEvaluation) -> None:
        summary = (
            f"[bold]Evaluación d1 (Ronda {evaluation.round_number}):[/bold] "
            f"PRO={evaluation.pro_score:.2f} | CON={evaluation.con_score:.2f} | "
            f"Concesión={evaluation.concession} | "
            f"Estancamiento={evaluation.stalemate_prob:.0%} | "
            f"Cerrar={evaluation.should_end_prob:.0%}\n"
            f"[dim]{evaluation.reason}[/dim]"
        )
        self.console.print(Panel(summary, border_style="blue"))

    def show_compaction(
        self, compacted_count: int, exhausted_count: int, compactions: int
    ) -> None:
        self.console.print(
            f"[magenta]⚙ Compactación #{compactions}: "
            f"{compacted_count} turnos resumidos | "
            f"{exhausted_count} argumentos marcados como agotados[/magenta]"
        )

    def show_verdict(self, result: DebateResult) -> None:
        verdict = result.verdict
        winner_label = WINNER_LABELS[verdict.winner]

        table = Table(title="Probabilidades Calibradas (d1:latest)", expand=True)
        table.add_column("Opción", style="bold")
        table.add_column("Probabilidad", justify="right")
        for key, label in WINNER_LABELS.items():
            prob = verdict.probabilities[key]
            table.add_row(label, f"{prob:.1%}")

        self.console.print()
        self.console.print(Rule("[bold cyan]VEREDICTO FINAL[/bold cyan]"))
        self.console.print(table)

        meta = (
            f"**Ganador:** {winner_label}  \n"
            f"**Confianza d1:** {verdict.confidence:.1%}  \n"
            f"**Factor decisivo:** `{verdict.decisive_factor}`  \n"
            f"**Puntuación media:** {SPEAKER_NAMES['pro']}: `{verdict.pro_avg_score:.2f}` | "
            f"{SPEAKER_NAMES['con']}: `{verdict.con_avg_score:.2f}`\n\n"
            f"---\n\n{verdict.explanation}"
        )
        self.console.print(
            Panel(
                Markdown(meta),
                title="[bold yellow]Fallo del Juez[/bold yellow]",
                border_style="yellow",
            )
        )

    def save_result(self, result: DebateResult, output_path: Path) -> None:
        verdict = result.verdict
        winner_label = WINNER_LABELS[verdict.winner]
        lines: list[str] = [
            f"# Debate Arena: {result.topic}",
            "",
            "## Transcripción del Debate",
            "",
        ]
        for turn in result.turns:
            lines.append(f"### [Ronda {turn.round_number}] {turn.speaker_name}")
            lines.append("")
            lines.append(turn.content)
            lines.append("")

        lines.extend(
            [
                "## Evaluaciones por Ronda (d1:latest)",
                "",
            ]
        )
        for ev in result.round_evaluations:
            lines.append(
                f"- **Ronda {ev.round_number}**: PRO={ev.pro_score:.2f}, "
                f"CON={ev.con_score:.2f}, concesión=`{ev.concession}`, "
                f"estancamiento={ev.stalemate_prob:.0%}, "
                f"cierre={ev.should_end_prob:.0%} — {ev.reason}"
            )

        lines.extend(
            [
                "",
                "## Veredicto Final",
                "",
                f"- **Ganador:** {winner_label}",
                f"- **Confianza:** {verdict.confidence:.1%}",
                f"- **Factor decisivo:** `{verdict.decisive_factor}`",
                f"- **Media PRO:** {verdict.pro_avg_score:.2f}",
                f"- **Media CON:** {verdict.con_avg_score:.2f}",
                f"- **Probabilidades:** PRO={verdict.probabilities['pro']:.1%}, "
                f"CON={verdict.probabilities['con']:.1%}, "
                f"Empate={verdict.probabilities['draw']:.1%}",
                "",
                "### Explicación del Juez",
                "",
                verdict.explanation,
                "",
            ]
        )
        output_path.write_text("\n".join(lines), encoding="utf-8")
        self.console.print(
            f"[green]✔ Transcripción guardada en {output_path}[/green]"
        )
