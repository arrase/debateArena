from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from debate_arena.domain.models import DebateResult


class ConsolePresenter:
    def __init__(self):
        self._console = Console()

    def present(self, result: DebateResult, output_file: Optional[Path] = None) -> None:
        self._console.print(Panel.fit(f"[bold]Debate topic:[/bold] {result.topic}", border_style="cyan"))
        for entry in result.transcript:
            color = "blue" if entry.role == "debater_a" else "green"
            self._console.print(f"[bold {color}]{entry.speaker}[/bold {color}]")
            self._console.print(Markdown(entry.content))
            self._console.print()

        self._console.print(
            Panel.fit(
                f"[bold]Winner:[/bold] {result.winner}\n[bold]Reason:[/bold] {result.reason}",
                border_style="magenta",
            )
        )
        if result.decisive_line:
            self._console.print(f"[bold]Decisive line:[/bold] {result.decisive_line}")
        if result.concessions_observed:
            self._console.print("[bold]Concessions observed:[/bold]")
            for concession in result.concessions_observed:
                self._console.print(f"- {concession}")
        if result.restrictions:
            self._console.print("[bold]Active restrictions:[/bold]")
            for restriction in result.restrictions:
                self._console.print(f"- {restriction}")
        if result.compact_summary:
            self._console.print("\n[bold]Compact summary:[/bold]")
            self._console.print(Markdown(result.compact_summary))

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(self._render_plaintext(result), encoding="utf-8")

    def _render_plaintext(self, result: DebateResult) -> str:
        lines = [f"Topic: {result.topic}", ""]
        for entry in result.transcript:
            lines.append(f"{entry.speaker} ({entry.role}) [round {entry.round_number}]:")
            lines.append(entry.content)
            lines.append("")
        lines.append(f"Winner: {result.winner}")
        lines.append(f"Reason: {result.reason}")
        if result.decisive_line:
            lines.append(f"Decisive line: {result.decisive_line}")
        if result.concessions_observed:
            lines.append("Concessions observed:")
            lines.extend(f"- {concession}" for concession in result.concessions_observed)
        if result.restrictions:
            lines.append("")
            lines.append("Restrictions:")
            lines.extend(f"- {restriction}" for restriction in result.restrictions)
        if result.compact_summary:
            lines.append("")
            lines.append("Compact summary:")
            lines.append(result.compact_summary)
        return "\n".join(lines).strip() + "\n"
