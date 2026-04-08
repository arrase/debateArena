from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


def render_transcript(entries: Sequence[Mapping[str, Any]]) -> str:
    if not entries:
        return "[No previous turns]"
    lines = []
    for entry in entries:
        lines.append(
            f"Round {entry['round_number']} - {entry['speaker']} ({entry['role']}): {entry['content']}".strip()
        )
    return "\n\n".join(lines)


def render_restrictions(restrictions: Sequence[str]) -> str:
    if not restrictions:
        return "- None."
    return "\n".join(f"- {restriction}" for restriction in restrictions)


def speaker_name_for_role(role: str) -> str:
    return {
        "debater_a": "Debater A",
        "debater_b": "Debater B",
        "referee": "Referee",
        "compactor": "Compactor",
    }.get(role, role.replace("_", " ").title())


def unique_lines(values: Iterable[Any]) -> List[str]:
    ordered: List[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return ordered
