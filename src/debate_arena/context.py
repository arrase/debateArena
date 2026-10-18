from dataclasses import dataclass, field

from debate_arena.config import AppConfig
from debate_arena.models import Role, Turn

SPEAKER_NAMES: dict[Role, str] = {
    "pro": "Debatiente A (A favor)",
    "con": "Debatiente B (En contra)",
}

STANCE_LABELS: dict[Role, str] = {
    "pro": "A FAVOR de la tesis",
    "con": "EN CONTRA de la tesis",
}


@dataclass
class DebateContext:
    topic: str
    config: AppConfig
    turns: list[Turn] = field(default_factory=list)
    compacted_turn_count: int = 0
    compact_summary: str = ""
    exhausted_arguments: list[str] = field(default_factory=list)
    compactions: int = 0
    last_prompt_tokens: int = 0

    def active_turns(self) -> list[Turn]:
        return self.turns[self.compacted_turn_count :]

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        self.last_prompt_tokens = turn.prompt_tokens

    def build_debater_messages(
        self, speaker: Role, retry_feedback: str = ""
    ) -> list[dict[str, str]]:
        system_content = self.config.render_prompt(
            "debater_system.j2",
            speaker_name=SPEAKER_NAMES[speaker],
            stance_label=STANCE_LABELS[speaker],
            topic=self.topic,
            language=self.config.debate.language,
            max_response_words=self.config.debate.max_response_words,
            compact_summary=self.compact_summary,
            exhausted_arguments=self.exhausted_arguments,
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        active = self.active_turns()

        if not active or active[0].speaker == speaker:
            opening_content = self.config.render_prompt(
                "opening_user.j2",
                is_opening=(self.compacted_turn_count == 0 and not active),
                topic=self.topic,
                stance_label=STANCE_LABELS[speaker],
                language=self.config.debate.language,
            )
            messages.append({"role": "user", "content": opening_content})

        for turn in active:
            role = "assistant" if turn.speaker == speaker else "user"
            messages.append({"role": role, "content": turn.content})

        if retry_feedback:
            messages[-1]["content"] = (
                f"{messages[-1]['content']}\n\n"
                f"Corrección obligatoria del juez: {retry_feedback}"
            )

        return messages

    def should_compact(self, round_number: int) -> bool:
        active_rounds = len(self.active_turns()) // 2
        if (
            round_number >= self.config.debate.max_rounds
            or active_rounds <= self.config.context.retain_recent_rounds
        ):
            return False

        token_threshold = int(
            self.config.context.num_ctx * self.config.context.compaction_threshold_ratio
        )
        return (
            round_number % self.config.context.compaction_interval_rounds == 0
            or self.last_prompt_tokens >= token_threshold
        )

    def turns_to_compact(self) -> list[Turn]:
        active = self.active_turns()
        keep_turns = self.config.context.retain_recent_rounds * 2
        if len(active) <= keep_turns:
            return []
        return active[:-keep_turns] if keep_turns > 0 else active

    def apply_compaction(
        self, compacted_count: int, new_summary: str, new_exhausted: list[str]
    ) -> None:
        self.compacted_turn_count += compacted_count
        self.compact_summary = new_summary.strip()
        for arg in new_exhausted:
            cleaned = arg.strip()
            if cleaned and cleaned not in self.exhausted_arguments:
                self.exhausted_arguments.append(cleaned)
        self.compactions += 1
        self.last_prompt_tokens = 0

    def format_turns(self, turns: list[Turn]) -> str:
        return "\n\n".join(
            f"[Ronda {t.round_number}] {t.speaker_name}: {t.content}" for t in turns
        )
