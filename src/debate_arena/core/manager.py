import time
import json
from typing import Dict, Any, Optional, List, Tuple
from debate_arena.agents.debater import DebateAgent
from debate_arena.agents.summarizer import SummarizerAgent, DebateSummary
from rich.console import Console
from rich.markdown import Markdown


class DebateManager:
    def __init__(self, config: Dict[str, Any], topic: str, output_file: Optional[str] = None):
        self.config = config
        self.topic = topic
        self.output_file = output_file
        self.max_turns = config["debate"]["max_turns"]
        self.language = config.get("debate", {}).get("response_language", "English")
        self.history: List[Tuple[str, str]] = []
        self.console = Console()

        checkpoint_config = config.get("checkpoint", {})
        self.checkpoint_interval = checkpoint_config.get("interval_turns", 5)
        self.max_violations = checkpoint_config.get("violation_limit", 3)

        self.agent_a = self._create_agent("debater_a", config["models"]["debater_a"])
        self.agent_b = self._create_agent("debater_b", config["models"]["debater_b"])

        judge_config = config["models"]["judge"]
        self.judge = self._create_agent("judge", judge_config)
        self.judge_evaluation_prompt = judge_config.get("evaluation_prompt", "")
        self.judge_forced_verdict_prompt = judge_config.get("forced_verdict_prompt", "")

        self.summarizer: Optional[SummarizerAgent] = None
        summarizer_config = config.get("models", {}).get("summarizer")
        self.summarizer_enabled = checkpoint_config.get("enabled", True)
        if self.summarizer_enabled:
            model_name = (summarizer_config or judge_config).get("name")
            temperature = (summarizer_config or {}).get("temperature", 0.1)
            self.summarizer = SummarizerAgent(
                model_name=model_name,
                temperature=temperature,
                language=self.language,
            )

        self.last_checkpoint_turn = 0
        self.current_restrictions = ""
        self.forced_end = False
        self.end_reason = ""
        
    def _create_agent(self, key: str, model_config: Dict[str, Any]) -> DebateAgent:
        system_prompt = model_config["system_prompt"].format(topic=self.topic, language=self.language)
        return DebateAgent(
            name=key.replace("_", " ").title(),
            model_name=model_config["name"],
            temperature=model_config["temperature"],
            system_prompt=system_prompt,
        )

    def _log_to_file(self, message: str):
        if self.output_file:
            try:
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(message + "\n")
            except Exception as e:
                self.console.print(f"[bold red][Warning] Failed to write to file: {e}[/bold red]")

    def _log(self, message: str):
        self.console.print(message)
        self._log_to_file(message)

    def _record_verdict(self, verdict: Dict[str, Any]):
        winner = verdict.get("winner", "draw")
        reason = verdict.get("reason", "")
        verdict_line = f"[Judge verdict] decision=end winner={winner} reason={reason}"
        self.history.append(("Judge", verdict_line))
        self._log(verdict_line)

    def _evaluate_with_judge(self, force_verdict: bool = False, exhausted_args_reason: str = "") -> Optional[Dict[str, Any]]:
        if not self.judge:
            return None
        transcript = "\n".join(f"{s}: {c}" for s, c in self.history[-10:])
        if force_verdict:
            judge_prompt = self.judge_forced_verdict_prompt.format(
                termination_reason=exhausted_args_reason or self.end_reason,
                topic=self.topic,
                transcript=transcript,
                language=self.language,
            )
        else:
            judge_prompt = self.judge_evaluation_prompt.format(
                topic=self.topic,
                transcript=transcript,
                language=self.language,
            )
        response = self.judge.run(judge_prompt)
        for payload in (response, response[response.find("{"):response.rfind("}") + 1]):
            try:
                return json.loads(payload)
            except Exception:
                continue
        if force_verdict:
            return {"decision": "end", "winner": "draw", "reason": exhausted_args_reason or self.end_reason}
        return None

    def _perform_checkpoint(self, turn: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        self._log(f"\n[Checkpoint at turn {turn}]")

        self._log("[Checkpoint] Judge evaluating for agreement or total refutation...")
        verdict = self._evaluate_with_judge()
        if verdict and verdict.get("decision") == "end":
            self._log(f"[Checkpoint] Judge determined debate should end: {verdict.get('reason', '')}")
            return True, verdict
        self._log("[Checkpoint] Judge: Debate should continue, no clear winner yet.")

        if not self.summarizer:
            return False, None
        self._log("[Checkpoint] Analyzing debate progress and exhausted arguments...")

        summary, should_end, end_reason = self.summarizer.analyze_debate(
            transcript=self.history[-(self.checkpoint_interval * 2 + 2):],
            topic=self.topic,
            previous_restrictions=self.current_restrictions or None,
        )
        new_restrictions = self.summarizer.get_restriction_text()

        if new_restrictions != self.current_restrictions:
            self.current_restrictions = new_restrictions
            last_a = last_b = None
            for speaker, message in reversed(self.history):
                if last_a is None and "Debater A" in speaker:
                    last_a = message
                if last_b is None and "Debater B" in speaker:
                    last_b = message
                if last_a and last_b:
                    break

            context_summary_a = self._generate_context_summary(
                summary,
                debater_name="Debater A",
                stance="PRO",
                stance_description=f"You are arguing IN FAVOR of: '{self.topic}'",
            )
            context_summary_b = self._generate_context_summary(
                summary,
                debater_name="Debater B",
                stance="CON",
                stance_description=f"You are arguing AGAINST: '{self.topic}'",
            )

            self._log("[Checkpoint] Updating debater prompts with new restrictions...")
            self.agent_a.reset_with_restrictions(
                restrictions=new_restrictions,
                context_summary=context_summary_a,
                last_exchange=last_b,
            )
            self.agent_b.reset_with_restrictions(
                restrictions=new_restrictions,
                context_summary=context_summary_b,
                last_exchange=last_a,
            )
            self._log(f"[Checkpoint] Exhausted arguments: {len(summary.exhausted_arguments)}")
            self._log(f"[Checkpoint] Total violations detected: {summary.total_violations}")

        if should_end:
            self.forced_end = True
            self.end_reason = end_reason or "Debate terminated: all argumentative lines have been exhausted"
            self._log(f"\n[Checkpoint] {self.end_reason}")
            verdict = self._evaluate_with_judge(
                force_verdict=True,
                exhausted_args_reason="Debate ended without consensus - no new arguments available",
            )
            return True, verdict

        if summary.total_violations >= self.max_violations:
            self.forced_end = True
            exhausted_list = ", ".join(summary.exhausted_arguments[:5]) if summary.exhausted_arguments else "multiple arguments"
            self.end_reason = (
                f"Debate terminated without consensus: {summary.total_violations} violations detected. "
                f"Debaters failed to present new arguments after exhausting: {exhausted_list}"
            )
            self._log(f"\n[Checkpoint] {self.end_reason}")
            verdict = self._evaluate_with_judge(force_verdict=True, exhausted_args_reason=self.end_reason)
            return True, verdict

        self.last_checkpoint_turn = turn
        self._log("[Checkpoint] Debate continues with updated restrictions.")
        return False, None
    
    def _generate_identity_block(self, debater_name: str, stance: str, stance_description: str) -> str:
        return f"""
=== ROLE REMINDER ===
You are: {debater_name}
Your assigned stance: {stance}
{stance_description}
=====================
"""

    def _generate_context_summary(
        self, summary: DebateSummary, debater_name: str = "", stance: str = "", stance_description: str = ""
    ) -> str:
        parts = []
        if debater_name and stance:
            parts.append(self._generate_identity_block(debater_name, stance, stance_description))
        if summary.current_focus:
            parts.append(f"Current focus: {summary.current_focus}")
        if summary.key_points:
            parts.append("Key developments:")
            parts.extend(f"  - {p}" for p in summary.key_points[-3:])
        return "\n".join(parts) if parts else ""

    def _run_turn(self, agent: DebateAgent, last_message: str, color: str) -> str:
        self.console.print(f"\n[yellow][{agent.name} is thinking...][/yellow]")
        response = agent.run(last_message)
        self.console.print(f"[bold {color}]{agent.name}:[/bold {color}]")
        self.console.print(Markdown(response))
        self._log_to_file(f"{agent.name}: {response}")
        self.history.append((agent.name, response))
        return response

    def run_debate(self):
        if self.output_file:
            try:
                with open(self.output_file, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception as e:
                self.console.print(f"[bold red][Warning] Could not initialize output file: {e}[/bold red]")

        self._log(f"\n=== STARTING DEBATE: {self.topic} ===\n")
        if self.summarizer_enabled:
            self._log(f"[Anti-loop mechanism enabled: checkpoint every {self.checkpoint_interval} turns]")

        last_message = f"The topic is: {self.topic}. Please present your opening argument."
        turn = 1
        while turn <= self.max_turns:
            self._log(f"\n--- Turn {turn}/{self.max_turns} ---")
            last_message = self._run_turn(self.agent_a, last_message, "blue")
            time.sleep(1)
            last_message = self._run_turn(self.agent_b, last_message, "green")

            if turn > 0 and turn % self.checkpoint_interval == 0:
                should_end, verdict = self._perform_checkpoint(turn)
                if should_end:
                    if verdict:
                        self._record_verdict(verdict)
                        winner = verdict.get("winner", "draw")
                        reason = verdict.get("reason", "")
                        if self.forced_end:
                            self._log(f"\n[Forced end] {reason}")
                        else:
                            self._log(f"\n[Early finish] Judge ended debate. Winner: {winner}. {reason}")
                    break
            turn += 1

        self._log("\n=== DEBATE FINISHED ===")
        if self.summarizer and self.summarizer.cumulative_summary:
            summary = self.summarizer.cumulative_summary
            self._log("\n=== DEBATE STATISTICS ===")
            self._log(f"Total turns: {turn}")
            self._log(f"Checkpoints performed: {self.agent_a.checkpoint_count}")
            self._log(f"Exhausted argument lines: {len(summary.exhausted_arguments)}")
            self._log(f"Rule violations: {summary.total_violations}")
