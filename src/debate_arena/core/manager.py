import time
import json
from typing import Dict, Any, Optional, List, Tuple
from debate_arena.agents.debater import DebateAgent
from debate_arena.agents.summarizer import SummarizerAgent, DebateSummary
from rich.console import Console
from rich.markdown import Markdown


class DebateManager:
    def __init__(self, config: Dict[str, Any], topic: str, output_file: Optional[str] = None):
        """Initialize the manager and both agents."""
        self.config = config
        self.topic = topic
        self.output_file = output_file
        self.max_turns = config['debate']['max_turns']
        self.language = config.get('debate', {}).get('language', 'English')
        self.history: List[Tuple[str, str]] = []
        self.console = Console()
        
        # Checkpoint configuration
        checkpoint_config = config.get('checkpoint', {})
        self.checkpoint_interval = checkpoint_config.get('interval_turns', 5)
        self.max_violations = checkpoint_config.get('max_violations', 3)
        
        # Create debater agents
        self.agent_a = self._create_agent('debater_a', config['models']['debater_a'])
        self.agent_b = self._create_agent('debater_b', config['models']['debater_b'])
        
        # Create judge (always enabled)
        judge_config = config['models']['judge']
        self.judge = self._create_agent('judge', judge_config)
        self.judge_evaluation_prompt = judge_config.get('evaluation_prompt', '')
        self.judge_forced_verdict_prompt = judge_config.get('forced_verdict_prompt', '')
        
        # Create summarizer for anti-loop mechanism
        self.summarizer: Optional[SummarizerAgent] = None
        summarizer_config = config.get('models', {}).get('summarizer')
        self.summarizer_enabled = checkpoint_config.get('enabled', True)
        if self.summarizer_enabled:
            model_name = summarizer_config.get('name') if summarizer_config else config['models']['judge']['name']
            temperature = summarizer_config.get('temperature', 0.1) if summarizer_config else 0.1
            self.summarizer = SummarizerAgent(
                model_name=model_name,
                temperature=temperature,
                language=self.language
            )
        
        # Track checkpoint state
        self.last_checkpoint_turn = 0
        self.current_restrictions = ""
        self.forced_end = False
        self.end_reason = ""
        
    def _create_agent(self, key: str, model_config: Dict[str, Any]) -> DebateAgent:
        system_prompt = model_config['system_prompt'].format(topic=self.topic, language=self.language)
        return DebateAgent(
            name=key.replace('_', ' ').title(),
            model_name=model_config['name'],
            temperature=model_config['temperature'],
            system_prompt=system_prompt,
        )

    def _log_to_file(self, message: str):
        """Append message to the output file if configured."""
        if self.output_file:
            try:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(message + "\n")
            except Exception as e:
                self.console.print(f"[bold red][Warning] Failed to write to file: {e}[/bold red]")

    def _log(self, message: str):
        """Print message to stdout and optional file."""
        self.console.print(message)
        self._log_to_file(message)

    def _record_verdict(self, verdict: Dict[str, Any]):
        """Record judge verdict in transcript and history."""
        winner = verdict.get("winner", "draw")
        reason = verdict.get("reason", "")
        verdict_line = f"[Judge verdict] decision=end winner={winner} reason={reason}"
        self.history.append(("Judge", verdict_line))
        self._log(verdict_line)

    def _evaluate_with_judge(self, force_verdict: bool = False, exhausted_args_reason: str = "") -> Optional[Dict[str, Any]]:
        """
        Ask the judge to evaluate the debate state.
        
        The judge checks for:
        1. Agreement between debaters
        2. One debater completely refuting all of the opponent's arguments
        3. Forced verdict due to argument exhaustion
        
        Args:
            force_verdict: If True, judge must provide final verdict
            exhausted_args_reason: Reason for termination due to exhausted arguments
        """
        if not self.judge:
            return None

        transcript_lines = []
        for speaker, content in self.history[-10:]:
            transcript_lines.append(f"{speaker}: {content}")
        transcript = "\n".join(transcript_lines)

        if force_verdict:
            termination_reason = exhausted_args_reason or self.end_reason
            judge_prompt = self.judge_forced_verdict_prompt.format(
                termination_reason=termination_reason,
                topic=self.topic,
                transcript=transcript,
                language=self.language
            )
        else:
            judge_prompt = self.judge_evaluation_prompt.format(
                topic=self.topic,
                transcript=transcript,
                language=self.language
            )
        
        response = self.judge.run(judge_prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass
            if force_verdict:
                return {"decision": "end", "winner": "draw", "reason": exhausted_args_reason or self.end_reason}
            return None

    def _perform_checkpoint(self, turn: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Perform a checkpoint with the following sequence:
        
        1. JUDGE EVALUATION: Check if there's agreement or total refutation
           - If one debater has irrefutably defeated all opponent's arguments -> END
           - If both debaters have reached consensus -> END
           
        2. SUMMARIZER ANALYSIS: If debate continues, analyze conversation
           - Generate list of exhausted arguments (no more room to develop)
           - Instruct debaters to use different argumentative lines
           - If no new arguments possible -> END without consensus
        
        Returns:
            Tuple[bool, Optional[Dict]]: (should_end, judge_verdict if any)
        """
        self._log(f"\n[Checkpoint at turn {turn}]")
        
        # =====================================================
        # STEP 1: JUDGE EVALUATION - Check for agreement/refutation
        # =====================================================
        self._log("[Checkpoint] Judge evaluating for agreement or total refutation...")
        verdict = self._evaluate_with_judge()
        if verdict and verdict.get("decision") == "end":
            self._log(f"[Checkpoint] Judge determined debate should end: {verdict.get('reason', '')}")
            return True, verdict
        self._log("[Checkpoint] Judge: Debate should continue, no clear winner yet.")
        
        # =====================================================
        # STEP 2: SUMMARIZER - Analyze and restrict exhausted arguments
        # =====================================================
        if not self.summarizer:
            return False, None
        
        self._log("[Checkpoint] Analyzing debate progress and exhausted arguments...")
        
        # Get recent history for analysis (since last checkpoint)
        history_slice = self.history[-(self.checkpoint_interval * 2 + 2):]
        
        # Analyze the debate
        summary, should_end, end_reason = self.summarizer.analyze_debate(
            transcript=history_slice,
            topic=self.topic,
            previous_restrictions=self.current_restrictions if self.current_restrictions else None
        )
        
        # Generate new restrictions
        new_restrictions = self.summarizer.get_restriction_text()
        
        if new_restrictions != self.current_restrictions:
            self.current_restrictions = new_restrictions
            
            # Get last exchanges for continuity - each debater needs their OPPONENT's last message
            # History format: [(speaker, message), ...] alternating A, B, A, B...
            last_message_from_a = None
            last_message_from_b = None
            for speaker, message in reversed(self.history):
                if last_message_from_a is None and "Debater A" in speaker:
                    last_message_from_a = message
                if last_message_from_b is None and "Debater B" in speaker:
                    last_message_from_b = message
                if last_message_from_a and last_message_from_b:
                    break
            
            # Generate context summaries with role clarity for each debater
            context_summary_a = self._generate_context_summary(
                summary, 
                debater_name="Debater A",
                stance="PRO",
                stance_description=f"You are arguing IN FAVOR of: '{self.topic}'"
            )
            context_summary_b = self._generate_context_summary(
                summary,
                debater_name="Debater B",
                stance="CON",
                stance_description=f"You are arguing AGAINST: '{self.topic}'"
            )
            
            # Reset both agents with new restrictions and their respective position context
            # CRITICAL: Each debater receives their OPPONENT's last message, not their own
            self._log("[Checkpoint] Updating debater prompts with new restrictions...")
            self.agent_a.reset_with_restrictions(
                restrictions=new_restrictions,
                context_summary=context_summary_a,
                last_exchange=last_message_from_b  # A receives B's last message
            )
            self.agent_b.reset_with_restrictions(
                restrictions=new_restrictions,
                context_summary=context_summary_b,
                last_exchange=last_message_from_a  # B receives A's last message
            )
            
            self._log(f"[Checkpoint] Exhausted arguments: {len(summary.exhausted_arguments)}")
            self._log(f"[Checkpoint] Total violations detected: {summary.total_violations}")
        
        # =====================================================
        # STEP 3: CHECK FOR TERMINATION CONDITIONS
        # =====================================================
        
        # Check if summarizer detected debate should end (argument exhaustion detected by LLM)
        if should_end:
            self.forced_end = True
            self.end_reason = end_reason or "Debate terminated: all argumentative lines have been exhausted"
            self._log(f"\n[Checkpoint] {self.end_reason}")
            verdict = self._evaluate_with_judge(
                force_verdict=True, 
                exhausted_args_reason="Debate ended without consensus - no new arguments available"
            )
            return True, verdict
        
        # Check for rule violations (debaters repeating exhausted arguments)
        if summary.total_violations >= self.max_violations:
            self.forced_end = True
            exhausted_list = ", ".join(summary.exhausted_arguments[:5]) if summary.exhausted_arguments else "multiple arguments"
            self.end_reason = (
                f"Debate terminated without consensus: {summary.total_violations} violations detected. "
                f"Debaters failed to present new arguments after exhausting: {exhausted_list}"
            )
            self._log(f"\n[Checkpoint] {self.end_reason}")
            verdict = self._evaluate_with_judge(
                force_verdict=True,
                exhausted_args_reason=self.end_reason
            )
            return True, verdict
        
        self.last_checkpoint_turn = turn
        self._log("[Checkpoint] Debate continues with updated restrictions.")
        return False, None
    
    def _generate_identity_block(self, debater_name: str, stance: str, stance_description: str) -> str:
        """Generate a clear identity reminder block.
        
        Args:
            debater_name: The debater's name (e.g., 'Debater A')
            stance: PRO or CON
            stance_description: Description of what position to defend
        """
        return f"""
=== ROLE REMINDER ===
You are: {debater_name}
Your assigned stance: {stance}
{stance_description}
=====================
"""

    def _generate_context_summary(self, summary: DebateSummary, debater_name: str = "", stance: str = "", stance_description: str = "") -> str:
        """Generate a brief context summary for debater continuity.
        
        Args:
            summary: The debate summary containing key points and focus
            debater_name: The debater's name (e.g., 'Debater A')
            stance: PRO or CON
            stance_description: Description of what position to defend
        """
        parts = []
        
        # CRITICAL: Include highly visible identity block to prevent role confusion
        if debater_name and stance:
            parts.append(self._generate_identity_block(debater_name, stance, stance_description))
        
        if summary.current_focus:
            parts.append(f"Current focus: {summary.current_focus}")
        
        if summary.key_points:
            parts.append("Key developments:")
            for point in summary.key_points[-3:]:
                parts.append(f"  - {point}")
        
        return "\n".join(parts) if parts else ""

    def run_debate(self):
        """Execute the debate loop."""
        if self.output_file:
            try:
                with open(self.output_file, 'w', encoding='utf-8') as f:
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

            self.console.print(f"\n[yellow][{self.agent_a.name} is thinking...][/yellow]")
            response_a = self.agent_a.run(last_message)
            self.console.print(f"[bold blue]{self.agent_a.name}:[/bold blue]")
            self.console.print(Markdown(response_a))
            self._log_to_file(f"{self.agent_a.name}: {response_a}")
            self.history.append((self.agent_a.name, response_a))
            last_message = response_a

            time.sleep(1)

            self.console.print(f"\n[yellow][{self.agent_b.name} is thinking...][/yellow]")
            response_b = self.agent_b.run(last_message)
            self.console.print(f"[bold green]{self.agent_b.name}:[/bold green]")
            self.console.print(Markdown(response_b))
            self._log_to_file(f"{self.agent_b.name}: {response_b}")
            self.history.append((self.agent_b.name, response_b))
            last_message = response_b

            # Perform checkpoint if interval reached (includes judge evaluation)
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

        self._log(f"\n=== DEBATE FINISHED ===")
        
        # Print final statistics if summarizer was used
        if self.summarizer and self.summarizer.cumulative_summary:
            summary = self.summarizer.cumulative_summary
            self._log(f"\n=== DEBATE STATISTICS ===")
            self._log(f"Total turns: {turn}")
            self._log(f"Checkpoints performed: {self.agent_a.checkpoint_count}")
            self._log(f"Exhausted argument lines: {len(summary.exhausted_arguments)}")
            self._log(f"Rule violations: {summary.total_violations}")
