import time
import json
from typing import Dict, Any, Optional, List, Tuple
from debate_arena.agents.debater import DebateAgent
from debate_arena.agents.summarizer import SummarizerAgent, DebateSummary


class DebateManager:
    def __init__(self, config: Dict[str, Any], topic: str, output_file: Optional[str] = None):
        """Initialize the manager and both agents."""
        self.config = config
        self.topic = topic
        self.output_file = output_file
        self.max_turns = config['debate']['max_turns']
        self.language = config.get('debate', {}).get('language', 'English')
        self.history: List[Tuple[str, str]] = []
        
        # Checkpoint configuration
        checkpoint_config = config.get('checkpoint', {})
        self.checkpoint_interval = checkpoint_config.get('interval_turns', 5)
        self.max_violations = checkpoint_config.get('max_violations', 3)
        
        # Create debater agents
        self.agent_a = self._create_agent('debater_a', config['models']['debater_a'])
        self.agent_b = self._create_agent('debater_b', config['models']['debater_b'])
        
        # Create judge if enabled
        self.judge = None
        judge_config = config.get('models', {}).get('judge')
        self.judge_enabled = config.get('debate', {}).get('judge_enabled', False)
        if self.judge_enabled and judge_config:
            self.judge = self._create_agent('judge', judge_config)
        
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

    def _log(self, message: str):
        """Print message to stdout and optional file."""
        print(message)
        if self.output_file:
            try:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(message + "\n")
            except Exception as e:
                print(f"[Warning] Failed to write to file: {e}")

    def _record_verdict(self, verdict: Dict[str, Any]):
        """Record judge verdict in transcript and history."""
        winner = verdict.get("winner", "draw")
        reason = verdict.get("reason", "")
        verdict_line = f"[Judge verdict] decision=end winner={winner} reason={reason}"
        self.history.append(("Judge", verdict_line))
        self._log(verdict_line)

    def _evaluate_with_judge(self, force_verdict: bool = False) -> Optional[Dict[str, Any]]:
        """Ask the judge to decide if the debate should end."""
        if not self.judge:
            return None

        transcript_lines = []
        for speaker, content in self.history[-10:]:
            transcript_lines.append(f"{speaker}: {content}")
        transcript = "\n".join(transcript_lines)

        if force_verdict:
            judge_prompt = (
                "The debate has been terminated due to rule violations or excessive repetition. "
                "You must provide a final verdict NOW. "
                "Respond ONLY with a valid JSON on a single line with the keys: "
                "decision (must be 'end'), winner (debater_a|debater_b|draw), reason. "
                f"The 'reason' key must be in {self.language}.\n\n"
                f"Termination reason: {self.end_reason}\n\n"
                f"Topic: {self.topic}\n\n"
                f"Transcript:\n{transcript}\n"
            )
        else:
            judge_prompt = (
                "Analyze the following segment of the debate and decide whether there is already agreement or a clear winner. "
                "Respond ONLY with a valid JSON on a single line with the keys: "
                "decision (continue|end), winner (debater_a|debater_b|draw), reason. "
                f"The 'reason' key must be in {self.language}.\n\n"
                f"Topic: {self.topic}\n\n"
                f"Transcript:\n{transcript}\n"
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
                return {"decision": "end", "winner": "draw", "reason": self.end_reason}
            return None

    def _perform_checkpoint(self, turn: int) -> bool:
        """
        Perform a checkpoint: analyze debate, update restrictions, reset agents if needed.
        
        This is the core anti-loop mechanism. Every N turns, we:
        1. Have the summarizer analyze the conversation
        2. Generate restrictions for exhausted arguments
        3. Reset debater agents with new prompts containing restrictions
        4. Check if debate should end due to violations
        
        Returns:
            bool: True if debate should end, False otherwise
        """
        if not self.summarizer:
            return False
        
        self._log(f"\n[Checkpoint at turn {turn}] Analyzing debate progress...")
        
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
            
            # Generate context summary for continuity
            context_summary = self._generate_context_summary(summary)
            
            # Get last exchange for continuity
            last_message = self.history[-1][1] if self.history else None
            
            # Reset both agents with new restrictions
            self._log("[Checkpoint] Updating debater prompts with new restrictions...")
            self.agent_a.reset_with_restrictions(
                restrictions=new_restrictions,
                context_summary=context_summary,
                last_exchange=last_message
            )
            self.agent_b.reset_with_restrictions(
                restrictions=new_restrictions,
                context_summary=context_summary,
                last_exchange=last_message
            )
            
            self._log(f"[Checkpoint] Exhausted arguments: {len(summary.exhausted_arguments)}")
            self._log(f"[Checkpoint] Total violations detected: {summary.total_violations}")
        
        # Check if we should force end
        if should_end:
            self.forced_end = True
            self.end_reason = end_reason or "Debate terminated due to argument exhaustion"
            self._log(f"\n[Checkpoint] Debate should end: {self.end_reason}")
            return True
        
        if summary.total_violations >= self.max_violations:
            self.forced_end = True
            self.end_reason = f"Debate terminated: {summary.total_violations} rule violations (repeated exhausted arguments)"
            self._log(f"\n[Checkpoint] {self.end_reason}")
            return True
        
        self.last_checkpoint_turn = turn
        return False
    
    def _generate_context_summary(self, summary: DebateSummary) -> str:
        """Generate a brief context summary for debater continuity."""
        parts = []
        
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
                print(f"[Warning] Could not initialize output file: {e}")

        self._log(f"\n=== STARTING DEBATE: {self.topic} ===\n")
        if self.summarizer_enabled:
            self._log(f"[Anti-loop mechanism enabled: checkpoint every {self.checkpoint_interval} turns]")
        
        last_message = f"The topic is: {self.topic}. Please present your opening argument."
        turn = 1
        
        while turn <= self.max_turns:
            self._log(f"\n--- Turn {turn}/{self.max_turns} ---")

            print(f"\n[{self.agent_a.name} is thinking...]")
            response_a = self.agent_a.run(last_message)
            self._log(f"{self.agent_a.name}: {response_a}")
            self.history.append((self.agent_a.name, response_a))
            last_message = response_a

            time.sleep(1)

            print(f"\n[{self.agent_b.name} is thinking...]")
            response_b = self.agent_b.run(last_message)
            self._log(f"{self.agent_b.name}: {response_b}")
            self.history.append((self.agent_b.name, response_b))
            last_message = response_b

            # Perform checkpoint if interval reached
            if self.summarizer_enabled and turn > 0 and turn % self.checkpoint_interval == 0:
                if self._perform_checkpoint(turn):
                    # Force judge verdict
                    if self.judge_enabled:
                        verdict = self._evaluate_with_judge(force_verdict=True)
                        if verdict:
                            self._record_verdict(verdict)
                            winner = verdict.get("winner", "draw")
                            reason = verdict.get("reason", "")
                            self._log(f"\n[Forced end] {reason}")
                    break

            # Regular judge evaluation
            if self.judge_enabled:
                verdict = self._evaluate_with_judge()
                if verdict and verdict.get("decision") == "end":
                    self._record_verdict(verdict)
                    winner = verdict.get("winner", "draw")
                    reason = verdict.get("reason", "")
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
