import time
import json
from typing import Dict, Any, Optional
from debate_arena.agents.debater import DebateAgent

class DebateManager:
    def __init__(self, config: Dict[str, Any], topic: str, output_file: Optional[str] = None):
        """Initialize the manager and both agents."""
        self.config = config
        self.topic = topic
        self.output_file = output_file
        self.max_turns = config['debate']['max_turns']
        self.language = config.get('debate', {}).get('language', 'English')
        self.history = []
        self.agent_a = self._create_agent('debater_a', config['models']['debater_a'])
        self.agent_b = self._create_agent('debater_b', config['models']['debater_b'])
        self.judge = None
        judge_config = config.get('models', {}).get('judge')
        self.judge_enabled = config.get('debate', {}).get('judge_enabled', False)
        if self.judge_enabled and judge_config:
            self.judge = self._create_agent('judge', judge_config)
        
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

    def _evaluate_with_judge(self) -> Optional[Dict[str, Any]]:
        """Ask the judge to decide if the debate should end."""
        if not self.judge:
            return None

        transcript_lines = []
        for speaker, content in self.history[-10:]:
            transcript_lines.append(f"{speaker}: {content}")
        transcript = "\n".join(transcript_lines)

        judge_prompt = (
            "Analiza el siguiente tramo de debate y decide si ya existe acuerdo o un vencedor claro. "
            "Responde SOLO con un JSON válido en una sola línea con las claves: "
            "decision (continue|end), winner (debater_a|debater_b|draw), reason. "
            f"La clave 'reason' debe estar en {self.language}.\n\n"
            f"Tema: {self.topic}\n\n"
            f"Transcripción:\n{transcript}\n"
        )
        response = self.judge.run(judge_prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None

    def run_debate(self):
        """Execute the debate loop."""
        if self.output_file:
            try:
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    f.write("")
            except Exception as e:
                print(f"[Warning] Could not initialize output file: {e}")

        self._log(f"\n=== STARTING DEBATE: {self.topic} ===\n")
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
