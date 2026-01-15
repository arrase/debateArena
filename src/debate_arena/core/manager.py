import time
from typing import Dict, Any, Optional
from debate_arena.agents.debater import DebateAgent

class DebateManager:
    def __init__(self, config: Dict[str, Any], topic: str, output_file: Optional[str] = None):
        """
        Initialize the DebateManager.
        
        Args:
            config (Dict[str, Any]): The full configuration dictionary.
            topic (str): The debate topic.
            output_file (Optional[str]): Path to file where debate transcript should be saved.
        """
        self.config = config
        self.topic = topic
        self.output_file = output_file
        self.max_turns = config['debate']['max_turns']
        self.history = []
        
        # Initialize Agents
        self.agent_a = self._create_agent('debater_a', config['models']['debater_a'])
        self.agent_b = self._create_agent('debater_b', config['models']['debater_b'])
        
    def _create_agent(self, key: str, model_config: Dict[str, Any]) -> DebateAgent:
        system_prompt = model_config['system_prompt'].format(topic=self.topic)
        return DebateAgent(
            name=key.replace('_', ' ').title(), # e.g. "Debater A"
            model_name=model_config['name'],
            temperature=model_config['temperature'],
            system_prompt=system_prompt
        )

    def _log(self, message: str):
        """Print message to stdout and optional file."""
        print(message)
        if self.output_file:
            try:
                # Open in append mode so we don't lose previous lines in the loop
                # However, for a fresh start each run, 'w' might be better used once or 
                # we assume the user manages the file.
                # A common pattern is 'a' for logs. But for a 'transcript', 
                # usually you want the whole thing.
                # If we open/close every time, 'a' is safer. 
                # If we wanted 'w' behavior we should clear it at start.
                pass 
                # Implementation detail: simplified to open append each time for safety 
                # regarding potential crashes, though less efficient.
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
            except Exception as e:
                print(f"[Warning] Failed to write to file: {e}")

    def run_debate(self):
        """Execute the debate loop."""
        
        # Clear file if it exists at start of run, to ensure fresh transcript
        if self.output_file:
             try:
                 with open(self.output_file, 'w', encoding='utf-8') as f:
                     f.write("")
             except Exception as e:
                 print(f"[Warning] Could not initialize output file: {e}")

        self._log(f"\n=== STARTING DEBATE: {self.topic} ===\n")
        
        # Initial trigger for Agent A (Pro)
        # We simulate a moderator start or just pass the topic as the first 'opponent' message to kickstart
        last_message = f"The topic is: {self.topic}. Please present your opening argument."
        
        turn = 1
        while turn <= self.max_turns:
            self._log(f"\n--- Turn {turn}/{self.max_turns} ---")
            
            # Agent A's Turn
            print(f"\n[{self.agent_a.name} is thinking...]") # Keep "thinking" status just to stdout to keep file clean?
            # Or log it? The user asked for "el debate se debe imprimir... y escribir".
            # Usually status messages aren't part of the "debate content". 
            # I will only _log the actual content and structural headers, but print the thinking status.
            
            response_a = self.agent_a.run(last_message)
            self._log(f"{self.agent_a.name}: {response_a}")
            self.history.append((self.agent_a.name, response_a))
            last_message = response_a
            
            # Check for stop conditions (e.g. if response is empty or specific stop token)
            # For now, just turn limit
            
            time.sleep(1) # Small pause for UX
            
            # Agent B's Turn
            print(f"\n[{self.agent_b.name} is thinking...]")
            response_b = self.agent_b.run(last_message)
            self._log(f"{self.agent_b.name}: {response_b}")
            self.history.append((self.agent_b.name, response_b))
            last_message = response_b
            
            turn += 1
            
        self._log(f"\n=== DEBATE FINISHED ===")
