from typing import List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

class DebateAgent:
    def __init__(self, name: str, model_name: str, temperature: float, system_prompt: str):
        """
        Initialize the DebateAgent.
        
        Args:
            name (str): The name/role of the agent (e.g., "Debater A").
            model_name (str): The Ollama model name (e.g., "gpt-oss:20b").
            temperature (float): The temperature for generation.
            system_prompt (str): The system prompt defining the persona.
        """
        self.name = name
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )
        self.system_prompt = system_prompt
        self.memory = ChatMessageHistory()
        
        # Add system prompt to memory initially
        self.memory.add_message(SystemMessage(content=system_prompt))

    def run(self, opponent_message: str) -> str:
        """
        Generate a response to the opponent's argument.
        
        Args:
            opponent_message (str): The argument from the opponent.
            
        Returns:
            str: The agent's response.
        """
        # Add opponent's message as HumanMessage (since it's external input to this agent)
        self.memory.add_message(HumanMessage(content=opponent_message))
        
        # Get messages from memory
        messages = self.memory.messages
        
        try:
            response = self.llm.invoke(messages)
            content = response.content
            
            # Store own response in memory
            self.memory.add_message(AIMessage(content=content))
            
            return content
        except Exception as e:
            return f"[Error generating response: {e}]"

    def reset(self):
        """Clear memory."""
        self.memory.clear()
        self.memory.add_message(SystemMessage(content=self.system_prompt))
