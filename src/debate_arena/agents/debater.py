from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory

class DebateAgent:
    def __init__(self, name: str, model_name: str, temperature: float, system_prompt: str):
        """Initialize the agent and its memory."""
        self.name = name
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.system_prompt = system_prompt
        self.memory = ChatMessageHistory()
        self.memory.add_message(SystemMessage(content=system_prompt))

    def run(self, opponent_message: str) -> str:
        """Generate a response to the opponent's argument."""
        self.memory.add_message(HumanMessage(content=opponent_message))

        try:
            response = self.llm.invoke(self.memory.messages)
            content = response.content
            self.memory.add_message(AIMessage(content=content))
            return content
        except Exception as e:
            return f"[Error generating response: {e}]"

    def reset(self):
        """Clear memory and re-apply system prompt."""
        self.memory.clear()
        self.memory.add_message(SystemMessage(content=self.system_prompt))
