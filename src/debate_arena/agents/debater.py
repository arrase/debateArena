from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory


class DebateAgent:
    def __init__(self, name: str, model_name: str, temperature: float, system_prompt: str):
        self.name = name
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.base_system_prompt = system_prompt
        self.current_system_prompt = system_prompt
        self.memory = ChatMessageHistory()
        self.memory.add_message(SystemMessage(content=system_prompt))
        self.checkpoint_count = 0

    def run(self, opponent_message: str) -> str:
        self.memory.add_message(HumanMessage(content=opponent_message))
        try:
            content = self.llm.invoke(self.memory.messages).content
        except Exception as e:
            return f"[Error generating response: {e}]"
        self.memory.add_message(AIMessage(content=content))
        return content

    def reset(self):
        self.memory.clear()
        self.memory.add_message(SystemMessage(content=self.current_system_prompt))
    
    def reset_with_restrictions(
        self,
        restrictions: str,
        context_summary: Optional[str] = None,
        last_exchange: Optional[str] = None,
    ):
        self.checkpoint_count += 1
        parts = []
        if context_summary:
            parts.extend([context_summary, ""])
        parts.append(self.base_system_prompt)
        if restrictions:
            parts.append(restrictions)
        self.current_system_prompt = "\n".join(parts)
        self.memory.clear()
        self.memory.add_message(SystemMessage(content=self.current_system_prompt))
        if last_exchange:
            self.memory.add_message(HumanMessage(content=last_exchange))
    
    def get_message_count(self) -> int:
        return len(self.memory.messages)
