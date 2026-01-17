from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory


class DebateAgent:
    def __init__(self, name: str, model_name: str, temperature: float, system_prompt: str):
        """Initialize the agent and its memory."""
        self.name = name
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.base_system_prompt = system_prompt
        self.current_system_prompt = system_prompt
        self.memory = ChatMessageHistory()
        self.memory.add_message(SystemMessage(content=system_prompt))
        self.checkpoint_count = 0

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
        """Clear memory and re-apply current system prompt."""
        self.memory.clear()
        self.memory.add_message(SystemMessage(content=self.current_system_prompt))
    
    def reset_with_restrictions(
        self, 
        restrictions: str, 
        context_summary: Optional[str] = None,
        last_exchange: Optional[str] = None
    ):
        """
        Reset the agent with new restrictions injected into the system prompt.
        
        This is the key mechanism to prevent argument loops in limited-context models.
        By restarting the conversation with updated restrictions, we ensure the model
        is aware of exhausted argument lines without requiring long context windows.
        
        Args:
            restrictions: Text describing forbidden/exhausted arguments
            context_summary: Optional summary of debate progress (includes identity block)
            last_exchange: Optional last message to continue from (includes role reminder)
        """
        self.checkpoint_count += 1
        
        # Build new system prompt with identity reinforcement at the TOP
        new_prompt_parts = []
        
        # CRITICAL: Add context summary (with identity block) FIRST for maximum visibility
        if context_summary:
            new_prompt_parts.append(context_summary)
            new_prompt_parts.append("")  # Empty line for separation
        
        # Then add the base system prompt
        new_prompt_parts.append(self.base_system_prompt)
        
        # Finally add restrictions
        if restrictions:
            new_prompt_parts.append(restrictions)
        
        self.current_system_prompt = "\n".join(new_prompt_parts)
        
        # Reset memory with new prompt
        self.memory.clear()
        self.memory.add_message(SystemMessage(content=self.current_system_prompt))
        
        # If there's a last exchange, add it to provide immediate context with role reminder
        if last_exchange:
            self.memory.add_message(HumanMessage(content=last_exchange))
    
    def get_message_count(self) -> int:
        """Get the number of messages in memory."""
        return len(self.memory.messages)
