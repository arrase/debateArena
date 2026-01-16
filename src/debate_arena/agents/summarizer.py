"""
Summarizer Agent - Analyzes debate progress and tracks exhausted argument lines.

This agent periodically reviews the conversation to:
1. Identify arguments that have been used by each debater
2. Determine which arguments have been successfully refuted
3. Track stalemates where neither side has made progress
4. Generate restrictions for debaters to avoid repetitive argumentation
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


@dataclass
class ArgumentStatus:
    """Represents the status of an argument in the debate."""
    argument: str
    owner: str  # 'debater_a' or 'debater_b'
    status: str  # 'active', 'refuted', 'stalemate'
    refuted_by: Optional[str] = None
    turns_discussed: int = 1


@dataclass
class DebateSummary:
    """Summary of the debate state including argument restrictions."""
    exhausted_arguments: List[str] = field(default_factory=list)
    refuted_arguments: Dict[str, List[str]] = field(default_factory=dict)  # {debater: [args]}
    stalemate_topics: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    current_focus: str = ""
    total_violations: int = 0
    
    def to_restriction_text(self, language: str = "Spanish") -> str:
        """Generate restriction text to inject into debater prompts."""
        lines = []
        
        if self.exhausted_arguments:
            lines.append("FORBIDDEN ARGUMENT LINES (already exhausted):")
            for arg in self.exhausted_arguments:
                lines.append(f"  - {arg}")
        
        if self.refuted_arguments:
            lines.append("\nREFUTED ARGUMENTS (do not use these):")
            for debater, args in self.refuted_arguments.items():
                if args:
                    lines.append(f"  {debater}:")
                    for arg in args:
                        lines.append(f"    - {arg}")
        
        if self.stalemate_topics:
            lines.append("\nSTALEMATE TOPICS (both sides failed to make progress):")
            for topic in self.stalemate_topics:
                lines.append(f"  - {topic}")
        
        if self.key_points:
            lines.append("\nDEBATE SUMMARY SO FAR:")
            for point in self.key_points:
                lines.append(f"  - {point}")
        
        if lines:
            restriction = "\n".join(lines)
            return f"""
=== DEBATE PROGRESS RESTRICTIONS ===
{restriction}

CRITICAL: You MUST NOT repeat any exhausted, refuted, or stalemate arguments.
You must bring NEW perspectives or evidence. If you have no new arguments,
acknowledge it honestly.
=================================
"""
        return ""


class SummarizerAgent:
    """Agent responsible for summarizing debate progress and tracking argument exhaustion."""
    
    ANALYSIS_PROMPT = """You are a debate analyst. Analyze the following debate transcript and provide a structured JSON analysis.

Your task:
1. Identify ALL distinct arguments made by each debater
2. Determine which arguments have been REFUTED (opponent provided irrefutable counter-evidence)
3. Identify STALEMATES (same argument repeated 2+ times without progress)
4. Detect if any debater is repeating previously refuted arguments (violation)

Respond with ONLY a valid JSON object (no markdown, no extra text):
{{
    "debater_a_arguments": ["arg1", "arg2"],
    "debater_b_arguments": ["arg1", "arg2"],
    "refuted_arguments": {{
        "debater_a": ["refuted arg by A"],
        "debater_b": ["refuted arg by B"]
    }},
    "stalemate_topics": ["topic stuck in loop"],
    "exhausted_lines": ["argument lines that should not be used anymore"],
    "key_points": ["important developments in the debate"],
    "violations_detected": 0,
    "current_focus": "what the debate is currently about",
    "should_end": false,
    "end_reason": ""
}}

IMPORTANT:
- Be concise in argument descriptions (max 15 words each)
- "refuted" means the opponent provided evidence/logic that completely dismantles the argument
- "stalemate" means the same point was argued back and forth without new evidence
- "violations_detected" = number of times a debater repeated a previously exhausted argument
- "should_end" = true if debate has devolved into pure repetition or one side has clearly won
- Write descriptions in {language}

Topic: {topic}

Transcript:
{transcript}
"""

    def __init__(self, model_name: str, temperature: float = 0.1, language: str = "Spanish"):
        """Initialize the summarizer agent."""
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.language = language
        self.cumulative_summary = DebateSummary()
        self.analysis_history: List[Dict[str, Any]] = []
    
    def analyze_debate(
        self, 
        transcript: List[Tuple[str, str]], 
        topic: str,
        previous_restrictions: Optional[str] = None
    ) -> Tuple[DebateSummary, bool, str]:
        """
        Analyze the debate transcript and return updated summary.
        
        Args:
            transcript: List of (speaker, message) tuples
            topic: The debate topic
            previous_restrictions: Previous restrictions that were given
            
        Returns:
            Tuple of (DebateSummary, should_end, end_reason)
        """
        transcript_text = self._format_transcript(transcript)
        
        if previous_restrictions:
            transcript_text = f"[Previous restrictions given to debaters:\n{previous_restrictions}]\n\n{transcript_text}"
        
        prompt = self.ANALYSIS_PROMPT.format(
            topic=topic,
            transcript=transcript_text,
            language=self.language
        )
        
        messages = [
            SystemMessage(content="You are a precise debate analyst. Output only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            analysis = self._parse_analysis(response.content)
            
            if analysis:
                self._update_cumulative_summary(analysis)
                self.analysis_history.append(analysis)
                
                should_end = analysis.get("should_end", False)
                end_reason = analysis.get("end_reason", "")
                violations = analysis.get("violations_detected", 0)
                
                # Force end if too many violations
                if violations >= 3:
                    should_end = True
                    end_reason = f"Debate terminated: {violations} rule violations detected (repeated exhausted arguments)"
                
                return self.cumulative_summary, should_end, end_reason
                
        except Exception as e:
            print(f"[Summarizer] Analysis error: {e}")
        
        return self.cumulative_summary, False, ""
    
    def _format_transcript(self, transcript: List[Tuple[str, str]]) -> str:
        """Format transcript for analysis."""
        lines = []
        for speaker, message in transcript:
            # Truncate very long messages to save context
            if len(message) > 500:
                message = message[:500] + "..."
            lines.append(f"{speaker}: {message}")
        return "\n\n".join(lines)
    
    def _parse_analysis(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from the LLM."""
        # Clean up common issues
        response = response.strip()
        
        # Try to extract JSON if wrapped in markdown
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass
        return None
    
    def _update_cumulative_summary(self, analysis: Dict[str, Any]):
        """Update cumulative summary with new analysis."""
        # Add new exhausted arguments (avoid duplicates)
        for arg in analysis.get("exhausted_lines", []):
            if arg and arg not in self.cumulative_summary.exhausted_arguments:
                self.cumulative_summary.exhausted_arguments.append(arg)
        
        # Update refuted arguments
        for debater, args in analysis.get("refuted_arguments", {}).items():
            if debater not in self.cumulative_summary.refuted_arguments:
                self.cumulative_summary.refuted_arguments[debater] = []
            for arg in args:
                if arg and arg not in self.cumulative_summary.refuted_arguments[debater]:
                    self.cumulative_summary.refuted_arguments[debater].append(arg)
        
        # Add stalemate topics
        for topic in analysis.get("stalemate_topics", []):
            if topic and topic not in self.cumulative_summary.stalemate_topics:
                self.cumulative_summary.stalemate_topics.append(topic)
        
        # Update key points (keep last 5)
        for point in analysis.get("key_points", []):
            if point:
                self.cumulative_summary.key_points.append(point)
        self.cumulative_summary.key_points = self.cumulative_summary.key_points[-5:]
        
        # Update current focus
        if analysis.get("current_focus"):
            self.cumulative_summary.current_focus = analysis["current_focus"]
        
        # Accumulate violations
        self.cumulative_summary.total_violations += analysis.get("violations_detected", 0)
    
    def get_restriction_text(self) -> str:
        """Get current restriction text for debater prompts."""
        return self.cumulative_summary.to_restriction_text(self.language)
    
    def reset(self):
        """Reset the summarizer state."""
        self.cumulative_summary = DebateSummary()
        self.analysis_history = []
