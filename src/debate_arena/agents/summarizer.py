"""Summarizer agent for debate analysis and exhausted-argument tracking."""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


@dataclass
class DebateSummary:
    exhausted_arguments: List[str] = field(default_factory=list)
    refuted_arguments: Dict[str, List[str]] = field(default_factory=dict)  # {debater: [args]}
    stalemate_topics: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    current_focus: str = ""
    total_violations: int = 0
    
    def to_restriction_text(self, language: str = "Spanish") -> str:
        lines = []
        
        if self.exhausted_arguments:
            lines.append("🚫 EXHAUSTED ARGUMENTS (no longer have any room for development):")
            for arg in self.exhausted_arguments:
                lines.append(f"   • {arg}")
        
        if self.refuted_arguments:
            lines.append("\n❌ REFUTED ARGUMENTS (completely dismantled - do not use):")
            for debater, args in self.refuted_arguments.items():
                if args:
                    lines.append(f"   {debater}:")
                    for arg in args:
                        lines.append(f"      • {arg}")
        
        if self.stalemate_topics:
            lines.append("\n⚠️ STALEMATE TOPICS (no progress possible on these):")
            for topic in self.stalemate_topics:
                lines.append(f"   • {topic}")
        
        if self.key_points:
            lines.append("\n📋 DEBATE PROGRESS SUMMARY:")
            for point in self.key_points:
                lines.append(f"   • {point}")
        
        if lines:
            restriction = "\n".join(lines)
            return f"""
╔══════════════════════════════════════════════════════════════╗
║           ⚠️  MANDATORY DEBATE RESTRICTIONS  ⚠️               ║
╚══════════════════════════════════════════════════════════════╝

{restriction}

┌──────────────────────────────────────────────────────────────┐
│ CRITICAL INSTRUCTIONS:                                        │
│                                                                │
│ 1. You MUST NOT repeat any argument listed above              │
│ 2. You MUST use DIFFERENT argumentative lines                 │
│ 3. Bring NEW perspectives, evidence, or angles                │
│ 4. If you have no new arguments, you MUST acknowledge this    │
│    honestly and the debate will end without consensus         │
│                                                                │
│ FAILURE TO COMPLY will terminate the debate immediately.      │
└──────────────────────────────────────────────────────────────┘
"""
        return ""


class SummarizerAgent:
    
    ANALYSIS_PROMPT = """You are an expert debate analyst. Your role is to track argument exhaustion and prevent repetitive debates.

ANALYZE the transcript and identify:

1. EXHAUSTED ARGUMENTS: Arguments that have no more room for development because:
   - They have been fully explored from all angles
   - They have been refuted with no valid counter-response
   - They keep being repeated without new evidence
   - Both sides have said everything possible about them

2. REFUTED ARGUMENTS: Arguments that have been completely dismantled by the opponent with irrefutable logic or evidence

3. STALEMATES: Topics where both debaters are going in circles without progress

4. VIOLATIONS: When a debater repeats an argument that was already exhausted/refuted

Respond with ONLY a valid JSON object (no markdown, no extra text):
{{
    "debater_a_arguments": ["arg1", "arg2"],
    "debater_b_arguments": ["arg1", "arg2"],
    "refuted_arguments": {{
        "debater_a": ["refuted arg by A"],
        "debater_b": ["refuted arg by B"]
    }},
    "stalemate_topics": ["topic stuck in loop"],
    "exhausted_lines": ["argument lines that have no more development possible"],
    "key_points": ["important developments in the debate"],
    "violations_detected": 0,
    "current_focus": "what the debate is currently about",
    "should_end": false,
    "end_reason": ""
}}

CRITICAL GUIDELINES:
- Be concise in argument descriptions (max 15 words each)
- "exhausted_lines" = arguments with NO MORE development possible (must use different angles)
- "violations_detected" = count of times a debater repeated an exhausted/refuted argument
- "should_end" = true ONLY if BOTH debaters are stuck repeating without any new arguments possible
- Write ALL descriptions in {language}

{previous_restrictions_block}

Topic: {topic}

Transcript:
{transcript}
"""

    def __init__(self, model_name: str, temperature: float = 0.1, language: str = "Spanish"):
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.language = language
        self.cumulative_summary = DebateSummary()
        self.analysis_history: List[Dict[str, Any]] = []
    
    def analyze_debate(
        self,
        transcript: List[Tuple[str, str]],
        topic: str,
        previous_restrictions: Optional[str] = None,
    ) -> Tuple[DebateSummary, bool, str]:
        transcript_text = self._format_transcript(transcript)
        previous_restrictions_block = ""
        if previous_restrictions:
            previous_restrictions_block = (
                "PREVIOUSLY RESTRICTED ARGUMENTS (debaters were told NOT to use these):\n"
                f"{previous_restrictions}\n\n"
                "If a debater uses any of these restricted arguments again, count it as a VIOLATION."
            )
        prompt = self.ANALYSIS_PROMPT.format(
            topic=topic,
            transcript=transcript_text,
            language=self.language,
            previous_restrictions_block=previous_restrictions_block
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
                return self.cumulative_summary, should_end, end_reason
        except Exception as e:
            print(f"[Summarizer] Analysis error: {e}")
        return self.cumulative_summary, False, ""
    
    def _format_transcript(self, transcript: List[Tuple[str, str]]) -> str:
        lines = []
        for speaker, message in transcript:
            if len(message) > 500:
                message = message[:500] + "..."
            lines.append(f"{speaker}: {message}")
        return "\n\n".join(lines)
    
    def _parse_analysis(self, response: str) -> Optional[Dict[str, Any]]:
        response = response.strip()
        if "```" in response:
            start = response.find("```json")
            start = (start + 7) if start >= 0 else response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        for payload in (response, response[response.find("{"):response.rfind("}") + 1]):
            try:
                return json.loads(payload)
            except Exception:
                continue
        return None
    
    def _update_cumulative_summary(self, analysis: Dict[str, Any]):
        for arg in analysis.get("exhausted_lines", []):
            if arg and arg not in self.cumulative_summary.exhausted_arguments:
                self.cumulative_summary.exhausted_arguments.append(arg)
        for debater, args in analysis.get("refuted_arguments", {}).items():
            bucket = self.cumulative_summary.refuted_arguments.setdefault(debater, [])
            for arg in args:
                if arg and arg not in bucket:
                    bucket.append(arg)
        for topic in analysis.get("stalemate_topics", []):
            if topic and topic not in self.cumulative_summary.stalemate_topics:
                self.cumulative_summary.stalemate_topics.append(topic)
        for point in analysis.get("key_points", []):
            if point:
                self.cumulative_summary.key_points.append(point)
        self.cumulative_summary.key_points = self.cumulative_summary.key_points[-5:]
        if analysis.get("current_focus"):
            self.cumulative_summary.current_focus = analysis["current_focus"]
        self.cumulative_summary.total_violations += analysis.get("violations_detected", 0)
    
    def get_restriction_text(self) -> str:
        return self.cumulative_summary.to_restriction_text(self.language)
    
    def reset(self):
        self.cumulative_summary = DebateSummary()
        self.analysis_history = []
