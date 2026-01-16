# Debate Arena

Debate Arena is a small experiment that runs an autonomous, turn‑based debate between two large‑language‑model agents on a configurable topic. The project explores how to keep long debates coherent while avoiding repetitive argument loops by periodically summarizing progress and injecting restrictions back into the agents’ prompts.

## What the experiment does

- Two agents debate **for** and **against** a topic (PRO vs CON).
- A **judge** can optionally stop the debate early when a clear winner or agreement emerges.
- A **summarizer/analyst** periodically reviews the transcript to detect repetition, refuted arguments, and stalemates.
- When loops are detected, the system **resets** both debaters with updated prompt restrictions that forbid exhausted lines of argument.

This creates a controlled environment to study:

- Argumentation dynamics over many turns.
- The effect of prompt‑based constraints on repetition and convergence.
- How summarization can extend effective context in limited‑context models.

## How the agents work

### Debater agents

Each debater is an instance of `DebateAgent` built on **LangChain + Ollama**:

- A **system prompt** defines the role (PRO/CON), topic, style, and language.
- Each response is appended to memory as a chat history.
- When a checkpoint triggers, both agents are **reset** with a new system prompt that injects **restrictions** plus a short **context summary**.

This reset keeps the debate on track without requiring a large context window.

### Summarizer agent (anti‑loop mechanism)

The summarizer analyzes recent transcript slices and returns a JSON analysis with:

- arguments made by each side,
- refuted arguments,
- stalemate topics,
- exhausted lines that should not be repeated,
- key points and current focus.

From this, the system generates restriction text and updates the debaters’ system prompts. If violations are too frequent, the debate is terminated early.

### Judge agent (optional)

If enabled, the judge periodically inspects the latest transcript and decides whether to end the debate. It can also be forced to deliver a final verdict after rule‑violation termination.

## Running the experiment

### 1) Configure settings

Edit [config/settings.yaml](config/settings.yaml) to set:

- `debate.topic` — the debate subject
- `debate.max_turns` — maximum number of turns
- `debate.language` — response language
- `checkpoint.interval_turns` — how often to analyze
- model names/temperatures for debaters, judge, and summarizer

### 2) Run from CLI

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -e .
debate-cli --config config/settings.yaml
```

Optional flags:

- `-p, --prompt` override topic
- `-f, --file` save transcript to file

## Key files

- [src/debate_arena/main.py](src/debate_arena/main.py) — CLI entry point
- [src/debate_arena/core/manager.py](src/debate_arena/core/manager.py) — debate loop, checkpoints, judge logic
- [src/debate_arena/agents/debater.py](src/debate_arena/agents/debater.py) — debater agent
- [src/debate_arena/agents/summarizer.py](src/debate_arena/agents/summarizer.py) — summarizer/analyst agent
- [config/settings.yaml](config/settings.yaml) — experiment configuration

## Notes

- The system is designed for **Ollama‑compatible** models via `langchain_ollama`.
- The debate can be run without a judge; in that case it will end at `max_turns` or via anti‑loop termination.
- Restriction injection is the core mechanism that prevents repetitive argumentation and keeps the debate progressing.
