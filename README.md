# Debate Arena

**Debate Arena** is an experimental autonomous CLI application that orchestrates turn-based debates between two large-language-model agents. Built with **LangChain** and **Ollama**, it explores mechanisms to keep long debates coherent while avoiding repetitive argument loops through periodic summarization and dynamic prompt injection.

---

## Features

- **Autonomous Debate System**: Two AI agents (PRO vs CON) debate a configurable topic without human intervention.
- **Anti-Loop Checkpoint System**: Periodically analyzes the debate for argument repetition and exhaustion, injecting restrictions to prevent loops.
- **Optional Judge Agent**: Can stop the debate early when a clear winner emerges, agreement is reached, or one debater concedes.
- **Summarizer/Analyst Agent**: Tracks arguments, detects stalemates, and generates restrictions for exhausted argument lines.
- **Rich Markdown Output**: Uses the `rich` library for beautiful terminal output with proper markdown rendering.
- **Flexible Configuration**: All settings (models, prompts, checkpoint intervals) are configurable via YAML.
- **Transcript Export**: Optionally save the full debate transcript to a file.

---

## How It Works

### Debate Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        DEBATE LOOP                               │
├──────────────────────────────────────────────────────────────────┤
│   ┌─────────────┐         ┌─────────────┐                        │
│   │  Debater A  │ ◄─────► │  Debater B  │                        │
│   │    (PRO)    │         │    (CON)    │                        │
│   └──────┬──────┘         └──────┬──────┘                        │
│          │                       │                               │
│          └───────────┬───────────┘                               │
│                      ▼                                           │
│          ┌───────────────────────┐                               │
│          │  Checkpoint Analysis  │ (every N turns)               │
│          └───────────┬───────────┘                               │
│                      │                                           │
│     ┌────────────────┼────────────────┐                          │
│     ▼                ▼                ▼                          │
│ ┌────────┐    ┌────────────┐    ┌─────────────┐                  │
│ │ Judge  │    │ Summarizer │    │  Inject     │                  │
│ │Evaluate│    │  Analyze   │    │ Restrictions│                  │
│ └────────┘    └────────────┘    └─────────────┘                  │
└──────────────────────────────────────────────────────────────────┘
```

### Agent Architecture

#### Debater Agents (`DebateAgent`)

Each debater is a LangChain-powered agent using `ChatOllama`:

- **System Prompt**: Defines the role (PRO/CON), topic, style, and language.
- **Memory Management**: Uses `ChatMessageHistory` to maintain conversation context.
- **Reset with Restrictions**: When checkpoints trigger, agents are reset with updated prompts containing forbidden argument lines and a context summary.

#### Summarizer Agent (`SummarizerAgent`)

Analyzes debate transcripts and returns structured JSON analysis:

- Arguments made by each side
- Refuted arguments
- Stalemate topics
- Exhausted lines that should not be repeated
- Key points and current debate focus

#### Judge Agent (Optional)

Periodically inspects the transcript and can decide to:

- **Continue**: Debate is still productive
- **End**: Agreement reached, total refutation, or concession detected

---

## Project Structure

```
debateArena/
├── config/
│   └── settings.yaml          # Main configuration file
├── docs/                      # Documentation and sample outputs
├── src/
│   └── debate_arena/
│       ├── __init__.py
│       ├── main.py            # CLI entry point
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── debater.py     # DebateAgent class
│       │   └── summarizer.py  # SummarizerAgent & DebateSummary
│       ├── core/
│       │   ├── __init__.py
│       │   └── manager.py     # DebateManager (orchestration)
│       └── utils/
│           ├── __init__.py
│           └── config_loader.py  # YAML configuration loader
├── pyproject.toml             # Package configuration
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running with your preferred models

### Setup

```bash
# Clone the repository
git clone https://github.com/arrase/debateArena.git
cd debateArena

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .
```

---

## Usage

### Basic Usage

```bash
# Run with topic from config
debate-cli --config config/settings.yaml

# Override topic via CLI
debate-cli -p "Is the use of artificial intelligence in medicine ethical?"

# Save transcript to file
debate-cli -p "Climate change is reversible" -f debate_output.txt
```

### CLI Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--config` | | Path to configuration file (default: `config/settings.yaml`) |
| `--prompt` | `-p` | Override the debate topic |
| `--file` | `-f` | Save transcript to specified file |

---

## Configuration

All settings are defined in `config/settings.yaml`:

### Debate Settings

```yaml
debate:
  max_turns: 100              # Maximum number of debate turns
  topic: "The Earth is flat"  # Default debate topic
  min_chars_per_turn: 50      # Minimum characters per response
  language: "Spanish"         # Response language
```

### Checkpoint System

```yaml
checkpoint:
  enabled: true
  interval_turns: 2           # Analyze every N turns
  max_violations: 1           # End after N violations
```

### Model Configuration

```yaml
models:
  debater_a:
    name: "nemotron-3-nano:30b"
    temperature: 0.7
    system_prompt: |
      You are Debater A. You are a passionate advocate for the PRO side...
  
  debater_b:
    name: "nemotron-3-nano:30b"
    temperature: 0.7
    system_prompt: |
      You are Debater B. You are a passionate advocate for the CON side...
  
  judge:
    name: "nemotron-3-nano:30b"
    temperature: 0.2
    system_prompt: |
      You are the impartial Judge of this debate...
  
  summarizer:
    name: "nemotron-3-nano:30b"
    temperature: 0.1
```

---

## Research Applications

This experiment is designed to study:

- **Argumentation dynamics** over extended multi-turn debates
- **Prompt-based constraints** and their effect on repetition/convergence
- **Context summarization** as a mechanism to extend effective context in limited-context models
- **LLM behavior** when faced with concession opportunities or logical defeat

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
