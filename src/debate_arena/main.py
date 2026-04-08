from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from debate_arena.config.loader import load_config
from debate_arena.graph.workflow import DebateWorkflow
from debate_arena.llm.ollama import OllamaChatFactory
from debate_arena.prompts.repository import PromptRepository
from debate_arena.services.presenter import ConsolePresenter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous CLI debate system")
    parser.add_argument("-p", "--prompt", help="Tema o tesis del debate")
    parser.add_argument("--config", default="config/settings.yaml", help="Ruta al fichero de configuración")
    parser.add_argument("-f", "--file", help="Ruta opcional para guardar la transcripción final")
    return parser


def resolve_config_path(config_argument: str) -> Path:
    config_path = Path(config_argument)
    if config_path.exists():
        return config_path
    package_root = Path(__file__).resolve().parents[2]
    candidate = package_root / config_argument
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Configuration file not found at: {config_argument}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(resolve_config_path(args.config))
        topic = (args.prompt or "").strip()
        if not topic:
            raise RuntimeError("No debate topic provided. Use -p/--prompt.")
        workflow = DebateWorkflow(
            config=config,
            prompt_repository=PromptRepository(config.prompts.directory),
            model_factory=OllamaChatFactory(config),
        )
        result = workflow.run(topic=topic)
        ConsolePresenter().present(result=result, output_file=Path(args.file) if args.file else None)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI entry point should surface the real error.
        print(f"Fatal Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
