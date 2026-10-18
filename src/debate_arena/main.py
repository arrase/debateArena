import argparse
from pathlib import Path

from ollama import Client

from debate_arena.arena import run_debate
from debate_arena.config import load_config
from debate_arena.presenter import ConsolePresenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="debate-cli",
        description="Arena de debate autónomo con gemma4:12b-it-qat y juez probabilístico d1:latest.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        help="Tesis o tema central sobre el que debatirán los modelos.",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Ruta al archivo de configuración YAML (por defecto: config/settings.yaml).",
    )
    parser.add_argument(
        "-f",
        "--file",
        default="",
        help="Ruta opcional para guardar la transcripción y el veredicto en Markdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    client = Client(
        host=config.ollama.base_url,
        timeout=config.ollama.timeout_seconds,
    )
    presenter = ConsolePresenter()
    result = run_debate(
        topic=args.prompt.strip(),
        config=config,
        client=client,
        presenter=presenter,
    )
    if args.file:
        presenter.save_result(result, Path(args.file))


if __name__ == "__main__":
    main()
