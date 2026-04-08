from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


class PromptRepository:
    def __init__(self, directory: Path):
        self._directory = directory
        self._environment = Environment(
            loader=FileSystemLoader(str(directory)),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, prompt_file: str, **context: object) -> str:
        return self._environment.get_template(prompt_file).render(**context).strip()
