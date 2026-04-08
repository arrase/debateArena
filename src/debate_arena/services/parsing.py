from __future__ import annotations

import json
from typing import Any, Dict


def extract_json_object(content: str) -> Dict[str, Any]:
    candidate = content.strip()
    payloads = [candidate]

    start_index = candidate.find("{")
    end_index = candidate.rfind("}")
    if start_index >= 0 and end_index > start_index:
        payloads.append(candidate[start_index : end_index + 1])

    for payload in payloads:
        cleaned = _strip_code_fences(payload)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(_escape_invalid_backslashes(cleaned))
            except json.JSONDecodeError:
                try:
                    parsed = json.loads(cleaned.replace("\\", "\\\\"))
                except json.JSONDecodeError:
                    continue
        if isinstance(parsed, dict):
            return parsed

    raise RuntimeError(f"Model did not return valid JSON: {content}")


def _strip_code_fences(payload: str) -> str:
    stripped = payload.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline >= 0:
            stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _escape_invalid_backslashes(payload: str) -> str:
    repaired: list[str] = []
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
    index = 0
    while index < len(payload):
        char = payload[index]
        if char == "\\":
            next_char = payload[index + 1] if index + 1 < len(payload) else ""
            if next_char not in valid_escapes:
                repaired.append("\\\\")
                index += 1
                continue
        repaired.append(char)
        index += 1
    return "".join(repaired)
