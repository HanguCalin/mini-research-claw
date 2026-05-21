"""LLM response parsing helpers.

Claude often returns JSON wrapped in markdown fences or with a prose preamble
despite explicit instructions otherwise. ``extract_json`` strips the fences,
locates the first balanced ``{...}`` or ``[...]`` block, and parses it.
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    r"""Parse JSON from an LLM response that may include fences or prose.

    Handles all of these:
      ``{"a": 1}``                              — clean
      ``\`\`\`json\n{"a": 1}\n\`\`\```          — markdown-fenced
      ``Here is the JSON:\n{"a": 1}``           — prose preamble
      ``{"a": 1}\nThat's all.``                 — trailing prose

    Raises ``ValueError`` if no JSON object/array can be found.
    """
    if not text or not text.strip():
        raise ValueError("LLM returned an empty response")

    cleaned = text.strip()

    fence_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?```",
        cleaned,
        flags=re.DOTALL,
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()

    obj_start = cleaned.find("{")
    arr_start = cleaned.find("[")

    candidates = [i for i in (obj_start, arr_start) if i != -1]
    if not candidates:
        raise ValueError(f"No JSON object/array in LLM response: {text[:200]!r}")

    start = min(candidates)
    open_char = cleaned[start]
    close_char = "}" if open_char == "{" else "]"

    end = cleaned.rfind(close_char)
    if end < start:
        raise ValueError(f"Unbalanced JSON in LLM response: {text[:200]!r}")

    snippet = cleaned[start : end + 1]

    try:
        return json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON parse failed at pos {exc.pos}: {exc.msg}\n"
            f"Snippet: {snippet[:300]!r}"
        ) from exc


def extract_text(response: Any) -> str:
    """Pull the text out of an Anthropic ``Message`` response.

    Defensive against empty content blocks (which trigger a confusing
    ``IndexError`` deep inside the SDK).
    """
    if not response.content:
        raise ValueError("Anthropic response has no content blocks")

    block = response.content[0]
    text = getattr(block, "text", "")
    if not text:
        raise ValueError("Anthropic response content is empty")

    return text
