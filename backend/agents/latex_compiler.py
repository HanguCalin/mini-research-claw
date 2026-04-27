"""Node 9: LaTeX Compiler with Repair Loop.

Type: Non-AI (pdflatex) + AI (Claude Haiku for repair).
Compiles the LaTeX draft, parses errors, and runs a targeted repair
loop up to max_latex_repair_attempts.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import anthropic

from backend.config import MODELS, THRESHOLDS
from backend.state import AutoResearchState
from backend.utils.llm_utils import extract_json, extract_text
from backend.utils.latex_utils import (
    apply_line_patch,
    compile_latex,
    format_error_for_repair,
    neutralize_missing_graphics,
    parse_log_errors,
)

logger = logging.getLogger(__name__)

REPAIR_SYSTEM_PROMPT = """\
You are a LaTeX repair agent. You receive a compilation error with the
error message, line number, and ±5 lines of context.

Produce a targeted LINE-LEVEL PATCH. Do NOT rewrite the entire document.

Return ONLY valid JSON:
{
  "line_number": 42,
  "old_line": "the exact broken line",
  "new_line": "the fixed line"
}
"""


def latex_compiler(state: AutoResearchState) -> dict[str, Any]:
    """Compile LaTeX, repair errors if needed, return PDF path or failure."""
    latex_draft = state.get("latex_draft", "")
    bibtex_source = state.get("bibtex_source", "")
    max_attempts = THRESHOLDS.max_latex_repair_attempts

    with tempfile.TemporaryDirectory(prefix="miniclaw_latex_") as work_dir_str:
        work_dir = Path(work_dir_str)
        tex_path = work_dir / "draft.tex"
        bib_path = work_dir / "references.bib"

        bib_path.write_text(bibtex_source, encoding="utf-8")

        current_source = neutralize_missing_graphics(latex_draft, work_dir)
        tex_path.write_text(current_source, encoding="utf-8")

        for attempt in range(max_attempts):
            success, raw_log = compile_latex(tex_path, work_dir)

            if success:
                pdf_path = work_dir / "draft.pdf"
                final_path = Path(tempfile.mktemp(suffix=".pdf", prefix="miniclaw_"))
                final_path.write_bytes(pdf_path.read_bytes())

                logger.info("LaTeX compilation succeeded on attempt %d", attempt + 1)
                return {
                    "final_pdf_path": str(final_path),
                    "latex_compile_log": raw_log[-2000:],
                    "latex_repair_attempts": attempt,
                    "pipeline_status": "success",
                }

            errors = parse_log_errors(raw_log, current_source)
            if not errors:
                logger.warning("Compilation failed but no parseable errors found")
                break

            error = errors[0]
            logger.info(
                "Repair attempt %d/%d: %s at line %s",
                attempt + 1, max_attempts,
                error.error_type, error.line_number,
            )

            patch = _get_repair_patch(error)
            if patch and error.line_number:
                current_source = apply_line_patch(
                    current_source,
                    patch["line_number"],
                    patch["old_line"],
                    patch["new_line"],
                )
                tex_path.write_text(current_source, encoding="utf-8")
            else:
                logger.warning("Repair agent failed to produce a valid patch")
                break

        logger.warning("LaTeX repair exhausted after %d attempts", max_attempts)
        return {
            "final_pdf_path": None,
            "latex_compile_log": raw_log[-2000:] if 'raw_log' in dir() else "",
            "latex_repair_attempts": max_attempts,
            "latex_draft": current_source,
            "pipeline_status": "failed_latex",
        }


def _get_repair_patch(error: Any) -> dict[str, Any] | None:
    """Ask Claude Haiku to produce a line-level patch for the error."""
    client = anthropic.Anthropic()
    error_context = format_error_for_repair(error)

    try:
        response = client.messages.create(
            model=MODELS.latex_repair,
            max_tokens=512,
            system=REPAIR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": error_context}],
        )
        return extract_json(extract_text(response))
    except (ValueError, anthropic.APIError) as exc:
        logger.warning("Repair LLM call failed: %s", exc)
        return None
