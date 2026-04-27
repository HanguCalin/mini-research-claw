"""LaTeX log parser and compilation helpers for Node 9 (LaTeX Compiler).

Parses pdflatex .log files to extract localized error context for the
repair agent — never feeds the entire manuscript, only the error snippet.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LatexError:
    """A single parsed LaTeX compilation error."""

    line_number: int | None
    error_type: str
    message: str
    context_lines: list[str]  # ±5 lines around the error


def compile_latex(
    tex_path: Path,
    work_dir: Path,
) -> tuple[bool, str]:
    """Run pdflatex + bibtex. Returns (success, raw_log).

    Always uses --no-shell-escape to block code execution from LLM-generated
    LaTeX.
    """
    stem = tex_path.stem

    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "--no-shell-escape", str(tex_path)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )

    subprocess.run(
        ["bibtex", stem],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=30,
    )

    final = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "--no-shell-escape", str(tex_path)],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    log_path = work_dir / f"{stem}.log"
    raw_log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else final.stdout
    pdf_path = work_dir / f"{stem}.pdf"
    success = pdf_path.exists() and final.returncode == 0

    return success, raw_log


def parse_log_errors(raw_log: str, tex_source: str) -> list[LatexError]:
    """Extract structured errors from a pdflatex .log file."""
    tex_lines = tex_source.splitlines()
    errors: list[LatexError] = []

    pattern = re.compile(r"^! (.+)", re.MULTILINE)
    line_pattern = re.compile(r"l\.(\d+)")

    for match in pattern.finditer(raw_log):
        error_msg = match.group(1).strip()
        start = match.start()
        surrounding = raw_log[start:start + 500]

        line_match = line_pattern.search(surrounding)
        line_num = int(line_match.group(1)) if line_match else None

        context: list[str] = []
        if line_num is not None and line_num > 0:
            lo = max(0, line_num - 6)
            hi = min(len(tex_lines), line_num + 5)
            context = [
                f"{i + 1}: {tex_lines[i]}" for i in range(lo, hi)
            ]

        error_type = _classify_error(error_msg)

        errors.append(LatexError(
            line_number=line_num,
            error_type=error_type,
            message=error_msg,
            context_lines=context,
        ))

    return errors


def _classify_error(msg: str) -> str:
    msg_lower = msg.lower()
    if "undefined control sequence" in msg_lower:
        return "undefined_command"
    if "missing" in msg_lower and "$" in msg_lower:
        return "math_mode"
    if "environment" in msg_lower:
        return "environment"
    if "file not found" in msg_lower or "not found" in msg_lower:
        return "file_not_found"
    if "missing" in msg_lower:
        return "missing_token"
    return "other"


def format_error_for_repair(error: LatexError) -> str:
    """Format a LatexError into a compact string for the repair agent."""
    parts = [
        f"Error type: {error.error_type}",
        f"Message: {error.message}",
    ]
    if error.line_number:
        parts.append(f"Line: {error.line_number}")
    if error.context_lines:
        parts.append("Context:")
        parts.extend(f"  {line}" for line in error.context_lines)
    return "\n".join(parts)


def apply_line_patch(
    tex_source: str,
    line_number: int,
    old_line: str,
    new_line: str,
) -> str:
    """Apply a targeted line-level patch to LaTeX source."""
    lines = tex_source.splitlines(keepends=True)
    idx = line_number - 1
    if 0 <= idx < len(lines):
        if lines[idx].strip() == old_line.strip():
            lines[idx] = new_line + "\n"
    return "".join(lines)


_INCLUDEGRAPHICS_RE = re.compile(
    r"\\includegraphics(?:\[(?P<opts>[^\]]*)\])?\{(?P<path>[^}]+)\}"
)


def neutralize_missing_graphics(tex_source: str, work_dir: Path) -> str:
    r"""Add ``,draft`` to any ``\includegraphics`` whose target file is absent.

    The Academic Writer often emits placeholder figure paths (e.g.
    ``figures/foo.pdf``) without producing the file. Without ``draft`` mode,
    pdflatex throws a hard ``File not found`` error that the LLM repair loop
    has historically failed to fix, burning all 5 attempts. This deterministic
    pre-pass renders missing graphics as labeled boxes so the rest of the
    document compiles, costing zero LLM calls.
    """
    def _patch(match: re.Match[str]) -> str:
        opts = match.group("opts") or ""
        path = match.group("path")
        if (work_dir / path).exists():
            return match.group(0)
        existing = [tok.strip() for tok in opts.split(",") if tok.strip()]
        if "draft" in existing:
            return match.group(0)
        new_opts = ",".join(existing + ["draft"])
        return f"\\includegraphics[{new_opts}]{{{path}}}"

    return _INCLUDEGRAPHICS_RE.sub(_patch, tex_source)
