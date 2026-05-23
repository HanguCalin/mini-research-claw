"""Unit tests for backend/utils/latex_utils.py

Tests:
- parse_log_errors extracts errors from pdflatex log output
- _classify_error assigns correct error types
- format_error_for_repair produces expected text block
- apply_line_patch replaces the correct line
- neutralize_missing_graphics adds draft option to missing images
"""

from pathlib import Path
import tempfile

import pytest

from backend.utils.latex_utils import (
    LatexError,
    apply_line_patch,
    format_error_for_repair,
    neutralize_missing_graphics,
    parse_log_errors,
)


# ─── parse_log_errors ─────────────────────────────────────────────────────────


def test_parse_log_detects_error():
    log = "! Undefined control sequence.\nl.42 \\badcommand\n"
    tex = "\n" * 50
    errors = parse_log_errors(log, tex)
    assert len(errors) == 1


def test_parse_log_captures_line_number():
    log = "! Undefined control sequence.\nl.42 \\badcommand\n"
    tex = "\n" * 50
    errors = parse_log_errors(log, tex)
    assert errors[0].line_number == 42


def test_parse_log_captures_error_message():
    log = "! Undefined control sequence.\nl.5 \\badcommand\n"
    errors = parse_log_errors(log, "line1\nline2\nline3\nline4\nline5\n")
    assert "Undefined control sequence" in errors[0].message


def test_parse_log_no_errors_returns_empty():
    assert parse_log_errors("This is fine.", "some tex") == []


def test_parse_log_captures_context_lines():
    tex = "\n".join(f"line {i}" for i in range(1, 20))
    log = "! Some error.\nl.10 \\bad\n"
    errors = parse_log_errors(log, tex)
    assert len(errors[0].context_lines) > 0


def test_parse_log_multiple_errors():
    log = "! Error one.\nl.5 \\bad1\n! Error two.\nl.10 \\bad2\n"
    tex = "\n" * 20
    errors = parse_log_errors(log, tex)
    assert len(errors) == 2


def test_parse_log_classifies_undefined_command():
    log = "! Undefined control sequence.\nl.1 \\bad\n"
    errors = parse_log_errors(log, "line1")
    assert errors[0].error_type == "undefined_command"


def test_parse_log_classifies_environment_error():
    log = "! LaTeX Error: environment tabular undefined.\nl.3 \\begin\n"
    errors = parse_log_errors(log, "\n\n\nline")
    assert errors[0].error_type == "environment"


# ─── format_error_for_repair ──────────────────────────────────────────────────


def test_format_includes_error_type():
    error = LatexError(
        line_number=5,
        error_type="undefined_command",
        message="Undefined control sequence",
        context_lines=["5: \\badcmd"],
    )
    text = format_error_for_repair(error)
    assert "undefined_command" in text


def test_format_includes_line_number():
    error = LatexError(line_number=5, error_type="other",
                       message="Some error", context_lines=[])
    assert "5" in format_error_for_repair(error)


def test_format_includes_context_lines():
    error = LatexError(line_number=5, error_type="other",
                       message="Error", context_lines=["5: \\begin{broken"])
    assert "\\begin{broken" in format_error_for_repair(error)


def test_format_works_without_line_number():
    error = LatexError(line_number=None, error_type="other",
                       message="Vague error", context_lines=[])
    text = format_error_for_repair(error)
    assert "Vague error" in text


# ─── apply_line_patch ─────────────────────────────────────────────────────────


def test_apply_patch_replaces_correct_line():
    source = "line one\nline two\nline three\n"
    result = apply_line_patch(source, 2, "line two", "line TWO fixed")
    assert "line TWO fixed" in result


def test_apply_patch_does_not_change_other_lines():
    source = "line one\nline two\nline three\n"
    result = apply_line_patch(source, 2, "line two", "fixed")
    assert "line one" in result
    assert "line three" in result


def test_apply_patch_noop_when_old_line_not_matching():
    source = "line one\nline two\nline three\n"
    result = apply_line_patch(source, 2, "WRONG LINE", "fixed")
    assert result == source


def test_apply_patch_noop_on_out_of_range_line():
    source = "only one line\n"
    result = apply_line_patch(source, 99, "anything", "fixed")
    assert result == source


# ─── neutralize_missing_graphics ─────────────────────────────────────────────


def test_neutralize_adds_draft_for_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        tex = r"\includegraphics{figures/missing.pdf}"
        result = neutralize_missing_graphics(tex, work_dir)
        assert "draft" in result


def test_neutralize_leaves_existing_file_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        img = work_dir / "existing.pdf"
        img.write_bytes(b"%PDF-1.4")
        tex = r"\includegraphics{existing.pdf}"
        result = neutralize_missing_graphics(tex, work_dir)
        assert "draft" not in result


def test_neutralize_does_not_double_add_draft():
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        tex = r"\includegraphics[draft]{figures/missing.pdf}"
        result = neutralize_missing_graphics(tex, work_dir)
        assert result.count("draft") == 1


def test_neutralize_preserves_existing_options():
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        tex = r"\includegraphics[width=0.5\textwidth]{figures/missing.pdf}"
        result = neutralize_missing_graphics(tex, work_dir)
        assert "width" in result
        assert "draft" in result


def test_neutralize_no_graphics_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        tex = r"\section{Introduction} No images here."
        result = neutralize_missing_graphics(tex, Path(tmp))
        assert result == tex
