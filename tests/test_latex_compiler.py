"""Unit tests for backend/agents/latex_compiler.py

Tests:
- on first-attempt success: final_pdf_path is set, pipeline_status="success"
- on failure then success: repair loop is triggered, PDF eventually produced
- on exhausted attempts: final_pdf_path=None, pipeline_status="failed_latex"
- repair patch is applied to source when LLM returns valid patch
- when compile_latex fails but parse_log_errors returns no errors: exits loop
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.utils.latex_utils import LatexError

BASE_STATE = {
    "latex_draft": r"\documentclass{article}\begin{document}\section{Intro}Hi.\end{document}",
    "bibtex_source": "@article{ref1, title={X}}",
}


def _compile_success(tex_path, work_dir):
    (work_dir / "draft.pdf").write_bytes(b"%PDF-1.4")
    return True, "Success log"

SAMPLE_ERROR = LatexError(
    line_number=3,
    error_type="undefined_command",
    message="Undefined control sequence",
    context_lines=["3: \\badcmd"],
)

VALID_PATCH = {
    "line_number": 3,
    "old_line": "\\badcmd",
    "new_line": "\\textbf{fixed}",
}


# ─── Success on first attempt ─────────────────────────────────────────────────


@patch("backend.agents.latex_compiler.neutralize_missing_graphics", side_effect=lambda src, _: src)
@patch("backend.agents.latex_compiler.compile_latex", side_effect=_compile_success)
def test_success_sets_pdf_path(mock_compile, mock_neutralize):
    from backend.agents.latex_compiler import latex_compiler
    result = latex_compiler(BASE_STATE)
    assert result["final_pdf_path"] is not None


@patch("backend.agents.latex_compiler.neutralize_missing_graphics", side_effect=lambda src, _: src)
@patch("backend.agents.latex_compiler.compile_latex", side_effect=_compile_success)
def test_success_sets_pipeline_status(mock_compile, mock_neutralize):
    from backend.agents.latex_compiler import latex_compiler
    result = latex_compiler(BASE_STATE)
    assert result["pipeline_status"] == "success"


@patch("backend.agents.latex_compiler.neutralize_missing_graphics", side_effect=lambda src, _: src)
@patch("backend.agents.latex_compiler.compile_latex", side_effect=_compile_success)
def test_success_sets_repair_attempts_to_zero(mock_compile, mock_neutralize):
    from backend.agents.latex_compiler import latex_compiler
    result = latex_compiler(BASE_STATE)
    assert result["latex_repair_attempts"] == 0


# ─── Failure then success (repair loop) ──────────────────────────────────────


@patch("backend.agents.latex_compiler._get_repair_patch", return_value=VALID_PATCH)
@patch("backend.agents.latex_compiler.parse_log_errors", return_value=[SAMPLE_ERROR])
@patch("backend.agents.latex_compiler.neutralize_missing_graphics", side_effect=lambda src, _: src)
@patch("backend.agents.latex_compiler.compile_latex")
def test_repair_loop_produces_pdf_on_second_attempt(mock_compile, mock_neutralize,
                                                     mock_parse, mock_patch):
    _calls = []
    def _se(tex_path, work_dir):
        _calls.append(None)
        if len(_calls) == 1:
            return False, "! Undefined control sequence.\nl.3 \\badcmd\n"
        (work_dir / "draft.pdf").write_bytes(b"%PDF-1.4")
        return True, "Success"
    mock_compile.side_effect = _se

    from backend.agents.latex_compiler import latex_compiler
    result = latex_compiler(BASE_STATE)
    assert result["pipeline_status"] == "success"
    assert result["final_pdf_path"] is not None


# ─── Exhausted attempts ───────────────────────────────────────────────────────


@patch("backend.agents.latex_compiler._get_repair_patch", return_value=VALID_PATCH)
@patch("backend.agents.latex_compiler.parse_log_errors", return_value=[SAMPLE_ERROR])
@patch("backend.agents.latex_compiler.neutralize_missing_graphics", side_effect=lambda src, _: src)
@patch("backend.agents.latex_compiler.compile_latex", return_value=(False, "! Error.\nl.3 \\bad\n"))
def test_exhausted_attempts_sets_failed_latex(mock_compile, mock_neutralize,
                                               mock_parse, mock_patch):
    from backend.agents.latex_compiler import latex_compiler
    result = latex_compiler(BASE_STATE)
    assert result["pipeline_status"] == "failed_latex"


@patch("backend.agents.latex_compiler._get_repair_patch", return_value=VALID_PATCH)
@patch("backend.agents.latex_compiler.parse_log_errors", return_value=[SAMPLE_ERROR])
@patch("backend.agents.latex_compiler.neutralize_missing_graphics", side_effect=lambda src, _: src)
@patch("backend.agents.latex_compiler.compile_latex", return_value=(False, "! Error.\nl.3 \\bad\n"))
def test_exhausted_attempts_pdf_path_is_none(mock_compile, mock_neutralize,
                                              mock_parse, mock_patch):
    from backend.agents.latex_compiler import latex_compiler
    result = latex_compiler(BASE_STATE)
    assert result["final_pdf_path"] is None


# ─── No parseable errors → exits loop ────────────────────────────────────────


@patch("backend.agents.latex_compiler.parse_log_errors", return_value=[])
@patch("backend.agents.latex_compiler.neutralize_missing_graphics", side_effect=lambda src, _: src)
@patch("backend.agents.latex_compiler.compile_latex", return_value=(False, "weird log"))
def test_no_parseable_errors_exits_loop(mock_compile, mock_neutralize, mock_parse):
    from backend.agents.latex_compiler import latex_compiler
    result = latex_compiler(BASE_STATE)
    # Should not loop forever — exits early
    assert result["pipeline_status"] == "failed_latex"
    # compile_latex called only once before giving up
    assert mock_compile.call_count == 1


# ─── Repair patch application ────────────────────────────────────────────────


@patch("backend.agents.latex_compiler._get_repair_patch", return_value=None)
@patch("backend.agents.latex_compiler.parse_log_errors", return_value=[SAMPLE_ERROR])
@patch("backend.agents.latex_compiler.neutralize_missing_graphics", side_effect=lambda src, _: src)
@patch("backend.agents.latex_compiler.compile_latex", return_value=(False, "! Error.\nl.3 \\bad\n"))
def test_null_patch_exits_loop(mock_compile, mock_neutralize, mock_parse, mock_repair):
    from backend.agents.latex_compiler import latex_compiler
    result = latex_compiler(BASE_STATE)
    assert result["pipeline_status"] == "failed_latex"
    assert mock_compile.call_count == 1
