"""Central eval runner — §7.4.

Executes all agent evals and prints a summary table.

Usage:
    python scripts/run_evals.py               # run all evals
    python scripts/run_evals.py --no-llm      # skip evals that require API key
    python scripts/run_evals.py --only coder  # run only evals whose name contains 'coder'

Environment:
    ANTHROPIC_API_KEY   required for LLM-based evals
    EVAL_RUNS           number of LLM runs per eval (default: 3)
"""

from __future__ import annotations

import argparse
import importlib
import io
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field

# Ensure project root is on path and load .env
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ─── Eval registry ────────────────────────────────────────────────────────────

@dataclass
class EvalEntry:
    name: str
    module: str
    needs_llm: bool
    needs_pdflatex: bool = False
    description: str = ""


EVALS: list[EvalEntry] = [
    # ── Deterministic (no API key) ─────────────────────────────────────────
    EvalEntry("claim_ledger_accuracy",  "tests.evals.eval_claim_ledger_accuracy",  needs_llm=False,
              description="Claim Ledger: evidence strength ratings >= 90% correct"),
    EvalEntry("claim_ledger_no_paper",  "tests.evals.eval_claim_ledger_no_paper",  needs_llm=False,
              description="Claim Ledger: No-Paper gate triggers correctly"),
    EvalEntry("dependency_resolver",    "tests.evals.eval_dependency_resolver",    needs_llm=False,
              description="Dependency Resolver: 100% recall on known imports"),
    EvalEntry("linter_detection",       "tests.evals.eval_linter_detection",       needs_llm=False,
              description="Linter: >= 9/10 structural issues detected"),
    EvalEntry("state_pruning",          "tests.evals.eval_state_pruning",          needs_llm=False,
              description="State Pruning: correct field scoping on all nodes"),

    # ── LLM-based (require ANTHROPIC_API_KEY) ─────────────────────────────
    EvalEntry("kg_extractor",           "tests.evals.eval_kg_extractor",           needs_llm=True,
              description="KG Extractor: schema compliance, dedup, polarity, depth"),
    EvalEntry("hypothesis",             "tests.evals.eval_hypothesis",             needs_llm=True,
              description="Hypothesis: anti-hallucination, delta, prior-art, novelty"),
    EvalEntry("experiment_designer",    "tests.evals.eval_experiment_designer",    needs_llm=True,
              description="Experiment Designer: all 6 spec fields present"),
    EvalEntry("coder",                  "tests.evals.eval_coder",                  needs_llm=True,
              description="Coder: syntax, debug prints, ML rigor, spec compliance, imports"),
    EvalEntry("writer",                 "tests.evals.eval_writer",                 needs_llm=True,
              description="Writer: IMRaD structure + LaTeX compilation"),
    EvalEntry("critique",               "tests.evals.eval_critique",               needs_llm=True,
              description="Critique Panel: diversity, debate survival, KG grounding"),
    EvalEntry("latex_repair",           "tests.evals.eval_latex_repair",           needs_llm=True,
              needs_pdflatex=True,
              description="LaTeX Repair: >= 7/10 injected errors fixed"),
]

# ─── Runner ───────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    name: str
    status: str          # PASS / FAIL / SKIP / ERROR
    duration_s: float
    output: str
    error: str = ""


def run_one(entry: EvalEntry) -> EvalResult:
    buf = io.StringIO()
    t0 = time.monotonic()
    try:
        mod = importlib.import_module(entry.module)
        with redirect_stdout(buf):
            mod.run_eval()
        output = buf.getvalue()
        duration = time.monotonic() - t0

        # Infer status from output keywords
        lower = output.lower()
        if "skip" in lower:
            status = "SKIP"
        elif "passed" in lower and "failed" not in lower:
            status = "PASS"
        elif "failed" in lower or "fail" in lower:
            status = "FAIL"
        else:
            status = "PASS"  # no explicit failure → treat as pass

        return EvalResult(name=entry.name, status=status,
                          duration_s=duration, output=output)
    except BaseException:
        duration = time.monotonic() - t0
        return EvalResult(name=entry.name, status="ERROR",
                          duration_s=duration, output=buf.getvalue(),
                          error=traceback.format_exc())


def _col(text: str, width: int) -> str:
    return text[:width].ljust(width)


def print_summary(results: list[EvalResult]) -> None:
    print("\n" + "=" * 72)
    print("  EVAL SUMMARY")
    print("=" * 72)
    header = f"  {'Eval':<28}  {'Status':<6}  {'Time':>6}  Description"
    print(header)
    print("-" * 72)

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}
    entry_map = {e.name: e for e in EVALS}

    for r in results:
        desc = entry_map[r.name].description
        desc_short = desc[:30] if len(desc) > 30 else desc
        print(f"  {_col(r.name, 28)}  {_col(r.status, 6)}  {r.duration_s:>5.1f}s  {desc_short}")
        counts[r.status] += 1

    print("=" * 72)
    print(f"  PASS={counts['PASS']}  FAIL={counts['FAIL']}  "
          f"SKIP={counts['SKIP']}  ERROR={counts['ERROR']}  "
          f"TOTAL={len(results)}")
    print("=" * 72)


def main() -> None:
    # Ensure emoji in eval output renders correctly on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run all agent evals")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip evals that require ANTHROPIC_API_KEY")
    parser.add_argument("--only", metavar="NAME",
                        help="Run only evals whose name contains NAME")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full output for each eval")
    args = parser.parse_args()

    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))

    selected = EVALS
    if args.only:
        selected = [e for e in EVALS if args.only.lower() in e.name.lower()]
        if not selected:
            print(f"No evals match '{args.only}'. Available: {[e.name for e in EVALS]}")
            sys.exit(1)

    results: list[EvalResult] = []

    for entry in selected:
        skip_reason = None
        if args.no_llm and entry.needs_llm:
            skip_reason = "--no-llm flag"
        elif entry.needs_llm and not has_api_key:
            skip_reason = "ANTHROPIC_API_KEY not set"

        print(f"\n{'─' * 60}")
        print(f"  Running: {entry.name}")
        print(f"  {entry.description}")

        if skip_reason:
            print(f"  SKIPPED ({skip_reason})")
            results.append(EvalResult(name=entry.name, status="SKIP",
                                      duration_s=0.0, output=""))
            continue

        result = run_one(entry)
        results.append(result)

        if args.verbose or result.status in ("FAIL", "ERROR"):
            print(result.output)
            if result.error:
                print(result.error)
        else:
            # Print last non-empty line as a quick summary
            lines = [l for l in result.output.splitlines() if l.strip()]
            if lines:
                print(f"  → {lines[-1].strip()}")

        print(f"  Status: {result.status}  ({result.duration_s:.1f}s)")

    print_summary(results)

    failed = sum(1 for r in results if r.status in ("FAIL", "ERROR"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
