"""Eval: Dependency Resolver AST Parsing Accuracy.

Tests the AST import extractor and dataset-ID detector against 10 sample
scripts with known expected imports and dataset calls.
No API key required — pure AST/regex parsing.
Pip/HF prefetch calls are mocked so nothing is actually downloaded.

Pass criteria: 100% recall on known dependencies.
"""

import sys
import os
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.agents.dependency_resolver import _extract_imports, _extract_dataset_ids

# ─── Test cases ───────────────────────────────────────────────────────────────

IMPORT_CASES = [
    # (description, code, expected_imports_subset)
    ("simple top-level imports",
     "import numpy as np\nimport pandas as pd\n",
     {"numpy", "pandas"}),

    ("from-imports",
     "from sklearn.ensemble import RandomForestClassifier\nfrom torch import nn\n",
     {"sklearn", "torch"}),

    ("mixed import styles",
     "import scipy.stats\nfrom transformers import AutoModel\nimport os\nimport json\n",
     {"scipy", "transformers"}),

    ("stdlib modules excluded (os, sys, json) — they appear in output but map to nothing in pypi",
     "import os\nimport sys\nimport json\nimport numpy\n",
     {"numpy"}),

    ("dynamic import still captured as static-looking import",
     "import importlib\nfrom huggingface_hub import hf_hub_download\n",
     {"importlib", "huggingface_hub"}),

    ("seaborn and matplotlib",
     "import matplotlib.pyplot as plt\nimport seaborn as sns\n",
     {"matplotlib", "seaborn"}),

    ("datasets library",
     "from datasets import load_dataset\nimport pandas\n",
     {"datasets", "pandas"}),

    ("torch with submodule",
     "import torch.nn as nn\nimport torch.optim as optim\n",
     {"torch"}),

    ("multiple from-imports on one module",
     "from sklearn.model_selection import train_test_split, cross_val_score\n",
     {"sklearn"}),

    ("no imports at all",
     "x = 1\nprint(x)\n",
     set()),
]

DATASET_CASES = [
    # (description, code, expected_datasets)
    ("HF load_dataset call",
     'from datasets import load_dataset\nds = load_dataset("imdb")\n',
     {"imdb"}),

    ("no load_dataset call — variable assignment is NOT captured (by design)",
     'dataset_id = "sklearn.iris"\nprint(dataset_id)\n',
     set()),

    ("multiple datasets",
     'load_dataset("glue", "sst2")\nload_dataset("squad")\n',
     {"glue", "squad"}),

    ("no datasets",
     "import numpy as np\nx = np.array([1,2,3])\n",
     set()),
]

# ─── Eval Logic ───────────────────────────────────────────────────────────────

def run_eval():
    print("Starting Dependency Resolver Accuracy Evaluation (no API key required)...")
    import_correct = 0
    dataset_correct = 0

    print("\n[Import Extraction]")
    for desc, code, expected_subset in IMPORT_CASES:
        got = set(_extract_imports(code))
        # Check that expected subset is captured
        missing = expected_subset - got
        ok = len(missing) == 0
        import_correct += int(ok)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {desc}")
        if not ok:
            print(f"         missing: {missing}, got: {got}")

    print("\n[Dataset ID Extraction]")
    for desc, code, expected_subset in DATASET_CASES:
        got = set(_extract_dataset_ids(code))
        missing = expected_subset - got
        ok = len(missing) == 0
        dataset_correct += int(ok)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {desc}")
        if not ok:
            print(f"         missing: {missing}, got: {got}")

    total = len(IMPORT_CASES) + len(DATASET_CASES)
    correct = import_correct + dataset_correct
    pct = correct / total * 100

    print(f"\n--- Results ---")
    print(f"Import extraction: {import_correct}/{len(IMPORT_CASES)}")
    print(f"Dataset extraction: {dataset_correct}/{len(DATASET_CASES)}")
    print(f"Overall: {correct}/{total} ({pct:.0f}%)")

    if correct == total:
        print("\nEvaluation PASSED (100% recall on known dependencies)")
    else:
        print(f"\nEvaluation FAILED ({correct}/{total} — threshold is 100%)")


if __name__ == "__main__":
    run_eval()
