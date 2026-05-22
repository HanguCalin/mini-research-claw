"""Unit tests for backend/agents/dependency_resolver.py

Tests:
- _extract_imports correctly identifies all import styles via AST
- stdlib modules are excluded from results
- _extract_dataset_ids detects load_dataset() and fetch_* patterns
- _map_to_pypi translates module names to PyPI package names
- dependency_resolver node returns correct state keys
- SyntaxError in generated code is handled gracefully
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.agents.dependency_resolver import (
    _extract_dataset_ids,
    _extract_imports,
    _map_to_pypi,
    dependency_resolver,
)


# ─── _extract_imports ─────────────────────────────────────────────────────────


def test_extract_simple_import():
    code = "import numpy\nimport pandas"
    result = _extract_imports(code)
    assert "numpy" in result
    assert "pandas" in result


def test_extract_from_import():
    code = "from sklearn.ensemble import RandomForestClassifier"
    result = _extract_imports(code)
    assert "sklearn" in result


def test_extract_submodule_import():
    code = "import matplotlib.pyplot as plt"
    result = _extract_imports(code)
    assert "matplotlib" in result
    # Only the top-level module name is kept
    assert "matplotlib.pyplot" not in result


def test_stdlib_modules_excluded():
    code = "import os\nimport sys\nimport json\nimport numpy"
    result = _extract_imports(code)
    assert "os" not in result
    assert "sys" not in result
    assert "json" not in result
    assert "numpy" in result


def test_returns_sorted_list():
    code = "import torch\nimport numpy\nimport pandas"
    result = _extract_imports(code)
    assert result == sorted(result)


def test_no_duplicate_modules():
    code = "import numpy\nimport numpy\nfrom numpy import array"
    result = _extract_imports(code)
    assert result.count("numpy") == 1


def test_syntax_error_returns_empty_list():
    code = "def broken(:"
    result = _extract_imports(code)
    assert result == []


def test_empty_code_returns_empty_list():
    result = _extract_imports("")
    assert result == []


# ─── _extract_dataset_ids ─────────────────────────────────────────────────────


def test_detects_load_dataset_call():
    code = 'from datasets import load_dataset\nds = load_dataset("imdb")'
    result = _extract_dataset_ids(code)
    assert "imdb" in result


def test_detects_datasets_load_dataset_qualified():
    code = 'import datasets\nds = datasets.load_dataset("glue")'
    result = _extract_dataset_ids(code)
    assert "glue" in result


def test_detects_fetch_openml():
    code = 'from sklearn.datasets import fetch_openml\nds = fetch_openml("mnist_784")'
    result = _extract_dataset_ids(code)
    assert "mnist_784" in result


def test_detects_fetch_with_name_kwarg():
    code = 'fetch_openml(name="adult")'
    result = _extract_dataset_ids(code)
    assert "adult" in result


def test_no_datasets_returns_empty():
    code = "import numpy\nx = numpy.array([1, 2, 3])"
    result = _extract_dataset_ids(code)
    assert result == []


def test_syntax_error_returns_empty():
    result = _extract_dataset_ids("def broken(:")
    assert result == []


# ─── _map_to_pypi ─────────────────────────────────────────────────────────────


def test_sklearn_mapped_to_scikit_learn():
    result = _map_to_pypi(["sklearn"])
    assert "scikit-learn" in result
    assert "sklearn" not in result


def test_cv2_mapped_to_opencv():
    result = _map_to_pypi(["cv2"])
    assert "opencv-python" in result


def test_unknown_module_passes_through():
    result = _map_to_pypi(["some_custom_package"])
    assert "some_custom_package" in result


def test_multiple_modules_all_mapped():
    result = _map_to_pypi(["numpy", "pandas", "sklearn"])
    assert "numpy" in result
    assert "pandas" in result
    assert "scikit-learn" in result


def test_returns_sorted_list():
    result = _map_to_pypi(["torch", "numpy", "pandas"])
    assert result == sorted(result)


def test_no_duplicates_in_output():
    # PIL and cv2 both map to different packages — no duplication expected
    result = _map_to_pypi(["numpy", "numpy"])
    assert result.count("numpy") == 1


# ─── dependency_resolver node ─────────────────────────────────────────────────


@patch("backend.agents.dependency_resolver._prefetch_pip")
@patch("backend.agents.dependency_resolver._prefetch_datasets")
def test_resolver_returns_required_keys(mock_prefetch_ds, mock_prefetch_pip):
    state = {
        "python_code": (
            "import numpy\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from datasets import load_dataset\n"
            'ds = load_dataset("imdb")\n'
        )
    }
    result = dependency_resolver(state)
    assert "resolved_dependencies" in result
    assert "resolved_datasets" in result
    assert "dataset_cache_path" in result


@patch("backend.agents.dependency_resolver._prefetch_pip")
@patch("backend.agents.dependency_resolver._prefetch_datasets")
def test_resolver_identifies_numpy_and_sklearn(mock_prefetch_ds, mock_prefetch_pip):
    state = {"python_code": "import numpy\nfrom sklearn.linear_model import LogisticRegression"}
    result = dependency_resolver(state)
    assert "numpy" in result["resolved_dependencies"]
    assert "scikit-learn" in result["resolved_dependencies"]


@patch("backend.agents.dependency_resolver._prefetch_pip")
@patch("backend.agents.dependency_resolver._prefetch_datasets")
def test_resolver_identifies_dataset(mock_prefetch_ds, mock_prefetch_pip):
    state = {"python_code": 'from datasets import load_dataset\nds = load_dataset("glue")'}
    result = dependency_resolver(state)
    assert "glue" in result["resolved_datasets"]


@patch("backend.agents.dependency_resolver._prefetch_pip")
@patch("backend.agents.dependency_resolver._prefetch_datasets")
def test_resolver_handles_empty_code(mock_prefetch_ds, mock_prefetch_pip):
    state = {"python_code": ""}
    result = dependency_resolver(state)
    assert result["resolved_dependencies"] == []
    assert result["resolved_datasets"] == []


@patch("backend.agents.dependency_resolver._prefetch_pip")
@patch("backend.agents.dependency_resolver._prefetch_datasets")
def test_resolver_handles_missing_code_key(mock_prefetch_ds, mock_prefetch_pip):
    result = dependency_resolver({})
    assert result["resolved_dependencies"] == []
    assert result["resolved_datasets"] == []
