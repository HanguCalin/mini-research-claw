"""Node 4b: Dependency Resolver.

Type: Non-AI (pure Python).
Uses AST parsing to extract pip dependencies and remote dataset IDs from
the generated code, then pre-fetches everything to the host-side cache
so the network-isolated sandbox can access them.
"""

from __future__ import annotations

import ast
import logging
import subprocess
from typing import Any

from backend.config import CACHE_ROOT
from backend.state import AutoResearchState

logger = logging.getLogger(__name__)

MODULE_TO_PYPI: dict[str, str] = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "transformers": "transformers",
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "datasets": "datasets",
    "huggingface_hub": "huggingface_hub",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
}

STDLIB_MODULES = {
    "os", "sys", "json", "re", "math", "random", "time", "datetime",
    "pathlib", "collections", "itertools", "functools", "typing",
    "copy", "io", "csv", "pickle", "hashlib", "logging", "unittest",
    "abc", "dataclasses", "enum", "statistics", "textwrap", "string",
    "operator", "contextlib", "warnings", "glob", "shutil", "tempfile",
}


def dependency_resolver(state: AutoResearchState) -> dict[str, Any]:
    """Parse code for imports and datasets, pre-fetch to host cache."""
    code = state.get("python_code", "")

    imports = _extract_imports(code)
    datasets = _extract_dataset_ids(code)

    pip_packages = _map_to_pypi(imports)

    pip_cache = str(CACHE_ROOT / "pip")
    hf_cache = str(CACHE_ROOT / "hf")
    sklearn_cache = str(CACHE_ROOT / "sklearn")

    _prefetch_pip(pip_packages, pip_cache)
    _prefetch_datasets(datasets, hf_cache, sklearn_cache)

    return {
        "resolved_dependencies": pip_packages,
        "resolved_datasets": datasets,
        "dataset_cache_path": str(CACHE_ROOT),
    }


def _extract_imports(code: str) -> list[str]:
    """Use AST to extract all top-level module names from import statements."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.warning("AST parse failed — returning empty import list")
        return []

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])

    return sorted(modules - STDLIB_MODULES)


def _extract_dataset_ids(code: str) -> list[str]:
    """Detect load_dataset("..."), fetch_openml(...), sklearn fetch_* patterns."""
    datasets: list[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return datasets

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func_name = _get_call_name(node)
        if not func_name:
            continue

        if func_name in ("load_dataset", "datasets.load_dataset"):
            if node.args and isinstance(node.args[0], ast.Constant):
                datasets.append(str(node.args[0].value))

        elif "fetch_" in func_name:
            if node.args and isinstance(node.args[0], ast.Constant):
                datasets.append(str(node.args[0].value))
            elif any(
                isinstance(kw.value, ast.Constant) and kw.arg == "name"
                for kw in node.keywords
            ):
                for kw in node.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                        datasets.append(str(kw.value.value))

    return datasets


def _get_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts = []
        current = node.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _map_to_pypi(modules: list[str]) -> list[str]:
    """Map module names to PyPI package names."""
    packages: set[str] = set()
    for mod in modules:
        pypi = MODULE_TO_PYPI.get(mod, mod)
        packages.add(pypi)
    return sorted(packages)


def _prefetch_pip(packages: list[str], dest: str) -> None:
    """Download pip wheels to the host cache."""
    if not packages:
        return
    try:
        subprocess.run(
            ["pip", "download", "--dest", dest, "--only-binary=:all:"] + packages,
            capture_output=True, text=True, timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("pip prefetch failed or timed out")


def _prefetch_datasets(datasets: list[str], hf_cache: str, sklearn_cache: str) -> None:
    """Pre-fetch HF datasets and scikit-learn datasets to cache."""
    for ds_id in datasets:
        if ds_id.startswith("sklearn."):
            _prefetch_sklearn(ds_id, sklearn_cache)
        else:
            _prefetch_hf(ds_id, hf_cache)


def _prefetch_hf(dataset_id: str, cache_dir: str) -> None:
    try:
        subprocess.run(
            ["huggingface-cli", "download", dataset_id, "--cache-dir", cache_dir,
             "--repo-type", "dataset"],
            capture_output=True, text=True, timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("HF prefetch failed for %s", dataset_id)


def _prefetch_sklearn(dataset_id: str, cache_dir: str) -> None:
    func_name = dataset_id.replace("sklearn.", "")
    try:
        import sklearn.datasets
        fetch_fn = getattr(sklearn.datasets, func_name, None)
        if fetch_fn:
            fetch_fn(data_home=cache_dir)
    except Exception:
        logger.warning("sklearn prefetch failed for %s", dataset_id)
