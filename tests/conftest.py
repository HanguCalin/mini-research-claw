import sys
from unittest.mock import MagicMock

# Block heavy/crashing imports for Python 3.14 compatibility
# This must happen before any backend code is imported.
for _mod in [
    "pyarrow", "pyarrow.lib", "pyarrow.dataset",
    "sentence_transformers",
    "datasets",
    "backend.utils.embeddings",
]:
    mock = MagicMock()
    if _mod == "pyarrow":
        mock.__version__ = "14.0.1"
    sys.modules.setdefault(_mod, mock)
