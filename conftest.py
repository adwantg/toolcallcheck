"""Repo-level pytest configuration for local source checkouts.

Ensures tests can import the package from ``src/`` and load the pytest
fixtures plugin without requiring an editable install first.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    distribution("toolcallcheck")
except PackageNotFoundError:
    pytest_plugins = ["toolcallcheck.fixtures"]
