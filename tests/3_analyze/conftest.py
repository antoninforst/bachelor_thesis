"""Conftest: ensure src/3_analyze is importable."""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(_PROJECT_ROOT / "src" / "3_analyze"))
