"""Root conftest: ensure src/1_process scripts are importable."""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(_PROJECT_ROOT / "src" / "1_process" / "1_filter"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "1_process" / "2_aggregate"))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "1_process" / "3_truncate"))
