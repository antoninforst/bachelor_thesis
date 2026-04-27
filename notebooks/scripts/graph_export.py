"""
Graph export utility for saving matplotlib figures to the graph_export folder.

Usage from any notebook:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[...] / "notebooks" / "graph_export"))
    # or simply:
    sys.path.insert(0, "../../graph_export")  # adjust depth as needed
    import graph_export

    fig, ax = plt.subplots()
    ...
    graph_export.save("my_graph", fig)
"""

import inspect
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPORT_DIR = _SCRIPT_DIR / "img"
_EXPORT_DIR.mkdir(exist_ok=True)
_REGISTRY_FILE = _SCRIPT_DIR / "graphs.yaml"


def save(name: str, fig, fmt: str = "png", **savefig_kwargs):
    """Save a matplotlib figure and update the registry.

    Parameters
    ----------
    name : str
        Graph identifier (used as filename without extension).
    fig : matplotlib.figure.Figure
        The figure to save.
    fmt : str
        File format (default "png"). Any format supported by matplotlib.
    **savefig_kwargs
        Extra keyword arguments forwarded to ``fig.savefig``
        (e.g. ``dpi=300``).
    """
    savefig_kwargs.setdefault("bbox_inches", "tight")
    savefig_kwargs.setdefault("dpi", 300)

    out_path = _EXPORT_DIR / f"{name}.{fmt}"
    fig.savefig(out_path, **savefig_kwargs)

    # Resolve the calling notebook / script
    source = _caller_source()

    # Update YAML registry
    registry = _load_registry()
    registry[name] = {
        "file": f"{name}.{fmt}",
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save_registry(registry)

    print(f"graph_export: saved {out_path.name}")


# ── internal helpers ──────────────────────────────────────────────


def _caller_source() -> str:
    """Best-effort detection of the calling notebook or script path."""
    # Walk up the call stack looking for a frame outside this module
    for frame_info in inspect.stack():
        fname = frame_info.filename
        # Skip this module
        if fname == __file__:
            continue
        # Jupyter kernels store the notebook path in globals
        globs = frame_info.frame.f_globals
        nb_name = globs.get("__vsc_ipynb_file__") or globs.get("__session__")
        if nb_name:
            return str(Path(nb_name).name)
        # ipykernel puts it differently sometimes
        if "ipykernel" in fname or "<ipython" in fname:
            return "<notebook>"
        return str(Path(fname).name)
    return "<unknown>"


def _load_registry() -> dict:
    if _REGISTRY_FILE.exists():
        with open(_REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    return {}


def _save_registry(registry: dict):
    with open(_REGISTRY_FILE, "w", encoding="utf-8") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=True)
