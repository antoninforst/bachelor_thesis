"""
Graph export utility for saving matplotlib figures to the thesis.

Usage from any notebook:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[...] / "notebooks" / "scripts"))
    import graph_export

    fig, ax = plt.subplots()
    ...
    graph_export.save("my_graph", fig)

CLI usage:
    python graph_export.py              # print help
    python graph_export.py -u/--update  # rerun all source notebooks
    python graph_export.py -c/--clear   # remove invalid entries from registry
    python graph_export.py -f/--force   # copy exported images to thesis/img/auto/
"""

import argparse
import inspect
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPORT_DIR = _SCRIPT_DIR / "img"
_EXPORT_DIR.mkdir(exist_ok=True)
_REGISTRY_FILE = _SCRIPT_DIR / "graphs.yaml"
_NOTEBOOKS_DIR = _SCRIPT_DIR.parent  # notebooks/
_THESIS_IMG_DIR = _SCRIPT_DIR.parents[1] / "thesis" / "img" / "auto"


def save(name: str, fig, fmt: str = "pdf", **savefig_kwargs):
    """Save a matplotlib figure and update the registry.

    Parameters
    ----------
    name : str
        Graph identifier (used as filename without extension).
    fig : matplotlib.figure.Figure
        The figure to save.
    fmt : str
        File format (default "pdf"). PDF is vector, sharp at any zoom.
        Other options: "png", "svg", "eps".
    **savefig_kwargs
        Extra keyword arguments forwarded to ``fig.savefig``
        (e.g. ``dpi=300``).
    """
    savefig_kwargs.setdefault("bbox_inches", "tight")
    if fmt == "png":
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
    for frame_info in inspect.stack():
        fname = frame_info.filename
        if fname == __file__:
            continue
        globs = frame_info.frame.f_globals
        nb_name = globs.get("__vsc_ipynb_file__") or globs.get("__session__")
        if nb_name:
            return str(Path(nb_name).name)
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


def _find_notebook(name: str) -> Path | None:
    """Find a notebook by filename under the notebooks/ directory."""
    matches = list(_NOTEBOOKS_DIR.rglob(name))
    return matches[0] if matches else None


# ── CLI commands ──────────────────────────────────────────────────


def _cmd_update():
    """Rerun all source notebooks listed in the registry."""
    registry = _load_registry()
    if not registry:
        print("Registry is empty, nothing to update.")
        return

    sources = sorted(set(entry["source"] for entry in registry.values()))
    print(f"Found {len(sources)} source notebook(s) to rerun:\n")

    for nb_name in sources:
        nb_path = _find_notebook(nb_name)
        if nb_path is None:
            print(f"  [INVALID] {nb_name} — not found under {_NOTEBOOKS_DIR}")
            continue

        print(f"  Running {nb_name} ({nb_path.relative_to(_NOTEBOOKS_DIR)})...")
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=600",
                str(nb_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  [OK] {nb_name}")
        else:
            print(f"  [FAILED] {nb_name}")
            print(f"    stderr: {result.stderr[:300]}")


def _cmd_clear():
    """Remove entries whose source notebook no longer exists."""
    registry = _load_registry()
    if not registry:
        print("Registry is empty.")
        return

    to_remove = []
    for name, entry in registry.items():
        nb_path = _find_notebook(entry["source"])
        img_path = _EXPORT_DIR / entry["file"]
        if nb_path is None:
            print(f"  [REMOVE] {name}: source '{entry['source']}' not found")
            to_remove.append(name)
        elif not img_path.exists():
            print(f"  [REMOVE] {name}: image '{entry['file']}' not found")
            to_remove.append(name)

    if not to_remove:
        print("All entries are valid.")
        return

    for name in to_remove:
        del registry[name]
    _save_registry(registry)
    print(f"\nRemoved {len(to_remove)} invalid entry/entries.")


def _cmd_force():
    """Copy all exported images to thesis/img/auto/."""
    registry = _load_registry()
    if not registry:
        print("Registry is empty, nothing to copy.")
        return

    _THESIS_IMG_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name, entry in registry.items():
        src = _EXPORT_DIR / entry["file"]
        if not src.exists():
            print(f"  [SKIP] {entry['file']} — not found in {_EXPORT_DIR.name}/")
            continue
        dst = _THESIS_IMG_DIR / entry["file"]
        shutil.copy2(src, dst)
        copied += 1
        print(f"  {entry['file']} -> thesis/img/auto/")

    print(f"\nCopied {copied} file(s) to {_THESIS_IMG_DIR}")


def _main():
    parser = argparse.ArgumentParser(
        description="Graph export manager — save, update, and publish figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python graph_export.py -u        Rerun all source notebooks to regenerate graphs
  python graph_export.py -c        Remove invalid entries from graphs.yaml
  python graph_export.py -f        Copy all images to thesis/img/auto/
  python graph_export.py -u -f     Rerun notebooks then copy to thesis
""",
    )
    parser.add_argument(
        "-u", "--update",
        action="store_true",
        help="rerun all source notebooks to regenerate graphs",
    )
    parser.add_argument(
        "-c", "--clear",
        action="store_true",
        help="remove entries with invalid source/image paths from the registry",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="copy exported images to thesis/img/auto/",
    )

    args = parser.parse_args()

    if not (args.update or args.clear or args.force):
        parser.print_help()
        return

    if args.clear:
        _cmd_clear()
    if args.update:
        _cmd_update()
    if args.force:
        _cmd_force()


if __name__ == "__main__":
    _main()
