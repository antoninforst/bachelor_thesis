"""
Path resolution utility for notebooks.

Resolves paths relative to the *calling* notebook/script location,
validates existence, and provides clear error messages.

Usage from any notebook::

    import paths

    csv = paths.existing("results/1_process/statistics.csv", n=3)
    out = paths.edit("results/1_process/new_report.csv", n=3)
    fresh = paths.new("results/1_process/brand_new.csv", n=3)

``n`` controls where the base directory is:
    * ``n=0``  – the folder containing the calling file (default)
    * ``n>0``  – go *n* levels up from the calling file
    * ``n=-1`` – treat *path* as absolute
"""

import inspect
from pathlib import Path


def _caller_dir() -> Path:
    """Return the directory of the file that called the public function."""
    # 1) Try VS Code notebook variable (__vsc_ipynb_file__)
    try:
        ip = get_ipython()  # type: ignore[name-defined]
        nb_file = ip.user_ns.get("__vsc_ipynb_file__")
        if nb_file:
            p = Path(nb_file).resolve()
            if p.exists():
                return p.parent
    except NameError:
        pass

    # 2) Normal .py caller
    frame = inspect.stack()[2]
    caller_file = frame.filename
    p = Path(caller_file).resolve()
    if p.exists():
        return p.parent

    # 3) Fallback to cwd (e.g. plain Jupyter without VS Code)
    return Path.cwd()


def _resolve_base(n: int) -> Path:
    """Compute the base directory according to *n*."""
    if n == -1:
        return Path("/")  # absolute – base is filesystem root
    base = _caller_dir()
    for _ in range(n):
        base = base.parent
    return base


def _find_first_missing(base: Path, rel: Path) -> tuple[Path, str]:
    """Walk *rel* parts from *base* and return (base, first_missing_part)."""
    current = base
    for part in rel.parts:
        candidate = current / part
        if not candidate.exists():
            return current, part
        current = candidate
    return current, ""


def _nice(p: Path) -> str:
    """Shorten a path for display."""
    try:
        return str(p.resolve())
    except OSError:
        return str(p)


# ── public API ──────────────────────────────────────────────


def exist(path: str, n: int = 0) -> Path:
    """Return *path* as a resolved :class:`Path`; it **must** already exist.

    Raises :class:`FileNotFoundError` with a helpful message when the
    target does not exist.
    """
    base = _resolve_base(n)
    target = (base / path).resolve() if n != -1 else Path(path).resolve()

    if target.exists():
        return target

    # Build error message
    rel = Path(path)
    search_dir, missing = _find_first_missing(base if n != -1 else Path(path).resolve().parent, rel)
    raise FileNotFoundError(
        f"Path does not exist.\n"
        f"  searched from : {_nice(base)}\n"
        f"  full path     : {_nice(target)}\n"
        f"  first missing : '{missing}' not found in {_nice(search_dir)}"
    )


def edit(path: str, n: int = 0) -> Path:
    """Return *path* as a resolved :class:`Path`.

    The **parent directory** must exist (so you can write/overwrite the
    file), but the file itself may or may not exist.

    Raises :class:`FileNotFoundError` when the parent directory is missing.
    """
    base = _resolve_base(n)
    target = (base / path).resolve() if n != -1 else Path(path).resolve()

    if target.parent.exists():
        return target

    # Build error message
    rel = Path(path)
    # Check up to the parent only
    parent_rel = Path(*rel.parts[:-1]) if len(rel.parts) > 1 else Path(".")
    search_dir, missing = _find_first_missing(base if n != -1 else Path(path).resolve().parents[-1], parent_rel)
    raise FileNotFoundError(
        f"Parent directory does not exist (cannot write file).\n"
        f"  searched from : {_nice(base)}\n"
        f"  full path     : {_nice(target)}\n"
        f"  first missing : '{missing}' not found in {_nice(search_dir)}"
    )


def new(path: str, n: int = 0) -> Path:
    """Return *path* as a resolved :class:`Path`.

    The **parent directory** must exist **and** no file/directory with
    this name may exist yet.

    Raises :class:`FileNotFoundError` when the parent is missing, or
    :class:`FileExistsError` when the target already exists.
    """
    base = _resolve_base(n)
    target = (base / path).resolve() if n != -1 else Path(path).resolve()

    if target.exists():
        raise FileExistsError(
            f"Path already exists (expected it to be new).\n"
            f"  searched from : {_nice(base)}\n"
            f"  full path     : {_nice(target)}"
        )

    if target.parent.exists():
        return target

    # Parent missing – build error
    rel = Path(path)
    parent_rel = Path(*rel.parts[:-1]) if len(rel.parts) > 1 else Path(".")
    search_dir, missing = _find_first_missing(base if n != -1 else Path(path).resolve().parents[-1], parent_rel)
    raise FileNotFoundError(
        f"Parent directory does not exist (cannot create file).\n"
        f"  searched from : {_nice(base)}\n"
        f"  full path     : {_nice(target)}\n"
        f"  first missing : '{missing}' not found in {_nice(search_dir)}"
    )