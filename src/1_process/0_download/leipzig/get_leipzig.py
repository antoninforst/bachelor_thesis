import argparse
import os
import time
import urllib.parse
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "0_raw")
TO_DOWNLOAD_PATH = os.path.join(SCRIPT_DIR, "to_download.tsv")
ALREADY_PATH = os.path.join(SCRIPT_DIR, "already.tsv")


def _build_url(name: str) -> str:
    """Build the Leipzig wordlist download URL for a corpus name."""
    encoded = urllib.parse.quote(name, safe="")
    return (
        f"https://text.wortschatz-leipzig.de/bonito/run.cgi/wordlist?"
        f"corpname={encoded}"
        f"&results_url=https%3A%2F%2Ftext.wortschatz-leipzig.de%2F%23wordlist"
        f"%3Fcorpname%3D{encoded}%26tab%3Dbasic%26include_nonwords%3D1"
        f"%26itemsPerPage%3D50%26cols%3D%255B%2522frq%2522%255D"
        f"%26showtimelines%3D0%26diaattr%3D%26showtimelineabs%3D0"
        f"%26timelinesthreshold%3D5%26showresults%3D1"
        f"&wlmaxitems=100000000&wlsort=frq&wlattr=lc&wlpat=.*"
        f"&wlminfreq=1&wlicase=1&wlmaxfreq=0&wltype=simple"
        f"&include_nonwords=0&random=0&relfreq=0&freqcls=0"
        f"&reldocf=0&wlpage=1&page=1&format=csv&format=csv"
    )


def _load_already() -> set[str]:
    """Load the set of corpus names we already downloaded."""
    if not os.path.exists(ALREADY_PATH):
        return set()
    with open(ALREADY_PATH, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _load_to_download() -> list[str]:
    """Read corpus names from the to_download.tsv (second column)."""
    names = []
    with open(TO_DOWNLOAD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                names.append(parts[1])
    return names


def _record_already(name: str) -> None:
    """Mark a corpus as downloaded (rewrites the file to stay in sync)."""
    existing = _load_already()
    existing.add(name)
    with open(ALREADY_PATH, "w", encoding="utf-8") as f:
        for entry in sorted(existing):
            f.write(entry + "\n")


def _resolve_filename(response, fallback_name: str) -> str:
    """Get the output filename from the response header or build a fallback."""
    content_disp = response.headers.get("Content-Disposition", "")
    filename = None
    if "filename=" in content_disp:
        filename = content_disp.split("filename=")[-1].strip().strip('"')

    if not filename:
        filename = f"wordlist_{fallback_name}.csv"

    # strip the "wordlist_" prefix the server adds
    if filename.startswith("wordlist_"):
        filename = filename[len("wordlist_"):]

    return filename


def _download_with_progress(response, progress: str) -> bytes:
    """Read the response body in chunks and print a progress bar."""
    total_size = response.headers.get("Content-Length")
    total_size = int(total_size) if total_size else None

    downloaded = 0
    chunks = []
    while True:
        chunk = response.read(8192)
        if not chunk:
            break
        chunks.append(chunk)
        downloaded += len(chunk)
        if total_size:
            pct = downloaded / total_size * 100
            filled = int(30 * downloaded / total_size)
            bar = "█" * filled + "░" * (30 - filled)
            print(f"\r{progress}   [{bar}] {pct:5.1f}% ({downloaded:,}/{total_size:,} bytes)", end="", flush=True)
        else:
            print(f"\r{progress}   Downloaded {downloaded:,} bytes", end="", flush=True)
    print()
    return b"".join(chunks)


def _download_file(name: str, progress: str) -> None:
    """Download one wordlist and save it to RAW_DIR."""
    url = _build_url(name)
    print(f"{progress} Downloading {name} ...")
    response = urllib.request.urlopen(url)

    filename = _resolve_filename(response, name)
    data = _download_with_progress(response, progress)

    dest = os.path.join(RAW_DIR, filename)
    with open(dest, "wb") as f:
        f.write(data)

    print(f"{progress}   Saved as {filename} ({len(data):,} bytes)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download wordlists from Leipzig Corpora")
    parser.add_argument(
        "-n", type=int, required=True,
        help="Number of files to download in this run",
    )
    parser.add_argument(
        "--delay", type=float, default=5.0,
        help="Delay in seconds between downloads (default: 5)",
    )
    return parser.parse_args()


def _get_pending_batch(n: int) -> list[str]:
    """Return up to n corpus names that haven't been downloaded yet."""
    already = _load_already()
    to_download = _load_to_download()
    pending = [name for name in to_download if name not in already]

    print(f"Already downloaded: {len(already)} files")
    print(f"Total in to_download: {len(to_download)} files")
    print(f"Pending: {len(pending)}")

    return pending[:n]


def _download_batch(batch: list[str], delay: float) -> None:
    """Download a list of corpora with a delay between requests."""
    total = len(batch)
    for i, name in enumerate(batch, 1):
        progress = f"[{i}/{total}]"
        try:
            _download_file(name, progress)
            _record_already(name)
        except Exception as e:
            print(f"{progress}   ERROR downloading {name}: {e}")
            continue

        if i < total:
            print(f"{progress}   Waiting {delay}s before next download...\n")
            time.sleep(delay)


def main():
    args = _parse_args()
    batch = _get_pending_batch(args.n)

    if not batch:
        print("Nothing to download – all files already downloaded.")
        return

    print(f"Downloading {len(batch)} this run.\n")
    _download_batch(batch, args.delay)
    print("\nDone.")


if __name__ == "__main__":
    main()
