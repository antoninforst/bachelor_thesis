import argparse
import os
import time
import urllib.parse
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "0_raw")
TO_DOWNLOAD_PATH = os.path.join(SCRIPT_DIR, "to_download.tsv")
ALREADY_PATH = os.path.join(SCRIPT_DIR, "already.tsv")


def geturl(name):
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


def load_already():
    if not os.path.exists(ALREADY_PATH):
        return set()
    with open(ALREADY_PATH, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_to_download():
    names = []
    with open(TO_DOWNLOAD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                names.append(parts[1])
    return names


def record_already(name):
    # Read existing entries, add new one, write everything back
    # (protects against VS Code overwriting the file with a stale buffer)
    existing = load_already()
    existing.add(name)
    with open(ALREADY_PATH, "w", encoding="utf-8") as f:
        for entry in sorted(existing):
            f.write(entry + "\n")


def download_file(name, progress):
    url = geturl(name)
    print(f"{progress} Downloading {name} ...")
    response = urllib.request.urlopen(url)

    total_size = response.headers.get("Content-Length")
    total_size = int(total_size) if total_size else None

    # Get the filename from Content-Disposition header, fall back to name.csv
    content_disp = response.headers.get("Content-Disposition", "")
    filename = None
    if "filename=" in content_disp:
        filename = content_disp.split("filename=")[-1].strip().strip('"')

    if not filename:
        filename = f"wordlist_{name}.csv"

    # Remove wordlist_ prefix
    if filename.startswith("wordlist_"):
        filename = filename[len("wordlist_"):]

    dest = os.path.join(RAW_DIR, filename)

    # Download with progress bar
    downloaded = 0
    chunk_size = 8192
    chunks = []
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
        downloaded += len(chunk)
        if total_size:
            pct = downloaded / total_size * 100
            bar_len = 30
            filled = int(bar_len * downloaded / total_size)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r{progress}   [{bar}] {pct:5.1f}% ({downloaded:,}/{total_size:,} bytes)", end="", flush=True)
        else:
            print(f"\r{progress}   Downloaded {downloaded:,} bytes", end="", flush=True)

    data = b"".join(chunks)
    print()  # newline after progress bar

    with open(dest, "wb") as f:
        f.write(data)

    print(f"{progress}   Saved as {filename} ({len(data):,} bytes)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download wordlists from Leipzig Corpora")
    parser.add_argument(
        "-n", type=int, required=True,
        help="Number of files to download in this run",
    )
    parser.add_argument(
        "--delay", type=float, default=5.0,
        help="Delay in seconds between downloads (default: 5)",
    )
    args = parser.parse_args()

    already = load_already()
    to_download = load_to_download()

    pending = [name for name in to_download if name not in already]
    batch = pending[: args.n]

    if not batch:
        print("Nothing to download – all files already downloaded.")
        return

    print(f"Already downloaded: {len(already)} files")
    print(f"Total in to_download: {len(to_download)} files")
    print(f"Pending: {len(pending)}, downloading {len(batch)} this run.\n")

    total = len(batch)
    for i, name in enumerate(batch, 1):
        progress = f"[{i}/{total}]"
        try:
            download_file(name, progress)
            record_already(name)
        except Exception as e:
            print(f"{progress}   ERROR downloading {name}: {e}")
            continue

        if i < total:
            print(f"{progress}   Waiting {args.delay}s before next download...\n")
            time.sleep(args.delay)

    print("\nDone.")


if __name__ == "__main__":
    main()
