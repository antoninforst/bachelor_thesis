"""
Download monolingual data from the TIL (Turkic Interlingua) corpus
and convert to word-frequency CSV files for data/0_raw/.

Usage:
    python src/0_data_processing/corpora/til/download_til.py
    python src/0_data_processing/corpora/til/download_til.py --langs az ba tr
    python src/0_data_processing/corpora/til/download_til.py --langs all
    python src/0_data_processing/corpora/til/download_til.py --skip-existing
"""

import argparse
import csv
import os
import sys
import urllib.request
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "0_raw")

# GCS bucket publicly accessible via HTTPS
BASE_URL = "https://storage.cloud.google.com/til-corpus/mono"

# TIL code -> ISO 639-3 code used in this project
TIL_TO_ISO3 = {
    "alt": "alt",   # Southern Altai
    "az":  "aze",   # Azerbaijani
    "ba":  "bak",   # Bashkir
    "cjs": "cjs",   # Shor
    "crh": "crh",   # Crimean Tatar
    "cv":  "chv",   # Chuvash
    "gag": "gag",   # Gagauz
    "kaa": "kaa",   # Karakalpak
    "kjh": "kjh",   # Khakas
    "kk":  "kaz",   # Kazakh
    "krc": "krc",   # Karachay-Balkar
    "kum": "kum",   # Kumyk
    "ky":  "kir",   # Kirghiz
    "sah": "sah",   # Sakha/Yakut
    "slr": "slr",   # Salar
    "tk":  "tuk",   # Turkmen
    "tr":  "tur",   # Turkish
    "tt":  "tat",   # Tatar
    "tyv": "tyv",   # Tuvan
    "ug":  "uig",   # Uyghur
    "uum": "uum",   # Urum
    "uz":  "uzb",   # Uzbek
}

# Skip English and Russian - already well-covered by other sources
SKIP_LANGS = {"en", "ru"}

ALL_LANGS = sorted(TIL_TO_ISO3.keys())

# Chunk size for streaming large files (64 KB)
CHUNK_SIZE = 65536


def stream_frequencies(til_code: str) -> Counter[str] | None:
    """Download and tokenize monolingual text, streaming to keep memory low."""
    url = f"{BASE_URL}/{til_code}.txt"
    print(f"  Downloading {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=300)
    except urllib.error.HTTPError as e:
        print(f"  HTTP error {e.code} for {til_code}: {e.reason}")
        return None
    except Exception as e:
        print(f"  Error downloading {til_code}: {e}")
        return None

    total_size = resp.headers.get("Content-Length")
    total_size = int(total_size) if total_size else None

    counter: Counter[str] = Counter()
    downloaded = 0
    leftover = ""

    try:
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            downloaded += len(chunk)
            text = leftover + chunk.decode("utf-8", errors="replace")
            lines = text.split("\n")
            # Last element might be incomplete line
            leftover = lines[-1]
            for line in lines[:-1]:
                tokens = line.split()
                counter.update(tokens)

            if total_size:
                pct = downloaded * 100 // total_size
                mb = downloaded / (1024 * 1024)
                print(f"\r  {mb:.1f} MB ({pct}%)", end="", flush=True)
            else:
                mb = downloaded / (1024 * 1024)
                print(f"\r  {mb:.1f} MB", end="", flush=True)

        # Process any remaining text
        if leftover.strip():
            counter.update(leftover.split())
    finally:
        resp.close()

    print()  # newline after progress
    return counter


def save_csv(freqs: list[tuple[str, int]], path: str) -> None:
    """Save frequency list as CSV matching the project format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        writer.writerows(freqs)


def main():
    parser = argparse.ArgumentParser(description="Download TIL monolingual data as frequency lists")
    parser.add_argument(
        "--langs", nargs="*", default=None,
        help=f"TIL language codes to download (default: all). Available: {', '.join(ALL_LANGS)}"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip languages that already have a TIL file in data/0_raw/"
    )
    args = parser.parse_args()

    langs = args.langs if args.langs else ALL_LANGS
    if langs == ["all"]:
        langs = ALL_LANGS

    # Validate
    invalid = [l for l in langs if l not in TIL_TO_ISO3 and l not in SKIP_LANGS]
    if invalid:
        print(f"Unknown language codes: {', '.join(invalid)}")
        print(f"Available: {', '.join(ALL_LANGS)}")
        sys.exit(1)

    # Filter out EN/RU
    langs = [l for l in langs if l not in SKIP_LANGS]

    print(f"Will download {len(langs)} language(s): {', '.join(langs)}")
    print(f"Output directory: {RAW_DIR}")
    print()

    success = 0
    skipped = 0
    failed = 0

    for i, til_code in enumerate(langs, 1):
        iso3 = TIL_TO_ISO3[til_code]
        out_name = f"{iso3}_til.csv"
        out_path = os.path.join(RAW_DIR, out_name)

        print(f"[{i}/{len(langs)}] {til_code} -> {out_name}")

        if args.skip_existing and os.path.exists(out_path):
            print(f"  Already exists, skipping.")
            skipped += 1
            continue

        counter = stream_frequencies(til_code)
        if counter is None:
            failed += 1
            continue

        if not counter:
            print(f"  No words found in downloaded text.")
            failed += 1
            continue

        # Sort by frequency descending, then alphabetically
        freqs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        total_tokens = sum(f for _, f in freqs)

        save_csv(freqs, out_path)
        print(f"  Saved {len(freqs):,} unique words, {total_tokens:,} total tokens")
        success += 1

    print()
    print(f"Done: {success} downloaded, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
