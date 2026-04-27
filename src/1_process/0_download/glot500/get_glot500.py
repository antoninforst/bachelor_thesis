"""
Download monolingual data from the Glot500 dataset (Hugging Face)
and convert to word-frequency CSV files for data/0_raw/.

Uses the HF datasets server Parquet API - no authentication needed.

Usage:
    python src/1_process/0_download/glot500/get_glot500.py
    python src/1_process/0_download/glot500/get_glot500.py --langs alt chv krc
    python src/1_process/0_download/glot500/get_glot500.py --skip-existing
    python src/1_process/0_download/glot500/get_glot500.py --list-configs
"""

import argparse
import csv
import json
import os
import queue
import shutil
import sys
import threading
import time
import urllib.request
from collections import Counter

import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "0_raw")
PARQUET_CACHE_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "0_3_other", "glot500_parquet")

DATASET = "cis-lmu/Glot500"
HF_PARQUET_API = f"https://huggingface.co/api/datasets/{DATASET}/parquet"
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
HF_SPLITS_API = "https://datasets-server.huggingface.co/splits"

# How the Glot500 config prefix maps to our ISO 639-3 code.
# Most are identical; only override the ones that differ or need special handling.
# If a code is not listed here, we use the Glot500 prefix as-is.
GLOT_TO_ISO3_OVERRIDES = {
    # Glot500 uses some macro/alternative codes
}


MAX_RETRIES = 5
RETRY_DELAYS = [5, 15, 30, 60, 120]


def fetch_json(url: str, timeout: int = 60):
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"    [dl] Retry {attempt+1}/{MAX_RETRIES} in {delay}s ({e})")
                time.sleep(delay)
            else:
                raise


def get_all_configs() -> list[str]:
    """Get all available configs from the Glot500 dataset."""
    data = fetch_json(f"{HF_SPLITS_API}?dataset={DATASET}")
    return sorted(set(s["config"] for s in data["splits"]))


def get_parquet_urls(config: str) -> list[str]:
    """Get Parquet file URLs for a given config."""
    url = f"{HF_PARQUET_API}/{config}/train"
    return fetch_json(url)


def validate_parquet(path: str) -> bool:
    """Check if a cached parquet file is valid."""
    try:
        pq.read_schema(path)
        return True
    except Exception:
        return False


def download_parquet(url: str, dest: str) -> None:
    """Download a Parquet file to a local path with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                total_size = resp.headers.get("Content-Length")
                total_size = int(total_size) if total_size else None
                downloaded = 0
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        mb = downloaded / (1024 * 1024)
                        if total_size:
                            pct = downloaded * 100 // total_size
                            print(f"\r    [dl] {mb:.1f} MB ({pct}%)", end="", flush=True)
                        else:
                            print(f"\r    [dl] {mb:.1f} MB", end="", flush=True)
            print()
            return
        except Exception as e:
            # Remove partial file on failure
            if os.path.exists(dest):
                os.unlink(dest)
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"\n    [dl] Retry {attempt+1}/{MAX_RETRIES} in {delay}s ({e})")
                time.sleep(delay)
            else:
                raise


def ensure_parquet_cached(purl: str, cached_path: str, label: str) -> None:
    """Download a parquet to cache if not already there and valid."""
    if os.path.exists(cached_path):
        if validate_parquet(cached_path):
            print(f"  [dl] {label} cached")
            return
        else:
            print(f"  [dl] {label} corrupt, re-downloading...")
            os.unlink(cached_path)
    else:
        print(f"  [dl] {label} downloading...")
    download_parquet(purl, cached_path)


# ---------------------------------------------------------------------------
# Pipeline: download thread + processor thread communicating via a queue
# ---------------------------------------------------------------------------

# Message types sent from download thread -> processor thread
# ("ready", config, iso3, [list of cached parquet paths])
# ("dl_error", config, iso3, exception)
# ("done", None, None, None)  -- downloader finished all work

def download_thread_func(
    jobs: list[tuple[str, str]],  # (config, iso3)
    skip_existing: bool,
    ready_queue: queue.Queue,
):
    """Continuously download parquets for all configs into cache.

    For each config, fetches parquet URLs, downloads all files, then
    puts a "ready" message on the queue so the processor can handle it.
    The downloader never waits for the processor -- it keeps going.
    """
    for config, iso3 in jobs:
        out_name = f"{iso3}_glot500_{config.split('_')[1].lower()}.csv"
        out_path = os.path.join(RAW_DIR, out_name)

        if skip_existing and os.path.exists(out_path):
            ready_queue.put(("skip", config, iso3, None))
            continue

        try:
            parquet_urls = get_parquet_urls(config)
        except Exception as e:
            ready_queue.put(("dl_error", config, iso3, e))
            continue

        if not parquet_urls:
            ready_queue.put(("dl_error", config, iso3, ValueError("no parquet files")))
            continue

        cache_dir = os.path.join(PARQUET_CACHE_DIR, config)
        os.makedirs(cache_dir, exist_ok=True)

        cached_paths = []
        failed = False
        for i, purl in enumerate(parquet_urls):
            cached_path = os.path.join(cache_dir, f"{i}.parquet")
            label = f"{config} parquet {i+1}/{len(parquet_urls)}"
            try:
                ensure_parquet_cached(purl, cached_path, label)
                cached_paths.append(cached_path)
            except Exception as e:
                ready_queue.put(("dl_error", config, iso3, e))
                failed = True
                break

        if not failed:
            ready_queue.put(("ready", config, iso3, cached_paths))

    ready_queue.put(("done", None, None, None))


def process_ready_config(
    config: str, iso3: str, cached_paths: list[str], clean_cache: bool
) -> bool:
    """Tokenize cached parquets into a frequency CSV.

    Reads one parquet at a time to keep memory low -- only the Counter
    and one table are in memory simultaneously.
    """
    out_name = f"{iso3}_glot500_{config.split('_')[1].lower()}.csv"
    out_path = os.path.join(RAW_DIR, out_name)

    counter: Counter[str] = Counter()

    for i, path in enumerate(cached_paths, 1):
        print(f"  [proc] Tokenizing parquet {i}/{len(cached_paths)}...")
        table = pq.read_table(path, columns=["text"])
        texts = table.column("text").to_pylist()
        del table  # free memory before counting
        for text in texts:
            if text:
                counter.update(text.split())
        del texts  # free memory

    if not counter:
        print(f"  [proc] No words found.")
        return False

    freqs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    total_tokens = sum(f for _, f in freqs)
    del counter  # free memory before writing

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        writer.writerows(freqs)

    print(f"  [proc] Saved {len(freqs):,} unique words, {total_tokens:,} total tokens -> {out_name}")
    del freqs

    if clean_cache:
        cache_dir = os.path.join(PARQUET_CACHE_DIR, config)
        shutil.rmtree(cache_dir, ignore_errors=True)

    return True


def main():
    parser = argparse.ArgumentParser(description="Download Glot500 data as frequency lists")
    parser.add_argument(
        "--langs", nargs="*", default=None,
        help="ISO 639-3 codes to download (downloads all matching Glot500 configs). "
             "If omitted, downloads all under-threshold languages."
    )
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="Specific Glot500 config names (e.g. alt_Cyrl chv_Cyrl). Overrides --langs."
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip configs that already have a file in data/0_raw/"
    )
    parser.add_argument(
        "--list-configs", action="store_true",
        help="List all available Glot500 configs and exit."
    )
    parser.add_argument(
        "--under-threshold", type=str, default=None,
        help="Path to languages_under_threshold.csv to download only those languages."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all available Glot500 configs."
    )
    parser.add_argument(
        "--clean-cache", action="store_true",
        help="Delete cached parquet files after successful CSV generation."
    )
    args = parser.parse_args()

    if args.list_configs:
        configs = get_all_configs()
        print(f"Available configs ({len(configs)}):")
        for c in configs:
            print(f"  {c}")
        return

    all_configs = get_all_configs()
    config_by_lang: dict[str, list[str]] = {}
    for c in all_configs:
        lang = c.split("_")[0]
        config_by_lang.setdefault(lang, []).append(c)

    # Determine which configs to download
    if args.configs:
        to_download = args.configs
        # Validate
        invalid = [c for c in to_download if c not in all_configs]
        if invalid:
            print(f"Unknown configs: {', '.join(invalid)}")
            sys.exit(1)
    elif args.all:
        to_download = list(all_configs)
        print(f"Downloading all {len(to_download)} configs")
    elif args.langs:
        to_download = []
        for lang in args.langs:
            iso3 = GLOT_TO_ISO3_OVERRIDES.get(lang, lang)
            if iso3 in config_by_lang:
                to_download.extend(config_by_lang[iso3])
            else:
                print(f"Warning: no Glot500 config for {lang}")
    elif args.under_threshold:
        # Read under-threshold file and match
        to_download = []
        with open(args.under_threshold, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = row["code"]
                if code and code in config_by_lang:
                    to_download.extend(config_by_lang[code])
        print(f"Found {len(to_download)} configs for under-threshold languages")
    else:
        # Default: use under-threshold file if it exists
        ut_path = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "3_results", "languages_under_threshold.csv")
        if os.path.exists(ut_path):
            to_download = []
            with open(ut_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    code = row["code"]
                    if code and code in config_by_lang:
                        to_download.extend(config_by_lang[code])
            print(f"Found {len(to_download)} configs for under-threshold languages")
        else:
            print("No --langs, --configs, or under-threshold file specified.")
            print("Use --list-configs to see available options.")
            sys.exit(1)

    print(f"Will download {len(to_download)} config(s)")
    print(f"Output directory: {RAW_DIR}")
    print(f"Parquet cache:   {PARQUET_CACHE_DIR}")
    print()

    # Build (config, iso3) job list
    jobs = []
    for config in sorted(to_download):
        iso3_prefix = config.split("_")[0]
        iso3 = GLOT_TO_ISO3_OVERRIDES.get(iso3_prefix, iso3_prefix)
        jobs.append((config, iso3))

    # Queue: download thread -> main thread (processor)
    # Unbounded so the downloader never blocks
    ready_q: queue.Queue = queue.Queue()

    dl = threading.Thread(
        target=download_thread_func,
        args=(jobs, args.skip_existing, ready_q),
        daemon=True,
    )
    dl.start()

    success = 0
    skipped = 0
    failed = 0
    total = len(jobs)
    idx = 0

    while True:
        msg_type, config, iso3, payload = ready_q.get()

        if msg_type == "done":
            break

        idx += 1

        if msg_type == "skip":
            print(f"[{idx}/{total}] {config} (-> {iso3}) -- already exists, skipping")
            skipped += 1
            continue

        if msg_type == "dl_error":
            print(f"[{idx}/{total}] {config} (-> {iso3}) -- DOWNLOAD FAILED: {payload}")
            failed += 1
            continue

        # msg_type == "ready"
        cached_paths = payload
        print(f"[{idx}/{total}] {config} (-> {iso3}) -- processing {len(cached_paths)} parquet(s)")
        try:
            ok = process_ready_config(config, iso3, cached_paths, clean_cache=args.clean_cache)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [proc] FAILED: {e}")
            failed += 1

    dl.join()
    print()
    print(f"Done: {success} downloaded, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
