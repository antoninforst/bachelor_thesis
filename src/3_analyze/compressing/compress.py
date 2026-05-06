"""
Compression analysis script.

For each language file in data/2_annotated/:
1. Scan ALL languages to collect every unique Unicode character.
2. Assign each character a single random 2-byte code (shared across all languages).
3. For each language, generate 100,000 tokens randomly according to word frequencies.
4. Encode the text using those 2-byte codes (space = 0x0000).
5. Save the binary file, then zip it.
6. Compare sizes before and after zipping.
7. Repeat NUM_RUNS times, average results.
"""

from __future__ import annotations

import csv
import io
import json
import multiprocessing
import os
import random
import struct
import zipfile
from pathlib import Path

from tqdm import tqdm

NUM_TOKENS = 20_000
NUM_RUNS = 6

AGGREGATED_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "2_annotated"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "3_1_compress"
RESULTS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "results" / "3_analyze" / "compress1M.csv"


def load_frequency_list(path: Path) -> list[tuple[str, int]]:
    """Load word,frequency CSV and return list of (word, freq) tuples."""
    words = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                word = row[0]
                try:
                    freq = int(row[1])
                except ValueError:
                    continue
                words.append((word, freq))
    return words


def generate_tokens(freq_list: list[tuple[str, int]], n: int, rng: random.Random) -> list[str]:
    """Generate n tokens randomly weighted by frequency."""
    words = [w for w, _ in freq_list]
    weights = [f for _, f in freq_list]
    return rng.choices(words, weights=weights, k=n)


def randomize_tokens(tokens: list[str], rng: random.Random) -> list[str]:
    """Shuffle characters within each token, preserving word lengths."""
    result = []
    for token in tokens:
        chars = list(token)
        rng.shuffle(chars)
        result.append("".join(chars))
    return result


CACHE_PATH = Path(__file__).resolve().parent / "unique_chars.json"


def get_unique_chars(csv_files: list[Path]) -> list[str]:
    """Get sorted unique characters, using cache if available."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            unique_chars = json.load(f)
        print(f"Loaded {len(unique_chars)} unique characters from cache")
        return unique_chars

    all_chars: set[str] = set()
    for path in csv_files:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    all_chars.update(row[0])
    all_chars.discard(" ")
    unique_chars = sorted(all_chars)

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(unique_chars, f, ensure_ascii=False)
    print(f"Scanned {len(unique_chars)} unique characters (cached to {CACHE_PATH.name})")
    return unique_chars


def build_global_char_mapping(csv_files: list[Path], rng: random.Random) -> dict[str, int]:
    """
    Get unique characters (from cache or scan), then
    assign each a random unique 2-byte code (shared for all languages).
    Code 0x0000 is reserved for the space delimiter.
    """
    unique_chars = get_unique_chars(csv_files)
    print(f"Global unique characters: {len(unique_chars)} (fits 2B: {len(unique_chars) <= 65535})")
    codes = rng.sample(range(1, 65536), len(unique_chars))
    return dict(zip(unique_chars, codes))


def encode_text(tokens: list[str], char_map: dict[str, int]) -> bytes:
    """
    Encode token list into binary:
    - Each character -> 2 bytes (big-endian unsigned short)
    - Space between words -> 0x0000 (2 null bytes)
    """
    parts = []
    for i, token in enumerate(tokens):
        if i > 0:
            parts.append(struct.pack(">H", 0))
        for ch in token:
            code = char_map.get(ch)
            if code is not None:
                parts.append(struct.pack(">H", code))
    return b"".join(parts)


def zip_in_memory(data: bytes, arcname: str) -> int:
    """Compress data in memory and return the zip size."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(arcname, data)
    return buf.tell()


def process_one_language(args: tuple) -> dict | None:
    """Process a single language for one run. Designed for multiprocessing."""
    lang_name, freq_list, char_map, run_idx = args
    rng = random.Random(42 + run_idx * 10000 + hash(lang_name) % 10000)

    if not freq_list:
        return None

    # Generate tokens
    tokens = generate_tokens(freq_list, NUM_TOKENS, rng)

    # Count unique chars
    lang_chars = set()
    for t in tokens:
        lang_chars.update(t)
    lang_chars.discard(" ")

    # Encode
    binary_data = encode_text(tokens, char_map)
    bin_size = len(binary_data)

    # Zip in memory
    zip_size = zip_in_memory(binary_data, f"{lang_name}.bin")

    # Randomize characters within words, encode & zip
    rand_tokens = randomize_tokens(tokens, rng)
    rand_binary = encode_text(rand_tokens, char_map)
    rand_zip_size = zip_in_memory(rand_binary, f"{lang_name}_rand.bin")

    # Ratio
    ratio = zip_size / rand_zip_size if rand_zip_size > 0 else 0.0
    reduction_pct = (1 - ratio) * 100

    return {
        "language": lang_name,
        "run_idx": run_idx,
        "unique_chars": len(lang_chars),
        "tokens": NUM_TOKENS,
        "bin_size": bin_size,
        "zip_size": zip_size,
        "rand_zip_size": rand_zip_size,
        "ratio": ratio,
        "reduction_pct": reduction_pct,
    }


def run_experiments_parallel(csv_files: list[Path], freq_lists: dict[str, list],
                             char_map: dict[str, int]) -> list[list[dict]]:
    """Run all experiments using multiprocessing across languages."""
    # Build task list: (lang_name, freq_list, char_map, run_idx) for each lang × run
    tasks = []
    for run_idx in range(NUM_RUNS):
        for csv_path in csv_files:
            lang_name = csv_path.stem
            tasks.append((lang_name, freq_lists[lang_name], char_map, run_idx))

    num_workers = min(multiprocessing.cpu_count(), len(tasks))
    print(f"Processing {len(tasks)} tasks on {num_workers} cores...")

    results_by_run: list[list[dict]] = [[] for _ in range(NUM_RUNS)]
    with multiprocessing.Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_one_language, tasks),
                           total=len(tasks), desc="Compressing", unit="lang"):
            if result:
                results_by_run[result["run_idx"]].append(result)

    return results_by_run


def save_example_files(csv_files: list[Path], freq_lists: dict[str, list],
                       char_map: dict[str, int], output_dir: Path):
    """Save one example set of files (from seed 42) to disk."""
    rng = random.Random(42)
    for csv_path in tqdm(csv_files, desc="Saving files", unit="lang"):
        lang_name = csv_path.stem
        freq_list = freq_lists[lang_name]
        if not freq_list:
            continue

        tokens = generate_tokens(freq_list, NUM_TOKENS, rng)

        # Save readable text
        txt_path = output_dir / f"{lang_name}.csv"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(" ".join(tokens))

        # Encode & save binary
        binary_data = encode_text(tokens, char_map)
        bin_path = output_dir / f"{lang_name}.bin"
        with open(bin_path, "wb") as f:
            f.write(binary_data)

        # Zip
        zip_path = output_dir / f"{lang_name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(bin_path, arcname=f"{lang_name}.bin")

        # Randomized
        rand_tokens = randomize_tokens(tokens, rng)
        rand_binary = encode_text(rand_tokens, char_map)
        rand_bin_path = output_dir / f"{lang_name}_rand.bin"
        with open(rand_bin_path, "wb") as f:
            f.write(rand_binary)
        rand_zip_path = output_dir / f"{lang_name}_rand.zip"
        with zipfile.ZipFile(rand_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(rand_bin_path, arcname=f"{lang_name}_rand.bin")


def average_results(all_runs: list[list[dict]]) -> list[dict]:
    """Average numeric results across runs, grouped by language."""
    from collections import defaultdict
    accum: dict[str, dict] = defaultdict(lambda: {
        "unique_chars": 0.0, "tokens": 0, "bin_size": 0.0,
        "zip_size": 0.0, "rand_zip_size": 0.0, "ratio": 0.0,
        "reduction_pct": 0.0, "count": 0,
    })

    for run_results in all_runs:
        for entry in run_results:
            lang = entry["language"]
            accum[lang]["unique_chars"] += entry["unique_chars"]
            accum[lang]["tokens"] = entry["tokens"]
            accum[lang]["bin_size"] += entry["bin_size"]
            accum[lang]["zip_size"] += entry["zip_size"]
            accum[lang]["rand_zip_size"] += entry["rand_zip_size"]
            accum[lang]["ratio"] += entry["ratio"]
            accum[lang]["reduction_pct"] += entry["reduction_pct"]
            accum[lang]["count"] += 1

    results = []
    for lang in sorted(accum.keys()):
        n = accum[lang]["count"]
        results.append({
            "language": lang,
            "unique_chars": round(accum[lang]["unique_chars"] / n),
            "tokens": accum[lang]["tokens"],
            "bin_size": round(accum[lang]["bin_size"] / n),
            "zip_size": round(accum[lang]["zip_size"] / n),
            "rand_zip_size": round(accum[lang]["rand_zip_size"] / n),
            "ratio": round(accum[lang]["ratio"] / n, 6),
            "reduction_pct": round(accum[lang]["reduction_pct"] / n, 2),
        })
    return results


def main():
    rng = random.Random(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(AGGREGATED_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} language files in {AGGREGATED_DIR}")

    freq_lists = {}
    for csv_path in csv_files:
        freq_lists[csv_path.stem] = load_frequency_list(csv_path)

    print("Building global character mapping...")
    char_map = build_global_char_mapping(csv_files, rng)
    print()

    all_runs = run_experiments_parallel(csv_files, freq_lists, char_map)

    results = average_results(all_runs)

    print("\nSaving example files...")
    save_example_files(csv_files, freq_lists, char_map, OUTPUT_DIR)

    if results:
        with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {RESULTS_PATH}")
        print(f"Processed {len(results)} languages (averaged over {NUM_RUNS} runs)")


if __name__ == "__main__":
    main()
