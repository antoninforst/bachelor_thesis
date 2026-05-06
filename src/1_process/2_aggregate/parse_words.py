"""
Segment compound words in aggregated frequency files for languages
that lack whitespace word boundaries (Chinese, Thai).

    python src/1_process/2_aggregate/parse_words.py
    python src/1_process/2_aggregate/parse_words.py --agg-dir data/1_aggregated
    python src/1_process/2_aggregate/parse_words.py --only-chinese
    python src/1_process/2_aggregate/parse_words.py --only-thai
"""

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import clean_freq_dict 

csv.field_size_limit(10 * 1024 * 1024)

AGG_DIR = Path("data/1_aggregated")

Segmenter = Callable[[str], list[str]]

SCRIPT_SEGMENTERS: dict[str, tuple[str, str]] = {
    "Hani": ("Chinese/jieba", "_segment_chinese"),
    "Thai": ("Thai/pythainlp", "_segment_thai"),
}


def _segment_chinese(text: str) -> list[str]:
    import jieba
    return list(jieba.cut(text))


def _segment_thai(text: str) -> list[str]:
    from pythainlp.tokenize import word_tokenize
    return word_tokenize(text, engine="newmm")


def _get_segmenter(script_code: str) -> tuple[Segmenter | None, str | None]:
    """Return (segmenter_function, label) for *script_code*, or (None, None)."""
    entry = SCRIPT_SEGMENTERS.get(script_code)
    if entry is None:
        return None, None
    label, func_name = entry
    func = globals()[func_name]
    return func, label


def _script_code_from_stem(stem: str) -> str | None:
    """Extract the 4-letter script code from a filename stem like 'zho_Hani'."""
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in SCRIPT_SEGMENTERS:
        return parts[1]
    return None


# -- parallel worker helpers ------------------------------------------------

_worker_segmenter = None


def _init_worker(script_code: str) -> None:
    """Called once per worker process to set up the segmenter."""
    global _worker_segmenter
    entry = SCRIPT_SEGMENTERS[script_code]
    _worker_segmenter = globals()[entry[1]]
    _worker_segmenter("test")  # force-load dictionary


def _process_chunk(chunk: list[tuple[str, int]]) -> tuple[dict[str, int], int]:
    """Segment a chunk of (word, count) pairs in a worker process."""
    new_freq: dict[str, int] = {}
    seg_count = 0
    for word, count in chunk:
        parts = _worker_segmenter(word)
        parts = [p for p in parts if p.strip()]
        if len(parts) <= 1:
            key = parts[0] if parts else word
            new_freq[key] = new_freq.get(key, 0) + count
        else:
            seg_count += 1
            for part in parts:
                new_freq[part] = new_freq.get(part, 0) + count
    return new_freq, seg_count


def _read_freq_file(path: Path) -> dict[str, int]:
    """Read a word,frequency CSV into a dict."""
    freq: dict[str, int] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                freq[row[0]] = freq.get(row[0], 0) + int(row[1])
            except ValueError:
                continue
    return freq


def _write_freq_file(path: Path, freq: dict[str, int]) -> None:
    """Write a frequency dict as a sorted CSV."""
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["word", "frequency"])
        writer.writerows(sorted_items)


def segment_file(
    path: Path,
    segmenter: Segmenter,
    *,
    script_code: str | None = None,
    jobs: int = 1,
) -> tuple[int, int, int]:
    """Segment words in *path* using *segmenter*. Returns (types_before, types_after, segmented_count)."""
    freq = _read_freq_file(path)
    types_before = len(freq)

    # Fast path: single-char words cannot be segmented further
    new_freq: dict[str, int] = {}
    to_segment: list[tuple[str, int]] = []
    for word, count in freq.items():
        if len(word) <= 1:
            new_freq[word] = new_freq.get(word, 0) + count
        else:
            to_segment.append((word, count))

    skipped = types_before - len(to_segment)
    print(f"  Skipped {skipped:,} single-char words, segmenting {len(to_segment):,}")

    segmented_count = 0

    if jobs > 1 and script_code and to_segment:
        # Parallel: split into small chunks for good load balancing
        chunk_size = max(1_000, len(to_segment) // (jobs * 4))
        chunks = [
            to_segment[i : i + chunk_size]
            for i in range(0, len(to_segment), chunk_size)
        ]
        done = 0
        with mp.Pool(jobs, initializer=_init_worker, initargs=(script_code,)) as pool:
            for chunk_freq, chunk_seg in pool.imap_unordered(_process_chunk, chunks):
                segmented_count += chunk_seg
                for w, c in chunk_freq.items():
                    new_freq[w] = new_freq.get(w, 0) + c
                done += 1
                print(f"\r  Chunks: {done}/{len(chunks)}", end="", flush=True)
        print()
    else:
        total = len(to_segment)
        for i, (word, count) in enumerate(to_segment, 1):
            if i % 50_000 == 0 or i == total:
                print(f"\r  Processing: {i:,}/{total:,}", end="", flush=True)
            parts = segmenter(word)
            parts = [p for p in parts if p.strip()]
            if len(parts) <= 1:
                key = parts[0] if parts else word
                new_freq[key] = new_freq.get(key, 0) + count
            else:
                segmented_count += 1
                for part in parts:
                    new_freq[part] = new_freq.get(part, 0) + count
        if to_segment:
            print()

    new_freq = clean_freq_dict(new_freq)
    types_after = len(new_freq)

    _write_freq_file(path, new_freq)
    return types_before, types_after, segmented_count


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment compound words in aggregated frequency files."
    )
    parser.add_argument(
        "langs",
        nargs="*",
        metavar="LANG",
        help="Language codes to limit processing (default: all files with a parsable script suffix).",
    )
    parser.add_argument(
        "--agg-dir",
        type=Path,
        default=AGG_DIR,
        help=f"Directory with aggregated CSVs (default: {AGG_DIR}).",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel workers for segmentation (default: 1).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only-chinese", "-o1", action="store_true",
        help="Only process Chinese (cmn) files.",
    )
    group.add_argument(
        "--only-thai", "-o2", action="store_true",
        help="Only process Thai (tha) files.",
    )
    group.add_argument(
        "--exclude-large", "-o3", action="store_true",
        help="Process all segmentable files except cmn and tha.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    lang_filter = {lang.lower() for lang in args.langs} if args.langs else None
    if args.only_chinese:
        lang_filter = {"cmn"}
    elif args.only_thai:
        lang_filter = {"tha"}
    files: list[tuple[str, str, Path]] = []

    for path in sorted(args.agg_dir.glob("*.csv")):
        script_code = _script_code_from_stem(path.stem)
        if script_code is None:
            continue
        lang = path.stem.rsplit("_", 1)[0]
        if lang_filter and lang not in lang_filter:
            continue
        if args.exclude_large and lang in ("cmn", "tha"):
            continue
        files.append((lang, script_code, path))

    if not files:
        print("No parsable aggregated files found.")
        sys.exit(1)

    print(f"Segmenting {len(files)} file(s)")
    print("=" * 50)

    for lang, script_code, path in files:
        segmenter, group = _get_segmenter(script_code)
        print(f"\n[{path.stem}] ({group})")
        before, after, seg_count = segment_file(
            path, segmenter, script_code=script_code, jobs=args.jobs,
        )
        print(f"  Types: {before:,} -> {after:,}")
        print(f"  Entries segmented: {seg_count:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
