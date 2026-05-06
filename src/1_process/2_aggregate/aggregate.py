"""
Aggregate raw frequency CSVs by language code.

    python src/1_process/2_aggregate/aggregate.py
    python src/1_process/2_aggregate/aggregate.py ces eng
    python src/1_process/2_aggregate/aggregate.py --repair
"""

import argparse
import csv
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import (  
    FALLBACK_SCRIPT, STRIP_CHARS,
    clean_word, clean_freq_dict, read_freq_csv, write_freq_csv,
    load_ignore_set, load_script_codes, load_lang_scripts,
    output_name, group_files, memory_ok, wait_for_memory,
)

RAW_DIR = Path("data/0_raw")
AGGREGATED_DIR = Path("data/1_aggregated")
REPORT_DIR = Path("results/1_process")
SCRIPTS_CSV = Path("metadata/scripts.csv")
LANGUAGE_OVERVIEW_CSV = Path("results/1_process/2_aggregate/language_overview.csv")


def _read_and_clean(path, collect_report):
    cleaned: dict[str, int] = {}
    raw_types: set[str] | None = set() if collect_report else None
    tokens_raw = tokens_clean = total_punc = 0

    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for line in reader:
            if line and line[0].strip().lower() in ("item", "word"):
                break
        for row in reader:
            if len(row) < 2:
                continue
            word = row[0]
            try:
                freq = int(row[1])
            except (ValueError, IndexError):
                continue
            tokens_raw += freq
            if raw_types is not None:
                raw_types.add(word)
            stripped = word.strip(STRIP_CHARS)
            if not stripped:
                continue
            if collect_report:
                total_punc += freq * (len(word) - len(stripped))
            key = clean_word(word)
            if key is None:
                continue
            tokens_clean += freq
            cleaned[key] = cleaned.get(key, 0) + freq

    if not collect_report:
        return cleaned, None
    return cleaned, {
        "tokens_raw": tokens_raw,
        "tokens_clean": tokens_clean,
        "types": len(cleaned),
        "ignored": len(raw_types) - len(cleaned),
        "avg_punc": round(total_punc / tokens_raw if tokens_raw else 0, 4),
    }


def _aggregate_lang(lang, files, out_dir, script_code, collect_report):
    merged: dict[str, int] = {}
    reports = []
    for path in files:
        wait_for_memory(f"{lang}/{path.name}")
        cleaned, stats = _read_and_clean(path, collect_report)
        new = sum(1 for w in cleaned if w not in merged)
        for w, f in cleaned.items():
            merged[w] = merged.get(w, 0) + f
        if stats:
            reports.append([lang, path.name, stats["tokens_raw"],
                            stats["tokens_clean"], stats["types"],
                            stats["ignored"], stats["avg_punc"], new])
    wait_for_memory(f"writing {lang}")
    write_freq_csv(out_dir / output_name(lang, script_code), merged)
    return reports


def _run_parallel(work_items, progress):
    items = sorted(work_items, key=lambda w: w["input_bytes"], reverse=True)
    big, small = items[:len(items) // 2], items[len(items) // 2:]
    results = []

    def do(item):
        if not memory_ok():
            wait_for_memory(item["lang"])
        reps = _aggregate_lang(item["lang"], item["files"], item["out_dir"],
                               item["script_code"], item["collect_report"])
        progress.update(1)
        progress.set_postfix_str(item["lang"])
        return reps

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(do, it): it for it in big + small}
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for f in done:
                results.extend(f.result())
                del futures[f]
    return results


def _repair(agg_dir, langs):
    files = sorted(agg_dir.glob("*.csv"))
    if langs:
        sel = {l.lower() for l in langs}
        files = [f for f in files if f.stem.split("_")[0] in sel]
    if not files:
        print("Nothing to repair.")
        return
    total = 0
    for path in tqdm(files, desc="Repairing", unit="file", file=sys.stdout):
        freq = read_freq_csv(path)
        before = len(freq)
        freq = clean_freq_dict(freq)
        total += before - len(freq)
        write_freq_csv(path, freq)
    print(f"Done. {total:,} words cleaned.")


def _write_report(rows, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "aggregation_report.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lng_shortcut", "file_name", "total_tokens",
                     "total_tokens_after_ignoring", "total_types(rows)",
                     "ignored_word_count", "avg_deleted_punc_per_token",
                     "new_types_from_previous_file"])
        w.writerows(rows)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Aggregate raw frequency CSVs.")
    p.add_argument("langs", nargs="*")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=AGGREGATED_DIR)
    p.add_argument("--ignore", type=Path, default=None)
    p.add_argument("--scripts-csv", type=Path, default=SCRIPTS_CSV)
    p.add_argument("--lang-overview", type=Path, default=LANGUAGE_OVERVIEW_CSV)
    p.add_argument("--repair", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-report", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.repair:
        _repair(args.out_dir, args.langs)
        return

    ignored = load_ignore_set(args.ignore) if args.ignore and args.ignore.exists() else set()

    groups = group_files(args.raw_dir)
    if ignored:
        groups = {k: [f for f in v if f.name not in ignored] for k, v in groups.items()}
        groups = {k: v for k, v in groups.items() if v}
    if not groups:
        print(f"No CSV files in {args.raw_dir}")
        sys.exit(1)

    lang_scripts: dict[str, str] = {}
    if args.scripts_csv.exists() and args.lang_overview.exists():
        lang_scripts = load_lang_scripts(args.lang_overview,
                                         load_script_codes(args.scripts_csv))

    if args.langs:
        sel = {l.lower() for l in args.langs}
        groups = {k: v for k, v in groups.items() if k in sel}
    if args.skip_existing:
        existing = {p.name for p in args.out_dir.glob("*.csv")}
        groups = {k: v for k, v in groups.items()
                  if output_name(k, lang_scripts.get(k, FALLBACK_SCRIPT)) not in existing}
    if not groups:
        print("Nothing to process.")
        sys.exit(1)

    collect_report = not args.no_report
    print(f"Aggregating {len(groups)} language(s) ({sum(len(f) for f in groups.values())} files)")

    work_items = [{"lang": lang, "files": groups[lang], "out_dir": args.out_dir,
                   "script_code": lang_scripts.get(lang, FALLBACK_SCRIPT),
                   "collect_report": collect_report,
                   "input_bytes": sum(f.stat().st_size for f in groups[lang])}
                  for lang in sorted(groups)]

    progress = tqdm(total=len(work_items), desc="Aggregating", unit="lang", file=sys.stdout)
    if len(work_items) == 1:
        item = work_items[0]
        reports = _aggregate_lang(item["lang"], item["files"], item["out_dir"],
                                  item["script_code"], item["collect_report"])
        progress.update(1)
    else:
        reports = _run_parallel(work_items, progress)
    progress.close()

    if collect_report:
        _write_report(reports, REPORT_DIR)


if __name__ == "__main__":
    main()
