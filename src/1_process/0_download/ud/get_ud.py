from __future__ import annotations

import csv
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from tqdm import tqdm


def is_ud_word_id(token_id: str) -> bool:
    """True for regular word IDs (1, 2, 3...), skips ranges (1-2) and empty nodes (5.1)."""
    return token_id.isdigit()


def count_conllu_stream(stream, field: str = "FORM") -> Counter[str]:
    """Count word frequencies from a CoNLL-U byte stream (FORM or LEMMA column)."""
    field_index = {"FORM": 1, "LEMMA": 2}[field]
    counter: Counter[str] = Counter()

    for raw_line in stream:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n")

        if not line or line.startswith("#"):
            continue

        cols = line.split("\t")
        if len(cols) != 10:
            continue

        token_id = cols[0]
        if not is_ud_word_id(token_id):
            continue

        value = cols[field_index].strip().lower()
        if value and value != "_":
            counter[value] += 1

    return counter


def write_frequency_csv(counter: Counter[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        for word, freq in counter.most_common():
            writer.writerow([word, freq])


def load_language_mapping(shortcuts_path: Path) -> dict[str, str]:
    """Read Leipzig shortcuts CSV -> {normalized_language_name: iso_code}."""
    mapping: dict[str, str] = {}
    with shortcuts_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang_normalized = row["language"].strip().lower().replace(" ", "_")
            mapping[lang_normalized] = row["code"].strip()
    return mapping


def load_ud_language_mapping(ud_mapping_path: Path) -> dict[str, str]:
    """Read UD-specific name overrides -> {normalized_ud_name: iso_code}."""
    mapping: dict[str, str] = {}
    with ud_mapping_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ud_name = row["ud_name"].strip().lower()
            mapping[ud_name] = row["code"].strip()
    return mapping


def parse_ud_treebank_name(name: str) -> tuple[str, str | None]:
    """'UD_Czech-CAC' -> ('Czech', 'CAC'), 'UD_Czech' -> ('Czech', None)."""
    rest = name[3:]  # strip 'UD_'
    if "-" in rest:
        lang, variant = rest.split("-", 1)
        return lang, variant
    return rest, None


def _collect_treebank_counts(
    archive_path: Path, field: str
) -> dict[str, Counter[str]]:
    """Read all .conllu files from the tar.gz and aggregate counts per treebank."""
    treebank_counters: dict[str, Counter[str]] = defaultdict(Counter)

    with tarfile.open(archive_path, "r:gz") as tar:
        conllu_members = [m for m in tar if m.isfile() and m.name.endswith(".conllu")]
        for member in tqdm(conllu_members, desc="Processing .conllu files"):
            path_parts = PurePosixPath(member.name).parts
            if len(path_parts) < 2:
                continue

            parent_dir = path_parts[-2]
            filename = path_parts[-1]
            treebank_name = parent_dir if parent_dir.startswith("UD_") else filename.replace(".conllu", "")

            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            counts = count_conllu_stream(extracted, field=field)
            treebank_counters[treebank_name].update(counts)

    return dict(treebank_counters)


def _group_by_language(
    treebank_counters: dict[str, Counter[str]]
) -> dict[str, list[tuple[str | None, Counter[str]]]]:
    """Group treebank counters by their language name."""
    lang_treebanks: dict[str, list[tuple[str | None, Counter[str]]]] = defaultdict(list)
    for tb_name, counter in treebank_counters.items():
        if not tb_name.startswith("UD_"):
            continue
        lang, variant = parse_ud_treebank_name(tb_name)
        lang_treebanks[lang].append((variant, counter))
    return dict(lang_treebanks)


def _write_all(
    lang_treebanks: dict[str, list[tuple[str | None, Counter[str]]]],
    output_dir: Path,
    lang_to_code: dict[str, str],
    ud_overrides: dict[str, str],
) -> list[str]:
    """Write frequency CSVs for each language. Returns list of unmatched languages."""
    unmatched: list[str] = []
    written = 0

    for lang, entries in sorted(lang_treebanks.items()):
        lang_key = lang.lower()
        code = ud_overrides.get(lang_key) or lang_to_code.get(lang_key)
        if code is None:
            unmatched.append(lang)
            continue

        if len(entries) == 1:
            out_file = output_dir / f"{code}-ud.csv"
            write_frequency_csv(entries[0][1], out_file)
            written += 1
        else:
            for variant, counter in entries:
                suffix = variant.lower() if variant else "default"
                out_file = output_dir / f"{code}-ud-{suffix}.csv"
                write_frequency_csv(counter, out_file)
                written += 1

    print(f"Done. Wrote {written} frequency files to {output_dir}")
    return unmatched


def process_ud_release_archive(
    archive_path: str | Path,
    output_dir: str | Path,
    shortcuts_path: str | Path,
    ud_mapping_path: str | Path,
    field: str = "FORM",
) -> None:
    """Top-level: extract UD archive, count words, write frequency CSVs."""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shortcuts_path = Path(shortcuts_path)
    ud_mapping_path = Path(ud_mapping_path)

    lang_to_code = load_language_mapping(shortcuts_path)
    ud_overrides = load_ud_language_mapping(ud_mapping_path)

    treebank_counters = _collect_treebank_counts(archive_path, field)
    lang_treebanks = _group_by_language(treebank_counters)
    unmatched = _write_all(lang_treebanks, output_dir, lang_to_code, ud_overrides)

    if unmatched:
        print(f"\nUnmatched UD languages ({len(unmatched)}):")
        for lang in sorted(unmatched):
            print(f"  {lang}")


if __name__ == "__main__":
    archive_path = Path("data/0_2_ud/ud-treebanks-v2.17.tgz")
    output_dir = Path("data/0_2_ud")
    shortcuts_path = Path("src/1_process/0_download/leipzig/lepzig_shortcuts.csv")
    ud_mapping_path = Path("src/1_process/0_download/ud/ud_language_mapping.csv")

    process_ud_release_archive(
        archive_path=archive_path,
        output_dir=output_dir,
        shortcuts_path=shortcuts_path,
        ud_mapping_path=ud_mapping_path,
        field="FORM",
    )