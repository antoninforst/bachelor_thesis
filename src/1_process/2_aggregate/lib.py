"""Shared helpers for the aggregation pipeline."""

import csv
import ctypes
import os
import re
import sys
import time
from pathlib import Path

csv.field_size_limit(10 * 1024 * 1024)

# ── word cleaning ─────────────────────────────────────────────────────────

_HAS_LETTER = re.compile(r"[^\W\d_]").search
STRIP_CHARS = '.,?!":()…\u201c \u201d \u201e \u201f \u2018 \u2019 \xab \xbb'
FALLBACK_SCRIPT = "xxxx"


def clean_word(word):
    stripped = word.strip(STRIP_CHARS)
    if not stripped:
        return None
    key = stripped.lower()
    return key if _HAS_LETTER(key) else None


def clean_freq_dict(freq_dict, punc_stats=None):
    cleaned: dict[str, int] = {}
    total_punc = 0
    for word, freq in freq_dict.items():
        stripped = word.strip(STRIP_CHARS)
        if not stripped:
            continue
        if punc_stats is not None:
            total_punc += freq * (len(word) - len(stripped))
        key = stripped.lower()
        if not _HAS_LETTER(key):
            continue
        cleaned[key] = cleaned.get(key, 0) + freq
    if punc_stats is not None:
        punc_stats[0] = total_punc
    return cleaned


# ── CSV I/O ───────────────────────────────────────────────────────────────

def read_freq_csv(path):
    freq: dict[str, int] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for line in reader:
            if line and line[0].strip().lower() in ("item", "word"):
                break
        for row in reader:
            if len(row) < 2:
                continue
            try:
                freq[row[0]] = freq.get(row[0], 0) + int(row[1])
            except (ValueError, IndexError):
                continue
    return freq


def write_freq_csv(path, freq):
    path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["word", "frequency"])
        w.writerows(items)


def load_ignore_set(path):
    ignored: set[str] = set()
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ignored.add(row["filename"])
    return ignored


# ── script codes ──────────────────────────────────────────────────────────

def load_script_codes(scripts_csv):
    mapping: dict[str, str] = {}
    with open(scripts_csv, encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) == 1 and ";" in row[0]:
                row = row[0].split(";")
            if len(row) >= 2 and row[0].strip():
                mapping[row[0].strip()] = row[1].strip()
    return mapping


def load_lang_scripts(overview_csv, script_codes):
    mapping: dict[str, str] = {}
    with open(overview_csv, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            lang = row["used_shortcut"].strip()
            script = row["primary_script"].strip()
            mapping[lang] = script_codes.get(script, FALLBACK_SCRIPT)
    return mapping


def output_name(lang, script_code):
    if script_code == FALLBACK_SCRIPT:
        return f"{lang}.csv"
    return f"{lang}_{script_code}.csv"


def group_files(raw_dir):
    groups: dict[str, list[Path]] = {}
    for p in sorted(raw_dir.glob("*.csv")):
        groups.setdefault(p.name[:3], []).append(p)
    return groups


# ── memory ────────────────────────────────────────────────────────────────

MAX_MEMORY_PCT = 90.0
_CHECK_INTERVAL = 5.0


def _used_memory_pct():
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return 100.0 * (status.ullTotalPhys - status.ullAvailPhys) / status.ullTotalPhys
        return None
    if hasattr(os, "sysconf"):
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            total = os.sysconf("SC_PHYS_PAGES")
            avail = os.sysconf("SC_AVPHYS_PAGES")
        except (OSError, ValueError):
            return None
        return 100.0 * (total - avail) / total if total > 0 else None
    return None


def memory_ok():
    pct = _used_memory_pct()
    return pct is None or pct < MAX_MEMORY_PCT


def wait_for_memory(label=""):
    if MAX_MEMORY_PCT >= 100:
        return
    warned = False
    while True:
        pct = _used_memory_pct()
        if pct is None or pct < MAX_MEMORY_PCT:
            return
        if not warned:
            print(f"Memory {pct:.0f}% — pausing {label}", flush=True)
            warned = True
        time.sleep(_CHECK_INTERVAL)
