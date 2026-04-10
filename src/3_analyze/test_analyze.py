"""
Hand-verified tests for every metric in analyze.py.

Test CSV (test_file.csv):
    word,frequency
    pref1 @root1 suff1,10
    pref2 @root2 inter1 @root3 suff2 suff3,5
    @root1 suff2,3
    @single,1

break_word results:
    Row 1 → whole="pref1root1suff1",   morphs=["pref1","root1","suff1"]
    Row 2 → whole="pref2root2inter1root3suff2suff3", morphs=["pref2","root2","inter1","root3","suff2","suff3"]
    Row 3 → whole="root1suff2",         morphs=["root1","suff2"]
    Row 4 → whole="single",             morphs=["single"]

whole_words Frequency.data (key → (total_freq, occurrence_count)):
    "pref1root1suff1"                → (10, 1)
    "pref2root2inter1root3suff2suff3"→ (5, 1)
    "root1suff2"                     → (3, 1)
    "single"                         → (1, 1)

morphs Frequency.data:
    "pref1"  → (10, 1)
    "root1"  → (13, 2)   # 10 from row1 + 3 from row3
    "suff1"  → (10, 1)
    "pref2"  → (5, 1)
    "root2"  → (5, 1)
    "inter1" → (5, 1)
    "root3"  → (5, 1)
    "suff2"  → (8, 2)    # 5 from row2 + 3 from row3
    "suff3"  → (5, 1)
    "single" → (1, 1)

MorphStats per row:
    Row 1: morphemes=["pref1","@root1","suff1"], roots=[1]→ roots=1, pref=1, suff=1, inter=0
    Row 2: morphemes=["pref2","@root2","inter1","@root3","suff2","suff3"], roots=[1,3]→ roots=2, pref=1, suff=2, inter=1
    Row 3: morphemes=["@root1","suff2"], roots=[0]→ roots=1, pref=0, suff=1, inter=0
    Row 4: morphemes=["@single"], roots=[0]→ roots=1, pref=0, suff=0, inter=0
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import (
    read_csv_rows, break_word, Frequency, MorphStats, METRICS,
    compute_metrics_for_file,
)

TEST_CSV = Path(__file__).resolve().parent / "test_file.csv"
PASS = 0
FAIL = 0


def check(name: str, got, expected, tol=1e-9):
    global PASS, FAIL
    ok = abs(got - expected) < tol if isinstance(expected, float) else got == expected
    if ok:
        PASS += 1
        print(f"  OK   {name}: {got}")
    else:
        FAIL += 1
        print(f"  FAIL {name}: got {got!r}, expected {expected!r}")


# ── 1. break_word ─────────────────────────────────────────

print("=== break_word ===")
w, s = break_word("pref1 @root1 suff1")
check("row1 whole", w, "pref1root1suff1")
check("row1 morphs", s, ["pref1", "root1", "suff1"])

w, s = break_word("pref2 @root2 inter1 @root3 suff2 suff3")
check("row2 whole", w, "pref2root2inter1root3suff2suff3")
check("row2 morphs", s, ["pref2", "root2", "inter1", "root3", "suff2", "suff3"])

w, s = break_word("@root1 suff2")
check("row3 whole", w, "root1suff2")
check("row3 morphs", s, ["root1", "suff2"])

w, s = break_word("@single")
check("row4 whole", w, "single")
check("row4 morphs", s, ["single"])


# ── 2. Build Frequency objects exactly as compute_metrics_for_file does ──

print("\n=== Building Frequency ===")
rows = read_csv_rows(TEST_CSV)
check("row_count", len(rows), 4)

whole_words = Frequency()
morphs = Frequency()
morph_stats = MorphStats()
for row in rows:
    word, segmentation = break_word(row.word)
    whole_words.add(word, row.frequency)
    for morph in segmentation:
        morphs.add(morph, row.frequency)
    morph_stats.add(row)


# ── 3. whole_words Frequency internals ────────────────────

print("\n=== whole_words Frequency ===")
check("ww unique_count", whole_words.get_unique_count(), 4)
check("ww total_count", whole_words.get_total_count(), 19)
# All 4 words appear in exactly 1 row → c=1 → all are "hapax" (occurrence-based)
check("ww hepax_count", whole_words.get_hepax_count(), 4)
# Only "single" has frequency=1 → fhepax=1
check("ww fhepax_count", whole_words.get_freq_hepax_count(), 1)


# ── 4. morphs Frequency internals ────────────────────────

print("\n=== morphs Frequency ===")
check("m unique_count", morphs.get_unique_count(), 10)
check("m total_count", morphs.get_total_count(), 67)
# "root1" and "suff2" each appear in 2 rows (c=2), rest in 1 row → 10-2=8 hapax
check("m hepax_count", morphs.get_hepax_count(), 8)
# Only "single" has total frequency=1
check("m fhepax_count", morphs.get_freq_hepax_count(), 1)

# Verify individual entries
check("m root1", morphs.get("root1"), (13, 2))
check("m suff2", morphs.get("suff2"), (8, 2))
check("m single", morphs.get("single"), (1, 1))
check("m pref1", morphs.get("pref1"), (10, 1))


# ── 5. word-level metrics ────────────────────────────────

print("\n=== Word-level metrics ===")

# count = unique words
check("word_count", METRICS["count"](whole_words), 4)

# total_frequency = 10+5+3+1 = 19
check("word_total_frequency", METRICS["total_frequency"](whole_words), 19)

# ttr = unique/total = 4/19
check("word_ttr", METRICS["ttr"](whole_words), 4 / 19)

# hapax = 4 (all words appear once in data)
check("word_hapax_count", METRICS["hapax_count"](whole_words), 4)
check("word_hapax_ratio", METRICS["hapax_ratio"](whole_words), 1.0)

# freq_hapax = 1 ("single" with f=1)
check("word_freq_hapax_count", METRICS["freq_hapax_count"](whole_words), 1)
check("word_freq_hapax_ratio", METRICS["freq_hapax_ratio"](whole_words), 0.25)

# avg_length = (15 + 31 + 10 + 6) / 4 = 62/4
check("word_avg_length", METRICS["avg_length"](whole_words), 62 / 4)

# avg_length_weighted = (15*10 + 31*5 + 10*3 + 6*1) / 19 = 341/19
check("word_avg_length_weighted", METRICS["avg_length_weighted"](whole_words), 341 / 19)

# entropy  (hand-computed with exact formula)
ww_freqs = [10, 5, 3, 1]
ww_total = 19
ww_entropy = -sum((f / ww_total) * math.log2(f / ww_total) for f in ww_freqs)
check("word_entropy", METRICS["frequency_entropy"](whole_words), ww_entropy)

# perplexity = 2^entropy
check("word_perplexity", METRICS["frequency_perplexity"](whole_words), 2 ** ww_entropy)


# ── 6. morph-level metrics ───────────────────────────────

print("\n=== Morph-level metrics ===")

check("morph_count", METRICS["count"](morphs), 10)
check("morph_total_frequency", METRICS["total_frequency"](morphs), 67)
check("morph_ttr", METRICS["ttr"](morphs), 10 / 67)
check("morph_hapax_count", METRICS["hapax_count"](morphs), 8)
check("morph_hapax_ratio", METRICS["hapax_ratio"](morphs), 8 / 10)
check("morph_freq_hapax_count", METRICS["freq_hapax_count"](morphs), 1)
check("morph_freq_hapax_ratio", METRICS["freq_hapax_ratio"](morphs), 1 / 10)

# avg_length: morph lengths are 5,5,5,5,5,6,5,5,5,6 → sum=52
check("morph_avg_length", METRICS["avg_length"](morphs), 52 / 10)

# avg_length_weighted: 5*10+5*13+5*10+5*5+5*5+6*5+5*5+5*8+5*5+6*1 = 341
check("morph_avg_length_weighted", METRICS["avg_length_weighted"](morphs), 341 / 67)

m_freqs = [10, 13, 10, 5, 5, 5, 5, 8, 5, 1]
m_total = 67
m_entropy = -sum((f / m_total) * math.log2(f / m_total) for f in m_freqs)
check("morph_entropy", METRICS["frequency_entropy"](morphs), m_entropy)
check("morph_perplexity", METRICS["frequency_perplexity"](morphs), 2 ** m_entropy)


# ── 7. MorphStats ────────────────────────────────────────

print("\n=== MorphStats ===")
ms = morph_stats.get_metrics()

# Totals: roots=5, prefixes=2, suffixes=4, interfixes=1, word_count=4
check("avg_root_count", ms["avg_root_count"], 5 / 4)
check("avg_prefix_count", ms["avg_prefix_count"], 2 / 4)
check("avg_suffix_count", ms["avg_suffix_count"], 4 / 4)
check("avg_interfix_count", ms["avg_interfix_count"], 1 / 4)

# Weighted totals: roots=1*10+2*5+1*3+1*1=24, pref=1*10+1*5+0*3+0*1=15,
#                  suff=1*10+2*5+1*3+0*1=23, inter=0*10+1*5+0*3+0*1=5, total_freq=19
check("avg_root_count_weighted", ms["avg_root_count_weighted"], 24 / 19)
check("avg_prefix_count_weighted", ms["avg_prefix_count_weighted"], 15 / 19)
check("avg_suffix_count_weighted", ms["avg_suffix_count_weighted"], 23 / 19)
check("avg_interfix_count_weighted", ms["avg_interfix_count_weighted"], 5 / 19)


# ── 8. Integration: compute_metrics_for_file ─────────────

print("\n=== compute_metrics_for_file integration ===")
result = compute_metrics_for_file(TEST_CSV, {}, {})
check("int_language", result["language"], "test_file")
check("int_word_count", result["word_count"], 4)
check("int_morph_count", result["morph_count"], 10)
check("int_word_total_frequency", result["word_total_frequency"], 19)
check("int_morph_total_frequency", result["morph_total_frequency"], 67)
check("int_avg_root_count", result["avg_root_count"], 5 / 4)


# ── Summary ──────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"PASSED: {PASS}  FAILED: {FAIL}")
if FAIL:
    sys.exit(1)
else:
    print("All tests passed!")
