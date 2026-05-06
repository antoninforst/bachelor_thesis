"""
Generate a printable language table from total_results.csv.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "results" / "3_analyze" / "total_results.csv"


def format_family(row: pd.Series) -> str:
    """Family column: for IE/Altaic show genus with family in brackets."""
    family = row.get("family", "")
    genus = row.get("genus", "")
    if pd.isna(family) or not family:
        return ""
    if family == "Indo-European":
        return f"{genus} (IE)" if pd.notna(genus) and genus else "(IE)"
    if family == "Altaic":
        return f"{genus} (Alt)" if pd.notna(genus) and genus else "(Alt)"
    return family


def format_name(row: pd.Series) -> str:
    """Language name; for deu/rus add segmentor in brackets."""
    name = row["language_name"] if pd.notna(row["language_name"]) else row["lang"]
    if row["lang"] in ("deu", "rus") and pd.notna(row.get("segmentor")):
        name = f"{name} ({row['segmentor']})"
    return name


def format_typology(val) -> str:
    if pd.isna(val) or not val:
        return ""
    return str(val)[:3]


def thousands(val) -> str:
    if pd.isna(val):
        return ""
    return f"{val / 1000:.1f}"


def fmt_float(val, decimals=3) -> str:
    if pd.isna(val):
        return ""
    return f"{val:.{decimals}f}"


def escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def main():
    df = pd.read_csv(DATA_PATH)

    columns = ["Code", "Name", "Family", "Typ", "oTkC(k)", "cTpC(k)",
               "TTR", "H", "Iws", "mH", "M/W", "CI"]

    lines = []
    lines.append(r"\begin{tabular}{ll l l rr rr r rr r}")
    lines.append(r"\toprule")
    lines.append(" & ".join(columns) + r" \\")
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        vals = [
            escape_latex(f"{r['lang']}_{r['script']}"),
            escape_latex(format_name(r)),
            escape_latex(format_family(r)),
            format_typology(r.get("typology")),
            thousands(r.get("original_token_count")),
            thousands(r.get("current_type_count")),
            fmt_float(r.get("type_token_ratio")),
            fmt_float(r.get("token_frequency_entropy")),
            fmt_float(r.get("compress_reduction_pct")),
            fmt_float(r.get("morph_entropy")),
            fmt_float(r.get("avg_morphs_per_word")),
            fmt_float(r.get("compounding_index")),
        ]
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path = ROOT / "results" / "3_analyze" / "print_table.tex"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
