"""
Reusable language color/marker system for all analysis notebooks.

Provides:
  - Color maps: IE_GENUS_COLORS, NON_IE_FAMILY_COLORS, SEGMENTOR_COLORS, TYPOLOGY_COLORS
  - Script markers: SCRIPT_MARKERS (script code → matplotlib marker)
  - Helpers: get_color(), get_marker()
  - Legend builders: build_color_legend(), build_script_legend()
  - Plot decorator: add_lang_markers() — adds colored script markers to y-axis labels
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath

# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------

IE_GENUS_COLORS = {
    "Germanic": "royalblue",
    "Romance": "crimson",
    "Slavic": "seagreen",
    "Indic": "darkorange",
    "Iranian": "mediumpurple",
    "Celtic": "goldenrod",
    "Baltic": "teal",
    "Greek": "deeppink",
    "Armenian": "saddlebrown",
    "Albanian": "olive",
}

NON_IE_FAMILY_COLORS = {
    "Niger-Congo": "goldenrod",
    "Uralic": "steelblue",
    "Altaic": "gold",
    "Austronesian": "coral",
    "Dravidian": "hotpink",
    "Kartvelian": "sienna",
    "Afro-Asiatic": "darkkhaki",
    "Sino-Tibetan": "darkslategray",
    "Tai-Kadai": "cadetblue",
    "Austro-Asiatic": "rosybrown",
}

SEGMENTOR_COLORS = {
    "Sadilar": "forestgreen",
    "kcis": "royalblue",
    "Morphynet": "crimson",
    "LDC_RLP": "darkorange",
    "MorphoChallenge": "mediumpurple",
    "CELEX": "teal",
    "Uniparser": "deeppink",
    "MorphoLex": "goldenrod",
}

TYPOLOGY_COLORS = {
    "Isolating": "crimson",
    "Fusional": "royalblue",
    "Agglutinative": "forestgreen",
    "Introflexive": "darkorange",
    "Polysynthetic": "mediumpurple",
}

# ---------------------------------------------------------------------------
# Custom marker paths
# ---------------------------------------------------------------------------

_rat = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)]
RIGHT_ANGLE_TRI = MPath(_rat, [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY])

_theta = np.linspace(0, np.pi, 30)
_hc = list(zip(np.cos(_theta), np.sin(_theta))) + [(1.0, 0.0)]
HALF_CIRCLE = MPath(_hc, [MPath.MOVETO] + [MPath.LINETO] * (len(_hc) - 2) + [MPath.CLOSEPOLY])

_tv = [(-0.5, 0.5), (0.5, 0.5), (0.5, 0.2), (0.15, 0.2),
       (0.15, -0.5), (-0.15, -0.5), (-0.15, 0.2), (-0.5, 0.2), (-0.5, 0.5)]
T_SHAPE = MPath(_tv, [MPath.MOVETO] + [MPath.LINETO] * 7 + [MPath.CLOSEPOLY])

# Script code → marker shape
SCRIPT_MARKERS = {
    # Alphabet
    "Latn": "^", "Cyrl": "v",
    "Grek": RIGHT_ANGLE_TRI, "Armn": RIGHT_ANGLE_TRI, "Geor": RIGHT_ANGLE_TRI,
    # Abjad
    "Arab": "s", "Hebr": "D", "Thaa": "D",
    # Abugida
    "Deva": "o", "Beng": "o", "Orya": "o", "Gujr": "o", "Guru": "o",
    "Sinh": "o", "Knda": "o", "Mlym": "o", "Taml": "o", "Telu": "o", "Thai": "o",
    # Ethiopic / Syllabary / Logographic
    "Ethi": HALF_CIRCLE, "Hang": T_SHAPE, "Cans": T_SHAPE, "Hani": "*",
}

# Legend display order: (label, marker)
SCRIPT_TYPE_LEGEND = [
    ("Latin", "^"), ("Cyrillic", "v"), ("Other alphabet", RIGHT_ANGLE_TRI),
    ("Arabic", "s"), ("Other abjad", "D"),
    ("Abugida", "o"), ("Ethiopic", HALF_CIRCLE),
    ("Syllabary", T_SHAPE), ("Logographic", "*"),
]

# ---------------------------------------------------------------------------
# Row-level helpers (expect dict-like with "family", "genus", "script" keys)
# ---------------------------------------------------------------------------

def get_color(row):
    """Return the color for a language row based on IE genus or non-IE family."""
    if row["family"] == "Indo-European":
        return IE_GENUS_COLORS.get(row["genus"], "gray")
    return NON_IE_FAMILY_COLORS.get(row["family"], "gray")


def get_marker(row):
    """Return the marker shape for a language row based on its script code."""
    return SCRIPT_MARKERS.get(row["script"], "x")


# ---------------------------------------------------------------------------
# Legend builders
# ---------------------------------------------------------------------------

def build_color_legend(data, marker="o", markersize=7, counts=False):
    """Legend handles for IE genus + non-IE family colors present in *data*.

    Parameters
    ----------
    data : DataFrame with "genus" and "family" columns.
    marker : marker style used in the legend swatches.
    markersize : size of the legend marker.
    counts : if True, append (n=...) to each label.
    """
    handles = [Line2D([], [], marker="", linestyle="", label="Indo-European:")]
    for genus, color in IE_GENUS_COLORS.items():
        mask = data["genus"] == genus
        if mask.any():
            lbl = f"  {genus} ({mask.sum()})" if counts else f"  {genus}"
            handles.append(Line2D([], [], marker=marker, color="w", markerfacecolor=color,
                                  markeredgecolor="k", markersize=markersize, label=lbl))
    handles.append(Line2D([], [], marker="", linestyle="", label="Other families:"))
    for fam, color in NON_IE_FAMILY_COLORS.items():
        mask = data["family"] == fam
        if mask.any():
            lbl = f"  {fam} ({mask.sum()})" if counts else f"  {fam}"
            handles.append(Line2D([], [], marker=marker, color="w", markerfacecolor=color,
                                  markeredgecolor="k", markersize=markersize, label=lbl))
    return handles


def build_script_legend(markersize=7):
    """Legend handles for script-type markers."""
    handles = [Line2D([], [], marker="", linestyle="", label="Scripts:")]
    for lbl, mkr in SCRIPT_TYPE_LEGEND:
        handles.append(Line2D([], [], marker=mkr, color="w", markerfacecolor="dimgray",
                              markeredgecolor="k", markersize=markersize, label=f"  {lbl}"))
    return handles


# ---------------------------------------------------------------------------
# Plot decorator: add colored script markers next to y-axis tick labels
# ---------------------------------------------------------------------------

def add_lang_markers(ax, data, x=-0.008, markersize=7, pad=18,
                     color_labels=True):
    """Place a colored script-shape marker next to every y-axis label.

    Parameters
    ----------
    ax : matplotlib Axes (horizontal bar chart with y-ticks at 0..n-1).
    data : DataFrame with "family", "genus", "script" columns,
           ordered the same as the y-axis labels.
    x : marker x-position in axes coordinates (negative = left of spine).
    markersize : marker size.
    pad : extra tick padding in points to leave room for the marker.
    color_labels : also recolor the y-tick label text.
    """
    ax.tick_params(axis="y", pad=pad)
    transform = ax.get_yaxis_transform()
    for i, (_, row) in enumerate(data.iterrows()):
        c = get_color(row)
        m = get_marker(row)
        ax.plot(x, i, marker=m, color=c, markersize=markersize,
                markeredgecolor="k", markeredgewidth=0.4,
                transform=transform, clip_on=False)
        if color_labels:
            ax.get_yticklabels()[i].set_color(c)
