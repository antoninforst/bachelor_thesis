"""
Interactive cutoff explorer.

Sliders let you change:
  - Coverage level  (picks the nearest available column)
  - Breakpoint position  (log-scale, draggable)

The plot updates in real time showing:
  - Scatter of original size vs truncated size (3 metrics)
  - Piecewise regression lines left/right of breakpoint
  - Left/right slopes, Pearson correlations, and language counts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import stats as sp_stats

# ── Settings ────────────────────────────────────────────────
USE_DISTINCT_WORDS = False   # x-axis: True = distinct_words, False = total_frequency
Y_USE_DISTINCT_WORDS = True  # y-axis: True = _types, False = _tokens
STATS_PATH = "results/1_process/3_truncate/statistics.csv"

# ── Load data ───────────────────────────────────────────────
stats = pd.read_csv(STATS_PATH)

y_sfx = "_types" if Y_USE_DISTINCT_WORDS else "_tokens"
x_col = "distinct_words" if USE_DISTINCT_WORDS else "total_frequency"

# Discover all coverage columns and their numeric levels
cov_cols = [c for c in stats.columns if c.startswith("cov_") and c.endswith(y_sfx)]
cov_levels = []
for c in cov_cols:
    inner = c[len("cov_"):-len(y_sfx)]
    parts = inner.split("_")
    if len(parts) == 2:
        cov_levels.append(float(f"{parts[0]}.{parts[1]}"))
    else:
        cov_levels.append(float(parts[0]))
cov_levels = np.array(cov_levels)

# Fixed metric columns (freq>=100, top 5% rank)
freq100_col = f"freq100{y_sfx}"
rank5_col = f"rank5pct{y_sfx}"

# Pre-compute log x for all valid rows
x_vals = stats[x_col].values.astype(float)

# ── Helper: nearest coverage column ─────────────────────────
def nearest_cov_col(target):
    idx = np.argmin(np.abs(cov_levels - target))
    return cov_cols[idx], cov_levels[idx]

# ── Helper: fit one segment ─────────────────────────────────
def fit_segment(log_x, log_y):
    if len(log_x) < 2:
        return np.nan, np.nan, np.nan
    sl, ic, r, _, _ = sp_stats.linregress(log_x, log_y)
    return sl, ic, r

# ── Initial state ───────────────────────────────────────────
init_cov = 91.5
cov_col, actual_cov = nearest_cov_col(init_cov)
y_vals = stats[cov_col].values.astype(float)

valid = (x_vals > 0) & (y_vals > 0)
log_x_all = np.log10(x_vals[valid])
log_y_all = np.log10(y_vals[valid])
init_bp_log = np.median(log_x_all)

# ── Build figure ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.25)

# Scatter artists (will be updated)
scat_cov = ax.scatter([], [], alpha=0.6, color="green", marker="o", s=25, label="", zorder=3)
scat_freq = ax.scatter([], [], alpha=0.6, color="red", marker="s", s=25, label="Freq >= 100", zorder=3)
scat_rank = ax.scatter([], [], alpha=0.6, color="blue", marker="^", s=25, label="Top 5% rank", zorder=3)

# Diagonal y=x
diag_line, = ax.plot([], [], "k--", alpha=0.2, linewidth=0.8)

# Regression lines
line_left, = ax.plot([], [], color="green", linewidth=2, alpha=0.7, label="")
line_right, = ax.plot([], [], color="darkgreen", linewidth=2, alpha=0.7, label="")

# Breakpoint line
bp_line = ax.axvline(10**init_bp_log, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(x_col.replace("_", " ").title())
y_label = "Distinct words after cutoff" if Y_USE_DISTINCT_WORDS else "Tokens after cutoff"
ax.set_ylabel(y_label)
ax.grid(True, which="both", alpha=0.3)

# Info text box
info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=9,
                    verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

# Legend placeholder
legend = ax.legend(fontsize=7, loc="lower right")

# ── Slider axes ─────────────────────────────────────────────
ax_cov = plt.axes([0.15, 0.12, 0.70, 0.03])
ax_bp = plt.axes([0.15, 0.06, 0.70, 0.03])

slider_cov = Slider(ax_cov, "Coverage %", float(cov_levels.min()), float(cov_levels.max()),
                    valinit=init_cov, valstep=0.5)
slider_bp = Slider(ax_bp, "Breakpoint (log₁₀)", float(log_x_all.min()) + 0.1,
                   float(log_x_all.max()) - 0.1, valinit=init_bp_log)

# ── Update function ─────────────────────────────────────────
def update(_=None):
    cov_target = slider_cov.val
    bp_log = slider_bp.val

    # Resolve coverage column
    col, actual = nearest_cov_col(cov_target)
    y_cov = stats[col].values.astype(float)

    # Valid mask for coverage scatter
    mask = (x_vals > 0) & (y_cov > 0)
    xv, yv = x_vals[mask], y_cov[mask]

    # Update coverage scatter
    scat_cov.set_offsets(np.column_stack([xv, yv]))
    scat_cov.set_label(f"{actual}% coverage")

    # Freq>=100 scatter
    y_f = stats[freq100_col].values.astype(float)
    mask_f = (x_vals > 0) & (y_f > 0)
    scat_freq.set_offsets(np.column_stack([x_vals[mask_f], y_f[mask_f]]))

    # Rank 5% scatter
    y_r = stats[rank5_col].values.astype(float)
    mask_r = (x_vals > 0) & (y_r > 0)
    scat_rank.set_offsets(np.column_stack([x_vals[mask_r], y_r[mask_r]]))

    # Diagonal
    all_max = max(x_vals[x_vals > 0].max(), yv.max() if len(yv) else 1) * 1.2
    diag_line.set_data([1, all_max], [1, all_max])

    # Log values for regression
    lx = np.log10(xv)
    ly = np.log10(yv)

    left_m = lx <= bp_log
    right_m = lx > bp_log
    n_left = int(left_m.sum())
    n_right = int(right_m.sum())

    # Left fit
    sl_l, ic_l, r_l = fit_segment(lx[left_m], ly[left_m])
    if not np.isnan(sl_l) and n_left >= 2:
        xl = np.logspace(lx[left_m].min(), bp_log, 50)
        line_left.set_data(xl, 10 ** (ic_l + sl_l * np.log10(xl)))
        line_left.set_label(f"Left fit (slope={sl_l:.3f})")
    else:
        line_left.set_data([], [])
        line_left.set_label("Left fit (N/A)")

    # Right fit
    sl_r, ic_r, r_r = fit_segment(lx[right_m], ly[right_m])
    if not np.isnan(sl_r) and n_right >= 2:
        xr = np.logspace(bp_log, lx[right_m].max(), 50)
        line_right.set_data(xr, 10 ** (ic_r + sl_r * np.log10(xr)))
        line_right.set_label(f"Right fit (slope={sl_r:.3f})")
    else:
        line_right.set_data([], [])
        line_right.set_label("Right fit (N/A)")

    # Breakpoint line
    bp_line.set_xdata([10**bp_log, 10**bp_log])

    # Pearson correlations (log-log)
    r_left_str = f"{r_l:.4f}" if not np.isnan(r_l) else "N/A"
    r_right_str = f"{r_r:.4f}" if not np.isnan(r_r) else "N/A"
    sl_l_str = f"{sl_l:.4f}" if not np.isnan(sl_l) else "N/A"
    sl_r_str = f"{sl_r:.4f}" if not np.isnan(sl_r) else "N/A"

    info = (
        f"Coverage: {actual}%\n"
        f"Breakpoint: {10**bp_log:,.0f}\n"
        f"─────────────────────────\n"
        f"Left:  n={n_left:>3d}  slope={sl_l_str}  r={r_left_str}\n"
        f"Right: n={n_right:>3d}  slope={sl_r_str}  r={r_right_str}\n"
        f"Languages right of BP: {n_right}"
    )
    info_text.set_text(info)

    ax.set_title(f"Original size vs truncated size — coverage {actual}%")

    # Re-do axis limits
    if len(xv) > 0:
        ax.set_xlim(0.5 * xv.min(), 2 * all_max)
        y_all = np.concatenate([yv, y_f[mask_f], y_r[mask_r]])
        y_all = y_all[y_all > 0]
        if len(y_all) > 0:
            ax.set_ylim(0.5 * y_all.min(), 2 * y_all.max())

    ax.legend(fontsize=7, loc="lower right")
    fig.canvas.draw_idle()

slider_cov.on_changed(update)
slider_bp.on_changed(update)

# Initial draw
update()
plt.show()
