"""Tests for the grouping framework used in metric_quality_matrix.ipynb."""
import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Re-implement the notebook functions so tests are self-contained
# ---------------------------------------------------------------------------
MIN_GROUP_SIZE = 2
USE_SQUARED_DISTANCE = True
WEIGHT_GROUPS = False


def normalized_values(values):
    values = pd.to_numeric(values, errors="coerce")
    std = (values - values.mean()) / values.std(ddof=0)
    if std.isna().all() or std.max() == std.min():
        return pd.Series(np.nan, index=values.index)
    return (std - std.min()) / (std.max() - std.min())


def clustering_index(frame, group_col=None, groups=None,
                     squared=USE_SQUARED_DISTANCE, weighted=WEIGHT_GROUPS):
    needed_cols = ["unit_id", "value"] + ([group_col] if group_col else [])
    work = frame[needed_cols].dropna(subset=["unit_id", "value"]).copy()
    if work.empty:
        return {"index": np.nan, "score": np.nan, "groups": 0, "pairs": 0}

    work["normalized"] = normalized_values(work["value"])
    work = work.dropna(subset=["normalized"])

    if groups is None:
        group_map = {
            str(n): g["unit_id"].tolist()
            for n, g in work.dropna(subset=[group_col]).groupby(group_col)
        }
    else:
        group_map = groups

    unit_to_val = work.set_index("unit_id")["normalized"].to_dict()
    group_distances, group_weights, pair_total = [], [], 0

    for unit_ids in group_map.values():
        vals = [unit_to_val[u] for u in unit_ids if u in unit_to_val]
        if len(vals) < MIN_GROUP_SIZE:
            continue
        dists = [
            (abs(l - r) ** 2 if squared else abs(l - r))
            for l, r in combinations(vals, 2)
        ]
        if not dists:
            continue
        group_distances.append(float(np.mean(dists)))
        group_weights.append(len(dists) if weighted else 1)
        pair_total += len(dists)

    if not group_distances:
        return {"index": np.nan, "score": np.nan, "groups": 0, "pairs": 0}

    idx = float(np.average(group_distances, weights=group_weights))
    return {"index": idx, "score": 1 - idx, "groups": len(group_distances), "pairs": pair_total}


def family_genus_score(frame):
    fam = clustering_index(frame, group_col="family")
    gen = clustering_index(frame, group_col="genus")
    vals = [v for v in [fam["index"], gen["index"]] if pd.notna(v)]
    idx = float(np.mean(vals)) if vals else np.nan
    return {
        "index": idx,
        "score": 1 - idx if pd.notna(idx) else np.nan,
        "groups": fam["groups"] + gen["groups"],
        "pairs": fam["pairs"] + gen["pairs"],
    }


def spearman_abs(a, b):
    both = pd.concat([a, b], axis=1).dropna()
    if len(both) < 3 or both.iloc[:, 0].nunique() < 2 or both.iloc[:, 1].nunique() < 2:
        return np.nan
    return abs(both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank()))


# ---------------------------------------------------------------------------
# normalized_values
# ---------------------------------------------------------------------------
class TestNormalizedValues:
    def test_basic_range(self):
        """Output must be in [0, 1]."""
        vals = pd.Series([10, 20, 30, 40, 50])
        normed = normalized_values(vals)
        assert normed.min() == pytest.approx(0.0)
        assert normed.max() == pytest.approx(1.0)

    def test_constant_returns_nan(self):
        """All-equal values cannot be normalized; expect NaN."""
        vals = pd.Series([5, 5, 5])
        normed = normalized_values(vals)
        assert normed.isna().all()

    def test_two_values(self):
        """Two distinct values should map to 0 and 1."""
        vals = pd.Series([100, 200])
        normed = normalized_values(vals)
        assert normed.iloc[0] == pytest.approx(0.0)
        assert normed.iloc[1] == pytest.approx(1.0)

    def test_with_nans(self):
        """NaN inputs stay NaN, rest normalized normally."""
        vals = pd.Series([1, np.nan, 3])
        normed = normalized_values(vals)
        assert pd.isna(normed.iloc[1])
        assert normed.iloc[0] == pytest.approx(0.0)
        assert normed.iloc[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# clustering_index
# ---------------------------------------------------------------------------
def _make_frame(unit_ids, values, group_col_name="group", group_labels=None):
    """Helper to build a DataFrame for clustering_index."""
    d = {"unit_id": unit_ids, "value": values}
    if group_labels is not None:
        d[group_col_name] = group_labels
    return pd.DataFrame(d)


class TestClusteringIndex:
    def test_identical_group_values_score_one(self):
        """If every member of every group has the same value, distance=0 → score=1."""
        frame = _make_frame(
            unit_ids=["a", "b", "c", "d"],
            values=[10, 10, 20, 20],
            group_labels=["G1", "G1", "G2", "G2"],
        )
        result = clustering_index(frame, group_col="group")
        assert result["score"] == pytest.approx(1.0)
        assert result["index"] == pytest.approx(0.0)
        assert result["groups"] == 2
        assert result["pairs"] == 2

    def test_maximally_spread_group(self):
        """One group spanning the full range: pair distance = (1-0)^2 = 1 → score=0."""
        frame = _make_frame(
            unit_ids=["a", "b"],
            values=[0, 100],
            group_labels=["G", "G"],
        )
        result = clustering_index(frame, group_col="group")
        assert result["index"] == pytest.approx(1.0)
        assert result["score"] == pytest.approx(0.0)

    def test_explicit_groups_dict(self):
        """Using explicit groups dict instead of group_col."""
        frame = _make_frame(
            unit_ids=["a", "b", "c"],
            values=[5, 5, 100],
        )
        groups = {"close_pair": ["a", "b"]}
        result = clustering_index(frame, groups=groups)
        assert result["score"] == pytest.approx(1.0)
        assert result["groups"] == 1
        assert result["pairs"] == 1

    def test_group_too_small_is_skipped(self):
        """Groups with fewer than MIN_GROUP_SIZE members are skipped."""
        frame = _make_frame(
            unit_ids=["a", "b", "c"],
            values=[1, 2, 3],
            group_labels=["G1", "G1", "G_solo"],
        )
        result = clustering_index(frame, group_col="group")
        assert result["groups"] == 1  # G_solo skipped
        assert result["pairs"] == 1

    def test_empty_frame(self):
        frame = pd.DataFrame({"unit_id": [], "value": [], "group": []})
        result = clustering_index(frame, group_col="group")
        assert math.isnan(result["index"])
        assert result["groups"] == 0

    def test_three_element_group_pair_count(self):
        """A group of 3 produces C(3,2)=3 pairs."""
        frame = _make_frame(
            unit_ids=["a", "b", "c"],
            values=[10, 10, 10],
            group_labels=["G", "G", "G"],
        )
        result = clustering_index(frame, group_col="group")
        assert result["pairs"] == 3

    def test_score_is_one_minus_index(self):
        """score == 1 - index for any result."""
        frame = _make_frame(
            unit_ids=["a", "b", "c", "d"],
            values=[1, 3, 10, 12],
            group_labels=["X", "X", "Y", "Y"],
        )
        result = clustering_index(frame, group_col="group")
        assert result["score"] == pytest.approx(1 - result["index"])

    def test_unsquared_distance(self):
        """With squared=False, distance = |a-b| instead of (a-b)^2."""
        frame = _make_frame(
            unit_ids=["a", "b"],
            values=[0, 100],
            group_labels=["G", "G"],
        )
        result = clustering_index(frame, group_col="group", squared=False)
        # normalized: 0 and 1; distance = |0-1| = 1
        assert result["index"] == pytest.approx(1.0)

    def test_known_numeric_example(self):
        """Hand-computed example: values [0, 4, 10] in one group.
        Normalized: std = [−1, −0.2, 1] (ddof=0, σ≈4.32),
        then min-max → [0, 0.4, 1].
        Squared pairwise: (0-0.4)^2=0.16, (0-1)^2=1, (0.4-1)^2=0.36.
        Mean = 0.5067.
        """
        frame = _make_frame(
            unit_ids=["a", "b", "c"],
            values=[0, 4, 10],
            group_labels=["G", "G", "G"],
        )
        result = clustering_index(frame, group_col="group")

        # Manual: normalize [0,4,10]: mean=14/3, std=sqrt(56/9)=~4.32
        vals = np.array([0.0, 4.0, 10.0])
        mu = vals.mean()
        sigma = vals.std(ddof=0)
        z = (vals - mu) / sigma
        normed = (z - z.min()) / (z.max() - z.min())
        pairs = list(combinations(normed, 2))
        expected_index = np.mean([(a - b) ** 2 for a, b in pairs])

        assert result["index"] == pytest.approx(expected_index, abs=1e-10)
        assert result["pairs"] == 3


# ---------------------------------------------------------------------------
# family_genus_score
# ---------------------------------------------------------------------------
class TestFamilyGenusScore:
    def test_combines_family_and_genus(self):
        """Score is the average of family and genus clustering indices."""
        frame = pd.DataFrame({
            "unit_id": ["a", "b", "c", "d"],
            "value": [10, 10, 10, 10],
            "family": ["F1", "F1", "F2", "F2"],
            "genus": ["G1", "G1", "G2", "G2"],
        })
        result = family_genus_score(frame)
        # All same value within groups → index 0 for both
        assert result["score"] == pytest.approx(1.0)
        assert result["groups"] == 4  # 2 family + 2 genus

    def test_no_groups(self):
        """If all groups are singletons, result is NaN."""
        frame = pd.DataFrame({
            "unit_id": ["a", "b"],
            "value": [1, 2],
            "family": ["F1", "F2"],
            "genus": ["G1", "G2"],
        })
        result = family_genus_score(frame)
        assert math.isnan(result["score"])


# ---------------------------------------------------------------------------
# spearman_abs
# ---------------------------------------------------------------------------
class TestSpearmanAbs:
    def test_perfect_positive(self):
        a = pd.Series([1, 2, 3, 4, 5])
        b = pd.Series([10, 20, 30, 40, 50])
        assert spearman_abs(a, b) == pytest.approx(1.0)

    def test_perfect_negative(self):
        a = pd.Series([1, 2, 3, 4, 5])
        b = pd.Series([50, 40, 30, 20, 10])
        assert spearman_abs(a, b) == pytest.approx(1.0)  # absolute

    def test_too_few_values(self):
        a = pd.Series([1, 2])
        b = pd.Series([3, 4])
        assert math.isnan(spearman_abs(a, b))

    def test_constant_series(self):
        a = pd.Series([5, 5, 5])
        b = pd.Series([1, 2, 3])
        assert math.isnan(spearman_abs(a, b))

    def test_with_nans(self):
        a = pd.Series([1, np.nan, 3, 4, 5])
        b = pd.Series([10, 20, np.nan, 40, 50])
        result = spearman_abs(a, b)
        assert 0 <= result <= 1
