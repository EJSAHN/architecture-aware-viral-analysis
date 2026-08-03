from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, silhouette_samples, silhouette_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

from .utils import benjamini_hochberg, bootstrap_ci, bootstrap_delta_ci


def _group_pseudo_f(X: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, int, int]:
    unique_labels = pd.Index(pd.unique(labels))
    n = X.shape[0]
    g = len(unique_labels)
    overall = X.mean(axis=0)
    ss_between = 0.0
    ss_within = 0.0
    for label in unique_labels:
        Xi = X[labels == label]
        centroid = Xi.mean(axis=0)
        ss_between += len(Xi) * float(((centroid - overall) ** 2).sum())
        ss_within += float(((Xi - centroid) ** 2).sum())
    df_between = g - 1
    df_within = n - g
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    return float(ms_between / ms_within), float(ss_between), float(ss_within), int(df_between), int(df_within)


def permutation_manova(
    X: np.ndarray,
    labels: np.ndarray,
    panel_name: str,
    label_name: str,
    n_permutations: int = 499,
    random_seed: int = 7,
) -> pd.DataFrame:
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.min(counts) < 2:
        return pd.DataFrame(
            [{
                "panel_name": panel_name,
                "label_name": label_name,
                "n_records": len(labels),
                "n_groups": len(unique),
                "groups_tested": "; ".join(map(str, unique)),
                "pseudo_f": np.nan,
                "r_squared": np.nan,
                "p_value": np.nan,
                "n_permutations": n_permutations,
            }]
        )

    observed_f, ss_between, ss_within, _, _ = _group_pseudo_f(X, labels)
    rng = np.random.default_rng(random_seed)
    permuted_f = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        permuted_labels = rng.permutation(labels)
        permuted_f[i] = _group_pseudo_f(X, permuted_labels)[0]
    p_value = (1.0 + float(np.sum(permuted_f >= observed_f))) / (n_permutations + 1.0)
    r_squared = ss_between / (ss_between + ss_within)

    return pd.DataFrame(
        [{
            "panel_name": panel_name,
            "label_name": label_name,
            "n_records": len(labels),
            "n_groups": len(unique),
            "groups_tested": "; ".join(map(str, unique)),
            "pseudo_f": observed_f,
            "r_squared": r_squared,
            "p_value": p_value,
            "n_permutations": n_permutations,
        }]
    )


def pairwise_permutation_manova(
    X: np.ndarray,
    labels: np.ndarray,
    panel_name: str,
    label_name: str,
    n_permutations: int = 199,
    random_seed: int = 7,
) -> pd.DataFrame:
    labels = np.asarray(labels)
    unique = pd.Index(pd.unique(labels))
    rows: list[dict[str, Any]] = []
    for i, label_a in enumerate(unique):
        for label_b in unique[i + 1 :]:
            mask = np.isin(labels, [label_a, label_b])
            Xi = X[mask]
            yi = labels[mask]
            result = permutation_manova(
                Xi,
                yi,
                panel_name=panel_name,
                label_name=label_name,
                n_permutations=n_permutations,
                random_seed=random_seed,
            ).iloc[0].to_dict()
            result["group_a"] = label_a
            result["group_b"] = label_b
            rows.append(result)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_fdr_bh"] = benjamini_hochberg(out["p_value"].tolist())
    return out


def dispersion_test(
    X: np.ndarray,
    labels: np.ndarray,
    panel_name: str,
    label_name: str,
    n_permutations: int = 499,
    random_seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.asarray(labels)
    unique = pd.Index(pd.unique(labels))
    if len(unique) < 2:
        empty_summary = pd.DataFrame(
            [{"panel_name": panel_name, "label_name": label_name, "pseudo_f": np.nan, "p_value": np.nan}]
        )
        return empty_summary, pd.DataFrame()

    distances = np.zeros(len(labels), dtype=float)
    per_record_rows: list[dict[str, Any]] = []
    for label in unique:
        mask = labels == label
        Xi = X[mask]
        centroid = Xi.mean(axis=0)
        d = np.sqrt(((Xi - centroid) ** 2).sum(axis=1))
        distances[mask] = d
        for value in d:
            per_record_rows.append(
                {
                    "panel_name": panel_name,
                    "label_name": label_name,
                    "group_label": label,
                    "distance_to_centroid": float(value),
                }
            )

    observed_f, _, _, _, _ = _group_pseudo_f(distances.reshape(-1, 1), labels)
    rng = np.random.default_rng(random_seed)
    perm_f = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        permuted_labels = rng.permutation(labels)
        perm_f[i] = _group_pseudo_f(distances.reshape(-1, 1), permuted_labels)[0]
    p_value = (1.0 + float(np.sum(perm_f >= observed_f))) / (n_permutations + 1.0)

    summary = pd.DataFrame(
        [
            {
                "panel_name": panel_name,
                "label_name": label_name,
                "n_records": len(labels),
                "n_groups": len(unique),
                "pseudo_f": observed_f,
                "p_value": p_value,
                "n_permutations": n_permutations,
            }
        ]
    )
    per_record = pd.DataFrame(per_record_rows)
    return summary, per_record


def pairwise_dispersion_tests(
    X: np.ndarray,
    labels: np.ndarray,
    panel_name: str,
    label_name: str,
    n_permutations: int = 199,
    random_seed: int = 7,
) -> pd.DataFrame:
    labels = np.asarray(labels)
    unique = pd.Index(pd.unique(labels))
    rows: list[dict[str, Any]] = []
    for i, label_a in enumerate(unique):
        for label_b in unique[i + 1 :]:
            mask = np.isin(labels, [label_a, label_b])
            summary, _ = dispersion_test(
                X[mask],
                labels[mask],
                panel_name=panel_name,
                label_name=label_name,
                n_permutations=n_permutations,
                random_seed=random_seed,
            )
            row = summary.iloc[0].to_dict()
            row["group_a"] = label_a
            row["group_b"] = label_b
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_fdr_bh"] = benjamini_hochberg(out["p_value"].tolist())
    return out


def silhouette_tables(X: np.ndarray, labels: np.ndarray, panel_name: str, label_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.min(counts) < 2:
        return (
            pd.DataFrame([{"panel_name": panel_name, "label_name": label_name, "silhouette_mean": np.nan}]),
            pd.DataFrame(),
        )
    score = float(silhouette_score(X, labels, metric="euclidean"))
    samples = silhouette_samples(X, labels, metric="euclidean")
    summary = pd.DataFrame(
        [
            {
                "panel_name": panel_name,
                "label_name": label_name,
                "n_records": len(labels),
                "n_groups": len(unique),
                "silhouette_mean": score,
            }
        ]
    )
    per_group = (
        pd.DataFrame({"group_label": labels, "silhouette_value": samples})
        .groupby("group_label", as_index=False)["silhouette_value"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "silhouette_mean", "median": "silhouette_median", "count": "n_records"})
    )
    per_group.insert(0, "label_name", label_name)
    per_group.insert(0, "panel_name", panel_name)
    return summary, per_group



def repeated_cv_classification(
    X: np.ndarray,
    labels: np.ndarray,
    panel_name: str,
    label_name: str,
    n_splits: int = 5,
    n_repeats: int = 5,
    n_permutations: int = 25,
    random_seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.min(counts) < n_splits:
        empty_summary = pd.DataFrame(
            [
                {
                    "panel_name": panel_name,
                    "label_name": label_name,
                    "n_records": len(labels),
                    "n_groups": len(unique),
                    "accuracy_mean": np.nan,
                    "balanced_accuracy_mean": np.nan,
                    "macro_f1_mean": np.nan,
                    "macro_f1_permutation_p_value": np.nan,
                }
            ]
        )
        return empty_summary, pd.DataFrame(), pd.DataFrame()

    min_class_size = int(np.min(counts))
    n_neighbors = max(1, min(3, min_class_size - 1))
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed)

    fold_rows: list[dict[str, Any]] = []
    split_cache: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X, labels), start=1):
        split_cache.append((train_idx, test_idx))
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", metric="euclidean")
        model.fit(X[train_idx], labels[train_idx])
        predicted = model.predict(X[test_idx])

        fold_rows.append(
            {
                "panel_name": panel_name,
                "label_name": label_name,
                "fold_index": fold_index,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_neighbors": n_neighbors,
                "accuracy": float(accuracy_score(labels[test_idx], predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(labels[test_idx], predicted)),
                "macro_f1": float(f1_score(labels[test_idx], predicted, average="macro")),
            }
        )

    fold_df = pd.DataFrame(fold_rows)

    rng = np.random.default_rng(random_seed)
    permutation_rows: list[dict[str, Any]] = []
    observed_macro_f1 = float(fold_df["macro_f1"].mean())
    null_macro_f1 = np.empty(n_permutations, dtype=float)
    for perm_index in range(n_permutations):
        permuted_labels = rng.permutation(labels)
        perm_scores = []
        for train_idx, test_idx in split_cache:
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", metric="euclidean")
            model.fit(X[train_idx], permuted_labels[train_idx])
            predicted = model.predict(X[test_idx])
            perm_scores.append(float(f1_score(permuted_labels[test_idx], predicted, average="macro")))
        null_macro_f1[perm_index] = float(np.mean(perm_scores))
        permutation_rows.append(
            {
                "panel_name": panel_name,
                "label_name": label_name,
                "permutation_index": perm_index + 1,
                "macro_f1_null": null_macro_f1[perm_index],
                "n_neighbors": n_neighbors,
            }
        )

    permutation_p = (1.0 + float(np.sum(null_macro_f1 >= observed_macro_f1))) / (n_permutations + 1.0)

    summary = pd.DataFrame(
        [
            {
                "panel_name": panel_name,
                "label_name": label_name,
                "n_records": len(labels),
                "n_groups": len(unique),
                "n_neighbors": n_neighbors,
                "accuracy_mean": float(fold_df["accuracy"].mean()),
                "accuracy_sd": float(fold_df["accuracy"].std(ddof=1)),
                "balanced_accuracy_mean": float(fold_df["balanced_accuracy"].mean()),
                "balanced_accuracy_sd": float(fold_df["balanced_accuracy"].std(ddof=1)),
                "macro_f1_mean": observed_macro_f1,
                "macro_f1_sd": float(fold_df["macro_f1"].std(ddof=1)),
                "macro_f1_permutation_p_value": permutation_p,
                "n_cv_folds_total": len(fold_df),
                "n_label_permutations": n_permutations,
            }
        ]
    )
    permutation_df = pd.DataFrame(permutation_rows)
    return summary, fold_df, permutation_df


def bootstrap_centroid_distances(
    X: np.ndarray,
    labels: np.ndarray,
    panel_name: str,
    label_name: str,
    n_bootstrap: int = 1000,
    random_seed: int = 7,
) -> pd.DataFrame:
    labels = np.asarray(labels)
    unique = pd.Index(pd.unique(labels))
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    for i, label_a in enumerate(unique):
        Xa = X[labels == label_a]
        for label_b in unique[i + 1 :]:
            Xb = X[labels == label_b]
            if len(Xa) == 0 or len(Xb) == 0:
                continue
            observed = float(np.linalg.norm(Xa.mean(axis=0) - Xb.mean(axis=0)))
            boot = np.empty(n_bootstrap, dtype=float)
            for b in range(n_bootstrap):
                Xa_draw = Xa[rng.integers(0, len(Xa), len(Xa))]
                Xb_draw = Xb[rng.integers(0, len(Xb), len(Xb))]
                boot[b] = float(np.linalg.norm(Xa_draw.mean(axis=0) - Xb_draw.mean(axis=0)))
            rows.append(
                {
                    "panel_name": panel_name,
                    "label_name": label_name,
                    "group_a": label_a,
                    "group_b": label_b,
                    "centroid_distance": observed,
                    "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                    "n_bootstrap": n_bootstrap,
                }
            )
    return pd.DataFrame(rows)


def panel_distance_summary(
    X: np.ndarray,
    labels: np.ndarray,
    panel_name: str,
    label_name: str,
) -> pd.DataFrame:
    labels = np.asarray(labels)
    if len(labels) < 2:
        return pd.DataFrame()
    distances = pdist(X, metric="euclidean")
    rows: list[dict[str, Any]] = []
    idx = 0
    for i in range(len(labels) - 1):
        for j in range(i + 1, len(labels)):
            rows.append(
                {
                    "panel_name": panel_name,
                    "label_name": label_name,
                    "group_a": labels[i],
                    "group_b": labels[j],
                    "same_group": bool(labels[i] == labels[j]),
                    "distance": float(distances[idx]),
                }
            )
            idx += 1
    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return pair_df
    summary = (
        pair_df.groupby(["panel_name", "label_name", "same_group"], as_index=False)["distance"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "distance_mean", "median": "distance_median", "count": "n_pairs"})
    )
    return summary



def _cohens_d(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = ((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2)
    if not np.isfinite(pooled) or pooled <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))



def _cliffs_delta(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    comparisons = np.subtract.outer(a, b)
    wins = np.sum(comparisons > 0)
    losses = np.sum(comparisons < 0)
    return float((wins - losses) / (a.size * b.size))





def _safe_correlation(x: np.ndarray, y: np.ndarray, method: str = "spearman") -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan"), float("nan")
    if method == "pearson":
        result = pearsonr(x, y)
    else:
        result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_bootstrap: int = 1000,
    random_seed: int = 7,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_seed)
    boot = []
    for _ in range(n_bootstrap):
        draw = rng.integers(0, x.size, x.size)
        xb = x[draw]
        yb = y[draw]
        statistic, _ = _safe_correlation(xb, yb, method=method)
        boot.append(float(statistic))
    boot_arr = np.asarray(boot, dtype=float)
    boot_arr = boot_arr[np.isfinite(boot_arr)]
    if boot_arr.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(boot_arr, 0.025)), float(np.quantile(boot_arr, 0.975))



def correlation_test_summary(
    x,
    y,
    panel_name: str,
    label_name: str,
    x_name: str,
    y_name: str,
    n_bootstrap: int = 1000,
    random_seed: int = 7,
    method: str = "spearman",
) -> pd.DataFrame:
    x_arr = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float)
    y_arr = np.asarray(pd.to_numeric(pd.Series(y), errors="coerce"), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 4:
        return pd.DataFrame(
            [{
                "panel_name": panel_name,
                "label_name": label_name,
                "metric_type": "correlation",
                "x_name": x_name,
                "y_name": y_name,
                "method": method,
                "n_records": int(x_arr.size),
                "statistic": np.nan,
                "p_value": np.nan,
                "bootstrap_ci_low": np.nan,
                "bootstrap_ci_high": np.nan,
            }]
        )
    statistic, p_value = _safe_correlation(x_arr, y_arr, method=method)
    ci_low, ci_high = _bootstrap_correlation_ci(x_arr, y_arr, method=method, n_bootstrap=n_bootstrap, random_seed=random_seed)
    return pd.DataFrame(
        [{
            "panel_name": panel_name,
            "label_name": label_name,
            "metric_type": "correlation",
            "x_name": x_name,
            "y_name": y_name,
            "method": method,
            "n_records": int(x_arr.size),
            "statistic": statistic,
            "p_value": p_value,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
        }]
    )



def group_value_summary(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    panel_name: str,
    label_name: str,
    n_bootstrap: int = 1000,
    random_seed: int = 7,
) -> pd.DataFrame:
    if df.empty or value_col not in df.columns or group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for group_value, sub in df.groupby(group_col):
        values = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        ci_low, ci_center, ci_high = bootstrap_ci(values, n_boot=n_bootstrap, seed=random_seed)
        rows.append(
            {
                "panel_name": panel_name,
                "label_name": label_name,
                "metric_type": "group_summary",
                "group_label": group_value,
                "n_records": int(values.size),
                "mean_value": float(np.mean(values)),
                "median_value": float(np.median(values)),
                "sd_value": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "bootstrap_center": ci_center,
            }
        )
    return pd.DataFrame(rows)



def pairwise_value_group_tests(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    panel_name: str,
    label_name: str,
    n_permutations: int = 499,
    n_bootstrap: int = 1000,
    random_seed: int = 7,
) -> pd.DataFrame:
    if df.empty or value_col not in df.columns or group_col not in df.columns:
        return pd.DataFrame()
    work = df[[group_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[group_col, value_col]).reset_index(drop=True)
    unique_groups = pd.Index(pd.unique(work[group_col]))
    rng = np.random.default_rng(random_seed)
    rows = []
    for i, group_a in enumerate(unique_groups):
        for group_b in unique_groups[i + 1 :]:
            values_a = work.loc[work[group_col] == group_a, value_col].to_numpy(dtype=float)
            values_b = work.loc[work[group_col] == group_b, value_col].to_numpy(dtype=float)
            if values_a.size < 2 or values_b.size < 2:
                continue
            observed = float(np.mean(values_a) - np.mean(values_b))
            combined = np.concatenate([values_a, values_b])
            null_stats = np.empty(n_permutations, dtype=float)
            for perm_index in range(n_permutations):
                shuffled = rng.permutation(combined)
                draw_a = shuffled[: values_a.size]
                draw_b = shuffled[values_a.size :]
                null_stats[perm_index] = float(np.mean(draw_a) - np.mean(draw_b))
            p_value = (1.0 + float(np.sum(np.abs(null_stats) >= abs(observed)))) / (n_permutations + 1.0)
            ci_low, ci_center, ci_high = bootstrap_delta_ci(values_a, values_b, n_boot=n_bootstrap, seed=random_seed)
            rows.append(
                {
                    "panel_name": panel_name,
                    "label_name": label_name,
                    "metric_type": "pairwise_group_difference",
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_group_a": int(values_a.size),
                    "n_group_b": int(values_b.size),
                    "mean_group_a": float(np.mean(values_a)),
                    "mean_group_b": float(np.mean(values_b)),
                    "difference_mean_a_minus_b": observed,
                    "difference_median_a_minus_b": float(np.median(values_a) - np.median(values_b)),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "bootstrap_center": ci_center,
                    "cohens_d": _cohens_d(values_a, values_b),
                    "cliffs_delta": _cliffs_delta(values_a, values_b),
                    "p_value": p_value,
                    "n_permutations": n_permutations,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_fdr_bh"] = benjamini_hochberg(out["p_value"].tolist())
    return out
