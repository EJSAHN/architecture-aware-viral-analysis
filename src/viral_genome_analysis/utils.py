from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def safe_sheet_name(name: str) -> str:
    clean = str(name).replace("/", "_").replace("\\", "_").replace("[", "(").replace("]", ")")
    clean = clean[:31]
    return clean or "Sheet"


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def flatten_config(prefix: str, value) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_config(new_prefix, nested))
        return rows
    if isinstance(value, (list, tuple)):
        rows.append({"key": prefix, "value": ", ".join(map(str, value))})
        return rows
    rows.append({"key": prefix, "value": "" if value is None else str(value)})
    return rows


def benjamini_hochberg(p_values: Sequence[float | int | None]) -> np.ndarray:
    arr = np.array([np.nan if p is None or (isinstance(p, float) and math.isnan(p)) else float(p) for p in p_values], dtype=float)
    valid = np.where(~np.isnan(arr))[0]
    result = np.full_like(arr, np.nan, dtype=float)
    if len(valid) == 0:
        return result
    vals = arr[valid]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = ranked[i] * n / rank
        prev = min(prev, adj)
        adjusted[i] = min(prev, 1.0)
    result[valid[order]] = adjusted
    return result


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, seed: int = 7, alpha: float = 0.05) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = rng.choice(arr, size=arr.size, replace=True)
        samples[i] = float(np.mean(draw))
    lower = float(np.quantile(samples, alpha / 2))
    median = float(np.mean(arr))
    upper = float(np.quantile(samples, 1 - alpha / 2))
    return (lower, median, upper)


def bootstrap_delta_ci(values_a: Sequence[float], values_b: Sequence[float], n_boot: int = 1000, seed: int = 7, alpha: float = 0.05) -> tuple[float, float, float]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw_a = rng.choice(a, size=a.size, replace=True)
        draw_b = rng.choice(b, size=b.size, replace=True)
        samples[i] = float(np.mean(draw_a) - np.mean(draw_b))
    observed = float(np.mean(a) - np.mean(b))
    lower = float(np.quantile(samples, alpha / 2))
    upper = float(np.quantile(samples, 1 - alpha / 2))
    return (lower, observed, upper)


def bool_to_yes_no(value: bool | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return "yes" if bool(value) else "no"


def top_n_or_all(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df.copy()
    return df.head(n).copy()
