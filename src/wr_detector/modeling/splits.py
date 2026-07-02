"""Stable source-id holdout masks and stratified cross-validation builders."""

from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold


def make_global_holdout_mask(
    source_ids: pd.Series,
    *,
    holdout_fraction: float,
    random_state: int,
    target: pd.Series | None = None,
) -> pd.Series:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1.")
    if target is None:
        return _hash_fraction_mask(source_ids, holdout_fraction=holdout_fraction, random_state=random_state)
    if len(source_ids) != len(target):
        raise ValueError("source_ids and target must have the same length.")
    out = pd.Series(False, index=source_ids.index)
    target_series = pd.Series(target, index=source_ids.index)
    for class_value in sorted(target_series.dropna().unique()):
        class_mask = target_series.eq(class_value)
        out.loc[class_mask] = _hash_fraction_mask(
            source_ids.loc[class_mask],
            holdout_fraction=holdout_fraction,
            random_state=random_state,
        )
    return out.astype(bool)


def _hash_fraction_mask(source_ids: pd.Series, *, holdout_fraction: float, random_state: int) -> pd.Series:
    threshold = int(holdout_fraction * 10_000)
    values = source_ids.astype("string").map(lambda value: _stable_bucket(str(value), random_state))
    return values.lt(threshold).astype(bool)


def make_cv(y: pd.Series, *, n_splits: int, n_repeats: int, random_state: int) -> RepeatedStratifiedKFold:
    class_counts = y.value_counts()
    if class_counts.empty or class_counts.min() < 2:
        raise ValueError("At least two rows per class are required for stratified CV.")
    effective_splits = min(int(n_splits), int(class_counts.min()))
    if effective_splits < 2:
        raise ValueError("At least two stratified folds are required.")
    return RepeatedStratifiedKFold(
        n_splits=effective_splits,
        n_repeats=int(n_repeats),
        random_state=int(random_state),
    )


def _stable_bucket(value: str, random_state: int) -> int:
    digest = hashlib.blake2b(f"{random_state}:{value}".encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % 10_000
