"""Match-grouped data partitions.

1. **Final test hold-out**: ~20% of eligible matches, drawn per stratum
   (competition, season) with a fixed seed. It is written to the split
   manifest and never loaded by the pilot (``assert_no_final_test``).
2. **Development pool**: the remaining matches.
3. **Pilot subset**: ``pilot_matches`` development matches, of which
   ``pilot_extra_time_matches`` are drawn from extra-time matches (so the
   extra-time handling is exercised) and the rest by proportional allocation
   over (competition, season) strata (largest remainder, at least one per
   stratum while the budget allows).
4. **Cross-validation**: StratifiedGroupKFold over evaluation rows, grouped by
   match, stratified by the target class; a match is in exactly one
   validation fold.

Every draw uses ``numpy.random.default_rng(seed)`` on sorted inputs, so the
partitions depend only on the seed and the match list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class LeakageError(RuntimeError):
    """A partition rule was violated (overlap, final-test access, ...)."""


@dataclass(frozen=True)
class FoldAssignment:
    fold_id: int
    train_match_ids: tuple[int, ...]
    validation_match_ids: tuple[int, ...]


def _largest_remainder(sizes: dict[str, int], budget: int, minimum: int = 0) -> dict[str, int]:
    """Allocate ``budget`` over strata proportionally to ``sizes``."""
    keys = sorted(sizes)
    allocation = {k: min(minimum, sizes[k]) for k in keys}
    remaining = budget - sum(allocation.values())
    if remaining < 0:
        raise ValueError("budget smaller than the per-stratum minimum")
    total = sum(sizes.values())
    quotas = {k: remaining * sizes[k] / total for k in keys}
    for k in keys:
        allocation[k] += min(int(np.floor(quotas[k])), sizes[k] - allocation[k])
    leftover = budget - sum(allocation.values())
    order = sorted(keys, key=lambda k: (-(quotas[k] - np.floor(quotas[k])), k))
    while leftover > 0:
        progressed = False
        for k in order:
            if leftover and allocation[k] < sizes[k]:
                allocation[k] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            raise ValueError("budget exceeds available matches")
    return allocation


def final_test_split(strata: pd.Series, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """(development ids, final test ids). ``strata``: match_id -> stratum label."""
    rng = np.random.default_rng(seed)
    test: list[int] = []
    for label in sorted(strata.unique()):
        members = sorted(int(m) for m in strata.index[strata == label])
        n_test = int(round(fraction * len(members)))
        test.extend(int(m) for m in rng.choice(members, size=n_test, replace=False))
    test_set = set(test)
    development = sorted(int(m) for m in strata.index if int(m) not in test_set)
    return development, sorted(test_set)


def pilot_subset(
    development_ids: list[int], strata: pd.Series, extra_time_ids: set[int], n_pilot: int, n_extra_time: int, seed: int
) -> list[int]:
    rng = np.random.default_rng(seed + 1)
    pool = sorted(development_ids)
    et_pool = [m for m in pool if m in extra_time_ids]
    chosen = [int(m) for m in rng.choice(et_pool, size=min(n_extra_time, len(et_pool)), replace=False)]
    remaining = [m for m in pool if m not in set(chosen)]
    labels = strata.loc[remaining]
    sizes = labels.value_counts().to_dict()
    allocation = _largest_remainder(sizes, n_pilot - len(chosen), minimum=1 if len(sizes) <= n_pilot - len(chosen) else 0)
    for label in sorted(allocation):
        members = sorted(int(m) for m in labels.index[labels == label])
        chosen.extend(int(m) for m in rng.choice(members, size=allocation[label], replace=False))
    return sorted(chosen)


def grouped_folds(rows: pd.DataFrame, n_folds: int, seed: int) -> list[FoldAssignment]:
    """StratifiedGroupKFold over evaluation rows (columns: match_id, target_id)."""
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    groups = rows["match_id"].to_numpy()
    folds = []
    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(rows)), rows["target_id"].to_numpy(), groups)):
        folds.append(FoldAssignment(
            fold_id=fold_id,
            train_match_ids=tuple(sorted(set(int(m) for m in groups[train_idx]))),
            validation_match_ids=tuple(sorted(set(int(m) for m in groups[val_idx]))),
        ))
    verify_folds(folds, sorted(set(int(m) for m in groups)))
    return folds


def verify_folds(folds: list[FoldAssignment], all_ids: list[int]) -> None:
    """Each match validates exactly once, never trains and validates in one fold."""
    seen: list[int] = []
    for fold in folds:
        overlap = set(fold.train_match_ids) & set(fold.validation_match_ids)
        if overlap:
            raise LeakageError(f"fold {fold.fold_id}: matches in train and validation {sorted(overlap)}")
        if set(fold.train_match_ids) | set(fold.validation_match_ids) != set(all_ids):
            raise LeakageError(f"fold {fold.fold_id}: partitions do not cover the match set")
        seen.extend(fold.validation_match_ids)
    if sorted(seen) != sorted(all_ids):
        raise LeakageError("validation folds do not partition the match set exactly once")


def assert_no_final_test(match_ids: list[int] | set[int], final_test_ids: list[int] | set[int]) -> None:
    leaked = set(int(m) for m in match_ids) & set(int(m) for m in final_test_ids)
    if leaked:
        raise LeakageError(f"final-test matches reached the development pipeline: {sorted(leaked)}")
