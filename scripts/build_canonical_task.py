"""Build the canonical task dataset ``actions15_360current_v1`` for the development matches.

Writes (git-ignored, regenerable) to the ``dataset_dir`` of configs/actions15_360current_v1.yaml:
  - observations.parquet : one row per canonical observation (TABLE_COLUMNS + tier)
  - manifest.json        : task version, counts, dataset sha256, raw source hash, partition
Also returns the same information to callers (scripts/run_canonical_baselines.py).

The 83 final-test matches are never loaded. Existing datasets are not touched.

Usage:
    .venv/bin/python scripts/build_canonical_task.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sequence_models.canonical_task import (  # noqa: E402
    CLASSES,
    TASK_VERSION,
    build_canonical_table,
    dataset_sha256,
    load_development_raw,
    load_task_spec,
)
from src.sequence_models.raw_events import EVENTS_DIR, THREE_SIXTY_DIR  # noqa: E402

LOGGER = logging.getLogger("build_canonical_task")


def raw_source_hashes(match_ids: list[int]) -> tuple[list[dict], str]:
    """sha256 of every raw events / three-sixty JSON used, and one digest over all of them."""
    rows = []
    for mid in sorted(match_ids):
        for kind, folder in (("events", EVENTS_DIR), ("three-sixty", THREE_SIXTY_DIR)):
            rows.append({"match_id": mid, "file": f"{kind}/{mid}.json",
                         "sha256": hashlib.sha256((folder / f"{mid}.json").read_bytes()).hexdigest()})
    digest = hashlib.sha256("\n".join(f"{r['file']} {r['sha256']}" for r in rows).encode()).hexdigest()
    return rows, digest


def build(write: bool = True) -> dict:
    spec = load_task_spec()
    raw, info = load_development_raw(spec)
    part = info["partition"]
    table = build_canonical_table(raw)
    tier = {m: "tier_a" for m in part.tier_a} | {m: "tier_b" for m in part.tier_b}
    table["tier"] = table["match_id"].map(tier)
    tb = table[table["tier"] == "tier_b"]
    source_rows, source_digest = raw_source_hashes(part.development)
    manifest = {
        "task_version": TASK_VERSION, "classes": list(CLASSES),
        "config": "configs/actions15_360current_v1.yaml",
        "matches": {"development": len(part.development), "tier_a": len(part.tier_a), "tier_b": len(part.tier_b),
                    "final_test_loaded": 0},
        "companion_records_removed_periods_1_4": table.attrs["companion_rule_counts"],
        "counts_development": {"observations": int(len(table)), "transitions": int((table["target_id"] >= 0).sum()),
                               "eligible_rows": int(table["row_eligible"].sum())},
        "counts_tier_b": {"observations": int(len(tb)), "transitions": int((tb["target_id"] >= 0).sum()),
                          "eligible_rows": int(tb["row_eligible"].sum()),
                          "eligible_rows_whose_target_lacks_360": int((tb["row_eligible"] & ~tb["target_has_360"].fillna(False).astype(bool)).sum())},
        "dataset_sha256_development": dataset_sha256(table),
        "dataset_sha256_tier_b": dataset_sha256(tb),
        "raw_source_digest_development": source_digest,
        "raw_source_files": len(source_rows),
    }
    if write:
        out = spec.path("dataset_dir")
        out.mkdir(parents=True, exist_ok=True)
        table.to_parquet(out / "observations.parquet", index=False)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        LOGGER.info("canonical dataset written to %s", out.relative_to(PROJECT_ROOT))
    return {"table": table, "manifest": manifest, "source_hashes": source_rows, "partition": part,
            "folds_payload": info["folds_payload"], "spec": spec}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = build(write=True)
    print(json.dumps(result["manifest"], indent=1))


if __name__ == "__main__":
    main()
