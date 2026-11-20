"""Canonical baselines on actions15_360current_v1: frequency, Markov 1, Markov 2, logistic F0.

Tier B (304 development matches), experiment 2's 5 match-grouped folds, Protocol B rows,
nested hyperparameter selection. Final-test matches are never loaded. The result directory is
never overwritten unless --overwrite is passed deliberately.

Usage (about 2-3 minutes):
    .venv/bin/python scripts/run_canonical_baselines.py
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_canonical_task import build  # noqa: E402

from src.sequence_models.canonical_baselines import Rows, evaluate, run_cv  # noqa: E402
from src.sequence_models.canonical_task import CLASSES, TASK_VERSION  # noqa: E402
from src.sequence_models.experiment import environment_info, sha256_file, write_json  # noqa: E402
from src.sequence_models.splits import FoldAssignment, assert_no_final_test, verify_folds  # noqa: E402

LOGGER = logging.getLogger("canonical_baselines")
CONFIG = PROJECT_ROOT / "configs" / "canonical_baselines_v1.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.time()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    out = PROJECT_ROOT / cfg["output_dir"]
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{cfg['output_dir']} exists; pass --overwrite to replace it deliberately")
    out.mkdir(parents=True, exist_ok=True)

    built = build(write=True)
    part, spec = built["partition"], built["spec"]
    folds = [FoldAssignment(int(f["fold_id"]), tuple(f["train_match_ids"]), tuple(f["validation_match_ids"]))
             for f in built["folds_payload"]["scenarioD15"]]
    verify_folds(folds, part.tier_b)
    for f in folds:
        assert_no_final_test(f.train_match_ids + f.validation_match_ids, part.final_test)
    table = built["table"]
    tb = table[table["tier"] == "tier_b"].reset_index(drop=True)
    rows = Rows.from_table(tb)
    LOGGER.info("%s: %d Tier B eligible rows in %d matches", TASK_VERSION, len(rows.y), len(set(rows.match_id)))

    cv = run_cv(rows, folds, cfg)
    tables = evaluate(rows, cv, cfg)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame(cv.choices).to_csv(out / "hyperparameter_choices.csv", index=False)
    pd.DataFrame(cv.fits).to_csv(out / "logistic_fits.csv", index=False)
    pd.DataFrame(cv.timings).to_csv(out / "timings.csv", index=False)
    pd.DataFrame(cv.counts).to_csv(out / "fold_counts.csv", index=False)
    pd.DataFrame(built["source_hashes"]).to_csv(out / "raw_source_hashes.csv", index=False)

    shutil.copy(CONFIG, out / "experiment_config.yaml")
    shutil.copy(PROJECT_ROOT / "configs" / f"{TASK_VERSION}.yaml", out / "task_config.yaml")
    write_json(out / "split_manifest.json", {
        "source_split_manifest": "results/markov_hmm_pilot/split_manifest.json",
        "source_split_manifest_sha256": sha256_file(PROJECT_ROOT / "results/markov_hmm_pilot/split_manifest.json"),
        "folds_file": spec.raw["partition"]["folds_file"],
        "folds_file_sha256": sha256_file(PROJECT_ROOT / spec.raw["partition"]["folds_file"]),
        "fold_method": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261027), groups = match (experiment 2)",
        "final_test_matches": len(part.final_test), "final_test_loaded": 0,
        "folds": [{"fold_id": f.fold_id, "train_match_ids": list(f.train_match_ids),
                   "validation_match_ids": list(f.validation_match_ids)} for f in folds]})
    write_json(out / "task_manifest.json", {**built["manifest"], "class_ids": {c: i for i, c in enumerate(CLASSES)}})
    env = environment_info()
    env.pop("platform", None)  # machine-specific; versions below suffice for reproduction
    write_json(out / "environment.json", {**env, "runtime_seconds": time.time() - started,
                                          "random_seeds": {"folds": 20261027, "bootstrap": cfg["evaluation"]["bootstrap_seed"],
                                                           "models": "none (count models and lbfgs are deterministic)"},
                                          "commands": ["python scripts/build_canonical_task.py",
                                                       "python scripts/run_canonical_baselines.py --overwrite"]})
    write_json(out / "checksums.json", {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob("*"))
                                        if p.is_file() and p.name not in ("checksums.json", "README.md")})
    LOGGER.info("done in %.0f s -> %s", time.time() - started, cfg["output_dir"])


if __name__ == "__main__":
    main()
