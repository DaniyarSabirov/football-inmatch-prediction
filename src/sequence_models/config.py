"""Explicit, validated experiment configuration (configs/markov_hmm_pilot.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.build_statsbomb_360_dataset import MEANINGFUL_EVENT_TYPES
from src.utils.paths import CONFIGS_DIR, PROJECT_ROOT

DEFAULT_CONFIG_PATH = CONFIGS_DIR / "markov_hmm_pilot.yaml"


class ConfigError(ValueError):
    """The experiment configuration is inconsistent or incomplete."""


@dataclass(frozen=True)
class TaskSpec:
    """One prediction-target definition.

    ``sequence_events``:
      - ``vocabulary_only``: the sequence is the subsequence of raw events whose
        type is in ``vocabulary`` (filtered-event task; other events are
        skipped by definition and counted per row as ``n_skipped_raw_events``);
      - ``all_events``: every raw event of the period is in the sequence; types
        outside the production types map to ``out_of_vocabulary_label``.
    """

    name: str
    description: str
    vocabulary: tuple[str, ...]
    sequence_events: str
    out_of_vocabulary_label: str | None
    prediction_point_types: tuple[str, ...]
    excluded_periods: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.sequence_events not in ("vocabulary_only", "all_events"):
            raise ConfigError(f"{self.name}: unknown sequence_events {self.sequence_events!r}")
        if self.sequence_events == "all_events" and not self.out_of_vocabulary_label:
            raise ConfigError(f"{self.name}: all_events needs an out_of_vocabulary_label")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ConfigError(f"{self.name}: duplicate vocabulary entries")


@dataclass(frozen=True)
class HMMConfig:
    n_states: tuple[int, ...]
    seeds: tuple[int, ...]
    init_method: str
    init_concentration: float
    init_self_transition: float
    max_iter: int
    tol: float
    pseudo_count: float
    variance_floor: float
    workers: int


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]
    output_dir: Path
    event_table_cache: Path
    production_event_types: tuple[str, ...]
    excluded_periods: tuple[int, ...]
    tasks: dict[str, TaskSpec]
    primary_task: str
    secondary_task: str
    split_seed: int
    final_test_fraction: float
    stratify_by: tuple[str, ...]
    pilot_matches: int
    pilot_extra_time_matches: int
    n_folds: int
    markov_alpha: float
    second_order_min_context_count: int
    baseline_alpha: float
    hmm: HMMConfig
    spatial_features: tuple[str, ...]
    bootstrap_replicates: int
    bootstrap_seed: int
    probability_tolerance: float


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    types = tuple(raw["production_event_types"])
    if set(types) != set(MEANINGFUL_EVENT_TYPES) or len(types) != len(MEANINGFUL_EVENT_TYPES):
        raise ConfigError(
            "production_event_types must equal MEANINGFUL_EVENT_TYPES exactly "
            f"(config {sorted(types)}, production {sorted(MEANINGFUL_EVENT_TYPES)})"
        )
    excluded_periods = tuple(int(p) for p in raw["excluded_periods"])
    tasks = {}
    for name, definition in raw["tasks"]["definitions"].items():
        oov = definition.get("out_of_vocabulary_label")
        vocabulary = types + ((oov,) if definition["sequence_events"] == "all_events" else ())
        tasks[name] = TaskSpec(
            name=name,
            description=" ".join(str(definition["description"]).split()),
            vocabulary=vocabulary,
            sequence_events=definition["sequence_events"],
            out_of_vocabulary_label=oov,
            prediction_point_types=types,
            excluded_periods=excluded_periods,
        )
    for role in ("primary", "secondary"):
        if raw["tasks"][role] not in tasks:
            raise ConfigError(f"tasks.{role} names an undefined task")
    h = raw["hmm"]
    return ExperimentConfig(
        path=Path(path),
        raw=raw,
        output_dir=PROJECT_ROOT / raw["output_dir"],
        event_table_cache=PROJECT_ROOT / raw["event_table_cache"],
        production_event_types=types,
        excluded_periods=excluded_periods,
        tasks=tasks,
        primary_task=raw["tasks"]["primary"],
        secondary_task=raw["tasks"]["secondary"],
        split_seed=int(raw["splits"]["seed"]),
        final_test_fraction=float(raw["splits"]["final_test_fraction"]),
        stratify_by=tuple(raw["splits"]["stratify_by"]),
        pilot_matches=int(raw["splits"]["pilot_matches"]),
        pilot_extra_time_matches=int(raw["splits"]["pilot_extra_time_matches"]),
        n_folds=int(raw["splits"]["n_folds"]),
        markov_alpha=float(raw["markov"]["smoothing_alpha"]),
        second_order_min_context_count=int(raw["markov"]["second_order_min_context_count"]),
        baseline_alpha=float(raw["baseline"]["smoothing_alpha"]),
        hmm=HMMConfig(
            n_states=tuple(int(k) for k in h["n_states"]),
            seeds=tuple(int(s) for s in h["seeds"]),
            init_method=str(h["init_method"]),
            init_concentration=float(h["init_concentration"]),
            init_self_transition=float(h["init_self_transition"]),
            max_iter=int(h["max_iter"]),
            tol=float(h["tol"]),
            pseudo_count=float(h["pseudo_count"]),
            variance_floor=float(h["variance_floor"]),
            workers=int(h["workers"]),
        ),
        spatial_features=tuple(raw["spatial"]["features"]),
        bootstrap_replicates=int(raw["evaluation"]["bootstrap_replicates"]),
        bootstrap_seed=int(raw["evaluation"]["bootstrap_seed"]),
        probability_tolerance=float(raw["evaluation"]["probability_tolerance"]),
    )
