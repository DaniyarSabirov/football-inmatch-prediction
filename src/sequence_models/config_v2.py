"""Validated configuration for experiment 2 (configs/markov_hmm_cost_sensitive_v2.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.build_statsbomb_360_dataset import MEANINGFUL_EVENT_TYPES
from src.utils.paths import CONFIGS_DIR, PROJECT_ROOT

from .config import ConfigError, TaskSpec
from .cost_sensitive import CostScheme

DEFAULT_CONFIG_PATH = CONFIGS_DIR / "markov_hmm_cost_sensitive_v2.yaml"
SCENARIO_D_RESEARCH = PROJECT_ROOT / "results" / "event_taxonomy_research_2026-10-27" / "scenarios.json"


@dataclass(frozen=True)
class ConfigV2:
    path: Path
    raw: dict[str, Any]
    output_dir: Path
    split_manifest: Path
    event_table_cache: Path
    tasks: dict[str, TaskSpec]
    cost_schemes: dict[str, CostScheme]


def load_config_v2(path: Path = DEFAULT_CONFIG_PATH) -> ConfigV2:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    excluded = tuple(int(p) for p in raw["excluded_periods"])
    tasks = {}
    for name, d in raw["taxonomies"].items():
        types = tuple(d["event_types"])
        tasks[name] = TaskSpec(name=name, description=" ".join(str(d["description"]).split()), vocabulary=types,
                               sequence_events="vocabulary_only", out_of_vocabulary_label=None,
                               prediction_point_types=types, excluded_periods=excluded)
    if set(tasks["filtered8"].vocabulary) != set(MEANINGFUL_EVENT_TYPES):
        raise ConfigError("filtered8 must equal the production MEANINGFUL_EVENT_TYPES")
    import json
    research = json.loads(SCENARIO_D_RESEARCH.read_text(encoding="utf-8"))["D_expanded_selected"]
    if set(tasks["scenarioD15"].vocabulary) != set(research["input_types"]) or \
            set(research["input_types"]) != set(research["target_types"]):
        raise ConfigError("scenarioD15 must equal the documented Scenario D input and target sets")
    schemes = {n: CostScheme(name=n, gamma=float(s["gamma"]), cap=float(s["cap"]),
                             false_alarm=float(s["false_alarm"]), description=s["description"])
               for n, s in raw["cost"]["schemes"].items()}
    if schemes["uniform"].gamma != 0:
        raise ConfigError("the uniform reference scheme must have gamma 0")
    return ConfigV2(path=Path(path), raw=raw, output_dir=PROJECT_ROOT / raw["output_dir"],
                    split_manifest=PROJECT_ROOT / raw["experiment1_split_manifest"],
                    event_table_cache=PROJECT_ROOT / raw["event_table_cache"], tasks=tasks, cost_schemes=schemes)
