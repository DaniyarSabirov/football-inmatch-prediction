r"""Authoritative 15-class event taxonomy (``scenarioD15``) and its class-id mapping.

Since 2026-11-17 every new experiment uses :math:`K = 15` target classes in exactly this
order. Class id ``c`` (0-based in code) is the position in :data:`SCENARIO_D15_CLASSES`.
In the mathematical text the classes are numbered :math:`c \in \{1, \dots, K\}`; code id
``c`` corresponds to class number ``c + 1``.

Provenance (verified 2026-11-17, see ``docs/thesis/00_repository_and_taxonomy_audit.md``):

- The class list and its order are the ``scenarioD15`` entry of
  ``configs/markov_hmm_cost_sensitive_v2.yaml`` (experiment 2). Experiment 2's prediction
  outputs, confusion matrices and per-class tables use this order.
- ``src/sequence_models/v3_taxonomy.py`` (experiment 3) defines the same tuple as
  ``D15 = FILTERED8 + (...)`` -- identical order.
- ``results/event_taxonomy_research_2026-10-27/scenarios.json`` (``D_expanded_selected``)
  defines the same SET, listed alphabetically. Its order is not used for class ids.
- The first 8 classes are the historical 8-class taxonomy (``filtered8``, the production
  ``MEANINGFUL_EVENT_TYPES``) in experiment 2's order, so 8-class ids are a prefix of the
  15-class ids.

The task is a *filtered-event* task: raw StatsBomb records whose type is not one of the 15
classes are not observations (they are skipped by definition and counted per row as
``n_skipped_raw_events``). Measured on the 334 development matches
(``results/context_aware_event_prediction_v3/taxonomy/skipped_records.csv``), the records
skipped between a current event and its target are mainly the derived companion records
``Ball Receipt*`` (331,650), ``Foul Won`` and ``Dribbled Past``, administrative and
stoppage records (``Substitution``, ``Injury Stoppage``, ``Tactical Shift``, ``Referee
Ball-Drop``, ``Player Off``/``Player On``), discipline/annotation records (``Bad
Behaviour``, ``Error``, ``Offside``, ``Own Goal For``/``Against``) and ``Shield`` (fewer
than 1,000 occurrences). Penalty-shootout periods (period 5) are excluded entirely. No two
raw types are merged into one class.

This module is deliberately free of side effects and of dependencies on other project
modules, so every model, test and report can import it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

TAXONOMY_NAME = "scenarioD15"

SCENARIO_D15_CLASSES: tuple[str, ...] = (
    "Pass",            # 0
    "Carry",           # 1
    "Ball Recovery",   # 2
    "Clearance",       # 3
    "Dribble",         # 4
    "Shot",            # 5
    "Miscontrol",      # 6
    "Dispossessed",    # 7
    "Pressure",        # 8
    "Duel",            # 9
    "Interception",    # 10
    "Block",           # 11
    "Goal Keeper",     # 12
    "Foul Committed",  # 13
    "50/50",           # 14
)

K: int = len(SCENARIO_D15_CLASSES)

#: Historical 8-class taxonomy (experiments 1-2); kept only for reproducibility.
HISTORICAL_FILTERED8: tuple[str, ...] = SCENARIO_D15_CLASSES[:8]

CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(SCENARIO_D15_CLASSES)}
ID_TO_CLASS: dict[int, str] = dict(enumerate(SCENARIO_D15_CLASSES))


class TaxonomyError(ValueError):
    """A vocabulary, class id or class name does not match the canonical taxonomy."""


def class_id(name: str) -> int:
    try:
        return CLASS_TO_ID[name]
    except KeyError:
        raise TaxonomyError(f"{name!r} is not one of the {K} canonical classes") from None


def class_name(identifier: int) -> str:
    try:
        return ID_TO_CLASS[int(identifier)]
    except KeyError:
        raise TaxonomyError(f"class id {identifier!r} outside 0..{K - 1}") from None


def assert_canonical_vocabulary(vocabulary: Sequence[str]) -> None:
    """Raise unless ``vocabulary`` is exactly the canonical 15 classes in canonical order."""
    vocabulary = tuple(vocabulary)
    if vocabulary != SCENARIO_D15_CLASSES:
        if set(vocabulary) == set(SCENARIO_D15_CLASSES):
            raise TaxonomyError("vocabulary has the canonical classes in a different order")
        raise TaxonomyError(f"vocabulary {vocabulary} is not the canonical {TAXONOMY_NAME} taxonomy")


def assert_valid_ids(ids: Iterable[int] | np.ndarray) -> None:
    arr = np.asarray(ids)
    if arr.size and (arr.min() < 0 or arr.max() >= K):
        raise TaxonomyError(f"class ids must lie in 0..{K - 1}")
