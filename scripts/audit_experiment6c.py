"""Independent final audit of experiment 6C (LSTM-H5).

Recomputes the critical quantities with implementations written in THIS script, from raw
StatsBomb JSON, the canonical table and the stored artefacts. Not used for verification: the
production history builder (``event_history``), the 6C module's NumPy reference
(``numpy_lstm_forward``), the 6C verifier, the 6A/6B audit's reconstruction, ``canonical_task``,
``evaluation_v2`` and ``xgboost_experiment``. Production code is imported only as the object
under test: the 6C model class / optimiser factory (PyTorch states, autograd, AdamW step) and
``FitResult`` (to unpickle the per-fit checkpoints).

Every check carries an evidence type:
  independent  - recomputed here with own code
  inspection   - established by reading the cited source lines (recorded as such, not as a computation)

Sections (results in ``results/audit_experiment6c_2026-11-26/``):
  0 provenance: checksums, artefacts, configuration, executed sources, run status, workers, checkpoints
  1 raw-JSON reconstruction of 60 complete Tier B matches (own K1-K3); task rules; history hash
  2 row identity of all stored predictions; folds; final-test exclusion
  3 mathematics: own NumPy LSTM on all rows / folds / seeds; layer-wise and gate-level states vs
    PyTorch; finite differences on trained E and F; loss; AdamW closed form; statelessness
  4 nested CV: all 175 fits re-scored from their stored best-epoch weights; seeds; stopping
    subsets; selection; data sets vs 6B; training populations
  5 metrics of all 10 models (own NumPy + scikit-learn), pooled / per fold / per class / confusion
  6 paired match-level bootstrap (own implementation of the seed-7 convention); seeds
  7 decomposition by current class (Pressure); Shot
  8 historical checksums

Usage:
    .venv/bin/python scripts/audit_experiment6c.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "audit_experiment6c_2026-11-26"
RAW = ROOT / "data" / "raw" / "statsbomb" / "data"
DATASET = ROOT / "data/interim/canonical_task/actions15_360current_v1/observations.parquet"
FOLDS = ROOT / "results/markov_hmm_cost_sensitive_v2/tiers/tier_b/scenarioD15/folds.json"
SPLIT = ROOT / "results/markov_hmm_pilot/split_manifest.json"
CONFIG = ROOT / "configs/lstm_h5_experiment6c_v1.yaml"
R6C = ROOT / "results/lstm_h5_experiment6c_actions15_360current_v1_2026-11-25"
R6B = ROOT / "results/mlp_h5_experiment6b_actions15_360current_v1_2026-11-23"
A6C = ROOT / "data/interim/lstm_h5_experiment6c"
PRED = {"exp5": ROOT / "data/interim/xgboost_experiment5/oof_predictions.parquet",
        "exp5b": ROOT / "data/interim/xgboost_experiment5b/oof_predictions_h2.parquet",
        "exp6a": ROOT / "data/interim/logistic_h5_experiment6a/oof_predictions_mlr_h5.parquet",
        "exp6b": ROOT / "data/interim/mlp_h5_experiment6b/oof_predictions_mlp_h5.parquet",
        "exp6c": A6C / "oof_predictions_lstm_h5.parquet"}
MODELS = {"frequency": "exp5", "markov1": "exp5", "markov2": "exp5", "logistic_F0": "exp5", "XGB-F0": "exp5",
          "XGB-H5": "exp5", "XGB-H2": "exp5b", "MLR-H5": "exp6a", "MLP-H5": "exp6b", "LSTM-H5": "exp6c"}
CLASSES = ["Pass", "Carry", "Ball Recovery", "Clearance", "Dribble", "Shot", "Miscontrol", "Dispossessed", "Pressure",
           "Duel", "Interception", "Block", "Goal Keeper", "Foul Committed", "50/50"]          # written out, not imported
CID = {c: i for i, c in enumerate(CLASSES)}
K, START, SHOT, PRESSURE = 15, 15, 5, 8
GATES = ("i", "f", "g", "o")
CHECKS: list[dict] = []


def check(section, name, ok, value=None, evidence="independent"):
    CHECKS.append({"section": section, "evidence": evidence, "check": name, "result": "PASS" if ok else "FAIL",
                   "value": json.dumps(value, default=float) if not isinstance(value, (str, type(None))) else value})
    print(f"[{section}] {'PASS' if ok else 'FAIL'} ({evidence})  {name}  {'' if value is None else value}", flush=True)


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha_arr(a) -> str:
    a = np.asarray(a)
    return hashlib.sha256("\n".join(map(str, a)).encode() if a.dtype == object else np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_matches(ms) -> str:
    return hashlib.sha256(",".join(map(str, sorted(int(m) for m in ms))).encode()).hexdigest()


# ============================================================================ own LSTM / softmax / seeds
def sig(a):
    e = np.exp(-np.abs(a))
    return np.where(a >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def lstm(w: dict, ids: np.ndarray, states: bool = False, order=GATES):
    """Own float64 LSTM; w uses PyTorch names; ``order`` = block order assumed in weight_ih/hh."""
    n = len(ids)
    x = np.zeros((n, 5, 16))
    x[np.arange(n)[:, None], np.arange(5)[None, :], ids] = 1.0
    layers = sum(1 for k in w if k.startswith("lstm.weight_ih_l"))
    seq, rec = x, []
    for k in range(layers):
        A, U = np.asarray(w[f"lstm.weight_ih_l{k}"], float), np.asarray(w[f"lstm.weight_hh_l{k}"], float)
        b = np.asarray(w[f"lstm.bias_ih_l{k}"], float) + np.asarray(w[f"lstm.bias_hh_l{k}"], float)
        H = U.shape[1]
        blk = {g: slice(j * H, (j + 1) * H) for j, g in enumerate(order)}
        h, c = np.zeros((n, H)), np.zeros((n, H))
        r = {q: [] for q in ("i", "f", "g", "o", "c", "h")}
        for s in range(5):
            a = seq[:, s] @ A.T + h @ U.T + b
            i, f, o = sig(a[:, blk["i"]]), sig(a[:, blk["f"]]), sig(a[:, blk["o"]])
            g = np.tanh(a[:, blk["g"]])
            c = f * c + i * g
            h = o * np.tanh(c)
            for q, v in (("i", i), ("f", f), ("g", g), ("o", o), ("c", c), ("h", h)):
                r[q].append(v)
        r = {q: np.stack(v, 1) for q, v in r.items()}
        rec.append(r)
        seq = r["h"]
    z = seq[:, -1] @ np.asarray(w["out.weight"], float).T + np.asarray(w["out.bias"], float)
    return (z, rec) if states else z


def nll(z, y):
    m = z.max(1)
    return float((m + np.log(np.exp(z - m[:, None]).sum(1)) - z[np.arange(len(y)), y]).mean())


def lstm_proba_unique(w, ids):
    code = ids @ (16 ** np.arange(4, -1, -1))
    u, first, inv = np.unique(code, return_index=True, return_inverse=True)
    return softmax(lstm(w, ids[first]))[inv]


def seed_of(*parts) -> int:
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1, dtype=np.uint32)[0])


def stop_split(ms, frac, seed):
    ms = sorted(int(m) for m in ms)
    n = max(1, int(round(frac * len(ms))))
    st = set(int(m) for m in np.random.default_rng(seed).choice(ms, n, replace=False))
    return [m for m in ms if m not in st], sorted(st)


# ============================================================================ 0 provenance
def section0(cfg):
    import torch
    cs = json.loads((R6C / "checksums.json").read_text())
    cs.update(json.loads((R6C / "checksums_supplementary.json").read_text())["files"])
    bad = [k for k, v in cs.items() if sha_file(R6C / k) != v]
    check(0, f"all {len(cs)} recorded result-file checksums of the 6C result directory", not bad and len(cs) == 36, bad or len(cs))
    env = json.loads((R6C / "environment.json").read_text())
    ok = {rel.split("/")[-1]: sha_file(ROOT / rel) == h for rel, h in env["artefacts_sha256"].items()}
    check(0, "3 large ignored artefacts (predictions, npz, pt) match their run-time hashes", all(ok.values()) and len(ok) == 3, ok)
    check(0, "configuration hash = run identifiers' config_sha256 = frozen value",
          sha_file(CONFIG) == env["checkpoint_identifiers"]["config_sha256"] ==
          "3741f7398a1f7aef454c1bf15d67182cf3633f31892b464714abc9e8ac63184a")
    cur = {}
    for name, h in env["source_files_sha256"].items():
        p = next(iter(list((ROOT / "src/sequence_models").glob(name)) + list((ROOT / "scripts").glob(name))))
        cur[name] = sha_file(p) == h
    check(0, f"all {len(cur)} executed source files byte-identical to the current files", all(cur.values()) and len(cur) == 14,
          {k: v for k, v in cur.items() if not v} or len(cur))
    st = json.loads((A6C / "run_status.json").read_text())
    check(0, "run status completed; experiment and audit stages exit 0 (one attempt each); COMPLETED marker present",
          st["state"] == "completed" and [(s["stage"], s["exit_code"], s["attempt"]) for s in st["stages"]] ==
          [("experiment", 0, 0), ("independent_audit", 0, 0)] and (A6C / "COMPLETED").exists(),
          {"run_id": st["run_id"], "supervisor": st["supervisor_pid"], "caffeinate": st["caffeinate_pid"],
           "experiment": st["experiment_pid"], "audit": st["audit_pid"]})
    wk = json.loads((A6C / "workers.json").read_text())
    ws = {k: json.loads((A6C / f"worker_fold{k}_status.json").read_text()) for k in range(5)}
    check(0, "5 fold workers: exit 0, state completed, each computed only its own fold, 1 thread, deterministic, no warnings",
          wk["exit_codes"] == {str(k): 0 for k in range(5)} and all(s["state"] == "completed" and s["computed_folds"] == [k] and s["torch_threads"] == 1
                                                  and s["deterministic"] and not s["warnings"] for k, s in ws.items()),
          {k: {"pid": s["pid"], "seconds": round(s["seconds"]), "rss_mb": round(s["maxrss_mb"])} for k, s in ws.items()})
    ck = A6C / "checkpoints"
    done, fc, ip = list(ck.rglob("*.done.pt")), list(ck.rglob("fold_complete.pt")), list(ck.rglob("*.inprogress.pt"))
    per_fold_ok = all(len(list((ck / f"fold{k}").glob("*.done.pt"))) == 35 and
                      all(p.name.startswith(f"f{k}_") for p in (ck / f"fold{k}").glob("*.done.pt")) for k in range(5))
    check(0, "checkpoints: 175 completed fits (35 per fold, each in its own fold directory), 5 fold records, no in-progress file",
          len(done) == 175 and len(fc) == 5 and not ip and per_fold_ok, {"done": len(done), "fold_complete": len(fc), "inprogress": len(ip)})
    ident = env["checkpoint_identifiers"]
    pt = torch.load(A6C / "fold_models_lstm_h5.pt", weights_only=False)
    npz = np.load(A6C / "fold_models_lstm_h5.npz")
    same, ids_ok = True, True
    for k in range(5):
        rec = torch.load(ck / f"fold{k}" / "fold_complete.pt", weights_only=False)
        ids_ok &= rec["identifiers"] == ident
        for r in range(3):
            for name, v in rec["states"][r].items():
                same &= torch.equal(v, pt["states"][(k, r)][name]) and np.array_equal(v.numpy(), npz[f"fold{k}_rep{r}__{name}"])
    check(0, "fold records' identifiers = run identifiers; weights identical in fold records, models.pt and models.npz (15 models)",
          bool(same and ids_ok and pt["identifiers"] == ident))
    return env, pt, npz


# ============================================================================ 1 raw reconstruction (own implementation)
def get(d, *ks):
    for k in ks:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def is_companion(ev, prev) -> bool:
    t = get(ev, "type", "name")
    if t == "Goal Keeper" and get(ev, "goalkeeper", "type", "name") in {"Shot Faced", "Goal Conceded", "Penalty Conceded"}:
        return True                                                                           # K1
    if prev is None:
        return False
    linked = prev["id"] in (ev.get("related_events") or [])
    same_ts = prev.get("timestamp") == ev.get("timestamp")
    pt = get(prev, "type", "name")
    if (t == "Duel" and get(ev, "duel", "type", "name") == "Tackle" and linked and same_ts
            and (pt == "Dispossessed" or (pt == "Dribble" and get(prev, "dribble", "outcome", "name") == "Incomplete"))):
        return True                                                                           # K2
    return t == "50/50" and pt == "50/50" and linked and same_ts                              # K3 (no K4)


def rebuild(mid: int):
    ev = json.loads((RAW / "events" / f"{mid}.json").read_text())
    p3 = RAW / "three-sixty" / f"{mid}.json"
    usable = {f["event_uuid"] for f in (json.loads(p3.read_text()) if p3.exists() else []) if f.get("freeze_frame")}
    out, info = [], {"period5_records": sum(e["period"] == 5 for e in ev), "companions": 0, "aerial_lost_kept": 0}
    for per in sorted({e["period"] for e in ev} - {5}):
        raw = sorted((e for e in ev if e["period"] == per), key=lambda e: e["index"])
        seq, prev, comp_at = [], None, []
        for e in raw:
            if is_companion(e, prev):
                info["companions"] += 1
                comp_at.append(len(seq))
            elif get(e, "type", "name") in CID:
                seq.append(e)
                info["aerial_lost_kept"] += int(get(e, "type", "name") == "Duel" and get(e, "duel", "type", "name") == "Aerial Lost")
            prev = e
        for p, e in enumerate(seq):
            h = [CID[seq[p - l]["type"]["name"]] if p >= l else START for l in (4, 3, 2, 1)] + [CID[e["type"]["name"]]]
            nx = seq[p + 1] if p + 1 < len(seq) else None
            out.append({"match_id": mid, "period": per, "pos": p, "event_id": e["id"], "index": e["index"], "h": h,
                        "target": CID[nx["type"]["name"]] if nx else -1, "target_event_id": nx["id"] if nx else None,
                        "usable": e["id"] in usable, "eligible": nx is not None and e["id"] in usable,
                        "period_start": p == 0, "period_end": p == len(seq) - 2,
                        "repeated_ts": p > 0 and seq[p - 1]["timestamp"] == e["timestamp"],
                        "companion_in_window": any(max(0, p - 4) < c <= p for c in comp_at),
                        "history_missing_360": any(seq[p - l]["id"] not in usable for l in (1, 2, 3, 4) if p >= l)})
    return pd.DataFrame(out), info


def own_history(tb: pd.DataFrame) -> np.ndarray:
    seq, pos, sym = tb["sequence_id"].to_numpy(), tb["seq_pos"].to_numpy(), tb["symbol_id"].to_numpy().astype(np.int64)
    out = np.full((len(sym), 5), START, dtype=np.int64)
    out[:, 4] = sym
    for lag in (1, 2, 3, 4):
        idx = np.arange(len(sym)) - lag
        ok = (idx >= 0) & (pos >= lag)
        ok[ok] &= seq[idx[ok]] == seq[ok]
        out[ok, 4 - lag] = sym[idx[ok]]
    return out


def section1(tb, final, env):
    hist_all = own_history(tb)
    e = tb["row_eligible"].to_numpy()
    ids = hist_all[e]
    check(1, "own full-table history (index arithmetic within sequence) hashes to the run's training input (feature_matrix_sha256)",
          sha_arr(ids) == env["checkpoint_identifiers"]["feature_matrix_sha256"], ids.shape)
    check(1, "task rules in the canonical table: no period 5; 15 symbols in canonical order; Duel class non-empty (K4 inactive)",
          5 not in set(tb["period"]) and set(tb["symbol_id"]) == set(range(15)) and int((tb["symbol_id"] == CID["Duel"]).sum()) > 0
          and tb.groupby("sequence_id")["period"].nunique().max() == 1 and tb.groupby("sequence_id")["match_id"].nunique().max() == 1,
          {"duel_rows": int((tb["symbol_id"] == CID["Duel"]).sum())})
    rng = np.random.default_rng(20261126)
    sample = sorted(int(m) for m in rng.choice(np.unique(tb["match_id"]), 60, replace=False))
    assert not set(sample) & final
    elig = tb[e].reset_index(drop=True)
    pos_of = {x: i for i, x in enumerate(elig["event_id"])}
    recs, obs_ok, info_all = [], True, {"period5_records": 0, "companions": 0, "aerial_lost_kept": 0}
    for mid in sample:
        r, info = rebuild(mid)
        for k in info_all:
            info_all[k] += info[k]
        prod = tb[tb["match_id"] == mid]
        obs_ok &= (list(r["event_id"]) == list(prod["event_id"]) and [h[4] for h in r["h"]] == list(prod["symbol_id"])
                   and list(r["period"]) == list(prod["period"]) and list(r["index"]) == list(prod["index"])
                   and list(r["usable"]) == list(prod["usable_360"]) and list(r["eligible"]) == list(prod["row_eligible"]))
        recs.append(r[r["eligible"]])
    rec = pd.concat(recs, ignore_index=True)
    i = rec["event_id"].map(pos_of).to_numpy()
    h_ok = np.array_equal(np.array(rec["h"].tolist()), ids[i])
    t_ok = np.array_equal(rec["target"].to_numpy(), elig["target_id"].to_numpy()[i]) and \
        list(rec["target_event_id"]) == list(elig["target_event_id"].to_numpy()[i])
    strata = {"matches": len(sample), "eligible_rows": len(rec), "period_starts": int(rec.period_start.sum()),
              "with_<START>": int((np.array(rec["h"].tolist())[:, 0] == START).sum()), "period_ends": int(rec.period_end.sum()),
              "repeated_timestamps": int(rec.repeated_ts.sum()), "companion_in_window": int(rec.companion_in_window.sum()),
              "history_event_without_360": int(rec.history_missing_360.sum()), **info_all,
              **{f"rows_with_{c}": int(((np.array(rec["h"].tolist())[:, 4] == CID[c]) | (rec.target == CID[c])).sum())
                 for c in ("50/50", "Goal Keeper", "Duel", "Shot", "Block", "Foul Committed")}}
    check(1, "60 complete Tier B matches rebuilt from raw events + three-sixty JSON (own K1-K3, no K4, period 5 dropped, index "
             "order): canonical observations (ids, order, class, period, index, usable 360, eligibility) identical", obs_ok, strata)
    check(1, f"rebuilt 5-position histories of all {len(rec)} eligible rows = the run's training input, row by row", h_ok, len(rec))
    check(1, "rebuilt targets (class and target event id) = canonical targets; targets need no 360 snapshot",
          t_ok and bool((~elig["target_event_id"].isin(set(tb.loc[tb.usable_360, "event_id"]))).any()), len(rec))
    check(1, "edge cases covered (every stratum non-empty, incl. period starts/ends, <START>, companions, missing 360, rare classes)",
          all(v > 0 for k, v in strata.items() if k != "period5_records"), None)
    # future-event mutation: histories of earlier rows unchanged
    t2 = tb.copy()
    t2.loc[t2["seq_pos"] >= 200, "symbol_id"] = (t2.loc[t2["seq_pos"] >= 200, "symbol_id"] + 7) % 15
    h2 = own_history(t2)
    early = tb["seq_pos"].to_numpy() < 200
    check(1, "adversarial: mutating every event at sequence position >= 200 leaves the histories of all earlier rows unchanged",
          np.array_equal(h2[early], hist_all[early]) and not np.array_equal(h2, hist_all), int(early.sum()))
    rec.drop(columns="h").to_csv(OUT / "raw_reconstruction_60_matches.csv", index=False)
    return ids, elig


# ============================================================================ 2 row identity
def section2(tb, elig, final, folds):
    split = json.loads(SPLIT.read_text())
    tier_b = set(split["development_match_ids"]) - set(split["pilot_match_ids"])
    check(2, "Tier B = development - pilot = 304 matches = dataset matches; 83 final-test ids disjoint from data and folds",
          tier_b == set(tb["match_id"]) and len(tier_b) == 304 and len(final) == 83 and not tier_b & final
          and not set(pd.read_parquet(DATASET, columns=["match_id"])["match_id"]) & final
          and not any((set(f["train_match_ids"]) | set(f["validation_match_ids"])) & final for f in folds))
    check(2, "690,147 eligible rows; 5 outer folds partition the 304 matches without train/validation overlap",
          len(elig) == 690_147 and sorted(m for f in folds for m in f["validation_match_ids"]) == sorted(tier_b)
          and all(not set(f["train_match_ids"]) & set(f["validation_match_ids"]) for f in folds), len(elig))
    fom = {int(m): f["fold_id"] for f in folds for m in f["validation_match_ids"]}
    fold = elig["match_id"].map(fom).to_numpy()
    proba = {}
    for key, path in PRED.items():
        o = pd.read_parquet(path)
        ok = (len(o) == 690_147 and o["event_id"].is_unique and np.array_equal(o["event_id"].to_numpy(), elig["event_id"].to_numpy())
              and np.array_equal(o["match_id"].to_numpy(), elig["match_id"].to_numpy())
              and np.array_equal(o["y"].to_numpy(), elig["target_id"].to_numpy()) and np.array_equal(o["fold_id"].to_numpy(), fold))
        check(2, f"{key} predictions: event id, match, target, fold identical element-wise to the canonical rows", ok)
        for m, k in MODELS.items():
            if k == key:
                cols = [f"{m}__p{c}" for c in range(K)]
                proba[m] = o[cols].to_numpy()
        if key == "exp6b":
            proba["MLP-H5_seed1"], proba["MLP-H5_seed2"] = (o[[f"MLP-H5_seed{r}__p{c}" for c in range(K)]].to_numpy() for r in (1, 2))
        if key == "exp6c":
            proba["LSTM-H5_seed1"], proba["LSTM-H5_seed2"] = (o[[f"LSTM-H5_seed{r}__p{c}" for c in range(K)]].to_numpy() for r in (1, 2))
            check(2, "6C prediction file: exactly 15 primary + 30 seed probability columns, in canonical class order p0..p14",
                  [c for c in o.columns if c.startswith("LSTM-H5__p")] == [f"LSTM-H5__p{c}" for c in range(K)] and o.shape[1] == 4 + 45)
    lp = proba["LSTM-H5"]
    freq = np.bincount(elig["target_id"], minlength=K) / len(elig)
    check(2, "LSTM-H5 probabilities (3 seeds): finite, > 0, row sums 1 (|dev| <= 1e-12); column means track class frequencies "
             "(|diff| < 0.01: class order not permuted)",
          all(np.isfinite(proba[m]).all() and (proba[m] > 0).all() and np.abs(proba[m].sum(1) - 1).max() <= 1e-12
              for m in ("LSTM-H5", "LSTM-H5_seed1", "LSTM-H5_seed2")) and np.abs(lp.mean(0) - freq).max() < 0.01,
          float(np.abs(lp.mean(0) - freq).max()))
    return proba, fold


# ============================================================================ 3 mathematics
def section3(cfg, ids, y, fold, proba, npz, pt):
    import torch
    import torch.nn.functional as F

    from src.sequence_models import lstm_h5_experiment as lm          # object under test only
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sel = {int(k): v for k, v in json.loads(str(npz["selected"])).items()}
    cands = {c["name"]: c for c in cfg["lstm"]["candidates"]}
    check(3, "selected configurations per fold: E (32 x 2) in folds 0, 4; F (64 x 2) in folds 1-3; equal to the frozen candidates",
          [sel[f]["name"] for f in range(5)] == ["E", "F", "F", "F", "E"] and all(sel[f] == cands[sel[f]["name"]] for f in range(5)))
    worst, shapes, counts = 0.0, True, True
    for f in range(5):
        s = fold == f
        H, L = sel[f]["hidden"], sel[f]["layers"]
        for r in range(3):
            w = {k.split("__", 1)[1]: npz[k] for k in npz.files if k.startswith(f"fold{f}_rep{r}__")}
            shapes &= (set(w) == {f"lstm.{a}_l{k}" for a in ("weight_ih", "weight_hh", "bias_ih", "bias_hh") for k in range(L)}
                       | {"out.weight", "out.bias"} and w["lstm.weight_ih_l0"].shape == (4 * H, 16)
                       and all(w[f"lstm.weight_hh_l{k}"].shape == (4 * H, H) and w[f"lstm.bias_ih_l{k}"].shape == (4 * H,) for k in range(L))
                       and (L == 1 or w["lstm.weight_ih_l1"].shape == (4 * H, H)) and w["out.weight"].shape == (15, H))
            counts &= sum(v.size for v in w.values()) == sum(4 * H * m + 4 * H * H + 8 * H for m in [16] + [H] * (L - 1)) + 15 * H + 15
            col = "LSTM-H5" if r == 0 else f"LSTM-H5_seed{r}"
            worst = max(worst, float(np.abs(lstm_proba_unique(w, ids[s]) - proba[col][s]).max()))
    check(3, "stored weights: 2 recurrent layers (E: H=32, F: H=64), layer-2 input dim = H, 4H gate blocks, 15 logits; parameter "
             "count = sum over layers (4Hm + 4H^2 + 8H) + 15H + 15 (E 15,343; F 55,247)", bool(shapes and counts))
    check(3, "own float64 NumPy LSTM (h_0 = c_0 = 0, gates i,f,g,o, bias_ih + bias_hh, final h of layer 2) of every stored model "
             "reproduces the stored probabilities on all 690,147 rows x 3 seeds (|dp| < 1e-5: float32 network vs float64 reference)",
          worst < 1e-5, worst)
    # sensitivity: wrong conventions are detected
    w0 = {k.split("__", 1)[1]: npz[k] for k in npz.files if k.startswith("fold0_rep0__")}
    xs = ids[fold == 0][:3000]
    p_ok = softmax(lstm(w0, xs))
    alt = {"gate order i,g,f,o": softmax(lstm(w0, xs, order=("i", "g", "f", "o"))),
           "gate order f,i,g,o": softmax(lstm(w0, xs, order=("f", "i", "g", "o"))),
           "bias_ih only": softmax(lstm({**w0, "lstm.bias_hh_l0": 0 * w0["lstm.bias_hh_l0"], "lstm.bias_hh_l1": 0 * w0["lstm.bias_hh_l1"]}, xs)),
           "reversed position order": softmax(lstm(w0, xs[:, ::-1].copy()))}
    stored = proba["LSTM-H5"][fold == 0][:3000]
    devs = {k: float(np.abs(v - stored).max()) for k, v in alt.items()}
    check(3, "sensitivity: exchanging gate blocks, dropping bias_hh or reversing the positions moves probabilities by >> 1e-5 "
             "(the reconstruction can detect convention errors)", all(v > 1e-3 for v in devs.values())
          and float(np.abs(p_ok - stored).max()) < 1e-5, devs)
    # layer-wise and gate-level states vs PyTorch (float64)
    gw = {"h": 0.0, "c": 0.0, "o": 0.0, "z": 0.0, "layer1_h": 0.0, "layer1_c": 0.0}
    for f in range(5):
        for r in range(3):
            cand = sel[f]
            m = lm.model_from_state(cand, pt["states"][(f, r)]).double().eval()
            xi = ids[fold == f][::50][:2000]
            w = {k: v.detach().numpy().astype(float) for k, v in m.state_dict().items()}
            z_np, st = lstm(w, xi, states=True)
            idx = torch.from_numpy(xi)
            with torch.no_grad():
                seq, (h_n, c_n) = m.lstm(m.encode(idx))
                z_t = m(idx).numpy()
                l1 = torch.nn.LSTM(16, cand["hidden"], 1, batch_first=True).double()
                l1.load_state_dict({f"{a}_l0": getattr(m.lstm, f"{a}_l0") for a in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")})
                seq1, _ = l1(m.encode(idx))
                cpref = [m.lstm(m.encode(idx[:, :s]))[1][1].numpy() for s in range(1, 6)]
            gw["h"] = max(gw["h"], float(np.abs(seq.numpy() - st[-1]["h"]).max()))
            gw["c"] = max(gw["c"], max(float(np.abs(cpref[s][k] - st[k]["c"][:, s]).max()) for s in range(5) for k in range(2)))
            tc = np.tanh(st[-1]["c"])
            okm = np.abs(tc) > 1e-3
            gw["o"] = max(gw["o"], float(np.abs(seq.numpy()[okm] / tc[okm] - st[-1]["o"][okm]).max()))
            gw["z"] = max(gw["z"], float(np.abs(z_t - z_np).max()))
            gw["layer1_h"] = max(gw["layer1_h"], float(np.abs(seq1.numpy() - st[0]["h"]).max()), float(np.abs(h_n[0].numpy() - st[0]["h"][:, -1]).max()))
            gw["layer1_c"] = max(gw["layer1_c"], float(np.abs(c_n[0].numpy() - st[0]["c"][:, -1]).max()))
    check(3, "float64, all 15 stored models, 2,000 windows each: layer-1 and layer-2 h_s and c_s at s = 1..5, recovered output gate "
             "(o_s = h_s / tanh c_s) and logits of PyTorch equal the own NumPy LSTM (< 1e-12; o_s < 1e-9)",
          max(gw["h"], gw["c"], gw["z"], gw["layer1_h"], gw["layer1_c"]) < 1e-12 and gw["o"] < 1e-9, gw)
    # finite differences on trained E and F (float64), all parameter tensors, against the own NumPy loss
    fd = {}
    for f in (0, 1):
        m = lm.model_from_state(sel[f], pt["states"][(f, 0)]).double()
        m.eval()
        rows = np.flatnonzero(fold == f)[:64]
        xi, yi = ids[rows], y[rows]
        m.zero_grad()
        F.cross_entropy(m(torch.from_numpy(xi)), torch.from_numpy(yi)).backward()
        w = {k: v.detach().numpy().astype(float).copy() for k, v in m.state_dict().items()}
        rng = np.random.default_rng(f)
        err, n = 0.0, 0
        for name, p in m.named_parameters():
            flat = w[name].reshape(-1)
            for j in rng.choice(flat.size, 6, replace=False):
                old = flat[j]
                flat[j] = old + 1e-6
                lp = nll(lstm(w, xi), yi)
                flat[j] = old - 1e-6
                lq = nll(lstm(w, xi), yi)
                flat[j] = old
                err, n = max(err, abs((lp - lq) / 2e-6 - p.grad.reshape(-1)[j].item())), n + 1
        fd[sel[f]["name"]] = {"entries": n, "tensors": len(list(m.named_parameters())), "max_abs_error": err}
    check(3, "finite differences (central, h = 1e-6, float64) of the own NumPy loss vs PyTorch autograd for every parameter tensor of "
             "the trained E and F models (10 tensors each; BPTT through 5 positions and 2 layers) < 1e-7",
          all(v["max_abs_error"] < 1e-7 and v["tensors"] == 10 for v in fd.values()), fd)
    # loss: own log-sum-exp = F.cross_entropy on logits = stored log loss; no double softmax
    m = lm.model_from_state(sel[0], pt["states"][(0, 0)])
    s = np.flatnonzero(fold == 0)
    with torch.no_grad():
        zz = lm.logits_of(m, torch.from_numpy(ids[s])).double().numpy()
    own, tl = nll(zz, y[s]), float(F.cross_entropy(torch.from_numpy(zz), torch.from_numpy(y[s])))
    stored = float(-np.log(proba["LSTM-H5"][s][np.arange(len(s)), y[s]]).mean())
    dbl = float(F.cross_entropy(torch.from_numpy(softmax(zz)), torch.from_numpy(y[s])))
    check(3, "fold 0: own log-sum-exp NLL of the stored model's logits = F.cross_entropy(logits) = stored predictions' log loss; "
             "a double softmax would give a clearly different value",
          abs(own - tl) < 1e-12 and abs(own - stored) < 1e-6 and abs(dbl - own) > 0.5, {"own": own, "torch": tl, "stored": stored, "double_softmax": dbl})
    # AdamW: two steps vs closed form (own NumPy), with the production optimiser groups
    cand = cands["F"]
    torch.manual_seed(0)
    m = lm.build_model(cand, 123)
    m.train()
    opt = lm.make_optimizer(m, cand, cfg["training"])
    decayed = {n for n, p in m.named_parameters() if any(p is q for q in opt.param_groups[0]["params"])}
    lr, wd, (b1, b2), eps = cand["lr"], cand["weight_decay"], cfg["training"]["betas"], cfg["training"]["eps"]
    theta = {n: p.detach().double().numpy().copy() for n, p in m.named_parameters()}
    mom = {n: 0.0 for n in theta}
    vel = {n: 0.0 for n in theta}
    worst_step = 0.0
    m.drop.p = 0.0
    for q in (1, 2):
        rows = np.arange(512 * (q - 1), 512 * q)
        opt.zero_grad()
        F.cross_entropy(m(torch.from_numpy(ids[rows])), torch.from_numpy(y[rows])).backward()
        grads = {n: p.grad.detach().double().numpy() for n, p in m.named_parameters()}
        opt.step()
        for n in theta:
            g = grads[n]
            mom[n] = b1 * mom[n] + (1 - b1) * g
            vel[n] = b2 * vel[n] + (1 - b2) * g * g
            mh, vh = mom[n] / (1 - b1 ** q), vel[n] / (1 - b2 ** q)
            lam = wd if n in decayed else 0.0
            theta[n] = theta[n] * (1 - lr * lam) - lr * mh / (np.sqrt(vh) + eps)
        worst_step = max(worst_step, max(float(np.abs(theta[n] - p.detach().double().numpy()).max()) for n, p in m.named_parameters()))
        theta = {n: p.detach().double().numpy().copy() for n, p in m.named_parameters()}   # re-anchor on float32 state
    check(3, "AdamW: two optimiser steps of the production optimiser equal the closed form theta(1 - eta*lambda) - eta*m_hat/(sqrt(v_hat)+eps) "
             "with bias correction (< 1e-6, float32); decay only on weight_ih/weight_hh/out.weight",
          worst_step < 1e-6 and decayed == {"lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.weight_ih_l1", "lstm.weight_hh_l1", "out.weight"},
          {"max_abs_dev": worst_step, "decayed": sorted(decayed)})
    # statelessness on a trained model (float64 to remove batch-size rounding effects)
    m = lm.model_from_state(sel[1], pt["states"][(1, 0)]).double().eval()
    xi = torch.from_numpy(ids[fold == 1][:400])
    with torch.no_grad():
        full = m(xi)
        alone = torch.cat([m(xi[i:i + 1]) for i in range(400)])
        perm = torch.randperm(400, generator=torch.Generator().manual_seed(1))
        permd = m(xi[perm])
        other = xi.clone()
        other[1:] = torch.from_numpy(ids[fold == 2][:399])
        changed = m(other)
        again = m(xi)
        rev = m(torch.flip(xi, [0]))
    ok = (torch.allclose(full, alone, atol=1e-12, rtol=0) and torch.allclose(full[perm], permd, atol=1e-12, rtol=0)
          and torch.allclose(changed[0], full[0], atol=1e-12, rtol=0) and torch.equal(full, again)
          and torch.allclose(rev, torch.flip(full, [0]), atol=1e-12, rtol=0))
    check(3, "statelessness of a trained model (fold 1, F): each window alone = in batch = after permutation / reversal of the batch; "
             "unchanged when all other windows change; repeated calls identical (no carried state)", bool(ok))
    check(3, "inspection: nn.LSTM is called without (h_0, c_0) (lstm_h5_experiment.py:114) so PyTorch initialises zeros per call; the "
             "only dropout is on h_5 (l.118), nn.LSTM(dropout=0.0) (l.105); logits go to F.cross_entropy (l.267); softmax only in "
             "predict_proba (l.183); epoch loss sum(|b| L_b)/n (l.276, 283); no clipping; no detach in forward (l.112-118)",
          True, None, evidence="inspection")
    m = lm.model_from_state(sel[1], pt["states"][(1, 0)])
    xb = xi[:50]
    m.train()
    with torch.no_grad():
        stochastic = not torch.equal(m(xb), m(xb))
    m.eval()
    with torch.no_grad():
        deterministic = torch.equal(m(xb), m(xb))
    check(3, "dropout semantics of a trained model: nn.LSTM.dropout = 0, unidirectional, batch_first, output dropout p = 0.1, train "
             "mode stochastic, eval mode deterministic", m.lstm.dropout == 0.0 and m.drop.p == 0.1 and not m.lstm.bidirectional
          and m.lstm.batch_first and stochastic and deterministic)


# ============================================================================ 4 nested CV
def section4(cfg, y, match, fold, folds, ids, npz):
    import torch

    import importlib
    importlib.import_module("src.sequence_models.lstm_h5_experiment")   # defines FitResult, needed to unpickle the fit checkpoints
    tc = cfg["training"]
    base, frac = int(tc["seed"]), float(tc["stopping_fraction"])
    cands = cfg["lstm"]["candidates"]
    names = [c["name"] for c in cands]
    fl = pd.read_csv(R6C / "fit_log.csv")
    th = pd.read_csv(R6C / "training_histories.csv")
    check(4, "175 fits = 5 folds x (4 inner x 8 candidates + 3 refits); 8 frozen candidates A..H",
          len(fl) == 175 and (fl.stage == "inner").sum() == 160 and (fl.stage == "outer_refit").sum() == 5
          and (fl.stage == "robustness_refit").sum() == 10 and names == list("ABCDEFGH"))
    fo = {f["fold_id"]: f for f in folds}
    sets, rows_tab, viol_seed, viol_rows, rescore = [], [], [], [], []
    ck = A6C / "checkpoints"
    inner_pooled = {}
    t0 = time.time()
    for f in range(5):
        val = set(fo[f]["validation_match_ids"])
        inner = [g for g in range(5) if g != f]
        for held in inner:
            itr = [m for g in inner if g != held for m in fo[g]["validation_match_ids"]]
            fit_m, stop_m = stop_split(itr, frac, seed_of(base, 7, f, held + 1))
            hv = np.flatnonzero(np.isin(match, fo[held]["validation_match_ids"]))
            st_rows = np.flatnonzero(np.isin(match, stop_m))
            for ci, c in enumerate(cands):
                sets += [(f, "inner_train", sha_matches(fit_m)), (f, "inner_stop", sha_matches(stop_m)),
                         (f, "inner_validation", sha_matches(fo[held]["validation_match_ids"]))]
                row = fl[(fl.fold_id == f) & (fl.stage == "inner") & (fl.held_inner_fold == held) & (fl.candidate == c["name"])].iloc[0]
                if int(row.seed) != seed_of(base, f, held + 1, ci, 0):
                    viol_seed.append((f, held, c["name"]))
                if int(row.n_train_rows) != int(np.isin(match, fit_m).sum()) or int(row.n_stop_rows) != len(st_rows):
                    viol_rows.append((f, held, c["name"]))
                r = torch.load(ck / f"fold{f}" / f"f{f}_inner{held}_{c['name']}_r0.done.pt", weights_only=False)["result"]
                w = {k: v.numpy() for k, v in r.best_state.items()}
                ph = lstm_proba_unique(w, ids[hv])
                ps = lstm_proba_unique(w, ids[st_rows])
                ll_h = float(-np.log(ph[np.arange(len(hv)), y[hv]]).mean())
                ll_s = float(-np.log(ps[np.arange(len(st_rows)), y[st_rows]]).mean())
                hist = [e["stop_log_loss"] for e in json.loads(th.loc[row.name, "history"])]
                rescore.append({"fold_id": f, "stage": "inner", "held": held, "candidate": c["name"], "replicate": 0,
                                "held_ll_own": ll_h, "held_ll_stored": float(row.inner_validation_log_loss),
                                "stop_ll_own": ll_s, "best_stop_ll_stored": float(row.best_stop_log_loss),
                                "best_epoch": int(row.best_epoch), "argmin_history": int(np.argmin(hist)),
                                "epochs": int(row.epochs_trained), "expected_epochs": min(int(np.argmin(hist)) + int(tc["patience"]), int(tc["max_epochs"])),
                                "result_matches_log": r.best_epoch == row.best_epoch and r.seed == row.seed and r.candidate == c})
                inner_pooled.setdefault((f, c["name"]), [0.0, 0])
                inner_pooled[(f, c["name"])][0] += ll_h * len(hv)
                inner_pooled[(f, c["name"])][1] += len(hv)
        fit_m, stop_m = stop_split(fo[f]["train_match_ids"], frac, seed_of(base, 7, f, 0))
        st_rows = np.flatnonzero(np.isin(match, stop_m))
        va = np.flatnonzero(fold == f)
        sel = min(names, key=lambda c: (round(inner_pooled[(f, c)][0] / inner_pooled[(f, c)][1], 12), names.index(c)))
        ci = names.index(sel)
        for rep in range(3):
            sets += [(f, "outer_train", sha_matches(fit_m)), (f, "outer_stop", sha_matches(stop_m)),
                     (f, "outer_validation_predict", sha_matches(val))]
            row = fl[(fl.fold_id == f) & (fl.stage != "inner") & (fl.replicate == rep)].iloc[0]
            if int(row.seed) != seed_of(base, f, 0, ci, rep) or row.candidate != sel:
                viol_seed.append((f, "refit", rep))
            if int(row.n_train_rows) != int(np.isin(match, fit_m).sum()) or int(row.n_stop_rows) != len(st_rows):
                viol_rows.append((f, "refit", rep))
            w = {k.split("__", 1)[1]: npz[k] for k in npz.files if k.startswith(f"fold{f}_rep{rep}__")}
            r = torch.load(ck / f"fold{f}" / f"f{f}_outer_{sel}_r{rep}.done.pt", weights_only=False)["result"]
            same_w = all(np.array_equal(r.best_state[k].numpy(), w[k]) for k in w)
            ps = lstm_proba_unique(w, ids[st_rows])
            pv = lstm_proba_unique(w, ids[va])
            hist = [e["stop_log_loss"] for e in json.loads(th.loc[row.name, "history"])]
            rescore.append({"fold_id": f, "stage": row.stage, "held": -1, "candidate": sel, "replicate": rep,
                            "held_ll_own": float(-np.log(pv[np.arange(len(va)), y[va]]).mean()),
                            "held_ll_stored": float(row.outer_validation_log_loss),
                            "stop_ll_own": float(-np.log(ps[np.arange(len(st_rows)), y[st_rows]]).mean()),
                            "best_stop_ll_stored": float(row.best_stop_log_loss), "best_epoch": int(row.best_epoch),
                            "argmin_history": int(np.argmin(hist)), "epochs": int(row.epochs_trained),
                            "expected_epochs": min(int(np.argmin(hist)) + int(tc["patience"]), int(tc["max_epochs"])),
                            "result_matches_log": bool(same_w and r.best_epoch == row.best_epoch and r.seed == row.seed)})
            rows_tab.append({"fold_id": f, "replicate": rep, "outer_train_matches": len(fo[f]["train_match_ids"]),
                             "outer_train_rows": int(np.isin(match, fo[f]["train_match_ids"]).sum()),
                             "update_matches": len(fit_m), "update_rows": int(np.isin(match, fit_m).sum()),
                             "stop_matches": len(stop_m), "stop_rows": len(st_rows),
                             "update_share_of_rows": np.isin(match, fit_m).sum() / np.isin(match, fo[f]["train_match_ids"]).sum(),
                             "overlap_update_stop": len(set(fit_m) & set(stop_m)), "overlap_with_outer_validation": len((set(fit_m) | set(stop_m)) & val),
                             "validation_matches": len(val), "validation_rows": len(va), "best_epoch": int(row.best_epoch),
                             "best_stop_log_loss": float(row.best_stop_log_loss)})
    rs = pd.DataFrame(rescore)
    rs.to_csv(OUT / "fit_rescoring.csv", index=False)
    pop = pd.DataFrame(rows_tab)
    pop.to_csv(OUT / "training_population.csv", index=False)
    print(f"  re-scored 175 fits in {time.time() - t0:.0f} s", flush=True)
    check(4, "fit seeds = SeedSequence(20261123, outer, inner+1 | 0, candidate index, replicate) for all 175 fits; refit candidate = selection",
          not viol_seed, viol_seed[:5])
    check(4, "training / stopping row counts of all 175 fits = own match-level stopping split (10 %, seed (base, 7, outer, inner+1 | 0))",
          not viol_rows, viol_rows[:5])
    dh = float((rs.held_ll_own - rs.held_ll_stored).abs().max())
    ds = float((rs.stop_ll_own - rs.best_stop_ll_stored).abs().max())
    check(4, "every fit's STORED best-epoch weights, re-scored with the own NumPy LSTM, reproduce its recorded held-out log loss "
             "(inner validation / outer validation) and its recorded best stopping loss (< 1e-6): the stored weights are the best-epoch "
             "weights", dh < 1e-6 and ds < 1e-6 and bool(rs.result_matches_log.all()), {"held": dh, "stop": ds})
    check(4, "best epoch = first argmin of the stopping curve (epoch 0 included as incumbent) and epochs = min(best + 5, 60) for all fits; "
             "no fit chose epoch 0", bool((rs.best_epoch == rs.argmin_history).all() and (rs.epochs == rs.expected_epochs).all()),
          {"best_epoch_0": int((rs.best_epoch == 0).sum()), "capped_at_60": int((rs.epochs == 60).sum())})
    hc = pd.read_csv(R6C / "hyperparameter_choices.csv").set_index("fold_id")
    own_sel = {f: min(names, key=lambda c: (round(inner_pooled[(f, c)][0] / inner_pooled[(f, c)][1], 12), names.index(c))) for f in range(5)}
    margins = {f: sorted(inner_pooled[(f, c)][0] / inner_pooled[(f, c)][1] for c in names)[:2] for f in range(5)}
    check(4, "selection recomputed from own held-inner-fold log losses (pooled over the 4 held folds) = stored choice (E, F, F, F, E)",
          all(own_sel[f] == hc.at[f, "candidate"] for f in range(5)),
          {f: {"selected": own_sel[f], "margin_to_second": round(v[1] - v[0], 6)} for f, v in margins.items()})
    da6c = pd.read_csv(R6C / "data_access_audit.csv")
    da6b = pd.read_csv(R6B / "data_access_audit.csv")
    key = lambda d: sorted(zip(d.outer_fold.astype(int), d.purpose, d.match_ids_sha256))   # noqa: E731
    check(4, "own reconstruction of all data sets: 75 distinct match sets, 525 uses; identical to the 6C AND the 6B access audits",
          sorted(sets) == key(da6c) == key(da6b) and len(set(sets)) == 75 and len(sets) == 525,
          {"distinct": len(set(sets)), "uses": len(sets)})
    check(4, "outer refits: update and stopping matches disjoint, inside outer training, never outer validation; update share 89.8-90.5 %",
          bool((pop.overlap_update_stop == 0).all() and (pop.overlap_with_outer_validation == 0).all()
               and pop.update_share_of_rows.between(0.897, 0.906).all()),
          pop[pop.replicate == 0][["fold_id", "update_matches", "update_rows", "stop_matches", "stop_rows", "update_share_of_rows"]].round(4).to_dict("records"))
    rob = pd.read_csv(R6C / "seed_robustness.csv")
    check(4, "primary prediction = replicate 0 (prespecified); replicates were not chosen by result (primary is not the best seed)",
          bool(rob.loc[rob.replicate == 0, "primary"].iat[0]) and rob.pooled_log_loss.idxmin() != rob.index[rob.replicate == 0][0],
          rob[["replicate", "pooled_log_loss"]].round(6).to_dict("records"))


# ============================================================================ 5 metrics
def metrics(y, p):
    n = len(y)
    d = p.argmax(1)
    cm = np.bincount(y * K + d, minlength=K * K).reshape(K, K)
    tp, sup, prd = np.diag(cm).astype(float), cm.sum(1).astype(float), cm.sum(0).astype(float)
    prec = np.divide(tp, prd, out=np.zeros(K), where=prd > 0)
    rec = np.divide(tp, sup, out=np.zeros(K), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    oh = np.zeros_like(p)
    oh[np.arange(n), y] = 1

    def ap(score, pos):
        o = np.argsort(-score, kind="stable")
        s, t = score[o], pos[o].astype(float)
        tpc, fpc = np.cumsum(t), np.cumsum(1 - t)
        last = np.r_[np.flatnonzero(np.diff(s) != 0), n - 1]
        tpc, fpc = tpc[last], fpc[last]
        return float(np.sum(np.diff(np.r_[0, tpc / tpc[-1]]) * tpc / (tpc + fpc)))
    return {"log_loss": float(-np.log(p[np.arange(n), y]).mean()), "brier": float(((p - oh) ** 2).sum(1).mean()),
            "accuracy": float(tp.sum() / n), "macro_f1": float(f1.mean()), "weighted_f1": float((f1 * sup).sum() / n),
            "shot_average_precision": ap(p[:, SHOT], y == SHOT), "shot_recall": float(rec[SHOT]), "shot_precision": float(prec[SHOT]),
            "shot_tp": int(tp[SHOT]), "shot_predicted": int(prd[SHOT]), "shot_support": int(sup[SHOT])}, cm, prec, rec, f1, sup


def section5(y, fold, proba):
    mc = pd.read_csv(R6C / "model_comparison.csv").set_index("model")
    bf = pd.read_csv(R6C / "metrics_by_fold.csv").set_index(["model", "fold_id"])
    pc = pd.read_csv(R6C / "per_class_metrics.csv").set_index(["model", "class_id"])
    cmt = pd.read_csv(R6C / "confusion_matrices.csv")
    keys = ["log_loss", "brier", "accuracy", "macro_f1", "weighted_f1", "shot_average_precision", "shot_recall", "shot_precision"]
    rows, worst, worst_f, sk = [], 0.0, 0.0, 0.0
    for m in MODELS:
        p = proba[m]
        mt, cm, prec, rec, f1, sup = metrics(y, p)
        rows.append({"model": m, **mt})
        worst = max(worst, max(abs(mt[k] - mc.at[m, k]) for k in keys))
        sk = max(sk, abs(mt["log_loss"] - log_loss(y, p, labels=list(range(K)))),
                 abs(mt["macro_f1"] - f1_score(y, p.argmax(1), labels=list(range(K)), average="macro", zero_division=0)),
                 abs(mt["weighted_f1"] - f1_score(y, p.argmax(1), labels=list(range(K)), average="weighted", zero_division=0)),
                 abs(mt["shot_average_precision"] - average_precision_score(y == SHOT, p[:, SHOT])))
        q = pc.loc[m]
        worst = max(worst, *(float(np.abs(q[c].to_numpy() - v).max()) for c, v in (("precision", prec), ("recall", rec), ("f1", f1), ("support", sup))))
        c = cmt[cmt.model == m].pivot(index="true_class", columns="predicted_class", values="count").loc[CLASSES, CLASSES].to_numpy()
        worst = max(worst, float(np.abs(c - cm).max()))
        for f in range(5):
            s = fold == f
            mf = metrics(y[s], p[s])[0]
            worst_f = max(worst_f, max(abs(mf[k] - bf.loc[(m, f), k]) for k in keys if k != "shot_precision"))
        rows[-1]["pooled_minus_mean_of_fold_log_losses"] = mt["log_loss"] - np.mean([bf.loc[(m, f), "log_loss"] for f in range(5)])
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "metrics_recomputed.csv", index=False)
    check(5, "pooled metrics, per-class precision/recall/F1/support and confusion matrices of all 10 models recomputed (own NumPy) = "
             "stored 6C tables (< 1e-9)", worst < 1e-9, worst)
    check(5, "per-fold metrics recomputed = stored (< 1e-9); pooled metrics are computed on all rows (pooled != mean of fold values)",
          worst_f < 1e-9 and (t.pooled_minus_mean_of_fold_log_losses.abs() > 0).all(), worst_f)
    check(5, "own implementations agree with scikit-learn (log loss, macro/weighted F1, average precision; < 1e-9)", sk < 1e-9, sk)
    ref = {"logistic_F0": (1.15850, 0.6505, 0.1004, 0.5497, 0.0336), "markov2": (1.06807, 0.6659, 0.1204, 0.5731, 0.0527),
           "XGB-H2": (1.06761, 0.6659, 0.1208, 0.5731, 0.0529), "XGB-H5": (1.05185, 0.6668, 0.1325, 0.5778, 0.0769),
           "MLR-H5": (1.10297, 0.6491, 0.1219, 0.5579, 0.0585), "MLP-H5": (1.05267, 0.6668, 0.1302, 0.5762, 0.0757),
           "LSTM-H5": (1.05147, 0.6669, 0.1298, 0.5767, 0.0808)}
    tt = t.set_index("model")
    dev = {m: max(abs(round(tt.at[m, "log_loss"], 5) - v[0]), *(abs(round(tt.at[m, k], 4) - v[j + 1]) for j, k in
                  enumerate(("accuracy", "macro_f1", "weighted_f1", "shot_average_precision")))) for m, v in ref.items()}
    check(5, "researcher's rounded reference table reproduced by the own recomputation (all 35 values equal after rounding)",
          all(v < 1e-9 for v in dev.values()), dev)
    lt = tt.loc["LSTM-H5"]
    check(5, "LSTM-H5 primary Shot: support 7,212; 28 correct of 75 argmax Shot predictions; recall 0.388 %; precision 37.3 %",
          lt.shot_support == 7212 and lt.shot_tp == 28 and lt.shot_predicted == 75,
          {"recall": lt.shot_recall, "precision": lt.shot_precision, "xgb_tp": int(tt.at["XGB-H5", "shot_tp"]), "mlp_tp": int(tt.at["MLP-H5", "shot_tp"])})
    print(t[["model", "log_loss", "brier", "accuracy", "macro_f1", "weighted_f1", "shot_average_precision", "shot_tp", "shot_predicted"]].round(5).to_string())
    return t


# ============================================================================ 6 bootstrap
def section6(y, match, fold, proba):
    mids, inv = np.unique(match, return_inverse=True)
    rng = np.random.default_rng(7)
    mult = np.stack([np.bincount(rng.integers(0, len(mids), len(mids)), minlength=len(mids)) for _ in range(1000)]).astype(float)
    n_g = np.bincount(inv).astype(float)
    L = {m: -np.log(p[np.arange(len(y)), y]) for m, p in proba.items()}
    st = pd.read_csv(R6C / "paired_differences.csv").query("metric == 'log_loss'").set_index("reference")
    out = []
    for r in ("MLP-H5", "XGB-H5", "MLR-H5", "markov2", "XGB-H2", "logistic_F0", "markov1", "XGB-F0"):
        d = L["LSTM-H5"] - L[r]
        boot = mult @ np.bincount(inv, weights=d) / (mult @ n_g)          # ratio of resampled sums (match sizes vary)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        fs = [float(d[fold == f].mean()) for f in range(5)]
        row = st.loc[r]
        ok = (abs(d.mean() - row.difference) < 1e-12 and abs(lo - row.ci_low) < 1e-12 and abs(hi - row.ci_high) < 1e-12
              and sum(v < 0 for v in fs) == row.folds_negative and abs(d.mean() / L[r].mean() - row.relative_difference) < 1e-12)
        out.append({"reference": r, "difference": d.mean(), "relative_percent": 100 * d.mean() / L[r].mean(), "ci_low": lo, "ci_high": hi,
                    "folds_lstm_better": sum(v < 0 for v in fs), "fold_differences": json.dumps([round(v, 6) for v in fs]),
                    "reproduces_stored": ok})
        check(6, f"LSTM-H5 - {r}: mean d_i, relative difference, 95 % CI (1,000 match resamples, seed 7, ratio of sums) and fold signs "
                 f"reproduce the stored values exactly (1e-12)", ok,
              {"diff": round(d.mean(), 6), "rel_%": round(100 * d.mean() / L[r].mean(), 4), "ci": [round(lo, 6), round(hi, 6)],
               "folds_better": sum(v < 0 for v in fs)})
    pd.DataFrame(out).to_csv(OUT / "paired_differences_recomputed.csv", index=False)
    seeds = []
    for r in (0, 1, 2):
        lc, mc = ("LSTM-H5", "MLP-H5") if r == 0 else (f"LSTM-H5_seed{r}", f"MLP-H5_seed{r}")
        d = L[lc] - L[mc]
        boot = mult @ np.bincount(inv, weights=d) / (mult @ n_g)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        seeds.append({"replicate": r, "lstm_log_loss": L[lc].mean(), "mlp_log_loss": L[mc].mean(), "diff_vs_mlp_same_replicate": d.mean(),
                      "ci_low": lo, "ci_high": hi, "diff_vs_xgb_h5": L[lc].mean() - L["XGB-H5"].mean(),
                      "folds_better_vs_mlp": int(sum(d[fold == f].mean() < 0 for f in range(5)))})
    sd = pd.DataFrame(seeds)
    sd.to_csv(OUT / "seed_comparisons.csv", index=False)
    check(6, "seed-matched comparisons (LSTM replicate r vs MLP replicate r, same training-seed derivation): all 3 negative with CIs "
             "excluding 0 (descriptive; replicates 1-2 are not primary)", bool((sd.ci_high < 0).all()),
          sd[["replicate", "diff_vs_mlp_same_replicate", "ci_low", "ci_high", "diff_vs_xgb_h5"]].round(6).to_dict("records"))


# ============================================================================ 7 decomposition, Shot
def section7(y, cur, proba):
    n = len(y)
    refs = ("MLP-H5", "XGB-H5", "MLR-H5", "markov2")
    L = {m: -np.log(proba[m][np.arange(n), y]) for m in ("LSTM-H5",) + refs}
    out = []
    for r in refs:
        d = L["LSTM-H5"] - L[r]
        for c in range(K):
            s = cur == c
            out.append({"reference": r, "current_class": CLASSES[c], "n_rows": int(s.sum()), "share_of_rows": s.mean(),
                        "conditional_mean_difference": d[s].mean(), "contribution": d[s].sum() / n,
                        "share_rows_lstm_better": float((d[s] < 0).mean())})
    t = pd.DataFrame(out)
    t.to_csv(OUT / "decomposition_by_current_class.csv", index=False)
    stored = pd.read_csv(R6C / "nll_difference_by_stratum.csv").query("stratum == 'current_class'")
    ok = set(stored.reference) == set(refs) and len(stored) == len(t) and all(abs(t[(t.reference == r.reference) & (t.current_class == r.level)].contribution.iat[0] - r.contribution) < 1e-12
             for r in stored.itertuples())
    tot = t.groupby("reference").contribution.sum()
    pr = t[t.current_class == "Pressure"].set_index("reference")
    check(7, "decomposition by current class recomputed = stored strata; contributions sum to the pooled differences (< 1e-12)",
          ok and all(abs(tot[r] - (L["LSTM-H5"] - L[r]).mean()) < 1e-12 for r in tot.index),
          {r: {"pressure_rows": int(pr.at[r, "n_rows"]), "share": round(pr.at[r, "share_of_rows"], 4),
               "pressure_conditional_mean": round(pr.at[r, "conditional_mean_difference"], 5),
               "pressure_contribution": round(pr.at[r, "contribution"], 6), "other_classes": round(tot[r] - pr.at[r, "contribution"], 6),
               "total": round(tot[r], 6)} for r in ("MLP-H5", "XGB-H5", "MLR-H5")})


# ============================================================================ 8 historical checksums
def section8():
    for d in ["canonical_baselines_actions15_360current_v1_2026-11-19", "xgboost_experiment5_actions15_360current_v1_2026-11-20",
              "xgboost_experiment5b_h2_actions15_360current_v1_2026-11-21", "logistic_h5_experiment6a_actions15_360current_v1_2026-11-23",
              "mlp_h5_experiment6b_actions15_360current_v1_2026-11-23", "audit_experiments6a_6b_2026-11-24"]:
        D = ROOT / "results" / d
        c = json.loads((D / "checksums.json").read_text())
        if (D / "checksums_supplementary.json").exists():
            c.update(json.loads((D / "checksums_supplementary.json").read_text())["files"])
        bad = [k for k, v in c.items() if sha_file(D / k) != v]
        check(8, f"historical checksums unchanged: {d}", not bad, bad or len(c))
    env = json.loads((R6C / "environment.json").read_text())
    before = env["historical_verification"]["before"]["files"]
    check(8, "historical predictions, models, canonical dataset and manifest unchanged since before the 6C run",
          all(sha_file(ROOT / k) == v for k, v in before.items()), len(before))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    cfg = yaml.safe_load(CONFIG.read_text())
    env, pt, npz = section0(cfg)
    final = set(int(m) for m in json.loads(SPLIT.read_text())["final_test_match_ids"])     # ids only; no final-test content read
    table = pd.read_parquet(DATASET)
    tb = table[table["tier"] == "tier_b"].reset_index(drop=True)
    del table
    folds = json.loads(FOLDS.read_text())["scenarioD15"]
    ids, elig = section1(tb, final, env)
    proba, fold = section2(tb, elig, final, folds)
    y, match = elig["target_id"].to_numpy().astype(np.int64), elig["match_id"].to_numpy()
    section3(cfg, ids, y, fold, proba, npz, pt)
    section4(cfg, y, match, fold, folds, ids, npz)
    section5(y, fold, proba)
    section6(y, match, fold, proba)
    section7(y, ids[:, 4], proba)
    section8()
    n_fail = sum(c["result"] == "FAIL" for c in CHECKS)
    pd.DataFrame(CHECKS).to_csv(OUT / "audit_checks.csv", index=False)
    (OUT / "audit_summary.json").write_text(json.dumps({
        "verdict_of_checks": "PASS" if not n_fail else "FAIL", "n_checks": len(CHECKS), "n_fail": n_fail,
        "by_evidence": pd.DataFrame(CHECKS).groupby("evidence").size().to_dict(), "runtime_seconds": round(time.time() - t0)}, indent=1))
    print(f"\n{len(CHECKS)} checks, {n_fail} FAIL, {time.time() - t0:.0f} s")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
