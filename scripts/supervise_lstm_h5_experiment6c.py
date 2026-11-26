"""Supervisor for experiment 6C: runs the experiment and its independent audit as one lifecycle,
independently of any interactive session (same architecture as the 6B supervisor).

- ``--detach`` re-launches the supervisor in a NEW SESSION (no controlling terminal, immune to
  SIGHUP when the launching terminal / tool session ends; stdin from /dev/null) and returns.
- A non-blocking ``fcntl`` lock (released by the kernel if the process dies) prevents a second
  supervisor; the runner and every fold worker hold their own locks as well.
- ``caffeinate -d -i -w <supervisor pid>`` prevents idle DISPLAY sleep and idle SYSTEM sleep for
  exactly the supervisor's lifetime (training, evaluation and audit) and exits with it. No power,
  screensaver, lock-screen or security setting is changed.
- Stages: runner (full or ``--smoke``), then the independent verifier (full mode only). Exit
  codes, PIDs, timestamps and log paths are written atomically to ``run_status.json``;
  ``COMPLETED`` is written only if every stage exited 0.
- Recovery: if the runner is killed by a signal (negative exit code, e.g. an out-of-memory kill),
  it is restarted at most ``MAX_RESTARTS`` times; it resumes from its atomic checkpoints and
  refuses incompatible ones. Ordinary failures (positive exit codes = failed integrity checks)
  are never retried.

Inspect a run with ``--status`` (never starts anything).

Usage:
    .venv/bin/python scripts/supervise_lstm_h5_experiment6c.py --detach            # full experiment + audit
    .venv/bin/python scripts/supervise_lstm_h5_experiment6c.py --detach --smoke    # lifecycle test (smoke only)
    .venv/bin/python scripts/supervise_lstm_h5_experiment6c.py --status
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ART = PROJECT_ROOT / "data" / "interim" / "lstm_h5_experiment6c"
MAX_RESTARTS = 2


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def write_status(path: Path, status: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(status, indent=1))
    os.replace(tmp, path)


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def show_status(tag: str) -> int:
    p = ART / f"run_status{tag}.json"
    if not p.exists():
        print("no run status file")
        return 1
    s = json.loads(p.read_text())
    s["supervisor_alive"] = alive(s.get("supervisor_pid"))
    s["caffeinate_alive"] = alive(s.get("caffeinate_pid"))
    s["current_stage_alive"] = alive(s.get("current_pid"))
    s["completed_marker"] = (ART / f"COMPLETED{tag}").exists()
    if not tag and (ART / "workers.json").exists():
        w = json.loads((ART / "workers.json").read_text())
        s["workers"] = {k: {**v, "alive": alive(v["pid"])} for k, v in w["workers"].items()}
        s["worker_exit_codes"] = w["exit_codes"]
        done = sorted(int(d.name[4:]) for d in (ART / "checkpoints").glob("fold*") if (d / "fold_complete.pt").exists()) \
            if (ART / "checkpoints").exists() else []
        s["completed_folds"] = done
        s["completed_fits"] = len(list((ART / "checkpoints").rglob("*.done.pt"))) if (ART / "checkpoints").exists() else 0
    print(json.dumps(s, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if a.smoke else ""
    if a.status:
        return show_status(tag)
    if a.detach:
        log = open(ART / f"supervisor{tag}.log", "a")
        p = subprocess.Popen([PY, str(Path(__file__).resolve())] + (["--smoke"] if a.smoke else []), cwd=PROJECT_ROOT,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                             close_fds=True)
        print(f"supervisor started: pid {p.pid} (new session); status: {ART / f'run_status{tag}.json'}")
        return 0

    lock = open(ART / f"supervisor{tag}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"{now()} another experiment-6C supervisor is running; refusing to start a duplicate", flush=True)
        return 3
    status_path = ART / f"run_status{tag}.json"
    marker = ART / f"COMPLETED{tag}"
    if marker.exists() and not a.smoke:
        print(f"{now()} {marker} exists: experiment already complete; nothing started", flush=True)
        return 4
    marker.unlink(missing_ok=True)
    caf = subprocess.Popen(["caffeinate", "-d", "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    status = {"run_id": f"exp6c-{dt.datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}", "mode": "smoke" if a.smoke else "full",
              "state": "running", "supervisor_pid": os.getpid(), "session_id": os.getsid(0), "caffeinate_pid": caf.pid,
              "caffeinate_command": f"caffeinate -d -i -w {os.getpid()}", "started": now(), "stages": [], "current_pid": None,
              "experiment_pid": None, "audit_pid": None}
    write_status(status_path, status)
    print(f"{now()} supervisor {os.getpid()} (session {os.getsid(0)}), caffeinate {caf.pid}, run {status['run_id']}", flush=True)

    def on_term(signum, _frame):     # record an external termination; the stage child (and its workers) is terminated too
        status["state"] = f"terminated (signal {signum})"
        status["ended"] = now()
        if status.get("current_pid") and alive(status["current_pid"]):
            os.kill(status["current_pid"], signal.SIGTERM)
        write_status(status_path, status)
        sys.exit(128 + signum)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    stages = [("experiment", [PY, "scripts/run_lstm_h5_experiment6c.py"] + (["--smoke"] if a.smoke else []), ART / f"run{tag}.log")]
    if not a.smoke:
        stages.append(("independent_audit", [PY, "scripts/verify_lstm_h5_experiment6c.py"], ART / "verify.log"))
    ok = True
    for name, cmd, logp in stages:
        attempt = 0
        while True:
            with open(logp, "a") as lf:
                child = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT)
                st = {"stage": name, "attempt": attempt, "command": " ".join(cmd[1:]), "pid": child.pid,
                      "log": str(logp.relative_to(PROJECT_ROOT)), "started": now(), "exit_code": None}
                status["stages"].append(st)
                status["current_pid"] = child.pid
                status["experiment_pid" if name == "experiment" else "audit_pid"] = child.pid
                write_status(status_path, status)
                code = child.wait()
            st["exit_code"], st["ended"] = code, now()
            status["current_pid"] = None
            write_status(status_path, status)
            print(f"{now()} stage {name} (attempt {attempt}) exited with {code}", flush=True)
            if code < 0 and name == "experiment" and attempt < MAX_RESTARTS:
                attempt += 1
                print(f"{now()} killed by signal {-code}: restarting from checkpoints", flush=True)
                continue
            break
        if code != 0:
            ok = False
            break
    status["state"] = "completed" if ok else "failed"
    status["ended"] = now()
    write_status(status_path, status)
    if ok:
        marker.write_text(json.dumps({"run_id": status["run_id"], "completed": status["ended"],
                                      "exit_codes": {s["stage"]: s["exit_code"] for s in status["stages"]}}, indent=1))
    caf.terminate()          # also exits on its own when this process exits (-w)
    caf.wait(timeout=10)
    print(f"{now()} supervisor finished: {status['state']}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
