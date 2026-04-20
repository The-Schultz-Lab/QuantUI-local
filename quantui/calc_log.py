"""
Performance and event logging for QuantUI-local.

Two separate log files, both stored in ``~/.quantui/logs/`` by default
(override with the ``QUANTUI_LOG_DIR`` environment variable):

``perf_log.jsonl``
    One record per completed calculation.  Kept indefinitely — the full
    history is needed to build reliable time-prediction models.

``event_log.jsonl``
    General app events (startup, calculation lifecycle, errors).
    Auto-pruned: entries older than 7 days are removed on every write.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()

# Rough relative cost of each method per SCF iteration.
# Used when no exact method+basis match exists in the history.
_METHOD_COST: dict[str, float] = {
    "RHF": 1.0,
    "UHF": 1.0,
    "B3LYP": 2.5,
    "PBE": 2.0,
    "PBE0": 2.5,
    "M06-2X": 3.0,
    "wB97X-D": 3.0,
    "CAM-B3LYP": 2.5,
    "M06-L": 2.0,
    "HSE06": 2.5,
    "PBE-D3": 2.1,
    "MP2": 8.0,
}


def _log_dir() -> Path:
    env = os.environ.get("QUANTUI_LOG_DIR")
    return Path(env) if env else Path.home() / ".quantui" / "logs"


def _perf_path() -> Path:
    return _log_dir() / "perf_log.jsonl"


def _event_path() -> Path:
    return _log_dir() / "event_log.jsonl"


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)


def _read_all(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with _LOCK:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    try:
                        records.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass
    return records


# ---------------------------------------------------------------------------
# Performance log
# ---------------------------------------------------------------------------


def log_calculation(
    formula: str,
    n_atoms: int,
    n_electrons: int,
    method: str,
    basis: str,
    n_iterations: int,
    elapsed_s: float,
    converged: bool,
) -> None:
    """Append one performance record to ``perf_log.jsonl``."""
    _append(
        _perf_path(),
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "formula": formula,
            "n_atoms": n_atoms,
            "n_electrons": n_electrons,
            "method": method,
            "basis": basis,
            "n_iterations": n_iterations,
            "elapsed_s": round(elapsed_s, 3),
            "converged": converged,
        },
    )


def estimate_time(
    n_atoms: int,
    n_electrons: int,
    method: str,
    basis: str,
) -> Optional[dict]:
    """
    Return a time estimate dict, or ``None`` if there is insufficient data.

    The returned dict has keys:

    * ``seconds``    – estimated wall time as a float
    * ``confidence`` – ``"high"``, ``"medium"``, or ``"low"``
    * ``n_samples``  – number of historical records used

    Prediction strategy (in priority order):

    1. **Exact match** (same method + basis, ≥ 2 converged runs):
       Median elapsed time scaled by ``(n_electrons / median_n_electrons)^2.7``.
       HF/DFT formal scaling is O(N³–N⁴); 2.7 is a practical empirical exponent
       for the sizes typical in a teaching context.
       Confidence: high (≥ 5 samples) or medium (2–4 samples).

    2. **Same basis, any method** (≥ 2 converged runs):
       Same electron-count scaling plus a relative method-cost factor from
       ``_METHOD_COST``.  Confidence: low.

    Returns ``None`` when fewer than 2 converged records are available for
    either strategy.
    """
    records = _read_all(_perf_path())
    converged = [r for r in records if r.get("converged")]
    if not converged:
        return None

    # ── Strategy 1: exact method + basis ────────────────────────────────────
    exact = [
        r for r in converged if r.get("method") == method and r.get("basis") == basis
    ]
    if len(exact) >= 2:
        median_ne = statistics.median(r["n_electrons"] for r in exact)
        median_t = statistics.median(r["elapsed_s"] for r in exact)
        scale = (n_electrons / median_ne) ** 2.7 if median_ne > 0 else 1.0
        return {
            "seconds": median_t * scale,
            "confidence": "high" if len(exact) >= 5 else "medium",
            "n_samples": len(exact),
        }

    # ── Strategy 2: same basis, any method ──────────────────────────────────
    same_basis = [r for r in converged if r.get("basis") == basis]
    if len(same_basis) >= 2:
        median_ne = statistics.median(r["n_electrons"] for r in same_basis)
        median_t = statistics.median(r["elapsed_s"] for r in same_basis)
        ref_cost = statistics.median(
            _METHOD_COST.get(r.get("method", "RHF"), 1.0) for r in same_basis
        )
        tgt_cost = _METHOD_COST.get(method, 1.0)
        ne_scale = (n_electrons / median_ne) ** 2.7 if median_ne > 0 else 1.0
        cost_scale = tgt_cost / ref_cost if ref_cost > 0 else 1.0
        return {
            "seconds": median_t * ne_scale * cost_scale,
            "confidence": "low",
            "n_samples": len(same_basis),
        }

    return None


def format_estimate(est: Optional[dict]) -> str:
    """
    Return an HTML string summarising *est* for display in the notebook.

    Returns an empty string when *est* is ``None``.
    """
    if est is None:
        return ""
    s = est["seconds"]
    conf = est["confidence"]
    n = est["n_samples"]

    if s < 5:
        time_str = "&lt; 5 s"
    elif s < 60:
        time_str = f"~{int(s)} s"
    elif s < 3600:
        time_str = f"~{int(s / 60)} min"
    else:
        time_str = f"~{s / 3600:.1f} hr"

    colour = {"high": "#22c55e", "medium": "#f59e0b", "low": "#94a3b8"}[conf]
    return (
        f'<span style="font-size:12px;color:#64748b">'
        f'Estimated time: <b style="color:{colour}">{time_str}</b>'
        f'&ensp;<span style="color:#94a3b8">({conf} confidence, {n} similar '
        f'run{"s" if n != 1 else ""})</span></span>'
    )


def get_perf_history() -> list[dict]:
    """Return all records from ``perf_log.jsonl`` as a list of dicts."""
    return _read_all(_perf_path())


def reset_perf_log() -> None:
    """Delete all records from ``perf_log.jsonl``.

    Removes the file entirely.  A fresh file is created automatically on the
    next :func:`log_calculation` call.  Time estimates will return ``None``
    until enough new records accumulate.
    """
    path = _perf_path()
    with _LOCK:
        if path.exists():
            path.unlink()


# ---------------------------------------------------------------------------
# Event log (7-day TTL)
# ---------------------------------------------------------------------------


def log_event(event_type: str, message: str, **extra: object) -> None:
    """
    Append one event to ``event_log.jsonl`` and prune entries > 7 days old.

    Args:
        event_type: Short category string, e.g. ``"startup"``, ``"calc_done"``,
                    ``"calc_error"``.
        message:    Human-readable description.
        **extra:    Any additional key-value pairs to include in the record.
    """
    record: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "message": message,
        **extra,
    }
    _append(_event_path(), record)
    prune_events()


def prune_events(days: int = 7) -> None:
    """Remove event-log entries older than *days* days (default: 7)."""
    path = _event_path()
    if not path.exists():
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    records = _read_all(path)
    kept: list[dict] = []
    for r in records:
        try:
            ts = datetime.fromisoformat(r["timestamp"])
            # fromisoformat on Python < 3.11 doesn't handle 'Z' suffix
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                kept.append(r)
        except (KeyError, ValueError):
            kept.append(r)  # keep malformed entries rather than silently drop
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_recent_events(n: int = 100) -> list[dict]:
    """Return the *n* most recent entries from ``event_log.jsonl``."""
    return _read_all(_event_path())[-n:]
