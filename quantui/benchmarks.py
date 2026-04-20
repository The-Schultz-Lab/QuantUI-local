"""
Timing calibration benchmark suite for QuantUI-local.

Runs a fixed set of small calculations that span the student-relevant
method/basis/molecule-size space.  Each completed step is logged to
``perf_log.jsonl`` via :func:`~quantui.calc_log.log_calculation` so that
:func:`~quantui.calc_log.estimate_time` immediately becomes useful on a
fresh install.

Typical usage (from the UI)::

    import threading
    from quantui.benchmarks import run_calibration

    stop = threading.Event()
    result = run_calibration(
        progress_cb=lambda *a: print(a),
        stop_event=stop,
        timeout_per_step=120,
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# Benchmark suite definition
# ---------------------------------------------------------------------------

#: Each entry: (label, atoms, coordinates, charge, multiplicity, method, basis)
#: Molecules are kept deliberately small so the full suite finishes quickly on
#: any modern laptop.
BENCHMARK_SUITE: list[tuple] = [
    (
        "H₂  RHF/STO-3G",
        ["H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        0,
        1,
        "RHF",
        "STO-3G",
    ),
    (
        "H₂O  RHF/STO-3G",
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        0,
        1,
        "RHF",
        "STO-3G",
    ),
    (
        "H₂O  B3LYP/STO-3G",
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        0,
        1,
        "B3LYP",
        "STO-3G",
    ),
    (
        "H₂O  RHF/6-31G*",
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        0,
        1,
        "RHF",
        "6-31G*",
    ),
    (
        "CH₄  RHF/STO-3G",
        ["C", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ],
        0,
        1,
        "RHF",
        "STO-3G",
    ),
    (
        "C₂H₄  RHF/STO-3G",
        ["C", "C", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.670],
            [0.0, 0.0, -0.670],
            [0.0, 0.924, 1.241],
            [0.0, -0.924, 1.241],
            [0.0, 0.924, -1.241],
            [0.0, -0.924, -1.241],
        ],
        0,
        1,
        "RHF",
        "STO-3G",
    ),
    (
        "C₂H₆O (ethanol)  RHF/STO-3G",
        ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        [
            [-1.232, 0.026, 0.000],
            [0.281, 0.026, 0.000],
            [0.829, 1.310, 0.000],
            [-1.566, 1.059, 0.000],
            [-1.609, -0.506, 0.880],
            [-1.609, -0.506, -0.880],
            [0.668, -0.497, 0.890],
            [0.668, -0.497, -0.890],
            [1.802, 1.311, 0.000],
        ],
        0,
        1,
        "RHF",
        "STO-3G",
    ),
    (
        "C₂H₆O (ethanol)  B3LYP/6-31G*",
        ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        [
            [-1.232, 0.026, 0.000],
            [0.281, 0.026, 0.000],
            [0.829, 1.310, 0.000],
            [-1.566, 1.059, 0.000],
            [-1.609, -0.506, 0.880],
            [-1.609, -0.506, -0.880],
            [0.668, -0.497, 0.890],
            [0.668, -0.497, -0.890],
            [1.802, 1.311, 0.000],
        ],
        0,
        1,
        "B3LYP",
        "6-31G*",
    ),
]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

_STATUS_OK = "ok"
_STATUS_TIMEOUT = "timed_out"
_STATUS_STOPPED = "stopped"
_STATUS_ERROR = "error"


@dataclass
class BenchmarkStep:
    """Result for a single benchmark step."""

    label: str
    method: str
    basis: str
    n_atoms: int
    n_electrons: int
    status: str  # "ok" | "timed_out" | "stopped" | "error"
    elapsed_s: float = 0.0
    error_msg: str = ""


@dataclass
class CalibrationResult:
    """Summary result from :func:`run_calibration`."""

    timestamp: str
    steps: List[BenchmarkStep] = field(default_factory=list)
    stopped_early: bool = False

    @property
    def n_completed(self) -> int:
        return sum(1 for s in self.steps if s.status == _STATUS_OK)

    @property
    def n_total(self) -> int:
        return len(BENCHMARK_SUITE)


# ---------------------------------------------------------------------------
# Main calibration runner
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int, str, str, float], None]
"""progress_cb(step_n, total, label, status, elapsed_s)"""


def _count_electrons(atoms: list[str], charge: int) -> int:
    """Rough electron count: sum of atomic numbers minus charge."""
    _Z = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
    }
    return sum(_Z.get(a, 6) for a in atoms) - charge


def run_calibration(
    progress_cb: Optional[ProgressCallback] = None,
    stop_event=None,
    timeout_per_step: float = 120.0,
) -> CalibrationResult:
    """Run the benchmark suite and populate ``perf_log.jsonl``.

    Args:
        progress_cb: Called after each step with
            ``(step_n, total, label, status, elapsed_s)``.
        stop_event: A :class:`threading.Event`; checked before each step.
            Set it to abort the suite cleanly.
        timeout_per_step: Wall-clock seconds allowed per step.  Steps that
            exceed this are marked ``"timed_out"`` and skipped.

    Returns:
        :class:`CalibrationResult` with per-step outcomes.
    """
    import concurrent.futures
    import json

    from quantui import calc_log as _calc_log
    from quantui.molecule import Molecule

    _pyscf_available = False
    try:
        import pyscf  # noqa: F401

        _pyscf_available = True
    except ImportError:
        pass

    timestamp = datetime.now(timezone.utc).isoformat()
    result = CalibrationResult(timestamp=timestamp)
    total = len(BENCHMARK_SUITE)

    for step_n, entry in enumerate(BENCHMARK_SUITE, start=1):
        label, atoms, coords, charge, mult, method, basis = entry

        # --- honour stop request ---
        if stop_event is not None and stop_event.is_set():
            result.stopped_early = True
            break

        step = BenchmarkStep(
            label=label,
            method=method,
            basis=basis,
            n_atoms=len(atoms),
            n_electrons=_count_electrons(atoms, charge),
            status=_STATUS_ERROR,
        )

        if not _pyscf_available:
            step.status = _STATUS_ERROR
            step.error_msg = "PySCF not available"
            result.steps.append(step)
            if progress_cb is not None:
                progress_cb(step_n, total, label, step.status, 0.0)
            continue

        def _run_step(
            atoms=atoms,
            coords=coords,
            charge=charge,
            mult=mult,
            method=method,
            basis=basis,
        ):
            from quantui.session_calc import run_in_session

            mol = Molecule(atoms, coords, charge=charge, multiplicity=mult)
            t0 = time.perf_counter()
            res = run_in_session(mol, method=method, basis=basis, verbose=0)
            return res, time.perf_counter() - t0

        t_start = time.perf_counter()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_step)
                try:
                    res, elapsed = future.result(timeout=timeout_per_step)
                    step.elapsed_s = elapsed
                    step.status = _STATUS_OK
                    # Log to perf_log.jsonl so estimate_time() can use it
                    _calc_log.log_calculation(
                        formula=res.formula,
                        n_atoms=step.n_atoms,
                        n_electrons=step.n_electrons,
                        method=method,
                        basis=basis,
                        n_iterations=res.n_iterations,
                        elapsed_s=elapsed,
                        converged=res.converged,
                    )
                except concurrent.futures.TimeoutError:
                    step.status = _STATUS_TIMEOUT
                    step.elapsed_s = time.perf_counter() - t_start
        except Exception as exc:
            step.status = _STATUS_ERROR
            step.error_msg = str(exc)
            step.elapsed_s = time.perf_counter() - t_start

        result.steps.append(step)
        if progress_cb is not None:
            progress_cb(step_n, total, label, step.status, step.elapsed_s)

    # --- persist calibration summary ---
    _cal_path = Path.home() / ".quantui" / "calibration.json"
    try:
        _cal_path.parent.mkdir(parents=True, exist_ok=True)
        _cal_path.write_text(
            json.dumps(
                {
                    "timestamp": result.timestamp,
                    "stopped_early": result.stopped_early,
                    "steps": [
                        {
                            "label": s.label,
                            "method": s.method,
                            "basis": s.basis,
                            "n_atoms": s.n_atoms,
                            "n_electrons": s.n_electrons,
                            "status": s.status,
                            "elapsed_s": round(s.elapsed_s, 3),
                            "error_msg": s.error_msg,
                        }
                        for s in result.steps
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass

    return result


def load_last_calibration() -> Optional[dict]:
    """Return the last calibration summary dict, or ``None`` if absent."""
    import json

    path = Path.home() / ".quantui" / "calibration.json"
    if not path.exists():
        return None
    try:
        data: dict = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception:
        return None
