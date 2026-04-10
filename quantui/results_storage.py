"""
results_storage — Persist and reload QuantUI-local calculation results.

Each calculation is saved to a timestamped subdirectory::

    <results_dir>/<timestamp>_<formula>_<method>_<basis>/
        result.json   — structured metadata + energy values (versioned)
        pyscf.log     — raw PySCF stdout (may be absent for short runs)

The ``result.json`` schema carries a ``_schema_version`` field so future
fields (geometry, IR/UV-Vis spectra file paths, etc.) can be added without
breaking existing readers.  A ``"spectra"`` key is reserved now as an empty
dict to make the intended extension point obvious.

Results directory
-----------------
Defaults to ``Path("results")`` relative to the working directory, or to
the value of the ``QUANTUI_RESULTS_DIR`` environment variable if set.
The Apptainer container sets this to ``$HOME/.quantui/results`` so that
results survive across kernel restarts and land in the user's home
directory (which is bind-mounted and writable).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .session_calc import SessionResult

_SCHEMA_VERSION = 1


def _default_results_dir() -> Path:
    env = os.environ.get("QUANTUI_RESULTS_DIR")
    return Path(env) if env else Path("results")


def _safe_name(s: str) -> str:
    """Replace characters that are unsafe in directory names with 'x'."""
    return re.sub(r"[^\w\-]", "x", s)


def save_result(
    result: SessionResult,
    pyscf_log: str = "",
    results_dir: Optional[Path] = None,
) -> Path:
    """Write *result* to a new timestamped subdirectory of *results_dir*.

    Parameters
    ----------
    result:
        Completed :class:`~quantui.session_calc.SessionResult`.
    pyscf_log:
        Raw PySCF stdout captured during the run.  Written to
        ``pyscf.log`` inside the result directory when non-empty.
    results_dir:
        Override the default results directory.

    Returns
    -------
    Path
        The directory that was created.
    """
    base = results_dir if results_dir is not None else _default_results_dir()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dirname = "_".join(
        [
            ts,
            _safe_name(result.formula),
            _safe_name(result.method),
            _safe_name(result.basis),
        ]
    )
    dest = base / dirname
    dest.mkdir(parents=True, exist_ok=True)

    data: dict = {
        "_schema_version": _SCHEMA_VERSION,
        "timestamp": ts,
        "formula": result.formula,
        "method": result.method,
        "basis": result.basis,
        "energy_hartree": result.energy_hartree,
        "energy_ev": result.energy_ev,
        "homo_lumo_gap_ev": result.homo_lumo_gap_ev,
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        # Reserved for future IR / UV-Vis spectra file paths.
        "spectra": {},
    }
    (dest / "result.json").write_text(json.dumps(data, indent=2))

    if pyscf_log:
        (dest / "pyscf.log").write_text(pyscf_log)

    return dest


def list_results(results_dir: Optional[Path] = None) -> list:
    """Return result directories sorted newest-first.

    Only directories containing a ``result.json`` file are included.
    """
    base = results_dir if results_dir is not None else _default_results_dir()
    if not base.exists():
        return []
    return sorted(
        (d for d in base.iterdir() if d.is_dir() and (d / "result.json").exists()),
        reverse=True,
    )


def load_result(result_dir: Path) -> dict:
    """Return the parsed ``result.json`` from *result_dir*."""
    data: dict = json.loads((result_dir / "result.json").read_text())
    return data
