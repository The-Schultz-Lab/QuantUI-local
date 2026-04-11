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
    pass  # result types accepted via duck typing; no hard import needed

_SCHEMA_VERSION = 2


def _default_results_dir() -> Path:
    env = os.environ.get("QUANTUI_RESULTS_DIR")
    return Path(env) if env else Path("results")


def _safe_name(s: str) -> str:
    """Replace characters that are unsafe in directory names with 'x'."""
    return re.sub(r"[^\w\-]", "x", s)


def save_result(
    result: object,
    pyscf_log: str = "",
    results_dir: Optional[Path] = None,
    calc_type: str = "single_point",
    spectra: Optional[dict] = None,
) -> Path:
    """Write *result* to a new timestamped subdirectory of *results_dir*.

    Accepts any result type that exposes ``.formula``, ``.method``,
    ``.basis``, ``.energy_hartree``, and ``.converged`` attributes
    (``SessionResult``, ``OptimizationResult``, ``FreqResult``,
    ``TDDFTResult``).  Missing optional fields (``homo_lumo_gap_ev``,
    ``n_iterations``) are stored as ``null``.

    Parameters
    ----------
    result:
        Any completed calculation result object.
    pyscf_log:
        Raw PySCF stdout captured during the run.  Written to
        ``pyscf.log`` inside the result directory when non-empty.
    results_dir:
        Override the default results directory.
    calc_type:
        Calculation type string stored in ``result.json`` for display
        in the History browser.  One of ``"single_point"``,
        ``"geometry_opt"``, ``"frequency"``, ``"tddft"``.
    spectra:
        Dict of spectra data (IR frequencies, UV-Vis excitations, …)
        stored under the ``"spectra"`` key in ``result.json``.

    Returns
    -------
    Path
        The directory that was created.
    """
    _HARTREE_TO_EV = 27.211386245988  # local fallback

    base = results_dir if results_dir is not None else _default_results_dir()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    dirname = "_".join(
        [
            ts,
            _safe_name(getattr(result, "formula", "unknown")),
            _safe_name(getattr(result, "method", "unknown")),
            _safe_name(getattr(result, "basis", "unknown")),
        ]
    )
    dest = base / dirname
    dest.mkdir(parents=True, exist_ok=True)

    _e_ha = getattr(result, "energy_hartree", float("nan"))
    # energy_ev may be a property (SessionResult) or absent (OptimizationResult
    # and new types also define it as a property, so getattr works for all).
    _e_ev = getattr(result, "energy_ev", _e_ha * _HARTREE_TO_EV)

    data: dict = {
        "_schema_version": _SCHEMA_VERSION,
        "timestamp": ts,
        "calc_type": calc_type,
        "formula": getattr(result, "formula", "?"),
        "method": getattr(result, "method", "?"),
        "basis": getattr(result, "basis", "?"),
        "energy_hartree": _e_ha,
        "energy_ev": _e_ev,
        "homo_lumo_gap_ev": getattr(result, "homo_lumo_gap_ev", None),
        "converged": getattr(result, "converged", None),
        "n_iterations": getattr(result, "n_iterations", -1),
        "spectra": spectra if spectra is not None else {},
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
