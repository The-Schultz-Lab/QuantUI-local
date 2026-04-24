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


def save_orbitals(result_dir: Path, result: object) -> None:
    """Persist MO data to *result_dir*/orbitals.npz and orbitals_meta.json.

    Saves ``mo_energy_hartree``, ``mo_occ``, and ``mo_coeff`` as a compressed
    NumPy archive and ``pyscf_mol_atom`` / ``pyscf_mol_basis`` as JSON so the
    orbital diagram and isosurface can be replayed from history.
    """
    import numpy as _np

    mo_e = getattr(result, "mo_energy_hartree", None)
    mo_occ = getattr(result, "mo_occ", None)
    mo_coeff = getattr(result, "mo_coeff", None)
    mol_atom = getattr(result, "pyscf_mol_atom", None)
    mol_basis = getattr(result, "pyscf_mol_basis", None)

    if mo_e is None and mo_occ is None:
        return

    arrays: dict = {}
    if mo_e is not None:
        arrays["mo_energy_hartree"] = _np.asarray(mo_e)
    if mo_occ is not None:
        arrays["mo_occ"] = _np.asarray(mo_occ)
    if mo_coeff is not None:
        arrays["mo_coeff"] = _np.asarray(mo_coeff)
    if arrays:
        _np.savez_compressed(str(result_dir / "orbitals.npz"), **arrays)

    meta: dict = {}
    if mol_atom is not None:
        # Convert list-of-tuples to JSON-safe list-of-lists.
        meta["mol_atom"] = [[sym, list(coords)] for sym, coords in mol_atom]
    if mol_basis is not None:
        meta["mol_basis"] = mol_basis
    if meta:
        (result_dir / "orbitals_meta.json").write_text(json.dumps(meta))


def load_orbitals(result_dir: Path):
    """Reload MO data saved by :func:`save_orbitals`.

    Returns a ``SimpleNamespace`` with ``mo_energy_hartree``, ``mo_occ``,
    ``mo_coeff``, ``pyscf_mol_atom``, ``pyscf_mol_basis``, and ``formula``
    (empty string if not known).

    Raises
    ------
    FileNotFoundError
        If ``orbitals.npz`` does not exist in *result_dir*.
    """
    import types

    import numpy as _np

    npz_path = result_dir / "orbitals.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = _np.load(str(npz_path))
    stub = types.SimpleNamespace(
        mo_energy_hartree=(
            data["mo_energy_hartree"] if "mo_energy_hartree" in data else None
        ),
        mo_occ=data["mo_occ"] if "mo_occ" in data else None,
        mo_coeff=data["mo_coeff"] if "mo_coeff" in data else None,
        pyscf_mol_atom=None,
        pyscf_mol_basis=None,
        formula="",
    )
    meta_path = result_dir / "orbitals_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        stub.pyscf_mol_atom = meta.get("mol_atom")
        stub.pyscf_mol_basis = meta.get("mol_basis")
    return stub


def save_trajectory(result_dir: Path, trajectory: list, energies: list) -> None:
    """Persist geometry-optimisation trajectory to *result_dir*/trajectory.json.

    Parameters
    ----------
    result_dir:
        Directory returned by :func:`save_result`.
    trajectory:
        List of ``Molecule`` objects (one per optimisation step).
    energies:
        List of total energies in Hartree, parallel to *trajectory*.
    """
    if not trajectory:
        return
    mol0 = trajectory[0]
    data = {
        "atoms": list(mol0.atoms),
        "charge": mol0.charge,
        "multiplicity": mol0.multiplicity,
        "steps": [
            {
                "coords": [list(row) for row in mol.coordinates],
                "energy": energies[i] if i < len(energies) else None,
            }
            for i, mol in enumerate(trajectory)
        ],
    }
    (result_dir / "trajectory.json").write_text(json.dumps(data))


def load_trajectory(result_dir: Path):
    """Reload a saved trajectory as (molecules, energies).

    Returns
    -------
    tuple[list, list]
        ``(trajectory, energies_hartree)`` where *trajectory* is a list of
        ``Molecule`` objects and *energies_hartree* is a parallel list of
        floats (``None`` entries are dropped to an empty list if all absent).

    Raises
    ------
    FileNotFoundError
        If ``trajectory.json`` does not exist in *result_dir*.
    """
    from quantui.molecule import Molecule

    raw = json.loads((result_dir / "trajectory.json").read_text())
    atoms = raw["atoms"]
    charge = raw.get("charge", 0)
    mult = raw.get("multiplicity", 1)
    trajectory = []
    energies = []
    for step in raw["steps"]:
        trajectory.append(
            Molecule(atoms, step["coords"], charge=charge, multiplicity=mult)
        )
        energies.append(step["energy"])
    # If every energy is None the list is meaningless; return empty instead.
    if all(e is None for e in energies):
        energies = []
    return trajectory, energies
