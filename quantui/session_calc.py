"""
In-session quantum chemistry calculation using ASE-PySCF.

Replaces the hand-rolled threading + screen-scraping runner in the
notebook with a clean, testable function that returns structured data.
PySCF's verbose output is routed through an optional stream parameter,
so the notebook can still display live SCF iterations in a widget without
needing to intercept mol.stdout manually.

Platform notes
--------------
PySCF is **Linux / macOS / WSL only** — not available on native Windows.
This module imports PySCF lazily inside :func:`run_in_session` so it can
be imported safely on any platform without raising at import time.
ASE >= 3.22 is required; ``ase.calculators.pyscf.PySCF`` is bundled.

Typical notebook usage
----------------------
>>> from quantui import run_in_session, SessionResult
>>> result = run_in_session(molecule, method="RHF", basis="6-31G")
>>> print(result.summary())
"""

from __future__ import annotations

import contextlib
import logging
import sys
from dataclasses import dataclass
from typing import IO, Optional

from .ase_bridge import ASE_AVAILABLE, molecule_to_atoms
from .molecule import Molecule

logger = logging.getLogger(__name__)

# NIST 2018 CODATA — consistent with PySCF's internal constant
HARTREE_TO_EV: float = 27.211386245988


# ============================================================================
# Result dataclass
# ============================================================================


@dataclass
class SessionResult:
    """
    Structured output from a completed in-session quantum chemistry calculation.

    Attributes:
        energy_hartree: Total SCF energy in Hartrees.
        homo_lumo_gap_ev: HOMO-LUMO gap in electronvolts, or ``None`` if the
            gap cannot be determined (e.g. open-shell UHF with complex orbital
            occupations, or too few occupied orbitals).
        converged: ``True`` if the SCF iterations reached the convergence
            threshold; ``False`` if the maximum iteration count was hit.
        n_iterations: Number of SCF macro-iterations completed.  May be
            ``-1`` if the underlying calculator does not expose this.
        method: Calculation method used (e.g. ``'RHF'``, ``'UHF'``).
        basis: Basis set used (e.g. ``'6-31G'``, ``'STO-3G'``).
        formula: Hill-notation molecular formula of the input molecule.
    """

    energy_hartree: float
    homo_lumo_gap_ev: Optional[float]
    converged: bool
    n_iterations: int
    method: str
    basis: str
    formula: str

    @property
    def energy_ev(self) -> float:
        """Total energy converted to electronvolts."""
        return self.energy_hartree * HARTREE_TO_EV

    def summary(self) -> str:
        """Return a multi-line human-readable result summary suitable for printing."""
        lines = [
            "=" * 60,
            "Calculation Results",
            "=" * 60,
            f"  Molecule      : {self.formula}",
            f"  Method/Basis  : {self.method}/{self.basis}",
            f"  SCF converged : {'Yes' if self.converged else '❌ NO — treat results with caution'}",
            f"  Iterations    : {self.n_iterations}",
            f"  Total energy  : {self.energy_hartree:.8f} Ha",
        ]
        if self.homo_lumo_gap_ev is not None:
            lines.append(f"  HOMO-LUMO gap : {self.homo_lumo_gap_ev:.4f} eV")
        lines += [
            "=" * 60,
            (
                "✅ Calculation completed successfully!"
                if self.converged
                else "⚠️  SCF did not converge — try a different starting geometry, basis, or method."
            ),
            "=" * 60,
        ]
        return "\n".join(lines)


# ============================================================================
# Main function
# ============================================================================


def run_in_session(
    molecule: Molecule,
    method: str = "RHF",
    basis: str = "6-31G",
    verbose: int = 3,
    progress_stream: Optional[IO[str]] = None,
) -> SessionResult:
    """
    Run a quantum chemistry calculation in the current kernel using ASE-PySCF.

    Uses ``ase.calculators.pyscf.PySCF`` (bundled with ASE ≥ 3.22) to set
    up and run the SCF calculation.  Returns a :class:`SessionResult` with
    structured data — energy, HOMO-LUMO gap, convergence status — rather
    than requiring callers to parse PySCF stdout.

    PySCF's verbose log is routed to *progress_stream* (or ``sys.stdout``
    if not provided) so live SCF iteration output can still be displayed in a
    Jupyter widget by passing a stream-backed output widget.

    Args:
        molecule: Validated :class:`~quantui.molecule.Molecule` object.
        method: SCF method — ``'RHF'`` for closed-shell molecules or
            ``'UHF'`` for open-shell / radical species.  Default: ``'RHF'``.
        basis: Basis set name as recognised by PySCF (e.g. ``'STO-3G'``,
            ``'6-31G'``, ``'6-31G*'``, ``'cc-pVDZ'``).  Default: ``'6-31G'``.
        verbose: PySCF verbosity level (0 = silent … 9 = very detailed).
            Level 3 prints per-iteration SCF energies; level 4 adds
            convergence diagnostics.  Default: 3.
        progress_stream: Optional writable text stream.  All PySCF output
            during the calculation is written here.  Pass a widget-backed
            stream (e.g. ``_WidgetStream``) in the notebook for live display;
            leave ``None`` to write to ``sys.stdout``.

    Returns:
        :class:`SessionResult` containing energy, HOMO-LUMO gap, convergence
        information, and metadata.

    Raises:
        ImportError: If ASE ≥ 3.22 or PySCF is not installed.
        RuntimeError: If PySCF raises an unexpected exception during the
            calculation (original exception is chained).
    """
    # --- Dependency checks ---
    if not ASE_AVAILABLE:
        raise ImportError(
            "ASE is not installed — cannot use ASE-PySCF calculator.\n"
            "  pip install 'ase>=3.22.0'\n"
            "  # or: conda install -c conda-forge ase"
        )

    try:
        from ase.calculators.pyscf import PySCF  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "ase.calculators.pyscf is not available.\n"
            "Ensure ASE >= 3.22.0 is installed: pip install 'ase>=3.22.0'"
        ) from exc

    try:
        import pyscf as _pyscf  # noqa: F401 — presence check
    except ImportError as exc:
        raise ImportError(
            "PySCF is not installed — cannot run in-session calculations.\n"
            "  conda install -c conda-forge pyscf\n"
            "Note: PySCF is Linux / macOS / WSL only."
        ) from exc

    # --- Set up ASE Atoms + calculator ---
    atoms = molecule_to_atoms(molecule)
    atoms.calc = PySCF(
        method=method,
        basis=basis,
        charge=molecule.charge,
        spin=molecule.multiplicity - 1,
        verbose=verbose,
    )

    stream: IO[str] = progress_stream if progress_stream is not None else sys.stdout

    # --- Run calculation (output redirected to stream) ---
    try:
        with contextlib.redirect_stdout(stream):
            energy_ev = atoms.get_potential_energy()
    except Exception as exc:
        raise RuntimeError(
            f"PySCF calculation failed for {molecule.get_formula()} "
            f"({method}/{basis}): {exc}"
        ) from exc

    energy_hartree = energy_ev / HARTREE_TO_EV

    # --- Extract results from the mean-field object ---
    mf = atoms.calc.mf  # type: ignore[attr-defined]
    converged = bool(getattr(mf, "converged", False))
    n_iterations = int(getattr(mf, "cycles", -1))

    homo_lumo_gap_ev: Optional[float] = None
    try:
        mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy
        import numpy as _np

        if isinstance(mo_energy, (list, _np.ndarray)) and hasattr(
            mo_energy[0], "__len__"
        ):
            # UHF: mo_energy is (2, n_mo) — use alpha spin for the gap estimate
            mo_energy_ref = mo_energy[0]
            mo_occ_ref = mo_occ[0]
        else:
            mo_energy_ref = mo_energy
            mo_occ_ref = mo_occ

        n_occ = int((_np.array(mo_occ_ref) > 0).sum())
        if 0 < n_occ < len(mo_energy_ref):
            homo_lumo_gap_ev = float(
                (mo_energy_ref[n_occ] - mo_energy_ref[n_occ - 1]) * HARTREE_TO_EV
            )
    except Exception:
        pass  # gap stays None — non-fatal

    formula = molecule.get_formula()
    logger.info(
        "Session calculation: %s %s/%s  E=%.8f Ha  converged=%s  iters=%d",
        formula,
        method,
        basis,
        energy_hartree,
        converged,
        n_iterations,
    )

    return SessionResult(
        energy_hartree=energy_hartree,
        homo_lumo_gap_ev=homo_lumo_gap_ev,
        converged=converged,
        n_iterations=n_iterations,
        method=method,
        basis=basis,
        formula=formula,
    )
