"""
QM geometry optimization using ASE-PySCF.

Performs a full quantum mechanical geometry optimization by coupling the
ASE BFGS optimizer with the PySCF mean-field calculator.  Atoms are
moved iteratively until the maximum force on any atom falls below the
convergence threshold (``fmax``).

Returns both the final optimized molecule and the complete list of
intermediate frames as a trajectory — enabling step-by-step
visualization of the relaxation path in the notebook.

Platform notes
--------------
Requires PySCF — **Linux / macOS / WSL only**.  ASE >= 3.22 required.
This module imports PySCF lazily so it can be imported safely on Windows.

Educational value
-----------------
* Students see the molecule relax step-by-step (trajectory slider in the
  notebook's 3D viewer).
* The energy-vs-step plot shows convergence behaviour.
* RMSD between initial and final geometry quantifies the structural change.
* Teaches that real molecular properties require an optimized geometry.

Typical usage
-------------
>>> from quantui import optimize_geometry
>>> result = optimize_geometry(molecule, method="RHF", basis="STO-3G")
>>> print(result.summary())
>>> # result.trajectory is a list[Molecule] for the step-through viewer
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, List, Optional, Tuple

from .ase_bridge import ASE_AVAILABLE, atoms_to_molecule, molecule_to_atoms
from .molecule import Molecule
from .session_calc import HARTREE_TO_EV

logger = logging.getLogger(__name__)

# Defaults also exposed in config.py for the notebook UI
DEFAULT_FMAX: float = 0.05       # eV/Å — tight enough for educational use
DEFAULT_OPT_STEPS: int = 200     # generous upper limit for small molecules


# ============================================================================
# Result dataclass
# ============================================================================


@dataclass
class OptimizationResult:
    """
    Structured output from a completed QM geometry optimization.

    Attributes:
        molecule: Final optimized :class:`~quantui.molecule.Molecule`.
        trajectory: All frames as a list of Molecule objects, starting from
            the *input* geometry and ending at the optimized geometry.
            Length is ``n_steps + 1``.
        energies_hartree: SCF energy in Hartrees at each trajectory frame.
            Same length as ``trajectory``.
        converged: ``True`` if the maximum atomic force dropped below
            ``fmax`` within the allowed number of steps.
        n_steps: Number of BFGS optimizer steps taken (``len(trajectory) - 1``).
        method: Calculation method used (e.g. ``'RHF'``).
        basis: Basis set used (e.g. ``'STO-3G'``).
        formula: Hill-notation molecular formula of the input molecule.
    """

    molecule: Molecule
    trajectory: List[Molecule]
    energies_hartree: List[float]
    converged: bool
    n_steps: int
    method: str
    basis: str
    formula: str

    @property
    def energy_hartree(self) -> float:
        """Final energy in Hartrees (last trajectory frame)."""
        return self.energies_hartree[-1] if self.energies_hartree else float("nan")

    @property
    def energy_ev(self) -> float:
        """Final energy in electronvolts."""
        return self.energy_hartree * HARTREE_TO_EV

    @property
    def energy_change_hartree(self) -> float:
        """Total energy change from the first to the last frame (Ha)."""
        if len(self.energies_hartree) < 2:
            return 0.0
        return self.energies_hartree[-1] - self.energies_hartree[0]

    @property
    def rmsd_angstrom(self) -> float:
        """
        Root-mean-square displacement (Å) between the initial and final geometry.

        Measures how much the structure changed during optimization.
        Uses pure Python so it works without numpy.
        """
        if len(self.trajectory) < 2:
            return 0.0
        initial = self.trajectory[0].coordinates
        final = self.trajectory[-1].coordinates
        n = len(initial)
        if n == 0:
            return 0.0
        total = sum(
            (fx - ix) ** 2 + (fy - iy) ** 2 + (fz - iz) ** 2
            for (ix, iy, iz), (fx, fy, fz) in zip(initial, final)
        )
        return math.sqrt(total / n)

    def summary(self) -> str:
        """Return a multi-line human-readable result summary."""
        lines = [
            "=" * 60,
            "Geometry Optimization Results",
            "=" * 60,
            f"  Molecule       : {self.formula}",
            f"  Method/Basis   : {self.method}/{self.basis}",
            f"  Converged      : {'Yes' if self.converged else '❌ NO — max steps reached'}",
            f"  Steps taken    : {self.n_steps}",
            f"  Final energy   : {self.energy_hartree:.8f} Ha",
            f"  Energy change  : {self.energy_change_hartree:+.6f} Ha",
            f"  Geometry RMSD  : {self.rmsd_angstrom:.4f} Å",
            "=" * 60,
        ]
        if self.converged:
            lines.append("✅ Optimization converged successfully!")
        else:
            lines.append(
                "⚠️  Optimization did not converge.\n"
                "   Try increasing Max Steps, loosening Force Threshold,\n"
                "   or using LJ pre-optimization to improve the starting geometry."
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# Main function
# ============================================================================


def optimize_geometry(
    molecule: Molecule,
    method: str = "RHF",
    basis: str = "STO-3G",
    fmax: float = DEFAULT_FMAX,
    steps: int = DEFAULT_OPT_STEPS,
    progress_stream: Optional[IO[str]] = None,
) -> OptimizationResult:
    """
    Optimize a molecular geometry at the QM level using ASE-BFGS + PySCF.

    Runs a BFGS quasi-Newton geometry optimization.  At each step the
    PySCF mean-field calculator provides the energy and analytical
    nuclear gradients (forces).  The BFGS optimizer moves the atoms
    toward lower energy until convergence.

    The full trajectory (one :class:`~quantui.molecule.Molecule` per
    optimizer step, including the initial geometry) is stored in
    :attr:`OptimizationResult.trajectory` for step-through visualization.

    Args:
        molecule: Starting geometry as a validated
            :class:`~quantui.molecule.Molecule`.
        method: SCF method — ``'RHF'`` or ``'UHF'``.  Default: ``'RHF'``.
            For optimization ``'RHF'`` is recommended unless the molecule
            is an open-shell radical.
        basis: Basis set.  ``'STO-3G'`` is fastest; ``'6-31G'`` or
            ``'6-31G*'`` give more chemically accurate geometries but
            are significantly slower.  Default: ``'STO-3G'``.
        fmax: Force convergence threshold in eV/Å.  Optimization stops
            when the maximum force on any atom is below this value.
            Default: 0.05 eV/Å (a standard tight threshold).
        steps: Maximum number of BFGS optimizer steps.  Default: 200.
        progress_stream: Optional writable text stream.  BFGS step
            progress (step number and maximum force) is written here.
            Pass a widget-backed stream in the notebook for live output;
            leave ``None`` to write to ``sys.stdout``.

    Returns:
        :class:`OptimizationResult` containing the optimized molecule,
        full trajectory, per-step energies, convergence status, and
        summary statistics.

    Raises:
        ImportError: If ASE ≥ 3.22 or PySCF is not installed.
        RuntimeError: If the optimization raises an unexpected exception
            (original exception is chained).

    Note:
        PySCF verbose output is suppressed during optimization to keep
        the progress stream clean.  BFGS writes a concise per-step table
        (step number and maximum force) to *progress_stream*.
    """
    # --- Dependency checks (mirrors session_calc) ---
    if not ASE_AVAILABLE:
        raise ImportError(
            "ASE is not installed — cannot run geometry optimization.\n"
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
            "PySCF is not installed — cannot run geometry optimization.\n"
            "  conda install -c conda-forge pyscf\n"
            "Note: PySCF is Linux / macOS / WSL only."
        ) from exc

    try:
        from ase.optimize import BFGS  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "ase.optimize.BFGS is not available.\n"
            "Ensure ASE >= 3.22.0: pip install 'ase>=3.22.0'"
        ) from exc

    # --- Set up ASE Atoms + PySCF calculator ---
    atoms = molecule_to_atoms(molecule)
    atoms.calc = PySCF(
        method=method,
        basis=basis,
        charge=molecule.charge,
        spin=molecule.multiplicity - 1,
        verbose=0,  # suppress per-step SCF output; BFGS table goes to progress_stream
    )

    _stream: IO[str] = progress_stream if progress_stream is not None else sys.stdout
    _null = io.StringIO()  # captures any stray PySCF stdout at verbose=0

    # --- Run optimization with trajectory file ---
    converged = False
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_path = Path(tmpdir) / "opt.traj"

            dyn = BFGS(
                atoms,
                trajectory=str(traj_path),
                logfile=_stream,  # BFGS step table → progress_stream
            )

            # Suppress any residual PySCF stdout even at verbose=0
            with contextlib.redirect_stdout(_null):
                converged = bool(dyn.run(fmax=fmax, steps=steps))

            # --- Read trajectory frames ---
            from ase.io.trajectory import Trajectory  # type: ignore[import]

            traj_frames = list(Trajectory(str(traj_path)))

    except Exception as exc:
        raise RuntimeError(
            f"Geometry optimization failed for {molecule.get_formula()} "
            f"({method}/{basis}): {exc}"
        ) from exc

    # Convert ASE frames → Molecule objects and extract stored energies
    charge = molecule.charge
    mult = molecule.multiplicity

    trajectory: List[Molecule] = []
    energies_hartree: List[float] = []

    for frame in traj_frames:
        mol_frame = atoms_to_molecule(frame, charge=charge, multiplicity=mult)
        trajectory.append(mol_frame)
        # Each frame has a SinglePointCalculator with the stored energy (eV)
        try:
            e_ev = frame.get_potential_energy()
            energies_hartree.append(e_ev / HARTREE_TO_EV)
        except Exception:
            energies_hartree.append(float("nan"))

    if not trajectory:
        # Edge case: no frames written — return the final atoms state
        trajectory = [atoms_to_molecule(atoms, charge=charge, multiplicity=mult)]
        try:
            e_ev = atoms.get_potential_energy()
            energies_hartree = [e_ev / HARTREE_TO_EV]
        except Exception:
            energies_hartree = [float("nan")]

    n_steps = max(0, len(trajectory) - 1)
    formula = molecule.get_formula()

    logger.info(
        "Geometry optimization: %s %s/%s  steps=%d  converged=%s  "
        "E_final=%.8f Ha  RMSD~%.4f Å",
        formula, method, basis, n_steps, converged,
        energies_hartree[-1] if energies_hartree else float("nan"),
        _rmsd(molecule, trajectory[-1]) if len(trajectory) > 1 else 0.0,
    )

    return OptimizationResult(
        molecule=trajectory[-1],
        trajectory=trajectory,
        energies_hartree=energies_hartree,
        converged=converged,
        n_steps=n_steps,
        method=method,
        basis=basis,
        formula=formula,
    )


def _rmsd(mol_a: Molecule, mol_b: Molecule) -> float:
    """Compute RMSD (Å) between two same-sized molecules (no alignment, pure Python)."""
    a = mol_a.coordinates
    b = mol_b.coordinates
    if len(a) != len(b) or not a:
        return float("nan")
    total = sum(
        (bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2
        for (ax, ay, az), (bx, by, bz) in zip(a, b)
    )
    return math.sqrt(total / len(a))
