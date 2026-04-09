"""
Fast force-field geometry pre-optimization using ASE.

Used as an optional step before quantum chemistry calculations to ensure
students start with a chemically reasonable geometry. The Lennard-Jones
force field is not physically accurate for molecular bonding, but is
fast (~0.1 s for small molecules) and effectively removes severe
steric clashes or unreasonably compressed bond lengths before the
expensive QM step.

Platform notes
--------------
This module is fully compatible with **Windows, Linux, and WSL** — it
only requires ASE and NumPy, with no PySCF or SLURM dependency.

Typical usage
-------------
>>> from quantui.preopt import preoptimize
>>> optimized_mol, rmsd = preoptimize(molecule)
>>> print(f"Geometry changed by {rmsd:.3f} Å (RMSD)")
"""

from __future__ import annotations

import logging
from typing import Tuple

from .ase_bridge import ASE_AVAILABLE, atoms_to_molecule, molecule_to_atoms
from .molecule import Molecule

logger = logging.getLogger(__name__)


def preoptimize(
    molecule: Molecule,
    fmax: float = 0.05,
    steps: int = 200,
) -> Tuple[Molecule, float]:
    """
    Run a fast Lennard-Jones force-field pre-optimization on a molecule.

    Uses the ASE Lennard-Jones calculator with a BFGS quasi-Newton
    minimizer to relax atomic positions to a local potential-energy
    minimum. The input ``molecule`` is **never mutated** — a new
    ``Molecule`` object is always returned.

    Args:
        molecule: Input molecule. May have a non-ideal starting geometry
            (e.g. from user-typed XYZ coordinates or a structure file).
        fmax: Force convergence threshold in eV/Å. The optimizer stops
            when the maximum atomic force falls below this value.
            Default: 0.05 eV/Å.
        steps: Maximum number of BFGS optimization steps. The optimizer
            also stops early if *fmax* is reached. Default: 200.

    Returns:
        A 2-tuple ``(optimized_molecule, rmsd)`` where

        * ``optimized_molecule`` is a :class:`~quantui.molecule.Molecule`
          with the relaxed coordinates and the same ``charge`` and
          ``multiplicity`` as the input.
        * ``rmsd`` is the root-mean-square displacement in Ångströms
          between the original and relaxed atomic positions — a rough
          measure of how much the geometry changed.

    Raises:
        ImportError: If ASE is not installed.

    Note:
        The LJ force field treats all atoms identically and does not
        model chemical bonds or angular geometry. RMSD values > ~1 Å
        suggest the starting geometry may have been very far from
        reasonable; students should visually check the result.
    """
    if not ASE_AVAILABLE:
        raise ImportError(
            "ASE is not installed — cannot run pre-optimization.\n"
            "  pip install ase\n"
            "  # or: conda install -c conda-forge ase"
        )

    import numpy as np
    from ase.calculators.lj import LennardJones  # type: ignore[import]
    from ase.optimize import BFGS  # type: ignore[import]

    atoms = molecule_to_atoms(molecule)
    original_positions = atoms.get_positions().copy()

    atoms.calc = LennardJones()
    dyn = BFGS(atoms, logfile=None)
    dyn.run(fmax=fmax, steps=steps)

    optimized_positions = atoms.get_positions()
    displacements = optimized_positions - original_positions
    rmsd = float(np.sqrt(np.mean(np.sum(displacements**2, axis=1))))

    optimized_molecule = atoms_to_molecule(
        atoms,
        charge=molecule.charge,
        multiplicity=molecule.multiplicity,
    )

    logger.info(
        "LJ pre-optimization complete: RMSD=%.4f Å (max %d steps, fmax=%.3f eV/Å)",
        rmsd,
        steps,
        fmax,
    )
    return optimized_molecule, rmsd
