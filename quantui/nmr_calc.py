"""
NMR chemical shift prediction using PySCF GIAO.

Computes isotropic NMR shielding tensors via GIAO (Gauge-Including
Atomic Orbitals) and converts to ¹H/¹³C chemical shifts relative to
TMS using tabulated reference constants from config.py.

Typical usage::

    from quantui.nmr_calc import run_nmr_calc
    result = run_nmr_calc(molecule, method="B3LYP", basis="6-31G*")
    for atom_idx, delta_ppm in result.h_shifts():
        print(f"H-{atom_idx+1}: {delta_ppm:.2f} ppm")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .molecule import Molecule


@dataclass
class NMRResult:
    """Structured output from an NMR shielding calculation."""

    atom_symbols: List[str]
    shielding_iso_ppm: List[float]
    chemical_shifts_ppm: Dict[int, float]  # atom_index → δ (ppm), ¹H and ¹³C only
    method: str
    basis: str
    formula: str
    reference_compound: str = "TMS"
    converged: bool = True

    def h_shifts(self) -> List[Tuple[int, float]]:
        """(atom_index, δ ppm) pairs for all H atoms in molecule order."""
        return [
            (i, d)
            for i, d in sorted(self.chemical_shifts_ppm.items())
            if self.atom_symbols[i] == "H"
        ]

    def c_shifts(self) -> List[Tuple[int, float]]:
        """(atom_index, δ ppm) pairs for all C atoms in molecule order."""
        return [
            (i, d)
            for i, d in sorted(self.chemical_shifts_ppm.items())
            if self.atom_symbols[i] == "C"
        ]


def run_nmr_calc(
    molecule: Molecule,
    method: str = "B3LYP",
    basis: str = "6-31G*",
    progress_stream=None,
) -> NMRResult:
    """Run NMR shielding calculation and return ¹H/¹³C chemical shifts.

    Uses PySCF GIAO (Gauge-Including Atomic Orbitals) formalism.
    Chemical shifts are reported relative to TMS using reference constants
    from :data:`~quantui.config.NMR_REFERENCE_SHIELDINGS`.

    Args:
        molecule: Validated :class:`~quantui.molecule.Molecule` object.
        method: SCF or DFT method. Recommended: B3LYP.
        basis: Basis set. Recommended: 6-31G* or better.
        progress_stream: Optional writable text stream for PySCF output.

    Returns:
        :class:`NMRResult` with per-atom shieldings and ¹H/¹³C shifts.

    Raises:
        ImportError: If PySCF is not installed.
        RuntimeError: If the SCF or GIAO-NMR calculation fails.
    """
    try:
        from pyscf import dft, gto, scf
    except ImportError as exc:
        raise ImportError(
            "PySCF is not installed — cannot run NMR calculations.\n"
            "Note: PySCF is Linux / macOS / WSL only."
        ) from exc

    import numpy as _np

    from . import config as _config
    from .session_calc import _XC_ALIAS

    stream = progress_stream if progress_stream is not None else sys.stdout

    mol = gto.Mole()
    mol.atom = molecule.to_pyscf_format()
    mol.basis = basis
    mol.charge = molecule.charge
    mol.spin = molecule.multiplicity - 1
    mol.verbose = 4
    mol.stdout = stream
    mol.build()

    method_upper = method.upper()
    if method_upper == "RHF":
        mf = scf.RHF(mol)
    elif method_upper == "UHF":
        mf = scf.UHF(mol)
    else:
        xc_string = _XC_ALIAS.get(method, method)
        mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        mf.xc = xc_string

    try:
        mf.kernel()
    except Exception as exc:
        raise RuntimeError(
            f"SCF failed for {molecule.get_formula()} ({method}/{basis}): {exc}"
        ) from exc

    converged = bool(getattr(mf, "converged", False))

    try:
        nmr_obj = mf.NMR()
        tensors = nmr_obj.kernel()
    except Exception as exc:
        raise RuntimeError(
            f"NMR shielding failed for {molecule.get_formula()}: {exc}"
        ) from exc

    shielding_iso: List[float] = []
    for tensor in tensors:
        arr = _np.array(tensor)
        if arr.ndim == 2:
            shielding_iso.append(float(_np.trace(arr) / 3.0))
        else:
            shielding_iso.append(float(arr))

    key = f"{method}/{basis}"
    ref_map = _config.NMR_REFERENCE_SHIELDINGS.get(key, _config.NMR_DEFAULT_REFERENCE)
    ref_H = float(ref_map.get("H", _config.NMR_DEFAULT_REFERENCE["H"]))
    ref_C = float(ref_map.get("C", _config.NMR_DEFAULT_REFERENCE["C"]))

    atoms = list(molecule.atoms)
    chemical_shifts: Dict[int, float] = {}
    for i, (atom, sigma) in enumerate(zip(atoms, shielding_iso)):
        if atom == "H":
            chemical_shifts[i] = round(ref_H - sigma, 2)
        elif atom == "C":
            chemical_shifts[i] = round(ref_C - sigma, 2)

    return NMRResult(
        atom_symbols=atoms,
        shielding_iso_ppm=shielding_iso,
        chemical_shifts_ppm=chemical_shifts,
        method=method,
        basis=basis,
        formula=molecule.get_formula(),
        converged=converged,
    )
