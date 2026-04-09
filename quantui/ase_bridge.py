"""
ASE (Atomic Simulation Environment) integration bridge for QuantUI.

Provides converters between QuantUI Molecule and ASE Atoms objects,
a file-format-agnostic structure reader built on ase.io, and a
curated molecule library for educational use.

All ASE imports are guarded by ASE_AVAILABLE so this module can be
imported safely even when ASE is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple, Union

if TYPE_CHECKING:
    import ase as _ase_type

    from .molecule import Molecule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft import — module works without ASE, just degrades gracefully
# ---------------------------------------------------------------------------

ASE_AVAILABLE = False
try:
    import ase as _ase  # noqa: F401 — presence check only

    ASE_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Curated educational molecule presets
#
# Each entry: display_label -> (ase_name, charge, multiplicity)
# Multiplicity > 1 is set for well-known open-shell ground states so the
# notebook can warn students about spin state automatically.
# ---------------------------------------------------------------------------

ASE_MOLECULE_PRESETS: Dict[str, Tuple[str, int, int]] = {
    # Diatomics
    "H₂ — Hydrogen": ("H2", 0, 1),
    "N₂ — Nitrogen": ("N2", 0, 1),
    "O₂ — Oxygen (triplet)": ("O2", 0, 3),
    "F₂ — Fluorine": ("F2", 0, 1),
    "HF — Hydrogen fluoride": ("HF", 0, 1),
    "HCl — Hydrogen chloride": ("HCl", 0, 1),
    "CO — Carbon monoxide": ("CO", 0, 1),
    # Triatomics
    "H₂O — Water": ("H2O", 0, 1),
    "CO₂ — Carbon dioxide": ("CO2", 0, 1),
    "SO₂ — Sulfur dioxide": ("SO2", 0, 1),
    "H₂S — Hydrogen sulfide": ("SH2", 0, 1),
    # Polyatomics
    "NH₃ — Ammonia": ("NH3", 0, 1),
    "PH₃ — Phosphine": ("PH3", 0, 1),
    "CH₄ — Methane": ("CH4", 0, 1),
    # Organic
    "C₂H₂ — Acetylene": ("C2H2", 0, 1),
    "C₂H₄ — Ethylene": ("C2H4", 0, 1),
    "C₂H₆ — Ethane": ("C2H6", 0, 1),
    "CH₃OH — Methanol": ("CH3OH", 0, 1),
    # Aromatic
    "C₆H₆ — Benzene": ("C6H6", 0, 1),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_ase_available() -> bool:
    """Return True if ASE is installed and importable."""
    return ASE_AVAILABLE


def molecule_to_atoms(molecule: Molecule) -> _ase_type.Atoms:
    """
    Convert a QuantUI Molecule to an ASE Atoms object.

    Charge and multiplicity are stored in ``atoms.info`` so they survive a
    round-trip through :func:`atoms_to_molecule`.

    Args:
        molecule: QuantUI Molecule object.

    Returns:
        ``ase.Atoms`` with identical symbols and positions (Ångströms).

    Raises:
        ImportError: If ASE is not installed.
    """
    if not ASE_AVAILABLE:
        raise ImportError(
            "ASE is not installed.\n"
            "  pip install ase\n"
            "  # or: conda install -c conda-forge ase"
        )
    from ase import Atoms  # type: ignore[import]

    atoms = Atoms(
        symbols=molecule.atoms,
        positions=molecule.coordinates,
    )
    atoms.info["charge"] = molecule.charge
    atoms.info["multiplicity"] = molecule.multiplicity
    return atoms


def atoms_to_molecule(
    atoms: _ase_type.Atoms,
    charge: int = 0,
    multiplicity: int = 1,
) -> Molecule:
    """
    Convert an ASE Atoms object to a QuantUI Molecule.

    Charge and multiplicity are read from ``atoms.info`` when present;
    otherwise the supplied default arguments are used.

    Args:
        atoms: ASE Atoms object.
        charge: Fallback total charge (used when not in ``atoms.info``).
        multiplicity: Fallback spin multiplicity (used when not in ``atoms.info``).

    Returns:
        Validated QuantUI Molecule object.
    """
    from .molecule import Molecule  # local import to avoid circular deps

    resolved_charge = int(atoms.info.get("charge", charge))
    resolved_mult = int(atoms.info.get("multiplicity", multiplicity))

    return Molecule(
        atoms=list(atoms.get_chemical_symbols()),
        coordinates=[list(map(float, pos)) for pos in atoms.get_positions()],
        charge=resolved_charge,
        multiplicity=resolved_mult,
    )


def read_structure_file(
    path: Union[str, Path],
    charge: int = 0,
    multiplicity: int = 1,
    **ase_kwargs: object,
) -> Molecule:
    """
    Read a molecular structure file using ``ase.io`` and return a Molecule.

    ASE auto-detects the format from the file extension.  Supported formats
    include XYZ (.xyz), PDB (.pdb), MDL MOL (.mol), SDF (.sdf), CIF (.cif),
    and many others.  MOL/SDF reading additionally requires rdkit.

    Args:
        path: Path to the structure file.
        charge: Total molecular charge (used when not encoded in the file).
        multiplicity: Spin multiplicity (used when not encoded in the file).
        **ase_kwargs: Extra keyword arguments forwarded to ``ase.io.read()``.

    Returns:
        QuantUI Molecule object.

    Raises:
        ImportError: If ASE is not installed.
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed.
    """
    if not ASE_AVAILABLE:
        raise ImportError(
            "ASE is not installed.\n"
            "  pip install ase\n"
            "  # or: conda install -c conda-forge ase"
        )
    import ase.io  # type: ignore[import]

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    try:
        atoms = ase.io.read(str(path), **ase_kwargs)
    except Exception as exc:
        raise ValueError(
            f"Could not read '{path.name}': {exc}\n"
            "Supported formats: XYZ, PDB, MOL, SDF, CIF, and others via ASE."
        ) from exc

    return atoms_to_molecule(atoms, charge=charge, multiplicity=multiplicity)


def ase_molecule_library(name: str) -> Molecule:
    """
    Fetch a molecule from ASE's built-in molecular library.

    Wraps ``ase.build.molecule()`` which provides pre-optimised geometries
    for ~100 small molecules including common diatomics, polyatomics, and
    simple organic molecules.

    Args:
        name: Molecule identifier as recognised by ``ase.build.molecule()``
              (e.g. ``'H2O'``, ``'CH4'``, ``'C6H6'``).

    Returns:
        QuantUI Molecule with the ASE-library geometry.

    Raises:
        ImportError: If ASE is not installed.
        KeyError: If the molecule name is not in the ASE library.
    """
    if not ASE_AVAILABLE:
        raise ImportError(
            "ASE is not installed.\n"
            "  pip install ase\n"
            "  # or: conda install -c conda-forge ase"
        )
    from ase.build import molecule as _ase_molecule  # type: ignore[import]

    try:
        atoms = _ase_molecule(name)
    except Exception as exc:
        raise KeyError(
            f"Molecule '{name}' not found in ASE library.\n"
            "See: https://wiki.fysik.dtu.dk/ase/ase/build/build.html#molecules"
        ) from exc

    logger.info("Loaded '%s' from ASE molecule library (%d atoms)", name, len(atoms))
    return atoms_to_molecule(atoms)
