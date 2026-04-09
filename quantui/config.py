"""
QuantUI-local Configuration Module

Configuration constants and defaults for the local teaching interface.
SLURM resource limits, job history paths, and cluster settings have been
removed — this version runs calculations in the current Jupyter session.
"""

from pathlib import Path
from typing import Dict, List, Any

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Supported quantum chemistry methods
SUPPORTED_METHODS = ["RHF", "UHF"]

# Supported basis sets
SUPPORTED_BASIS_SETS = [
    "STO-3G",
    "3-21G",
    "6-31G",
    "6-31G*",
    "6-31G**",
    "cc-pVDZ",
    "cc-pVTZ",
]

# Default calculation settings
DEFAULT_METHOD = "RHF"
DEFAULT_BASIS = "6-31G"
DEFAULT_CHARGE = 0
DEFAULT_MULTIPLICITY = 1

# Geometry optimization defaults
DEFAULT_FMAX: float = 0.05     # eV/Å force convergence threshold
DEFAULT_OPT_STEPS: int = 200   # maximum BFGS optimizer steps

# Widget styling
WIDGET_LAYOUT = {
    "width": "400px",
}

DESCRIPTION_WIDTH = "150px"

# Molecule presets — 20+ curated educational molecules
MOLECULE_LIBRARY: Dict[str, Dict[str, Any]] = {
    # ========== SIMPLE DIATOMIC MOLECULES ==========
    "H2": {
        "atoms": ["H", "H"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        "charge": 0,
        "multiplicity": 1,
        "description": "Hydrogen molecule (simplest molecule)",
    },
    "O2": {
        "atoms": ["O", "O"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.21]],
        "charge": 0,
        "multiplicity": 3,  # Triplet ground state
        "description": "Oxygen molecule (triplet ground state)",
    },
    "N2": {
        "atoms": ["N", "N"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.10]],
        "charge": 0,
        "multiplicity": 1,
        "description": "Nitrogen molecule (triple bond)",
    },
    "CO": {
        "atoms": ["C", "O"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]],
        "charge": 0,
        "multiplicity": 1,
        "description": "Carbon monoxide",
    },
    "HF": {
        "atoms": ["H", "F"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]],
        "charge": 0,
        "multiplicity": 1,
        "description": "Hydrogen fluoride (polar bond)",
    },
    "HCl": {
        "atoms": ["H", "Cl"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.27]],
        "charge": 0,
        "multiplicity": 1,
        "description": "Hydrogen chloride",
    },

    # ========== SIMPLE TRIATOMIC MOLECULES ==========
    "H2O": {
        "atoms": ["O", "H", "H"],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.587],
            [0.0, -0.757, 0.587],
        ],
        "charge": 0,
        "multiplicity": 1,
        "description": "Water molecule (bent geometry)",
    },
    "CO2": {
        "atoms": ["C", "O", "O"],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.16],
            [0.0, 0.0, -1.16],
        ],
        "charge": 0,
        "multiplicity": 1,
        "description": "Carbon dioxide (linear)",
    },
    "O3": {
        "atoms": ["O", "O", "O"],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [1.07, 0.65, 0.0],
            [-1.07, 0.65, 0.0],
        ],
        "charge": 0,
        "multiplicity": 1,
        "description": "Ozone (bent, resonance structure)",
    },
    "H2O2": {
        "atoms": ["H", "O", "O", "H"],
        "coordinates": [
            [0.74, -0.54, 0.48],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.45],
            [-0.74, -0.54, 1.93],
        ],
        "charge": 0,
        "multiplicity": 1,
        "description": "Hydrogen peroxide (non-planar)",
    },

    # ========== SIMPLE ORGANIC MOLECULES ==========
    "CH4": {
        "atoms": ["C", "H", "H", "H", "H"],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ],
        "charge": 0,
        "multiplicity": 1,
        "description": "Methane (tetrahedral)",
    },
    "NH3": {
        "atoms": ["N", "H", "H", "H"],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [0.0, 0.94, 0.33],
            [0.81, -0.47, 0.33],
            [-0.81, -0.47, 0.33],
        ],
        "charge": 0,
        "multiplicity": 1,
        "description": "Ammonia (pyramidal)",
    },
    "C2H6": {
        "atoms": ["C", "C", "H", "H", "H", "H", "H", "H"],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [-0.51, 0.89, 0.0],
            [-0.51, -0.44, 0.89],
            [-0.51, -0.44, -0.89],
            [2.05, 0.89, 0.0],
            [2.05, -0.44, 0.89],
            [2.05, -0.44, -0.89],
        ],
        "charge": 0,
        "multiplicity": 1,
        "description": "Ethane (single bond)",
    },
    "C2H4": {
        "atoms": ["C", "C", "H", "H", "H", "H"],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [1.34, 0.0, 0.0],
            [-0.51, 0.93, 0.0],
            [-0.51, -0.93, 0.0],
            [1.85, 0.93, 0.0],
            [1.85, -0.93, 0.0],
        ],
        "charge": 0,
        "multiplicity": 1,
        "description": "Ethylene (double bond, planar)",
    },
    "C2H2": {
        "atoms": ["C", "C", "H", "H"],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [1.20, 0.0, 0.0],
            [-1.06, 0.0, 0.0],
            [2.26, 0.0, 0.0],
        ],
        "charge": 0,
        "multiplicity": 1,
        "description": "Acetylene (triple bond, linear)",
    },
    "CH3OH": {
        "atoms": ["C", "O", "H", "H", "H", "H"],
        "coordinates": [
            [0.66, -0.02, 0.0],
            [-0.75, 0.09, 0.0],
            [1.03, -0.54, 0.89],
            [1.03, -0.54, -0.89],
            [1.05, 0.99, 0.0],
            [-1.07, -0.83, 0.0],
        ],
        "charge": 0,
        "multiplicity": 1,
        "description": "Methanol (alcohol)",
    },
    "CH2O": {
        "atoms": ["C", "O", "H", "H"],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.21],
            [0.94, 0.0, -0.59],
            [-0.94, 0.0, -0.59],
        ],
        "charge": 0,
        "multiplicity": 1,
        "description": "Formaldehyde (carbonyl group)",
    },
    "CH3CHO": {
        "atoms": ["C", "C", "O", "H", "H", "H", "H"],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [1.51, 0.0, 0.0],
            [2.17, 1.03, 0.0],
            [-0.36, -0.52, 0.89],
            [-0.36, -0.52, -0.89],
            [-0.39, 1.02, 0.0],
            [1.89, -1.02, 0.0],
        ],
        "charge": 0,
        "multiplicity": 1,
        "description": "Acetaldehyde (aldehyde group)",
    },
    "CH3COOH": {
        "atoms": ["C", "C", "O", "O", "H", "H", "H", "H"],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [1.51, 0.0, 0.0],
            [2.09, 1.09, 0.0],
            [2.16, -1.18, 0.0],
            [-0.36, -0.52, 0.89],
            [-0.36, -0.52, -0.89],
            [-0.39, 1.02, 0.0],
            [3.11, -1.09, 0.0],
        ],
        "charge": 0,
        "multiplicity": 1,
        "description": "Acetic acid (carboxylic acid)",
    },

    # ========== AROMATIC MOLECULES ==========
    "C6H6": {
        "atoms": ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
        "coordinates": [
            [0.0, 1.40, 0.0],
            [1.21, 0.70, 0.0],
            [1.21, -0.70, 0.0],
            [0.0, -1.40, 0.0],
            [-1.21, -0.70, 0.0],
            [-1.21, 0.70, 0.0],
            [0.0, 2.48, 0.0],
            [2.15, 1.24, 0.0],
            [2.15, -1.24, 0.0],
            [0.0, -2.48, 0.0],
            [-2.15, -1.24, 0.0],
            [-2.15, 1.24, 0.0],
        ],
        "charge": 0,
        "multiplicity": 1,
        "description": "Benzene (aromatic, hexagonal)",
    },
}

# Valid atomic symbols (periodic table subset commonly used)
VALID_ATOMS = [
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
]

# Quick-start templates for the notebook UI
# The `notes` and `learning_goals` fields are shown to students.
# The `calc_settings` fields pre-fill the method/basis dropdowns.
QUICK_START_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "beginner_water": {
        "name": "Beginner: Water Molecule",
        "description": "Perfect first calculation — fast and reliable",
        "molecule": {
            "xyz": "O  0.0  0.0  0.0\nH  0.757  0.587  0.0\nH  -0.757  0.587  0.0",
            "charge": 0,
            "multiplicity": 1,
        },
        "calc_settings": {"method": "RHF", "basis": "6-31G"},
        "notes": "Water (H2O) with 6-31G basis. Expected energy: ~-76.03 Ha.",
        "learning_goals": [
            "Run your first local calculation",
            "Understand basic molecule structure",
            "See SCF convergence in action",
        ],
    },
    "basis_comparison": {
        "name": "Learn: Basis Set Effect (CO2)",
        "description": "Compare different basis sets on carbon dioxide",
        "molecule": {
            "xyz": "C  0.0  0.0  0.0\nO  1.16  0.0  0.0\nO  -1.16  0.0  0.0",
            "charge": 0,
            "multiplicity": 1,
        },
        "calc_settings": {"method": "RHF", "basis": "STO-3G"},
        "notes": "Try with STO-3G first, then re-run with 6-31G to see the difference!",
        "learning_goals": [
            "Understand basis set accuracy trade-offs",
            "Compare computational costs",
            "See energy convergence with basis size",
        ],
    },
    "radical_oxygen": {
        "name": "Advanced: Oxygen Radical (O2)",
        "description": "Learn about multiplicity with triplet oxygen",
        "molecule": {
            "xyz": "O  0.0  0.0  0.0\nO  1.21  0.0  0.0",
            "charge": 0,
            "multiplicity": 3,
        },
        "calc_settings": {"method": "UHF", "basis": "6-31G"},
        "notes": "O2 has two unpaired electrons (triplet state). Try mult=1 first to see it fail!",
        "learning_goals": [
            "Understand multiplicity for radicals",
            "Learn UHF vs RHF",
            "See why O2 is paramagnetic",
        ],
    },
    "benzene": {
        "name": "Intermediate: Benzene Ring",
        "description": "Medium-sized aromatic molecule",
        "molecule": {
            "xyz": (
                "C  0.000  1.396  0.000\nC  1.209  0.698  0.000\nC  1.209 -0.698  0.000\n"
                "C  0.000 -1.396  0.000\nC -1.209 -0.698  0.000\nC -1.209  0.698  0.000\n"
                "H  0.000  2.479  0.000\nH  2.147  1.240  0.000\nH  2.147 -1.240  0.000\n"
                "H  0.000 -2.479  0.000\nH -2.147 -1.240  0.000\nH -2.147  1.240  0.000"
            ),
            "charge": 0,
            "multiplicity": 1,
        },
        "calc_settings": {"method": "RHF", "basis": "6-31G"},
        "notes": "Benzene (C6H6) — classic aromatic molecule. Larger system, takes longer locally.",
        "learning_goals": [
            "Work with larger molecules",
            "Understand aromatic systems",
            "See how calculation time scales with molecule size",
        ],
    },
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# PySCF calculation script template
# Used by calculator.py to generate a standalone .py file students can
# download and run independently ("Export Script" feature).
PYSCF_SCRIPT_TEMPLATE = """#!/usr/bin/env python3
\"\"\"
PySCF Calculation Script
Generated by QuantUI-local

Calculation: {job_name}
Method: {method}
Basis: {basis}
\"\"\"

import sys
from pathlib import Path
from pyscf import gto, scf
import numpy as np

def main():
    # Define molecule
    mol = gto.Mole()
    mol.atom = '''
{geometry}
    '''
    mol.basis = '{basis}'
    mol.charge = {charge}
    mol.spin = {spin}
    mol.build()

    print("=" * 60)
    print("Molecule Information")
    print("=" * 60)
    print(f"Number of electrons: {{mol.nelectron}}")
    print(f"Nuclear repulsion energy: {{mol.energy_nuc():.8f}} Ha")
    print()

    # Run calculation
    print("=" * 60)
    print("Starting {method}/{basis} Calculation")
    print("=" * 60)

    try:
        if '{method}' == 'RHF':
            mf = scf.RHF(mol)
        elif '{method}' == 'UHF':
            mf = scf.UHF(mol)
        else:
            raise ValueError(f"Unsupported method: {method}")

        mf.verbose = 4  # Detailed output
        energy = mf.kernel()

        if mf.converged:
            print()
            print("=" * 60)
            print("Calculation Results")
            print("=" * 60)
            print(f"SCF converged: Yes")
            print(f"Total energy: {{energy:.8f}} Ha")
            print(
                f"HOMO-LUMO gap: "
                f"{{(mf.mo_energy[mol.nelectron//2] - mf.mo_energy[mol.nelectron//2-1]) * 27.211:.4f}} eV"
            )
            print()

            # Save results next to the script so the path is predictable
            results_path = str(Path(__file__).parent / 'results.npz')
            np.savez(results_path,
                     energy=energy,
                     mo_energy=mf.mo_energy,
                     mo_coeff=mf.mo_coeff,
                     converged=mf.converged)
            print(f"Results saved to {{results_path}}")
            print("=" * 60)

            sys.exit(0)
        else:
            print("ERROR: SCF did not converge!")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: Calculation failed with exception: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
"""

# ============================================================================
# Local Configuration Override (optional)
# ============================================================================
# Create config.local.py in the project root to override settings without
# modifying this tracked file. config.local.py is git-ignored.
#
# Supported overrides: PUBCHEM_API_KEY, DEBUG
#
# Example config.local.py:
#   PUBCHEM_API_KEY = "your-key-here"
#   DEBUG = True

import os
import importlib.util

_local_config_path = PROJECT_ROOT / "config.local.py"

if _local_config_path.exists():
    spec = importlib.util.spec_from_file_location("config_local", _local_config_path)
    if spec and spec.loader:
        config_local = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_local)

        for attr in ("PUBCHEM_API_KEY", "DEBUG"):
            if hasattr(config_local, attr):
                value = getattr(config_local, attr)
                if value is not None:
                    globals()[attr] = value
