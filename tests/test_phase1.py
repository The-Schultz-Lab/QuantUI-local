#!/usr/bin/env python3
"""
QuantUI-local Smoke Test

Quick validation of core local functionality.
Ported from QuantUI test_phase1.py with SLURM-specific checks removed:
  - Storage (JobStorage, JobMetadata) — removed module
  - Resource estimation (estimate_resources) — SLURM-only feature
  - PlotlyMol visualization — py3Dmol only in local version
"""

import sys
import pytest
from pathlib import Path


def _run_imports_check() -> bool:
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import quantui
        from quantui import config, utils, molecule, calculator
        print("  All imports successful")
        return True
    except Exception as e:
        print(f"  Import failed: {e}")
        return False


def test_imports():
    """Test that all modules can be imported."""
    assert _run_imports_check()


def _run_molecule_check() -> bool:
    """Test molecule creation and validation."""
    print("\nTesting molecule module...")
    try:
        from quantui import Molecule, parse_xyz_input

        # Test parsing
        xyz_text = """
        O  0.0  0.0  0.0
        H  0.757  0.587  0.0
        H  -0.757  0.587  0.0
        """
        atoms, coords = parse_xyz_input(xyz_text)
        print(f"  Parsed {len(atoms)} atoms")

        # Test molecule creation
        mol = Molecule(atoms, coords, charge=0, multiplicity=1)
        print(f"  Created molecule: {mol.get_formula()}")
        print(f"  Electrons: {mol.get_electron_count()}")

        # Test PySCF format
        pyscf_format = mol.to_pyscf_format()
        print(f"  PySCF format generated ({len(pyscf_format)} chars)")

        print("  Molecule module working")
        return True

    except Exception as e:
        print(f"  Molecule test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_molecule():
    """Test molecule creation and validation."""
    assert _run_molecule_check()


def _run_calculator_check() -> bool:
    """Test calculation script generation."""
    print("\nTesting calculator module...")
    try:
        from quantui import Molecule, create_calculation

        # Create simple molecule
        atoms = ['H', 'H']
        coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
        mol = Molecule(atoms, coords)

        # Create calculation
        calc = create_calculation(mol, 'RHF', '6-31G')
        print(f"  Created calculation: {calc.method}/{calc.basis}")

        # Test script generation
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "test_calc.py"
            calc.generate_calculation_script(script_path)
            print(f"  Generated script: {script_path.stat().st_size} bytes")

        print("  Calculator module working")
        return True

    except Exception as e:
        print(f"  Calculator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_calculator():
    """Test calculation script generation."""
    assert _run_calculator_check()


def _run_utils_check() -> bool:
    """Test utility functions."""
    print("\nTesting utils module...")
    try:
        from quantui import utils

        # Test username detection
        try:
            username = utils.get_username()
            print(f"  Detected username: {username}")
        except RuntimeError:
            print("  Username detection skipped (no env vars)")

        # Test sanitization
        sanitized = utils.sanitize_filename("Test File! @#$.txt")
        print(f"  Sanitized filename: {sanitized}")

        # Test validation
        assert utils.validate_atom_symbol('H') is True
        assert utils.validate_atom_symbol('Xx') is False
        print("  Validation functions working")

        # Test session resource detection
        cores, mem = utils.get_session_resources()
        print(f"  Session resources: {cores} core(s), {mem} GB RAM")

        print("  Utils module working")
        return True

    except Exception as e:
        print(f"  Utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions."""
    assert _run_utils_check()


def _run_visualization_check() -> bool:
    """Test py3Dmol visualization module (optional dependency)."""
    print("\nTesting visualization module...")
    try:
        from quantui.visualization_py3dmol import (
            is_visualization_available,
            visualize_molecule,
            display_molecule,
            PY3DMOL_AVAILABLE,
        )
        print("  Visualization module imports successfully")

        from quantui import Molecule
        mol = Molecule(
            atoms=["O", "H", "H"],
            coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
            charge=0,
            multiplicity=1,
        )

        available = is_visualization_available()
        print(f"  Visualization availability: {available}")
        print(f"  py3Dmol installed: {PY3DMOL_AVAILABLE}")

        if available and PY3DMOL_AVAILABLE:
            view = visualize_molecule(mol)
            assert view is not None
            print("  Molecule visualization works")
        else:
            print("  py3Dmol not installed (optional dependency)")
            print("  To enable: pip install py3Dmol")

        print("  Visualization module tests passed")
        return True

    except ImportError as e:
        print(f"  Visualization module not available: {e}")
        print("  This is OK — visualization is optional")
        return True  # Not a failure
    except Exception as e:
        print(f"  Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization module (optional dependency)."""
    assert _run_visualization_check()


def _run_session_calc_check() -> bool:
    """Test in-session PySCF runner availability (optional — Linux/WSL only)."""
    print("\nTesting in-session calculator...")
    try:
        from quantui import run_in_session, SessionResult
        print("  run_in_session imported successfully (PySCF available)")
        return True
    except (ImportError, AttributeError):
        print("  PySCF not available (Linux/WSL required) — OK on Windows")
        return True


def test_session_calc():
    """Test in-session calculator import (optional)."""
    assert _run_session_calc_check()


def main():
    """Run all checks and print summary."""
    print("=" * 60)
    print("QuantUI-local Smoke Test")
    print("=" * 60)

    results = []

    results.append(("Imports", _run_imports_check()))
    results.append(("Utils", _run_utils_check()))
    results.append(("Molecule", _run_molecule_check()))
    results.append(("Calculator", _run_calculator_check()))
    results.append(("Visualization", _run_visualization_check()))
    results.append(("Session calc", _run_session_calc_check()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:20s} {status}")

    all_passed = all(r[1] for r in results)

    print("=" * 60)
    if all_passed:
        print("All checks passed! Core functionality is working.")
        print("\nNote: PySCF (Linux/WSL only) tested via import check only")
        return 0
    else:
        print("Some checks failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
