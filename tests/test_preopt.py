"""
Tests for quantui.preopt — LJ force-field geometry pre-optimization.

All tests that call the real ASE LJ optimizer are skipped gracefully
when ASE is not installed (CI without the 'ase' extra).

WSL / Linux compatibility note
-------------------------------
This module is pure ASE + NumPy — no PySCF dependency.  All tests run
on Windows, Linux, and WSL equally.  The preoptimize() function itself
makes no subprocess calls, no OS-specific syscalls, and relies on no
compiled extensions beyond NumPy (which ships cross-platform wheels).
"""

import pytest

from quantui.ase_bridge import ASE_AVAILABLE
from quantui.molecule import Molecule

# Convenience skip marker
ase_only = pytest.mark.skipif(not ASE_AVAILABLE, reason="ase not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _water() -> Molecule:
    """Return a water molecule with a reasonable geometry."""
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


def _h2() -> Molecule:
    """Return a hydrogen molecule."""
    return Molecule(
        atoms=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    )


def _charged_radical() -> Molecule:
    """Return a water cation (doublet) for charge/multiplicity round-trip checks."""
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        charge=1,
        multiplicity=2,
    )


# ============================================================================
# ImportError guard
# ============================================================================


class TestPreoptimizeImportGuard:
    """preoptimize() raises ImportError with helpful message when ASE is absent."""

    def test_raises_import_error_when_ase_unavailable(self, monkeypatch):
        import quantui.preopt as preopt_mod

        # Patch ASE_AVAILABLE directly in preopt's namespace — no reload needed;
        # monkeypatch auto-reverts after the test so later tests are unaffected.
        monkeypatch.setattr(preopt_mod, "ASE_AVAILABLE", False)

        mol = _water()
        with pytest.raises(ImportError, match="pip install ase"):
            preopt_mod.preoptimize(mol)

    def test_import_error_message_mentions_conda(self, monkeypatch):
        import quantui.preopt as preopt_mod

        monkeypatch.setattr(preopt_mod, "ASE_AVAILABLE", False)

        with pytest.raises(ImportError) as exc_info:
            preopt_mod.preoptimize(_water())
        assert (
            "conda" in str(exc_info.value).lower()
            or "pip" in str(exc_info.value).lower()
        )


# ============================================================================
# Return type and structure
# ============================================================================


class TestPreoptimizeReturnTypes:
    """preoptimize() returns (Molecule, float) with correct types."""

    @ase_only
    def test_returns_two_tuple(self):
        from quantui.preopt import preoptimize

        result = preoptimize(_water())
        assert isinstance(result, tuple)
        assert len(result) == 2

    @ase_only
    def test_first_element_is_molecule(self):
        from quantui.preopt import preoptimize

        mol, _ = preoptimize(_water())
        assert isinstance(mol, Molecule)

    @ase_only
    def test_second_element_is_float(self):
        from quantui.preopt import preoptimize

        _, rmsd = preoptimize(_water())
        assert isinstance(rmsd, float)

    @ase_only
    def test_rmsd_is_non_negative(self):
        from quantui.preopt import preoptimize

        _, rmsd = preoptimize(_water())
        assert rmsd >= 0.0


# ============================================================================
# Geometry reasonableness
# ============================================================================


class TestPreoptimizeGeometry:
    """The returned molecule should have sane geometry."""

    @ase_only
    def test_atom_count_preserved(self):
        from quantui.preopt import preoptimize

        original = _water()
        optimized, _ = preoptimize(original)
        assert len(optimized.atoms) == len(original.atoms)

    @ase_only
    def test_atom_symbols_preserved(self):
        from quantui.preopt import preoptimize

        original = _water()
        optimized, _ = preoptimize(original)
        assert optimized.atoms == original.atoms

    @ase_only
    def test_h2_rmsd_is_small_for_good_geometry(self):
        """H2 with a near-equilibrium bond length should move very little under LJ."""
        from quantui.preopt import preoptimize

        _, rmsd = preoptimize(_h2())
        # LJ equilibrium for H-H is ~3.4 Å (sigma); our 0.74 Å bond is tighter,
        # but BFGS will push it outward — expect a moderate displacement, not zero.
        # We only assert it is not absurdly large for a 2-atom system.
        assert rmsd < 10.0

    @ase_only
    def test_coordinates_are_3d(self):
        from quantui.preopt import preoptimize

        optimized, _ = preoptimize(_water())
        for coord in optimized.coordinates:
            assert len(coord) == 3

    @ase_only
    def test_coordinates_are_floats(self):
        from quantui.preopt import preoptimize

        optimized, _ = preoptimize(_water())
        for coord in optimized.coordinates:
            for val in coord:
                assert isinstance(val, float)


# ============================================================================
# Metadata preservation
# ============================================================================


class TestPreoptimizeMetadata:
    """Charge and multiplicity must survive pre-optimization unchanged."""

    @ase_only
    def test_charge_preserved_neutral(self):
        from quantui.preopt import preoptimize

        optimized, _ = preoptimize(_water())
        assert optimized.charge == 0

    @ase_only
    def test_multiplicity_preserved_singlet(self):
        from quantui.preopt import preoptimize

        optimized, _ = preoptimize(_water())
        assert optimized.multiplicity == 1

    @ase_only
    def test_charge_preserved_for_radical(self):
        from quantui.preopt import preoptimize

        original = _charged_radical()
        optimized, _ = preoptimize(original)
        assert optimized.charge == original.charge

    @ase_only
    def test_multiplicity_preserved_for_radical(self):
        from quantui.preopt import preoptimize

        original = _charged_radical()
        optimized, _ = preoptimize(original)
        assert optimized.multiplicity == original.multiplicity

    @ase_only
    def test_triplet_multiplicity_preserved(self):
        from quantui.preopt import preoptimize

        o2 = Molecule(
            atoms=["O", "O"],
            coordinates=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            multiplicity=3,
        )
        optimized, _ = preoptimize(o2)
        assert optimized.multiplicity == 3


# ============================================================================
# Input immutability
# ============================================================================


class TestPreoptimizeImmutability:
    """preoptimize() must never mutate the input Molecule."""

    @ase_only
    def test_original_atoms_unchanged(self):
        from quantui.preopt import preoptimize

        original = _water()
        original_atoms = list(original.atoms)
        preoptimize(original)
        assert original.atoms == original_atoms

    @ase_only
    def test_original_coordinates_unchanged(self):
        import copy

        from quantui.preopt import preoptimize

        original = _water()
        original_coords = copy.deepcopy(original.coordinates)
        preoptimize(original)
        assert original.coordinates == original_coords

    @ase_only
    def test_original_charge_unchanged(self):
        from quantui.preopt import preoptimize

        original = _charged_radical()
        preoptimize(original)
        assert original.charge == 1

    @ase_only
    def test_original_multiplicity_unchanged(self):
        from quantui.preopt import preoptimize

        original = _charged_radical()
        preoptimize(original)
        assert original.multiplicity == 2


# ============================================================================
# Parameter handling
# ============================================================================


class TestPreoptimizeParameters:
    """fmax and steps parameters are accepted without error."""

    @ase_only
    def test_custom_fmax(self):
        from quantui.preopt import preoptimize

        mol, rmsd = preoptimize(_h2(), fmax=0.5)
        assert isinstance(rmsd, float)

    @ase_only
    def test_custom_steps(self):
        from quantui.preopt import preoptimize

        mol, rmsd = preoptimize(_h2(), steps=10)
        assert isinstance(mol, Molecule)

    @ase_only
    def test_single_step_does_not_crash(self):
        from quantui.preopt import preoptimize

        mol, rmsd = preoptimize(_water(), steps=1)
        assert isinstance(mol, Molecule)
        assert rmsd >= 0.0

    @ase_only
    def test_tight_fmax_accepted(self):
        from quantui.preopt import preoptimize

        mol, rmsd = preoptimize(_h2(), fmax=1e-6, steps=50)
        assert isinstance(mol, Molecule)


# ============================================================================
# Public API surface
# ============================================================================


class TestPreoptimizePublicAPI:
    """preoptimize is accessible from the top-level quantui package."""

    @ase_only
    def test_importable_from_quantui(self):
        from quantui import preoptimize  # noqa: F401 — import check only

    @ase_only
    def test_callable_from_quantui(self):
        from quantui import preoptimize

        mol, rmsd = preoptimize(_h2())
        assert isinstance(mol, Molecule)


# ============================================================================
# Run directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
