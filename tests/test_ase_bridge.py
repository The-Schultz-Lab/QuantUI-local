"""
Tests for quantui.ase_bridge

Covers ASE availability guard, Molecule ↔ ASE Atoms round-trips,
structure file reading via ase.io, and the curated molecule library.

All tests that exercise live ASE functionality are skipped gracefully
when ASE is not installed (CI without the 'ase' extra).
"""

import pytest

from quantui.ase_bridge import (
    ASE_AVAILABLE,
    ASE_MOLECULE_PRESETS,
    ase_molecule_library,
    atoms_to_molecule,
    is_ase_available,
    molecule_to_atoms,
    read_structure_file,
)
from quantui.molecule import Molecule

# Convenience marker — skip the test when ASE is absent
ase_only = pytest.mark.skipif(not ASE_AVAILABLE, reason="ase not installed")


# ============================================================================
# is_ase_available / ASE_AVAILABLE flag
# ============================================================================


class TestIsAseAvailable:
    """Unit tests for is_ase_available() and the module-level ASE_AVAILABLE flag."""

    def test_returns_bool(self):
        assert isinstance(is_ase_available(), bool)

    def test_matches_module_flag(self):
        assert is_ase_available() == ASE_AVAILABLE


# ============================================================================
# ASE_MOLECULE_PRESETS dictionary
# ============================================================================


class TestAseMoleculePresets:
    """Tests for the curated ASE_MOLECULE_PRESETS dictionary structure."""

    def test_is_dict(self):
        assert isinstance(ASE_MOLECULE_PRESETS, dict)

    def test_non_empty(self):
        assert len(ASE_MOLECULE_PRESETS) > 0

    def test_values_are_three_tuples(self):
        for label, entry in ASE_MOLECULE_PRESETS.items():
            assert isinstance(entry, tuple), f"'{label}': expected tuple, got {type(entry)}"
            assert len(entry) == 3, f"'{label}': expected 3-tuple (ase_name, charge, mult)"

    def test_ase_names_are_nonempty_strings(self):
        for label, (ase_name, charge, mult) in ASE_MOLECULE_PRESETS.items():
            assert isinstance(ase_name, str) and ase_name, f"'{label}': ase_name must be non-empty string"

    def test_charges_are_ints(self):
        for label, (_, charge, _) in ASE_MOLECULE_PRESETS.items():
            assert isinstance(charge, int), f"'{label}': charge must be int"

    def test_multiplicities_are_positive_ints(self):
        for label, (_, _, mult) in ASE_MOLECULE_PRESETS.items():
            assert isinstance(mult, int) and mult >= 1, f"'{label}': multiplicity must be int >= 1"

    def test_contains_water(self):
        ase_names = {v[0] for v in ASE_MOLECULE_PRESETS.values()}
        assert "H2O" in ase_names

    def test_contains_hydrogen(self):
        ase_names = {v[0] for v in ASE_MOLECULE_PRESETS.values()}
        assert "H2" in ase_names

    def test_contains_methane(self):
        ase_names = {v[0] for v in ASE_MOLECULE_PRESETS.values()}
        assert "CH4" in ase_names

    def test_o2_is_triplet(self):
        """O2 ground state is a triplet — verify preset encodes this correctly."""
        o2_entries = [(label, v) for label, v in ASE_MOLECULE_PRESETS.items() if v[0] == "O2"]
        assert o2_entries, "O2 should be in ASE_MOLECULE_PRESETS"
        _, (_, _, mult) = o2_entries[0]
        assert mult == 3, "O2 ground state should be multiplicity 3 (triplet)"

    @ase_only
    def test_all_presets_loadable(self):
        """Every preset label should produce a valid Molecule via ase_molecule_library."""
        for label, (ase_name, _charge, _mult) in ASE_MOLECULE_PRESETS.items():
            mol = ase_molecule_library(ase_name)
            assert isinstance(mol, Molecule), f"Preset '{label}' ({ase_name}) did not return a Molecule"
            assert len(mol.atoms) > 0, f"Preset '{label}' ({ase_name}) returned empty Molecule"


# ============================================================================
# molecule_to_atoms
# ============================================================================


class TestMoleculeToAtoms:
    """Tests for molecule_to_atoms()."""

    def test_raises_import_error_when_ase_unavailable(self, monkeypatch):
        import quantui.ase_bridge as bridge

        monkeypatch.setattr(bridge, "ASE_AVAILABLE", False)
        mol = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        with pytest.raises(ImportError, match="pip install ase"):
            molecule_to_atoms(mol)

    @ase_only
    def test_symbol_count_h2(self):
        mol = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        atoms = molecule_to_atoms(mol)
        assert len(atoms) == 2
        assert list(atoms.get_chemical_symbols()) == ["H", "H"]

    @ase_only
    def test_symbol_count_water(self):
        mol = Molecule(
            ["O", "H", "H"],
            [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        )
        atoms = molecule_to_atoms(mol)
        assert list(atoms.get_chemical_symbols()) == ["O", "H", "H"]

    @ase_only
    def test_positions_preserved(self):
        import numpy as np

        coords = [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]]
        mol = Molecule(["O", "H", "H"], coords)
        atoms = molecule_to_atoms(mol)
        np.testing.assert_allclose(atoms.get_positions(), coords, atol=1e-6)

    @ase_only
    def test_charge_stored_in_atoms_info(self):
        mol = Molecule(
            ["O", "H", "H"],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            charge=1,
            multiplicity=2,
        )
        atoms = molecule_to_atoms(mol)
        assert atoms.info["charge"] == 1

    @ase_only
    def test_multiplicity_stored_in_atoms_info(self):
        mol = Molecule(
            ["O", "O"],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            multiplicity=3,
        )
        atoms = molecule_to_atoms(mol)
        assert atoms.info["multiplicity"] == 3


# ============================================================================
# atoms_to_molecule
# ============================================================================


class TestAtomsToMolecule:
    """Tests for atoms_to_molecule()."""

    @ase_only
    def test_basic_water_symbols(self):
        from ase import Atoms

        coords = [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]]
        atoms = Atoms(symbols=["O", "H", "H"], positions=coords)
        mol = atoms_to_molecule(atoms)
        assert isinstance(mol, Molecule)
        assert mol.atoms == ["O", "H", "H"]

    @ase_only
    def test_reads_charge_from_info(self):
        from ase import Atoms

        atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        )
        atoms.info["charge"] = -1
        atoms.info["multiplicity"] = 2
        mol = atoms_to_molecule(atoms)
        assert mol.charge == -1
        assert mol.multiplicity == 2

    @ase_only
    def test_fallback_defaults_when_info_absent(self):
        from ase import Atoms

        atoms = Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [0, 0, 0.74]])
        mol = atoms_to_molecule(atoms, charge=0, multiplicity=1)
        assert mol.charge == 0
        assert mol.multiplicity == 1

    @ase_only
    def test_info_overrides_defaults(self):
        from ase import Atoms

        atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        )
        atoms.info["charge"] = 1
        atoms.info["multiplicity"] = 2
        # Pass different defaults — info should win
        mol = atoms_to_molecule(atoms, charge=0, multiplicity=1)
        assert mol.charge == 1
        assert mol.multiplicity == 2

    @ase_only
    def test_positions_roundtrip(self):
        import numpy as np
        from ase import Atoms

        coords = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
        atoms = Atoms(symbols=["N", "N"], positions=coords)
        mol = atoms_to_molecule(atoms)
        for original, result in zip(coords, mol.coordinates):
            np.testing.assert_allclose(result, original, atol=1e-6)

    @ase_only
    def test_returns_valid_molecule(self):
        """atoms_to_molecule should produce a Molecule that passes internal validation."""
        from ase import Atoms

        atoms = Atoms(symbols=["C", "H", "H", "H", "H"], positions=[[0, 0, 0]] * 5)
        mol = atoms_to_molecule(atoms)
        assert mol.get_electron_count() > 0


# ============================================================================
# Round-trip: Molecule → Atoms → Molecule
# ============================================================================


class TestMoleculeAtomsRoundTrip:
    """Verify that converting Molecule → Atoms → Molecule is lossless."""

    @ase_only
    def test_roundtrip_water(self):
        original = Molecule(
            atoms=["O", "H", "H"],
            coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
            charge=0,
            multiplicity=1,
        )
        restored = atoms_to_molecule(molecule_to_atoms(original))
        assert restored.atoms == original.atoms
        assert restored.charge == original.charge
        assert restored.multiplicity == original.multiplicity

    @ase_only
    def test_roundtrip_preserves_positions(self):
        import numpy as np

        coords = [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]]
        original = Molecule(["O", "H", "H"], coords)
        restored = atoms_to_molecule(molecule_to_atoms(original))
        np.testing.assert_allclose(restored.coordinates, original.coordinates, atol=1e-6)

    @ase_only
    def test_roundtrip_charged_radical(self):
        original = Molecule(
            atoms=["O", "H", "H"],
            coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            charge=1,
            multiplicity=2,
        )
        restored = atoms_to_molecule(molecule_to_atoms(original))
        assert restored.charge == 1
        assert restored.multiplicity == 2

    @ase_only
    def test_roundtrip_triplet(self):
        original = Molecule(
            atoms=["O", "O"],
            coordinates=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            multiplicity=3,
        )
        restored = atoms_to_molecule(molecule_to_atoms(original))
        assert restored.multiplicity == 3


# ============================================================================
# read_structure_file
# ============================================================================


class TestReadStructureFile:
    """Tests for read_structure_file()."""

    def test_raises_import_error_when_ase_unavailable(self, monkeypatch, tmp_path):
        import quantui.ase_bridge as bridge

        monkeypatch.setattr(bridge, "ASE_AVAILABLE", False)
        fake_file = tmp_path / "mol.xyz"
        fake_file.write_text("2\n\nH 0 0 0\nH 0 0 1\n")
        with pytest.raises(ImportError, match="pip install ase"):
            read_structure_file(fake_file)

    @ase_only
    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_structure_file(tmp_path / "nonexistent.xyz")

    @ase_only
    def test_read_xyz_water(self, tmp_path):
        xyz_content = (
            "3\n"
            "Water\n"
            "O  0.0  0.0  0.0\n"
            "H  0.757  0.587  0.0\n"
            "H  -0.757  0.587  0.0\n"
        )
        xyz_file = tmp_path / "water.xyz"
        xyz_file.write_text(xyz_content)
        mol = read_structure_file(xyz_file)
        assert isinstance(mol, Molecule)
        assert mol.atoms == ["O", "H", "H"]

    @ase_only
    def test_read_xyz_h2_with_explicit_charge_mult(self, tmp_path):
        xyz_content = "2\nH2\nH  0.0  0.0  0.0\nH  0.0  0.0  0.74\n"
        xyz_file = tmp_path / "h2.xyz"
        xyz_file.write_text(xyz_content)
        mol = read_structure_file(xyz_file, charge=0, multiplicity=1)
        assert mol.atoms == ["H", "H"]
        assert mol.charge == 0
        assert mol.multiplicity == 1

    @ase_only
    def test_read_xyz_methane(self, tmp_path):
        xyz_content = (
            "5\n"
            "Methane\n"
            "C   0.000   0.000   0.000\n"
            "H   0.631   0.631   0.631\n"
            "H  -0.631  -0.631   0.631\n"
            "H  -0.631   0.631  -0.631\n"
            "H   0.631  -0.631  -0.631\n"
        )
        xyz_file = tmp_path / "methane.xyz"
        xyz_file.write_text(xyz_content)
        mol = read_structure_file(xyz_file)
        assert mol.get_formula() == "CH4"
        assert len(mol.atoms) == 5

    @ase_only
    def test_read_invalid_raises_value_error(self, tmp_path):
        bad_file = tmp_path / "bad.xyz"
        bad_file.write_text("this is not a valid xyz file at all\n????garbage\n")
        with pytest.raises(ValueError, match="Could not read"):
            read_structure_file(bad_file)

    @ase_only
    def test_accepts_string_path(self, tmp_path):
        xyz_content = "2\nH2\nH  0.0  0.0  0.0\nH  0.0  0.0  0.74\n"
        xyz_file = tmp_path / "h2.xyz"
        xyz_file.write_text(xyz_content)
        mol = read_structure_file(str(xyz_file))  # str, not Path
        assert mol.atoms == ["H", "H"]


# ============================================================================
# ase_molecule_library
# ============================================================================


class TestAseMoleculeLibrary:
    """Tests for ase_molecule_library()."""

    def test_raises_import_error_when_ase_unavailable(self, monkeypatch):
        import quantui.ase_bridge as bridge

        monkeypatch.setattr(bridge, "ASE_AVAILABLE", False)
        with pytest.raises(ImportError, match="pip install ase"):
            ase_molecule_library("H2O")

    @ase_only
    def test_load_water(self):
        mol = ase_molecule_library("H2O")
        assert isinstance(mol, Molecule)
        assert mol.get_formula() == "H2O"

    @ase_only
    def test_load_h2(self):
        mol = ase_molecule_library("H2")
        assert mol.get_formula() == "H2"
        assert len(mol.atoms) == 2

    @ase_only
    def test_load_methane(self):
        mol = ase_molecule_library("CH4")
        assert mol.get_formula() == "CH4"
        assert len(mol.atoms) == 5

    @ase_only
    def test_load_ammonia(self):
        mol = ase_molecule_library("NH3")
        assert mol.get_formula() == "H3N"
        assert len(mol.atoms) == 4

    @ase_only
    def test_load_benzene(self):
        mol = ase_molecule_library("C6H6")
        assert mol.get_formula() == "C6H6"
        assert len(mol.atoms) == 12

    @ase_only
    def test_invalid_name_raises_key_error(self):
        with pytest.raises(KeyError, match="not found in ASE library"):
            ase_molecule_library("NotAMolecule_XYZ12345")

    @ase_only
    def test_returns_valid_molecule_with_electrons(self):
        mol = ase_molecule_library("NH3")
        assert mol.get_electron_count() > 0

    @ase_only
    def test_all_coordinates_are_3d(self):
        mol = ase_molecule_library("H2O")
        assert all(len(c) == 3 for c in mol.coordinates)

    @ase_only
    def test_default_charge_is_zero(self):
        mol = ase_molecule_library("H2O")
        assert mol.charge == 0

    @ase_only
    def test_default_multiplicity_is_singlet(self):
        mol = ase_molecule_library("H2O")
        assert mol.multiplicity == 1


# ============================================================================
# Run directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
