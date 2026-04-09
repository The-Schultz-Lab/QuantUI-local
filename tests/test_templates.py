"""
Tests for QUICK_START_TEMPLATES in quantui/config.py.

Validates that the template data is well-formed, uses only supported
methods/basis sets, and can be used to create valid Molecule objects.
"""

import pytest
from quantui.config import QUICK_START_TEMPLATES, SUPPORTED_METHODS, SUPPORTED_BASIS_SETS
from quantui import Molecule, parse_xyz_input

# 'resources' key removed from local templates (no SLURM resource allocation needed)
# 'job_name' removed from calc_settings (no batch job submission)
EXPECTED_KEYS = {"beginner_water", "basis_comparison", "radical_oxygen", "benzene"}
REQUIRED_FIELDS = {"name", "description", "molecule", "calc_settings", "learning_goals"}
REQUIRED_MOLECULE_FIELDS = {"xyz", "charge", "multiplicity"}
REQUIRED_CALC_FIELDS = {"method", "basis"}


class TestQuickStartTemplatesStructure:

    def test_all_expected_keys_present(self):
        assert set(QUICK_START_TEMPLATES.keys()) == EXPECTED_KEYS

    def test_each_template_has_required_top_level_fields(self):
        for tid, t in QUICK_START_TEMPLATES.items():
            missing = REQUIRED_FIELDS - set(t.keys())
            assert not missing, f"{tid} missing fields: {missing}"

    def test_each_molecule_has_required_fields(self):
        for tid, t in QUICK_START_TEMPLATES.items():
            missing = REQUIRED_MOLECULE_FIELDS - set(t["molecule"].keys())
            assert not missing, f"{tid}.molecule missing fields: {missing}"

    def test_each_calc_settings_has_required_fields(self):
        for tid, t in QUICK_START_TEMPLATES.items():
            missing = REQUIRED_CALC_FIELDS - set(t["calc_settings"].keys())
            assert not missing, f"{tid}.calc_settings missing fields: {missing}"

    def test_learning_goals_is_non_empty_list(self):
        for tid, t in QUICK_START_TEMPLATES.items():
            goals = t["learning_goals"]
            assert isinstance(goals, list) and len(goals) > 0, \
                f"{tid} has empty or invalid learning_goals"


class TestQuickStartTemplatesValidity:

    def test_methods_are_supported(self):
        for tid, t in QUICK_START_TEMPLATES.items():
            method = t["calc_settings"]["method"]
            assert method in SUPPORTED_METHODS, \
                f"{tid} uses unsupported method '{method}'"

    def test_basis_sets_are_supported(self):
        for tid, t in QUICK_START_TEMPLATES.items():
            basis = t["calc_settings"]["basis"]
            assert basis in SUPPORTED_BASIS_SETS, \
                f"{tid} uses unsupported basis '{basis}'"

    def test_charges_are_integers(self):
        for tid, t in QUICK_START_TEMPLATES.items():
            charge = t["molecule"]["charge"]
            assert isinstance(charge, int), f"{tid} charge is not int: {charge!r}"

    def test_multiplicities_are_positive_integers(self):
        for tid, t in QUICK_START_TEMPLATES.items():
            mult = t["molecule"]["multiplicity"]
            assert isinstance(mult, int) and mult >= 1, \
                f"{tid} multiplicity invalid: {mult!r}"

    def test_xyz_can_be_parsed_to_molecule(self):
        """Each template XYZ string should parse into a valid Molecule."""
        for tid, t in QUICK_START_TEMPLATES.items():
            xyz = t["molecule"]["xyz"]
            charge = t["molecule"]["charge"]
            multiplicity = t["molecule"]["multiplicity"]
            try:
                atoms, coords = parse_xyz_input(xyz)
                mol = Molecule(
                    atoms=atoms,
                    coordinates=coords,
                    charge=charge,
                    multiplicity=multiplicity,
                )
                assert len(atoms) > 0, f"{tid} produced empty atom list"
            except Exception as e:
                pytest.fail(f"{tid} XYZ failed to parse into Molecule: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
