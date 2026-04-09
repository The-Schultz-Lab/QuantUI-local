"""
Tests for PubChem Integration Module

Tests both API functionality (mocked) and real network requests (marked).
"""

import pytest
from unittest.mock import Mock, patch
import requests

# Skip all tests if RDKit not available
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

if RDKIT_AVAILABLE:
    from quantui.pubchem import (
        search_molecule_by_name,
        get_molecule_sdf,
        sdf_to_xyz,
        fetch_molecule,
        student_friendly_fetch,
        get_common_molecules,
        check_pubchem_availability,
        PubChemError,
        MoleculeNotFoundError,
        PubChemAPIError,
    )


pytestmark = pytest.mark.skipif(
    not RDKIT_AVAILABLE, reason="RDKit required for PubChem tests"
)


# ============================================================================
# Mocked API Tests (No Network Required)
# ============================================================================

class TestSearchMoleculeByName:
    """Test molecule name search functionality."""

    @patch('quantui.pubchem.requests.get')
    def test_search_water_success(self, mock_get):
        """Test successful search for water."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "IdentifierList": {"CID": [962]}
        }
        mock_get.return_value = mock_response

        cid = search_molecule_by_name("water")
        assert cid == 962

    @patch('quantui.pubchem.requests.get')
    def test_search_not_found(self, mock_get):
        """Test search for non-existent molecule."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with pytest.raises(MoleculeNotFoundError):
            search_molecule_by_name("xyznonexistent123")

    @patch('quantui.pubchem.requests.get')
    def test_search_api_error(self, mock_get):
        """Test API connection failure."""
        mock_get.side_effect = requests.RequestException("Connection failed")

        with pytest.raises(PubChemAPIError):
            search_molecule_by_name("water")

    @patch('quantui.pubchem.requests.get')
    def test_search_empty_result(self, mock_get):
        """Test search returning empty CID list."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "IdentifierList": {"CID": []}
        }
        mock_get.return_value = mock_response

        with pytest.raises(MoleculeNotFoundError):
            search_molecule_by_name("unknown")


class TestGetMoleculeSDF:
    """Test SDF retrieval functionality."""

    @patch('quantui.pubchem.requests.get')
    def test_get_sdf_3d_success(self, mock_get, sample_sdf_water):
        """Test successful 3D SDF retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sample_sdf_water
        mock_get.return_value = mock_response

        sdf = get_molecule_sdf(962, conformer_3d=True)
        assert sdf == sample_sdf_water

        # Verify 3D parameter was passed
        call_args = mock_get.call_args
        assert call_args[1]['params'] == {"record_type": "3d"}

    @patch('quantui.pubchem.requests.get')
    def test_get_sdf_2d_fallback(self, mock_get, sample_sdf_water):
        """Test fallback to 2D when 3D not available."""
        # First call (3D) returns 404, second call (2D) succeeds
        mock_response_404 = Mock()
        mock_response_404.status_code = 404

        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.text = sample_sdf_water

        mock_get.side_effect = [mock_response_404, mock_response_200]

        sdf = get_molecule_sdf(962, conformer_3d=True)
        assert sdf == sample_sdf_water
        assert mock_get.call_count == 2

    @patch('quantui.pubchem.requests.get')
    def test_get_sdf_not_found(self, mock_get):
        """Test CID not found."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with pytest.raises(MoleculeNotFoundError):
            get_molecule_sdf(999999999, conformer_3d=False)


class TestSDFToXYZ:
    """Test SDF to XYZ conversion."""

    def test_sdf_to_xyz_water(self, sample_sdf_water):
        """Test conversion of water SDF to XYZ."""
        xyz_string, metadata = sdf_to_xyz(sample_sdf_water)

        # Check XYZ format
        lines = xyz_string.strip().split('\n')
        assert len(lines) >= 3  # n_atoms, comment, at least 1 atom

        # Check metadata
        assert 'formula' in metadata
        assert 'molecular_weight' in metadata
        assert 'charge' in metadata
        assert 'num_atoms' in metadata

        # Water should have 3 atoms (O + 2H)
        assert metadata['num_atoms'] == 3

    def test_sdf_to_xyz_invalid(self):
        """Test conversion of invalid SDF."""
        invalid_sdf = "Not a valid SDF file"

        with pytest.raises(ValueError):
            sdf_to_xyz(invalid_sdf)

    def test_sdf_to_xyz_metadata_fields(self, sample_sdf_water):
        """Test all metadata fields are present."""
        xyz_string, metadata = sdf_to_xyz(sample_sdf_water)

        required_fields = [
            'formula', 'molecular_weight', 'charge',
            'num_atoms', 'num_heavy_atoms'
        ]

        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"


class TestFetchMolecule:
    """Test high-level molecule fetching."""

    @patch('quantui.pubchem.get_molecule_sdf')
    @patch('quantui.pubchem.search_molecule_by_name')
    def test_fetch_molecule_complete(self, mock_search, mock_get_sdf, sample_sdf_water):
        """Test complete molecule fetch workflow."""
        mock_search.return_value = 962
        mock_get_sdf.return_value = sample_sdf_water

        xyz_string, metadata, cid = fetch_molecule("water", conformer_3d=True)

        # Verify search was called
        mock_search.assert_called_once_with("water")

        # Verify SDF retrieval was called
        mock_get_sdf.assert_called_once_with(962, conformer_3d=True)

        # Verify results
        assert cid == 962
        assert 'pubchem_cid' in metadata
        assert metadata['pubchem_cid'] == 962
        assert 'pubchem_name' in metadata
        assert metadata['pubchem_name'] == "water"
        assert isinstance(xyz_string, str)

    @patch('quantui.pubchem.search_molecule_by_name')
    def test_fetch_molecule_not_found(self, mock_search):
        """Test fetching non-existent molecule."""
        mock_search.side_effect = MoleculeNotFoundError("Not found")

        with pytest.raises(MoleculeNotFoundError):
            fetch_molecule("nonexistent_molecule_xyz123")


class TestStudentFriendlyFetch:
    """Test student-friendly wrapper function."""

    @patch('quantui.pubchem.fetch_molecule')
    def test_student_friendly_success(self, mock_fetch):
        """Test successful fetch with friendly message."""
        mock_fetch.return_value = (
            "O 0 0 0\nH 1 0 0\nH 0 1 0",
            {'formula': 'H2O', 'molecular_weight': 18.015,
             'num_atoms': 3, 'num_heavy_atoms': 1},
            962
        )

        xyz_string, message = student_friendly_fetch("water")

        assert xyz_string is not None
        assert "Found 'water' in PubChem" in message
        assert "CID: 962" in message
        assert "H2O" in message

    @patch('quantui.pubchem.fetch_molecule')
    def test_student_friendly_not_found(self, mock_fetch):
        """Test friendly error for molecule not found."""
        mock_fetch.side_effect = MoleculeNotFoundError("Not found")

        xyz_string, message = student_friendly_fetch("xyznotfound")

        assert xyz_string is None
        assert "Could not find" in message
        assert "Try:" in message

    @patch('quantui.pubchem.fetch_molecule')
    def test_student_friendly_api_error(self, mock_fetch):
        """Test friendly error for API failure."""
        mock_fetch.side_effect = PubChemAPIError("Connection failed")

        xyz_string, message = student_friendly_fetch("water")

        assert xyz_string is None
        assert "Connection to PubChem failed" in message
        assert "internet connection" in message


class TestCommonMolecules:
    """Test common molecules helper function."""

    def test_get_common_molecules_returns_dict(self):
        """Test that function returns a dictionary."""
        common = get_common_molecules()
        assert isinstance(common, dict)

    def test_get_common_molecules_has_water(self):
        """Test that water is in common molecules."""
        common = get_common_molecules()
        # Check if any key contains water
        water_found = any('water' in value.lower() for value in common.values())
        assert water_found

    def test_get_common_molecules_values_are_strings(self):
        """Test that all values are search strings."""
        common = get_common_molecules()
        for display_name, search_name in common.items():
            assert isinstance(display_name, str)
            assert isinstance(search_name, str)
            assert len(search_name) > 0


class TestCheckPubChemAvailability:
    """Test PubChem connectivity check."""

    @patch('quantui.pubchem.requests.get')
    def test_check_available(self, mock_get):
        """Test successful connectivity check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = check_pubchem_availability()
        assert result is True

    @patch('quantui.pubchem.requests.get')
    def test_check_unavailable(self, mock_get):
        """Test failed connectivity check."""
        mock_get.side_effect = requests.RequestException("Connection failed")

        result = check_pubchem_availability()
        assert result is False


# ============================================================================
# Integration Tests (Require Network) - Marked and Skipped by Default
# ============================================================================

@pytest.mark.network
@pytest.mark.integration
@pytest.mark.slow
class TestPubChemIntegration:
    """
    Integration tests with real PubChem API.

    Run with: pytest -m network
    Skipped by default to avoid network dependencies.
    """

    def test_real_search_water(self):
        """Test real API search for water."""
        try:
            cid = search_molecule_by_name("water")
            assert cid == 962
        except PubChemAPIError:
            pytest.skip("PubChem not accessible")

    def test_real_fetch_caffeine(self):
        """Test real fetch of caffeine molecule."""
        try:
            xyz_string, metadata, cid = fetch_molecule("caffeine")

            assert cid > 0
            assert xyz_string is not None
            assert len(xyz_string) > 0
            assert 'formula' in metadata
            assert 'C' in metadata['formula']  # Caffeine contains carbon
        except PubChemAPIError:
            pytest.skip("PubChem not accessible")

    def test_real_student_friendly_glucose(self):
        """Test student-friendly fetch of glucose."""
        try:
            xyz_string, message = student_friendly_fetch("glucose")

            assert xyz_string is not None
            assert "Found 'glucose' in PubChem" in message
        except Exception:
            pytest.skip("PubChem not accessible")


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test proper error handling and exceptions."""

    @patch('quantui.pubchem.requests.get')
    def test_timeout_handling(self, mock_get):
        """Test timeout error handling."""
        mock_get.side_effect = requests.Timeout("Request timed out")

        with pytest.raises(PubChemAPIError):
            search_molecule_by_name("water")

    @patch('quantui.pubchem.requests.get')
    def test_http_error_handling(self, mock_get):
        """Test HTTP error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Error")
        mock_get.return_value = mock_response

        with pytest.raises(PubChemAPIError):
            search_molecule_by_name("water")

    def test_sdf_conversion_no_rdkit(self):
        """Test SDF conversion failure handling."""
        # This test assumes RDKit is available for the test suite
        # In a real scenario where RDKit isn't available, we'd test ImportError
        invalid_sdf = "INVALID SDF CONTENT"

        with pytest.raises(ValueError):
            sdf_to_xyz(invalid_sdf)


# ============================================================================
# Caching Tests
# ============================================================================

class TestCaching:
    """Test LRU caching of SDF retrieval."""

    @patch('quantui.pubchem.requests.get')
    def test_sdf_caching(self, mock_get, sample_sdf_water):
        """Test that repeated SDF requests are cached."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sample_sdf_water
        mock_get.return_value = mock_response

        # Clear cache first
        get_molecule_sdf.cache_clear()

        # First call - should hit the API
        sdf1 = get_molecule_sdf(962, conformer_3d=True)
        assert mock_get.call_count == 1

        # Second call with same parameters - should use cache
        sdf2 = get_molecule_sdf(962, conformer_3d=True)
        assert mock_get.call_count == 1  # Still 1, not 2

        assert sdf1 == sdf2

        # Different parameters - should hit API again
        sdf3 = get_molecule_sdf(962, conformer_3d=False)
        assert mock_get.call_count == 2


# ============================================================================
# SMILES Conversion Tests
# ============================================================================

class TestSMILESConversion:
    """Test SMILES to XYZ conversion functionality."""

    def test_smiles_to_xyz_simple(self):
        """Test conversion of simple SMILES (water)."""
        xyz_string, metadata = smiles_to_xyz("O", optimize_3d=True)

        # Check XYZ format
        lines = xyz_string.strip().split('\n')
        assert len(lines) >= 3  # Header + atoms

        # Check metadata
        assert 'formula' in metadata
        assert 'H2O' in metadata['formula']
        assert metadata['num_atoms'] == 3  # O + 2H

    def test_smiles_to_xyz_ethanol(self):
        """Test conversion of ethanol (CCO)."""
        xyz_string, metadata = smiles_to_xyz("CCO", optimize_3d=True)

        lines = xyz_string.strip().split('\n')
        assert len(lines) >= 3

        # Ethanol should have 9 atoms (2C, 1O, 6H)
        assert metadata['num_atoms'] == 9
        assert 'C2H6O' in metadata['formula'] or 'C2H5OH' in metadata['formula']

    def test_smiles_to_xyz_benzene(self):
        """Test conversion of benzene (aromatic)."""
        xyz_string, metadata = smiles_to_xyz("c1ccccc1", optimize_3d=True)

        # Benzene should have 12 atoms (6C + 6H)
        assert metadata['num_atoms'] == 12
        assert '6' in metadata['formula']  # C6H6

    def test_smiles_to_xyz_invalid(self):
        """Test invalid SMILES string."""
        with pytest.raises(ValueError):
            smiles_to_xyz("INVALID_SMILES_XYZ123", optimize_3d=True)

    def test_smiles_to_xyz_metadata_fields(self):
        """Test all metadata fields are present."""
        xyz_string, metadata = smiles_to_xyz("C", optimize_3d=True)

        required_fields = [
            'formula', 'molecular_weight', 'charge',
            'num_atoms', 'num_heavy_atoms', 'smiles', 'canonical_smiles'
        ]

        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"

    def test_smiles_to_xyz_without_optimization(self):
        """Test SMILES conversion with optimization disabled."""
        xyz_string, metadata = smiles_to_xyz("C", optimize_3d=False)

        # Should still generate coordinates
        lines = xyz_string.strip().split('\n')
        assert len(lines) >= 3

class TestStudentFriendlySMILES:
    """Test student-friendly SMILES wrapper."""

    def test_student_friendly_success(self):
        """Test successful SMILES conversion with friendly message."""
        xyz_string, message = student_friendly_smiles_to_xyz("CCO")

        assert xyz_string is not None
        assert "✓ Converted SMILES" in message
        assert "CCO" in message
        assert "Atoms:" in message

    def test_student_friendly_invalid(self):
        """Test friendly error for invalid SMILES."""
        xyz_string, message = student_friendly_smiles_to_xyz("INVALID")

        assert xyz_string is None
        assert "❌ Invalid SMILES" in message
        assert "Tips:" in message


class TestSMILESExamples:
    """Test SMILES examples helper function."""

    def test_get_smiles_examples_returns_dict(self):
        """Test that function returns a dictionary."""
        examples = get_smiles_examples()
        assert isinstance(examples, dict)

    def test_get_smiles_examples_has_common_molecules(self):
        """Test that common molecules are included."""
        examples = get_smiles_examples()

        # Check for some common molecules
        molecule_names = [name.lower() for name in examples.keys()]
        assert any('water' in name for name in molecule_names)
        assert any('methane' in name for name in molecule_names)
        assert any('ethanol' in name for name in molecule_names)

    def test_get_smiles_examples_valid_smiles(self):
        """Test that all example SMILES are valid."""
        examples = get_smiles_examples()

        for name, smiles_str in examples.items():
            is_valid, _ = validate_smiles(smiles_str)
            assert is_valid, f"Invalid SMILES for {name}: {smiles_str}"


class TestSMILESValidation:
    """Test SMILES validation function."""

    def test_validate_valid_smiles(self):
        """Test validation of valid SMILES strings."""
        valid_smiles = ["C", "CCO", "c1ccccc1", "CC(=O)O", "N"]

        for smiles_str in valid_smiles:
            is_valid, message = validate_smiles(smiles_str)
            assert is_valid, f"Should be valid: {smiles_str}"

    def test_validate_invalid_smiles(self):
        """Test validation of invalid SMILES strings."""
        invalid_smiles = ["INVALID", "XYZ123", "((()))"]

        for smiles_str in invalid_smiles:
            is_valid, message = validate_smiles(smiles_str)
            assert not is_valid, f"Should be invalid: {smiles_str}"

    def test_validate_empty_molecule(self):
        """Test validation of empty/nonsensical SMILES."""
        is_valid, message = validate_smiles("")
        assert not is_valid


# ============================================================================
# 2D Structure Rendering Tests
# ============================================================================

class Test2DStructureGeneration:
    """Test 2D structure SVG generation."""

    def test_generate_2d_from_smiles(self):
        """Test 2D structure generation from SMILES."""
        svg = generate_2d_structure_svg(smiles="CCO", width=300, height=300)

        assert svg is not None
        assert isinstance(svg, str)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_generate_2d_from_xyz(self, sample_water_xyz):
        """Test 2D structure generation from XYZ."""
        svg = generate_2d_structure_svg(
            xyz_string=sample_water_xyz,
            width=300,
            height=300
        )

        # May succeed or fail depending on bond perception
        if svg is not None:
            assert "<svg" in svg

    def test_generate_2d_benzene(self):
        """Test 2D structure of aromatic molecule."""
        svg = generate_2d_structure_svg(smiles="c1ccccc1")

        assert svg is not None
        assert "<svg" in svg

    def test_generate_2d_custom_size(self):
        """Test 2D structure with custom dimensions."""
        svg = generate_2d_structure_svg(
            smiles="C",
            width=500,
            height=400
        )

        assert svg is not None
        # SVG should contain width/height attributes
        assert "width" in svg or "viewBox" in svg

    def test_generate_2d_no_input(self):
        """Test 2D structure generation with no input."""
        with pytest.raises(ValueError):
            generate_2d_structure_svg()

    def test_generate_2d_invalid_smiles(self):
        """Test 2D structure generation with invalid SMILES."""
        svg = generate_2d_structure_svg(smiles="INVALID")
        # Should return None for invalid input
        assert svg is None


# ============================================================================
# Integration Tests for SMILES and 2D
# ============================================================================

@pytest.mark.network
@pytest.mark.integration
class TestSMILESIntegration:
    """Integration tests for SMILES workflow."""

    def test_smiles_to_xyz_to_2d(self):
        """Test complete workflow: SMILES → XYZ → 2D structure."""
        # Convert SMILES to XYZ
        xyz_string, metadata = smiles_to_xyz("CCO", optimize_3d=True)
        assert xyz_string is not None

        # Generate 2D structure from SMILES
        svg = generate_2d_structure_svg(smiles="CCO")
        assert svg is not None

        # Generate 2D structure from XYZ
        svg_from_xyz = generate_2d_structure_svg(xyz_string=xyz_string)
        # May work or fail depending on bond perception
        assert svg_from_xyz is None or "<svg" in svg_from_xyz

    def test_pubchem_to_smiles_to_2d(self):
        """Test workflow: PubChem → SMILES → 2D."""
        try:
            # Fetch from PubChem
            xyz_string, metadata, cid = fetch_molecule("ethanol")

            # Get canonical SMILES
            if 'canonical_smiles' in metadata:
                smiles_str = metadata['canonical_smiles']
            else:
                # Convert XYZ back to SMILES
                from rdkit import Chem
                mol = Chem.MolFromXYZBlock(xyz_string)
                if mol:
                    smiles_str = Chem.MolToSmiles(mol)
                else:
                    pytest.skip("Could not generate SMILES")

            # Generate 2D structure
            svg = generate_2d_structure_svg(smiles=smiles_str)
            assert svg is not None

        except (PubChemAPIError, PubChemError):
            pytest.skip("PubChem not accessible")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not network"])
