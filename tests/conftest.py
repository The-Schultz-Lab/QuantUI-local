"""
Pytest Configuration and Fixtures

Shared test fixtures for QuantUI test suite.
"""

import pytest


@pytest.fixture
def sample_water_xyz():
    """Simple water molecule XYZ coordinates."""
    return """O  0.0  0.0  0.0
H  0.757  0.587  0.0
H  -0.757  0.587  0.0"""


@pytest.fixture
def sample_methane_xyz():
    """Simple methane molecule XYZ coordinates."""
    return """C  0.0  0.0  0.0
H  0.63  0.63  0.63
H  -0.63  -0.63  0.63
H  -0.63  0.63  -0.63
H  0.63  -0.63  -0.63"""


@pytest.fixture
def sample_sdf_water():
    """Sample SDF content for water molecule."""
    return """
  Mrv2311 02131511003D

  3  2  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.7570    0.5870    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7570    0.5870    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
$$$$
"""


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory for test files."""
    test_dir = tmp_path / "quantui_test"
    test_dir.mkdir()
    return test_dir


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "network: tests that require network connectivity"
    )
    config.addinivalue_line("markers", "slow: tests that take significant time to run")
    config.addinivalue_line(
        "markers", "integration: integration tests requiring external services"
    )
