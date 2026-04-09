"""
Tests for quantui.orbital_visualization

Covers OrbitalInfo construction, load helpers, the matplotlib energy-level
diagram, and the summary HTML.  Cube-file functions are tested with a
synthetic minimal cube string.
"""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from quantui.orbital_visualization import (
    OrbitalInfo,
    load_orbital_info,
    orbital_info_from_arrays,
    orbital_summary_html,
    parse_cube_file,
    plot_orbital_diagram,
    HARTREE_TO_EV,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_mo_data(tmp_path):
    """Create a minimal results.npz with 4 MOs (2 occ, 2 virt)."""
    mo_energy = np.array([-0.6, -0.3, 0.2, 0.8])  # Hartrees
    mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
    np.savez(
        tmp_path / "results.npz",
        mo_energy=mo_energy,
        mo_occ=mo_occ,
        energy=-75.0,
        converged=True,
    )
    return tmp_path / "results.npz"


@pytest.fixture()
def uhf_mo_data(tmp_path):
    """Create results.npz with UHF-style (2, n_mo) arrays."""
    alpha = np.array([-0.7, -0.4, 0.1, 0.5])
    beta = np.array([-0.65, -0.35, 0.15, 0.55])
    mo_energy = np.stack([alpha, beta])  # shape (2, 4)
    mo_occ = np.array([[1.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0]])
    np.savez(
        tmp_path / "results.npz",
        mo_energy=mo_energy,
        mo_occ=mo_occ,
        energy=-74.5,
        converged=True,
    )
    return tmp_path / "results.npz"


@pytest.fixture()
def minimal_cube_file(tmp_path):
    """Write a trivially small cube file (1 atom, 2x2x2 grid)."""
    content = textwrap.dedent("""\
        Comment line 1
        Comment line 2
         1  0.000000  0.000000  0.000000
         2  0.500000  0.000000  0.000000
         2  0.000000  0.500000  0.000000
         2  0.000000  0.000000  0.500000
         1  0.000000  0.000000  0.000000  0.000000
         0.1  0.2  0.3  0.4
         0.5  0.6  0.7  0.8
    """)
    p = tmp_path / "test.cube"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# OrbitalInfo construction
# ---------------------------------------------------------------------------

class TestLoadOrbitalInfo:

    def test_basic_load(self, simple_mo_data):
        info = load_orbital_info(simple_mo_data, formula="H2O")
        assert info.n_occupied == 2
        assert info.n_virtual == 2
        assert info.formula == "H2O"

    def test_homo_lumo_values(self, simple_mo_data):
        info = load_orbital_info(simple_mo_data)
        expected_homo = -0.3 * HARTREE_TO_EV
        expected_lumo = 0.2 * HARTREE_TO_EV
        assert abs(info.homo_energy_ev - expected_homo) < 1e-6
        assert abs(info.lumo_energy_ev - expected_lumo) < 1e-6

    def test_gap_positive(self, simple_mo_data):
        info = load_orbital_info(simple_mo_data)
        assert info.homo_lumo_gap_ev > 0

    def test_uhf_uses_alpha(self, uhf_mo_data):
        info = load_orbital_info(uhf_mo_data)
        # Alpha HOMO is at index 1 = -0.4 Ha
        expected_homo = -0.4 * HARTREE_TO_EV
        assert abs(info.homo_energy_ev - expected_homo) < 1e-6

    def test_formula_fallback_to_stem(self, simple_mo_data):
        info = load_orbital_info(simple_mo_data)
        assert info.formula == "results"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_orbital_info(tmp_path / "nonexistent.npz")

    def test_external_mo_occ(self, tmp_path):
        """Pass mo_occ explicitly, overriding what's in the file."""
        mo_energy = np.array([-1.0, -0.5, 0.3, 1.0])
        np.savez(tmp_path / "results.npz", mo_energy=mo_energy, energy=-50.0, converged=True)
        info = load_orbital_info(
            tmp_path / "results.npz",
            mo_occ=np.array([2.0, 0.0, 0.0, 0.0]),
            formula="test",
        )
        assert info.n_occupied == 1


class TestOrbitalInfoFromArrays:

    def test_basic(self):
        info = orbital_info_from_arrays(
            mo_energy=np.array([-0.5, -0.2, 0.1, 0.6]),
            mo_occ=np.array([2.0, 2.0, 0.0, 0.0]),
            formula="CH4",
        )
        assert info.n_occupied == 2
        assert info.formula == "CH4"

    def test_gap_calculation(self):
        info = orbital_info_from_arrays(
            mo_energy=np.array([-0.5, 0.5]),
            mo_occ=np.array([2.0, 0.0]),
        )
        expected_gap = 1.0 * HARTREE_TO_EV
        assert abs(info.homo_lumo_gap_ev - expected_gap) < 1e-6

    def test_all_occupied_raises(self):
        with pytest.raises(ValueError, match="Cannot determine HOMO/LUMO"):
            orbital_info_from_arrays(
                mo_energy=np.array([-0.5, -0.2]),
                mo_occ=np.array([2.0, 2.0]),
            )

    def test_none_occupied_raises(self):
        with pytest.raises(ValueError, match="Cannot determine HOMO/LUMO"):
            orbital_info_from_arrays(
                mo_energy=np.array([0.5, 1.0]),
                mo_occ=np.array([0.0, 0.0]),
            )


# ---------------------------------------------------------------------------
# Matplotlib diagram
# ---------------------------------------------------------------------------

class TestPlotOrbitalDiagram:

    def test_returns_figure(self, simple_mo_data):
        import matplotlib
        matplotlib.use("Agg")
        info = load_orbital_info(simple_mo_data, formula="H2O")
        fig = plot_orbital_diagram(info)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_custom_title(self, simple_mo_data):
        import matplotlib
        matplotlib.use("Agg")
        info = load_orbital_info(simple_mo_data, formula="H2O")
        fig = plot_orbital_diagram(info, title="Custom Title")
        # Title should be set on the axes
        ax = fig.axes[0]
        assert ax.get_title() == "Custom Title"

    def test_max_orbitals_limits_lines(self, tmp_path):
        """With 20 MOs and max_orbitals=6, only ~6 levels are drawn."""
        import matplotlib
        matplotlib.use("Agg")
        mo_e = np.linspace(-2, 2, 20)
        mo_occ = np.zeros(20)
        mo_occ[:10] = 2.0
        np.savez(tmp_path / "results.npz", mo_energy=mo_e, mo_occ=mo_occ,
                 energy=-100.0, converged=True)
        info = load_orbital_info(tmp_path / "results.npz", formula="big")
        fig = plot_orbital_diagram(info, max_orbitals=6)
        # Should have plotted 6 lines (each is a Line2D object)
        ax = fig.axes[0]
        lines = [c for c in ax.get_children()
                 if hasattr(c, "get_xdata") and len(c.get_xdata()) == 2]
        assert len(lines) == 6


# ---------------------------------------------------------------------------
# Summary HTML
# ---------------------------------------------------------------------------

class TestOrbitalSummaryHtml:

    def test_contains_formula(self, simple_mo_data):
        info = load_orbital_info(simple_mo_data, formula="H2O")
        html = orbital_summary_html(info)
        assert "H2O" in html

    def test_contains_occupied_count(self, simple_mo_data):
        info = load_orbital_info(simple_mo_data, formula="H2O")
        html = orbital_summary_html(info)
        assert "Occupied MOs: 2" in html

    def test_contains_gap(self, simple_mo_data):
        info = load_orbital_info(simple_mo_data, formula="H2O")
        html = orbital_summary_html(info)
        assert "HOMO–LUMO gap" in html


# ---------------------------------------------------------------------------
# Cube-file parser
# ---------------------------------------------------------------------------

class TestParseCubeFile:

    def test_parse_basic(self, minimal_cube_file):
        cube = parse_cube_file(minimal_cube_file)
        assert cube["nx"] == 2
        assert cube["ny"] == 2
        assert cube["nz"] == 2
        assert cube["data"].shape == (2, 2, 2)

    def test_atom_count(self, minimal_cube_file):
        cube = parse_cube_file(minimal_cube_file)
        assert len(cube["atoms"]) == 1

    def test_origin(self, minimal_cube_file):
        cube = parse_cube_file(minimal_cube_file)
        np.testing.assert_allclose(cube["origin"], [0.0, 0.0, 0.0])

    def test_data_values(self, minimal_cube_file):
        cube = parse_cube_file(minimal_cube_file)
        flat = cube["data"].flatten()
        np.testing.assert_allclose(flat, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
