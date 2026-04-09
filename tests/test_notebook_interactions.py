"""
Tests that emulate notebook button-press logic and generate visualization
artifacts.

The notebook callbacks call the same package functions tested here.
Running this file is equivalent to clicking notebook buttons with known
inputs.

Usage:
    pytest tests/test_notebook_interactions.py -v        # run all tests
    pytest tests/test_notebook_interactions.py -v -s     # show artifact paths
"""

from unittest.mock import patch

import pytest

import quantui.visualization_py3dmol as viz_mod
from quantui.molecule import Molecule, parse_xyz_input

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WATER_XYZ = "O  0.0  0.0  0.0\n" "H  0.757  0.587  0.0\n" "H  -0.757  0.587  0.0"

H2_XYZ = "H  0.0  0.0  0.0\nH  0.0  0.0  0.74"


@pytest.fixture
def water():
    atoms, coords = parse_xyz_input(WATER_XYZ)
    return Molecule(atoms=atoms, coordinates=coords, charge=0, multiplicity=1)


@pytest.fixture
def h2():
    atoms, coords = parse_xyz_input(H2_XYZ)
    return Molecule(atoms=atoms, coordinates=coords, charge=0, multiplicity=1)


py3dmol_only = pytest.mark.skipif(
    not viz_mod.PY3DMOL_AVAILABLE,
    reason="py3Dmol not installed",
)


# ---------------------------------------------------------------------------
# 1. Emulate: "Validate Molecule" button
#    The on_validate_molecule callback: parse XYZ -> build Molecule -> display
# ---------------------------------------------------------------------------


class TestValidateMoleculeButton:
    """Logic invoked by the Validate Molecule button in the notebook."""

    def test_parse_xyz_water_gives_correct_atoms(self):
        atoms, coords = parse_xyz_input(WATER_XYZ)
        assert atoms == ["O", "H", "H"]
        assert len(coords) == 3

    def test_molecule_formula(self, water):
        assert water.get_formula() == "H2O"

    def test_molecule_electron_count(self, water):
        assert water.get_electron_count() == 10

    def test_invalid_xyz_raises(self):
        with pytest.raises((ValueError, KeyError)):
            parse_xyz_input("NOTANELEMENT 0.0 0.0 0.0")

    def test_charged_molecule(self):
        atoms, coords = parse_xyz_input(WATER_XYZ)
        mol = Molecule(atoms=atoms, coordinates=coords, charge=1, multiplicity=2)
        assert mol.charge == 1
        assert mol.multiplicity == 2
        assert mol.get_electron_count() == 9  # one electron removed

    @py3dmol_only
    def test_validate_then_visualize_does_not_raise(self, water):
        """End-to-end: parse -> Molecule -> visualize (no display call)."""
        view = viz_mod.visualize_molecule_py3dmol(water, style="stick")
        assert view is not None


# ---------------------------------------------------------------------------
# 2. Emulate: "Visualize" button
#    The on_manual_visualize callback: display_molecule(mol, style=...)
# ---------------------------------------------------------------------------


class TestVisualizeMoleculeButton:
    """Logic invoked by the Visualize button."""

    @py3dmol_only
    def test_visualize_returns_view_object(self, water):
        import py3Dmol

        view = viz_mod.visualize_molecule_py3dmol(water, style="stick")
        assert isinstance(view, py3Dmol.view)

    @py3dmol_only
    @pytest.mark.parametrize("style", ["stick", "sphere", "line"])
    def test_all_display_styles_succeed(self, water, style):
        view = viz_mod.visualize_molecule_py3dmol(water, style=style)
        assert view is not None

    def test_invalid_style_raises(self, water):
        with pytest.raises(ValueError, match="style must be one of"):
            viz_mod._validate_py3dmol_style("invalid_style")

    @py3dmol_only
    def test_display_molecule_does_not_print_none(self, water, capsys):
        """
        Regression for the 'None' bug: display(viz.show()) passed a None
        return value to display(), printing 'None' in the cell output.
        The fix is display(viz), which calls _repr_html_() instead and
        also works in JupyterLab (show() uses Javascript injection which
        is blocked by JupyterLab's content-security-policy).
        """
        with patch("IPython.display.display"):
            viz_mod.display_molecule(water, style="stick", show_info=False)
        captured = capsys.readouterr()
        assert "None" not in captured.out

    @py3dmol_only
    def test_display_molecule_does_not_raise(self, water):
        with patch("IPython.display.display"):
            viz_mod.display_molecule(water, style="stick", show_info=False)

    @py3dmol_only
    def test_display_molecule_with_info_does_not_raise(self, water):
        with patch("IPython.display.display"):
            viz_mod.display_molecule(water, style="sphere", show_info=True)

    @py3dmol_only
    def test_display_molecule_h2(self, h2):
        """Smallest possible molecule."""
        with patch("IPython.display.display"):
            viz_mod.display_molecule(h2, style="stick", show_info=False)


# ---------------------------------------------------------------------------
# Shared helpers for artifact generation
# ---------------------------------------------------------------------------

# CPK colour scheme + van-der-Waals radii (Angstrom) scaled to scatter size
_ATOM_COLOR = {
    "H": "#dddddd",
    "C": "#555555",
    "N": "#3355ff",
    "O": "#ff2222",
    "F": "#22aa22",
    "S": "#ddcc00",
    "Cl": "#22cc22",
    "P": "#ff8800",
    "Br": "#993300",
}
_ATOM_SIZE = {
    "H": 150,
    "C": 400,
    "N": 380,
    "O": 360,
    "F": 320,
    "S": 600,
    "Cl": 550,
    "P": 520,
    "Br": 700,
}


def _mol_to_png(mol, out_path, elev=20, azim=30):
    """
    Render molecule atom positions as a matplotlib 3D scatter plot and save
    as a PNG.  Works offline — no CDN, no WebGL required.
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend, safe in tests
    import matplotlib.pyplot as plt
    import numpy as np

    coords = np.array(mol.coordinates)
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    colors = [_ATOM_COLOR.get(a, "#ff69b4") for a in mol.atoms]
    sizes = [_ATOM_SIZE.get(a, 350) for a in mol.atoms]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xs,
        ys,
        zs,
        c=colors,
        s=sizes,
        edgecolors="black",
        linewidth=0.4,
        alpha=0.92,
        zorder=5,
    )

    for atom, (x, y, z) in zip(mol.atoms, mol.coordinates):
        ax.text(
            x,
            y,
            z + 0.12,
            atom,
            fontsize=9,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title(f"{mol.get_formula()}  ({len(mol.atoms)} atoms)", pad=20)
    ax.set_xlabel("X (A)")
    ax.set_ylabel("Y (A)")
    ax.set_zlabel("Z (A)")
    ax.view_init(elev=elev, azim=azim)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close()


def _artifact_dir(tmp_path):
    import os
    from pathlib import Path

    env_dir = os.environ.get("QUANTUI_ARTIFACT_DIR")
    if env_dir:
        d = Path(env_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d
    return tmp_path


# ---------------------------------------------------------------------------
# 3. Generate PNG visualization artifacts (offline, no CDN needed)
#
#    Open the .png files in any image viewer to confirm atom positions and
#    element colours look correct.
#
#    Run:
#        $env:QUANTUI_ARTIFACT_DIR = ".\viz-artifacts"
#        pytest tests/test_notebook_interactions.py \
#            ::TestGenerateVisualizationArtifacts -v -s --no-cov
# ---------------------------------------------------------------------------


class TestGenerateVisualizationArtifacts:
    """
    Produce offline PNG images via matplotlib — no CDN, no WebGL needed.

    The HTML files that py3Dmol generates load 3Dmol.js from jsDelivr CDN.
    Browsers opening local HTML files (file://) may block the CDN script
    load, resulting in a blank white box.  These PNG tests bypass that
    entirely and give immediate visual confirmation.

    Run:
        $env:QUANTUI_ARTIFACT_DIR = ".\\viz-artifacts"
        pytest tests/test_notebook_interactions.py \
            ::TestGenerateVisualizationArtifacts -v -s --no-cov
    """

    def test_water_png(self, water, tmp_path):
        out = _artifact_dir(tmp_path) / "water.png"
        _mol_to_png(water, out)
        assert out.exists()
        assert out.stat().st_size > 1000  # non-trivial file
        print(f"\n  [artifact] {out}")

    def test_h2_png(self, h2, tmp_path):
        out = _artifact_dir(tmp_path) / "h2.png"
        _mol_to_png(h2, out)
        assert out.exists()

    def test_benzene_png(self, tmp_path):
        """Benzene from Quick Start Templates — larger molecule."""
        from quantui.config import QUICK_START_TEMPLATES

        template = QUICK_START_TEMPLATES["benzene"]
        atoms, coords = parse_xyz_input(template["molecule"]["xyz"])
        mol = Molecule(atoms=atoms, coordinates=coords, charge=0, multiplicity=1)
        out = _artifact_dir(tmp_path) / "benzene.png"
        _mol_to_png(mol, out)
        assert out.exists()
        print(f"\n  [artifact] {out}")

    def test_oxygen_radical_png(self, tmp_path):
        """Radical from Quick Start Templates — open-shell molecule."""
        from quantui.config import QUICK_START_TEMPLATES

        template = QUICK_START_TEMPLATES["radical_oxygen"]
        atoms, coords = parse_xyz_input(template["molecule"]["xyz"])
        mol = Molecule(
            atoms=atoms,
            coordinates=coords,
            charge=template["molecule"]["charge"],
            multiplicity=template["molecule"]["multiplicity"],
        )
        out = _artifact_dir(tmp_path) / "o2_radical.png"
        _mol_to_png(mol, out)
        assert out.exists()

    @pytest.mark.parametrize("azim", [0, 45, 90, 135])
    def test_water_four_angles(self, water, tmp_path, azim):
        """Generate water from four rotation angles."""
        out = _artifact_dir(tmp_path) / f"water_az{azim}.png"
        _mol_to_png(water, out, azim=azim)
        assert out.exists()
