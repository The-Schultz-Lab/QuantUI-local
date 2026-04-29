"""
M2.4 tests for post-calculation 3D visualization helpers.

Covers:
  - ``create_trajectory_animation()`` in plotlyMol
  - ``QuantUIApp._build_vib_data_from_freq_result()``

No PySCF required — all calc results are mocked.
plotlyMol + RDKit are required for most tests; tests are skipped when absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from quantui.molecule import Molecule

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

try:
    from plotlymol3d import create_trajectory_animation

    _PLOTLYMOL_AVAILABLE = True
except ImportError:
    _PLOTLYMOL_AVAILABLE = False

plotlymol_only = pytest.mark.skipif(
    not _PLOTLYMOL_AVAILABLE,
    reason="plotlyMol (and RDKit) required",
)

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _water() -> Molecule:
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


def _water_xyzblocks(n: int = 3) -> list:
    """Return n slightly different XYZ blocks for water (trajectory mock)."""
    # Perturb O position slightly for each step.
    blocks = []
    for i in range(n):
        o_z = i * 0.01
        block = (
            f"3\nH2O step {i}\n"
            f"O  0.0   0.0   {o_z:.4f}\n"
            f"H  0.757 0.587 0.0\n"
            f"H -0.757 0.587 0.0"
        )
        blocks.append(block)
    return blocks


# ---------------------------------------------------------------------------
# create_trajectory_animation (plotlyMol)
# ---------------------------------------------------------------------------


class TestCreateTrajectoryAnimation:
    @plotlymol_only
    def test_returns_figure(self):
        """create_trajectory_animation returns a Plotly Figure."""
        import plotly.graph_objects as go

        blocks = _water_xyzblocks(3)
        fig = create_trajectory_animation(blocks)
        assert isinstance(fig, go.Figure)

    @plotlymol_only
    def test_correct_frame_count(self):
        """Figure has exactly one frame per input XYZ block."""
        blocks = _water_xyzblocks(4)
        fig = create_trajectory_animation(blocks)
        assert len(fig.frames) == 4

    @plotlymol_only
    def test_minimum_two_frames(self):
        """Two frames is the minimum valid input."""
        blocks = _water_xyzblocks(2)
        import plotly.graph_objects as go

        fig = create_trajectory_animation(blocks)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 2

    @plotlymol_only
    def test_single_frame_raises(self):
        """Fewer than 2 frames raises ValueError."""
        blocks = _water_xyzblocks(1)
        with pytest.raises(ValueError, match="at least 2 frames"):
            create_trajectory_animation(blocks)

    @plotlymol_only
    def test_energy_labels(self):
        """Energies are reflected in frame layout titles."""
        blocks = _water_xyzblocks(3)
        energies = [-75.0, -75.3, -75.6]
        fig = create_trajectory_animation(blocks, energies_hartree=energies)
        # At least one frame title should contain an energy value.
        titles = [
            f.layout.title.text for f in fig.frames if f.layout and f.layout.title
        ]
        assert any("-75" in (t or "") for t in titles)

    @plotlymol_only
    def test_has_slider(self):
        """Figure layout contains a slider for step navigation."""
        blocks = _water_xyzblocks(3)
        fig = create_trajectory_animation(blocks)
        assert len(fig.layout.sliders) > 0
        assert len(fig.layout.sliders[0].steps) == 3

    @plotlymol_only
    def test_initial_data_populated(self):
        """Initial figure data (frame 0) is not empty."""
        blocks = _water_xyzblocks(2)
        fig = create_trajectory_animation(blocks)
        assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# QuantUIApp._build_vib_data_from_freq_result
# ---------------------------------------------------------------------------


@dataclass
class _MockFreqResult:
    """Minimal mock of FreqResult sufficient for _build_vib_data_from_freq_result."""

    frequencies_cm1: List[float] = field(default_factory=list)
    ir_intensities: List[float] = field(default_factory=list)
    displacements: Optional[List] = None


def _make_mock_freq_result(n_atoms: int = 3, include_displacements: bool = True):
    """Return a mock FreqResult for water (3N = 9 modes including trans/rot)."""
    import numpy as np

    n_modes = 3 * n_atoms  # 9 for water
    freqs = list(range(-2, n_modes - 2))  # includes a couple of near-zero and negative
    freqs = [float(f) * 100 for f in range(n_modes)]  # 0, 100, 200, ..., 800 cm-1
    ir_intensities = [float(i) * 10 for i in range(n_modes)]

    if include_displacements:
        displ = np.random.rand(n_modes, n_atoms, 3).tolist()
    else:
        displ = None

    return _MockFreqResult(
        frequencies_cm1=freqs,
        ir_intensities=ir_intensities,
        displacements=displ,
    )


class TestBuildVibData:
    @plotlymol_only
    def test_returns_vib_data(self):
        """Returns a VibrationalData object when prerequisites are met."""
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        mol = _water()
        result = _make_mock_freq_result(n_atoms=3)
        vib_data = app._build_vib_data_from_freq_result(result, mol)
        assert vib_data is not None

    @plotlymol_only
    def test_coordinate_shape(self):
        """VibrationalData.coordinates has shape (n_atoms, 3)."""
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        mol = _water()
        result = _make_mock_freq_result(n_atoms=3)
        vib_data = app._build_vib_data_from_freq_result(result, mol)
        assert vib_data.coordinates.shape == (3, 3)

    @plotlymol_only
    def test_mode_count(self):
        """VibrationalData has the same number of modes as input frequencies."""
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        mol = _water()
        n_modes = 3 * 3  # 9 for water (3N)
        result = _make_mock_freq_result(n_atoms=3)
        vib_data = app._build_vib_data_from_freq_result(result, mol)
        assert len(vib_data.modes) == n_modes

    @plotlymol_only
    def test_displacement_shape_per_mode(self):
        """Each VibrationalMode has displacement_vectors of shape (n_atoms, 3)."""
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        mol = _water()
        result = _make_mock_freq_result(n_atoms=3)
        vib_data = app._build_vib_data_from_freq_result(result, mol)
        for mode in vib_data.modes:
            assert mode.displacement_vectors.shape == (3, 3)

    @plotlymol_only
    def test_atomic_numbers_populated(self):
        """Atomic numbers are assigned for O, H, H of water."""
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        mol = _water()
        result = _make_mock_freq_result(n_atoms=3)
        vib_data = app._build_vib_data_from_freq_result(result, mol)
        assert vib_data.atomic_numbers == [8, 1, 1]  # O, H, H

    @plotlymol_only
    def test_imaginary_modes_flagged(self):
        """Modes with negative frequency are flagged as imaginary."""
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        mol = _water()
        result = _MockFreqResult(
            frequencies_cm1=[-500.0, 100.0, 200.0],
            ir_intensities=[0.0, 1.0, 2.0],
            displacements=[
                [[0.1, 0.0, 0.0]] * 3,
                [[0.0, 0.1, 0.0]] * 3,
                [[0.0, 0.0, 0.1]] * 3,
            ],
        )
        vib_data = app._build_vib_data_from_freq_result(result, mol)
        assert vib_data is not None
        assert vib_data.modes[0].is_imaginary is True
        assert vib_data.modes[1].is_imaginary is False

    @plotlymol_only
    def test_none_when_no_displacements(self):
        """Returns None when FreqResult.displacements is None."""
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        mol = _water()
        result = _make_mock_freq_result(n_atoms=3, include_displacements=False)
        vib_data = app._build_vib_data_from_freq_result(result, mol)
        assert vib_data is None

    def test_none_when_plotlymol_missing(self, monkeypatch):
        """Returns None gracefully when plotlyMol is not installed."""
        import builtins

        original_import = builtins.__import__

        def _block_plotlymol(name, *args, **kwargs):
            if "plotlymol3d" in name:
                raise ImportError("plotlyMol blocked for test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_plotlymol)

        from quantui.app import QuantUIApp

        app = QuantUIApp()
        mol = _water()
        result = _make_mock_freq_result(n_atoms=3)
        vib_data = app._build_vib_data_from_freq_result(result, mol)
        assert vib_data is None


# ---------------------------------------------------------------------------
# FreqResult.displacements field
# ---------------------------------------------------------------------------


class TestFreqResultDisplacements:
    def test_displacements_field_exists(self):
        """FreqResult has a displacements field defaulting to None."""
        from quantui.freq_calc import FreqResult

        r = FreqResult(
            energy_hartree=-75.0,
            homo_lumo_gap_ev=None,
            converged=True,
            n_iterations=5,
            method="RHF",
            basis="STO-3G",
            formula="H2O",
        )
        assert r.displacements is None

    def test_displacements_can_be_set(self):
        """FreqResult.displacements accepts a nested list value."""
        from quantui.freq_calc import FreqResult

        mock_displ = [[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]]
        r = FreqResult(
            energy_hartree=-75.0,
            homo_lumo_gap_ev=None,
            converged=True,
            n_iterations=5,
            method="RHF",
            basis="STO-3G",
            formula="H2O",
            displacements=mock_displ,
        )
        assert r.displacements is mock_displ
