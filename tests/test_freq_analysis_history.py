"""Integration tests for the frequency-analysis history roundtrip.

Covers the complete path from "calculation finishes" to "history panels activate":

  (1) Spectra structure  — save_result stores result.json with the correct
                           keys that history-load needs (ir, molecule,
                           displacements).
  (2) History context    — _build_history_context loads calc_type + spectra
                           correctly from disk.
  (3) Panel activation   — _apply_analysis_context activates Vibrational and
                           IR Spectrum panels from a history context.
  (4) _do_run end-to-end — Full path: patched run_freq_calc → disk write →
                           _history_load_analysis → panels activate.
                           (PySCF-gated; skipped on Windows.)

These tests are the canary for BUG-FREQ-ANA class failures — any break in the
data pipeline will show up here before a user notices in the UI.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from quantui.app import QuantUIApp
from quantui.molecule import Molecule
from quantui.results_storage import list_results, load_result, save_result

try:
    from quantui.app import _PYSCF_AVAILABLE
except ImportError:
    _PYSCF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _water():
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )


def _make_freq_result():
    """Realistic SimpleNamespace mirroring a FreqResult from run_freq_calc.

    Covers all attributes that _do_run reads when building save_spectra,
    calling save_result, save_orbitals, and _apply_analysis_context.
    """
    return SimpleNamespace(
        formula="H2O",
        method="RHF",
        basis="STO-3G",
        energy_hartree=-75.0,
        energy_ev=-2040.5,
        homo_lumo_gap_ev=10.5,
        converged=True,
        n_iterations=8,
        # Frequency-specific
        frequencies_cm1=[100.0, 1500.0, 3800.0],
        ir_intensities=[5.0, 50.0, 10.0],
        zpve_hartree=0.021,
        displacements=[
            [[0.0, 0.0, 0.1], [0.0, 0.07, -0.05], [0.0, -0.07, -0.05]],
            [[0.0, 0.0, 0.1], [0.07, 0.0, -0.05], [-0.07, 0.0, -0.05]],
            [[0.0, 0.1, 0.0], [0.0, -0.05, 0.07], [0.0, -0.05, -0.07]],
        ],
        thermo=None,
        # MO data for orbital diagram / save_orbitals
        mo_energy_hartree=np.array([-20.0, -1.3, -0.7, -0.5, -0.3]),
        mo_occ=np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        mo_coeff=None,
        pyscf_mol_atom=[
            ("O", [0.0, 0.0, 0.0]),
            ("H", [0.757, 0.586, 0.0]),
            ("H", [-0.757, 0.586, 0.0]),
        ],
        pyscf_mol_basis="sto-3g",
    )


def _make_freq_spectra(result, mol):
    """Build the spectra dict exactly as _do_run does for a Frequency calc."""
    disps = None
    if result.displacements is not None:
        disps = np.asarray(result.displacements).tolist()
    return {
        "ir": {
            "frequencies_cm1": result.frequencies_cm1,
            "ir_intensities": result.ir_intensities,
            "zpve_hartree": result.zpve_hartree,
            "displacements": disps,
        },
        "molecule": {
            "atoms": list(mol.atoms),
            "coords": [list(map(float, row)) for row in mol.coordinates],
            "charge": mol.charge,
            "multiplicity": mol.multiplicity,
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    a = QuantUIApp()
    a._set_molecule(_water())
    return a


@pytest.fixture
def freq_result():
    return _make_freq_result()


@pytest.fixture
def water_mol():
    return _water()


# ---------------------------------------------------------------------------
# Part 1: save_result stores the correct JSON structure for frequency
# ---------------------------------------------------------------------------


class TestFreqSpectraStructure:
    def test_result_json_has_frequency_calc_type(
        self, tmp_path, freq_result, water_mol
    ):
        spectra = _make_freq_spectra(freq_result, water_mol)
        saved = save_result(
            freq_result, results_dir=tmp_path, calc_type="frequency", spectra=spectra
        )
        data = json.loads((saved / "result.json").read_text())
        assert data["calc_type"] == "frequency"

    def test_spectra_ir_frequencies_present(self, tmp_path, freq_result, water_mol):
        spectra = _make_freq_spectra(freq_result, water_mol)
        saved = save_result(
            freq_result, results_dir=tmp_path, calc_type="frequency", spectra=spectra
        )
        data = json.loads((saved / "result.json").read_text())
        assert data["spectra"]["ir"]["frequencies_cm1"] == pytest.approx(
            [100.0, 1500.0, 3800.0]
        )

    def test_spectra_ir_displacements_shape(self, tmp_path, freq_result, water_mol):
        spectra = _make_freq_spectra(freq_result, water_mol)
        saved = save_result(
            freq_result, results_dir=tmp_path, calc_type="frequency", spectra=spectra
        )
        data = json.loads((saved / "result.json").read_text())
        disps = data["spectra"]["ir"]["displacements"]
        assert (
            disps is not None
        ), "displacements must be stored — _pop_vibrational needs them"
        assert len(disps) == 3  # 3 modes for H2O (3N-6)
        assert len(disps[0]) == 3  # 3 atoms
        assert len(disps[0][0]) == 3  # x, y, z

    def test_spectra_molecule_atoms_present(self, tmp_path, freq_result, water_mol):
        spectra = _make_freq_spectra(freq_result, water_mol)
        saved = save_result(
            freq_result, results_dir=tmp_path, calc_type="frequency", spectra=spectra
        )
        data = json.loads((saved / "result.json").read_text())
        assert data["spectra"]["molecule"]["atoms"] == ["O", "H", "H"]

    def test_spectra_molecule_coords_present(self, tmp_path, freq_result, water_mol):
        spectra = _make_freq_spectra(freq_result, water_mol)
        saved = save_result(
            freq_result, results_dir=tmp_path, calc_type="frequency", spectra=spectra
        )
        data = json.loads((saved / "result.json").read_text())
        coords = data["spectra"]["molecule"]["coords"]
        assert len(coords) == 3
        assert coords[0] == pytest.approx([0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Part 2: _build_history_context reconstructs the context correctly
# ---------------------------------------------------------------------------


class TestFreqHistoryContext:
    def _save(self, tmp_path, freq_result, water_mol):
        spectra = _make_freq_spectra(freq_result, water_mol)
        return save_result(
            freq_result, results_dir=tmp_path, calc_type="frequency", spectra=spectra
        )

    def test_context_has_correct_calc_type(self, tmp_path, app, freq_result, water_mol):
        saved = self._save(tmp_path, freq_result, water_mol)
        ctx = app._build_history_context(saved)
        assert ctx is not None
        assert ctx.calc_type == "frequency"

    def test_context_spectra_data_has_ir_key(
        self, tmp_path, app, freq_result, water_mol
    ):
        saved = self._save(tmp_path, freq_result, water_mol)
        ctx = app._build_history_context(saved)
        assert (
            "ir" in ctx.spectra_data
        ), "spectra_data must have 'ir' key for panel dispatch"

    def test_context_spectra_data_has_molecule_key(
        self, tmp_path, app, freq_result, water_mol
    ):
        saved = self._save(tmp_path, freq_result, water_mol)
        ctx = app._build_history_context(saved)
        assert (
            "molecule" in ctx.spectra_data
        ), "spectra_data must have 'molecule' key for _pop_vibrational"

    def test_context_spectra_ir_has_displacements(
        self, tmp_path, app, freq_result, water_mol
    ):
        saved = self._save(tmp_path, freq_result, water_mol)
        ctx = app._build_history_context(saved)
        disps = ctx.spectra_data.get("ir", {}).get("displacements")
        assert disps is not None, "displacements must survive disk roundtrip"

    def test_context_live_result_is_none(self, tmp_path, app, freq_result, water_mol):
        saved = self._save(tmp_path, freq_result, water_mol)
        ctx = app._build_history_context(saved)
        assert ctx.live_result is None


# ---------------------------------------------------------------------------
# Part 3: _apply_analysis_context activates the correct panels
# ---------------------------------------------------------------------------


class TestFreqAnalysisPanelActivation:
    def _save(self, tmp_path, freq_result, water_mol, spectra=None):
        if spectra is None:
            spectra = _make_freq_spectra(freq_result, water_mol)
        return save_result(
            freq_result, results_dir=tmp_path, calc_type="frequency", spectra=spectra
        )

    def test_vibrational_panel_activates(self, tmp_path, app, freq_result, water_mol):
        saved = self._save(tmp_path, freq_result, water_mol)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Vibrational" in app._ana_available

    def test_ir_spectrum_panel_activates(self, tmp_path, app, freq_result, water_mol):
        saved = self._save(tmp_path, freq_result, water_mol)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "IR Spectrum" in app._ana_available

    def test_navigate_button_visible_when_panels_activate(
        self, tmp_path, app, freq_result, water_mol
    ):
        saved = self._save(tmp_path, freq_result, water_mol)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert app._to_analysis_btn.layout.display == ""

    def test_no_vibrational_when_displacements_missing(
        self, tmp_path, app, freq_result, water_mol
    ):
        spectra = _make_freq_spectra(freq_result, water_mol)
        spectra["ir"]["displacements"] = None
        saved = self._save(tmp_path, freq_result, water_mol, spectra=spectra)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Vibrational" not in app._ana_available

    def test_ir_spectrum_still_activates_when_displacements_missing(
        self, tmp_path, app, freq_result, water_mol
    ):
        spectra = _make_freq_spectra(freq_result, water_mol)
        spectra["ir"]["displacements"] = None
        saved = self._save(tmp_path, freq_result, water_mol, spectra=spectra)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert (
            "IR Spectrum" in app._ana_available
        ), "IR Spectrum only needs frequencies_cm1, not displacements"

    def test_no_panels_when_spectra_empty(self, tmp_path, app, freq_result):
        saved = save_result(
            freq_result, results_dir=tmp_path, calc_type="frequency", spectra={}
        )
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Vibrational" not in app._ana_available
        assert "IR Spectrum" not in app._ana_available

    def test_no_panels_when_calc_type_wrong(self, tmp_path, app, freq_result):
        saved = save_result(freq_result, results_dir=tmp_path, calc_type="", spectra={})
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert len(app._ana_available) == 0
        assert app._to_analysis_btn.layout.display == "none"

    def test_empty_html_hidden_when_panels_activate(
        self, tmp_path, app, freq_result, water_mol
    ):
        saved = self._save(tmp_path, freq_result, water_mol)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert app._analysis_empty_html.layout.display == "none"


# ---------------------------------------------------------------------------
# Part 4: _do_run end-to-end (PySCF-gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _PYSCF_AVAILABLE, reason="PySCF not available on this platform")
class TestFreqDoRunEndToEnd:
    """Full pipeline: patched run_freq_calc → disk → _history_load_analysis → panels.

    Does NOT mock save_result so real disk writes happen.
    Uses QUANTUI_RESULTS_DIR env var to redirect writes to tmp_path.
    """

    def _run_freq(self, app, tmp_dir, monkeypatch):
        """Run a real Frequency calc via _do_run, redirecting saves to tmp_dir."""
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_dir))
        app.calc_type_dd.value = "Frequency"
        app._do_run()
        return list_results(tmp_dir)

    @pytest.fixture
    def running_app(self):
        a = QuantUIApp()
        a._set_molecule(_water())
        return a

    def test_do_run_saves_calc_type_frequency(self, tmp_path, running_app, monkeypatch):
        saved = self._run_freq(running_app, tmp_path, monkeypatch)
        assert saved, "No result saved to disk"
        data = load_result(saved[0])
        assert data["calc_type"] == "frequency"

    def test_do_run_saves_ir_frequencies(self, tmp_path, running_app, monkeypatch):
        saved = self._run_freq(running_app, tmp_path, monkeypatch)
        data = load_result(saved[0])
        freqs = data.get("spectra", {}).get("ir", {}).get("frequencies_cm1")
        assert (
            freqs is not None and len(freqs) > 0
        ), "frequencies_cm1 must be saved in spectra.ir"

    def test_do_run_saves_molecule_atoms(self, tmp_path, running_app, monkeypatch):
        saved = self._run_freq(running_app, tmp_path, monkeypatch)
        data = load_result(saved[0])
        atoms = data.get("spectra", {}).get("molecule", {}).get("atoms")
        assert atoms == [
            "O",
            "H",
            "H",
        ], "molecule.atoms must be saved for history replay"

    def test_do_run_saves_displacements(self, tmp_path, running_app, monkeypatch):
        saved = self._run_freq(running_app, tmp_path, monkeypatch)
        data = load_result(saved[0])
        disps = data.get("spectra", {}).get("ir", {}).get("displacements")
        assert (
            disps is not None
        ), "displacements must be saved — _pop_vibrational needs them"
        assert len(disps) == 3

    def test_history_load_activates_vibrational_panel(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_freq(running_app, tmp_path, monkeypatch)
        assert saved
        running_app._deactivate_all_ana_panels()
        running_app._history_load_analysis(saved[0])
        assert "Vibrational" in running_app._ana_available

    def test_history_load_activates_ir_spectrum_panel(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_freq(running_app, tmp_path, monkeypatch)
        running_app._deactivate_all_ana_panels()
        running_app._history_load_analysis(saved[0])
        assert "IR Spectrum" in running_app._ana_available

    def test_history_load_shows_navigate_button(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_freq(running_app, tmp_path, monkeypatch)
        running_app._deactivate_all_ana_panels()
        running_app._history_load_analysis(saved[0])
        assert running_app._to_analysis_btn.layout.display == ""
