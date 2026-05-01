"""Integration tests for the TD-DFT (UV-Vis) analysis history roundtrip.

Covers the complete path from "calculation finishes" to "history panels activate":

  (1) Spectra structure  — save_result stores result.json with calc_type "tddft"
                           and the correct uv_vis spectra keys.
  (2) History context    — _build_history_context loads the spectra correctly.
  (3) Panel activation   — _apply_analysis_context activates the UV-Vis panel
                           from a history context.
  (4) _do_run end-to-end — Full path: real RHF/STO-3G + 3 excited states →
                           disk write → _history_load_analysis → panel activates.
                           (PySCF-gated; skipped on Windows.)
"""

from __future__ import annotations

import json
from types import SimpleNamespace

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


def _make_tddft_result():
    """Minimal namespace mirroring a real RHF/STO-3G TD-DFT result on water."""
    energies = [6.5, 7.2, 8.1]
    return SimpleNamespace(
        formula="H2O",
        method="RHF",
        basis="STO-3G",
        energy_hartree=-75.0,
        energy_ev=-2040.5,
        homo_lumo_gap_ev=10.5,
        converged=True,
        n_iterations=8,
        excitation_energies_ev=energies,
        oscillator_strengths=[0.05, 0.12, 0.03],
        wavelengths_nm=[1240.0 / e for e in energies],
    )


def _make_tddft_spectra(result=None):
    """Build the spectra dict as _do_run does for a TD-DFT calculation."""
    if result is None:
        result = _make_tddft_result()
    return {
        "uv_vis": {
            "excitation_energies_ev": result.excitation_energies_ev,
            "oscillator_strengths": result.oscillator_strengths,
            "wavelengths_nm": result.wavelengths_nm,
        }
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
def tddft_result():
    return _make_tddft_result()


@pytest.fixture
def tddft_spectra(tddft_result):
    return _make_tddft_spectra(tddft_result)


# ---------------------------------------------------------------------------
# Part 1: save_result stores the correct JSON structure for tddft
# ---------------------------------------------------------------------------


class TestTDDFTSpectraStructure:
    def test_result_json_has_tddft_calc_type(
        self, tmp_path, tddft_result, tddft_spectra
    ):
        saved = save_result(
            tddft_result, results_dir=tmp_path, calc_type="tddft", spectra=tddft_spectra
        )
        data = json.loads((saved / "result.json").read_text())
        assert data["calc_type"] == "tddft"

    def test_all_uv_vis_keys_present(self, tmp_path, tddft_result, tddft_spectra):
        saved = save_result(
            tddft_result, results_dir=tmp_path, calc_type="tddft", spectra=tddft_spectra
        )
        data = json.loads((saved / "result.json").read_text())
        uv = data["spectra"]["uv_vis"]
        assert "excitation_energies_ev" in uv
        assert "oscillator_strengths" in uv
        assert "wavelengths_nm" in uv

    def test_wavelengths_non_empty(self, tmp_path, tddft_result, tddft_spectra):
        saved = save_result(
            tddft_result, results_dir=tmp_path, calc_type="tddft", spectra=tddft_spectra
        )
        data = json.loads((saved / "result.json").read_text())
        wl = data["spectra"]["uv_vis"]["wavelengths_nm"]
        assert len(wl) > 0
        assert all(w > 0 for w in wl)


# ---------------------------------------------------------------------------
# Part 2: _build_history_context reconstructs the context correctly
# ---------------------------------------------------------------------------


class TestTDDFTHistoryContext:
    def _save(self, tmp_path, tddft_result, tddft_spectra):
        return save_result(
            tddft_result, results_dir=tmp_path, calc_type="tddft", spectra=tddft_spectra
        )

    def test_context_has_correct_calc_type(
        self, tmp_path, app, tddft_result, tddft_spectra
    ):
        saved = self._save(tmp_path, tddft_result, tddft_spectra)
        ctx = app._build_history_context(saved)
        assert ctx is not None
        assert ctx.calc_type == "tddft"

    def test_context_spectra_data_has_excitation_energies(
        self, tmp_path, app, tddft_result, tddft_spectra
    ):
        saved = self._save(tmp_path, tddft_result, tddft_spectra)
        ctx = app._build_history_context(saved)
        uv = ctx.spectra_data.get("uv_vis", {})
        assert uv.get(
            "excitation_energies_ev"
        ), "spectra_data must have uv_vis.excitation_energies_ev for panel dispatch"


# ---------------------------------------------------------------------------
# Part 3: _apply_analysis_context activates the correct panels
# ---------------------------------------------------------------------------


class TestTDDFTPanelActivation:
    def _save(self, tmp_path, tddft_result, spectra):
        return save_result(
            tddft_result, results_dir=tmp_path, calc_type="tddft", spectra=spectra
        )

    def test_uv_vis_panel_activates_with_data(
        self, tmp_path, app, tddft_result, tddft_spectra
    ):
        saved = self._save(tmp_path, tddft_result, tddft_spectra)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "UV-Vis" in app._ana_available

    def test_uv_vis_absent_when_spectra_empty(self, tmp_path, app, tddft_result):
        saved = save_result(
            tddft_result, results_dir=tmp_path, calc_type="tddft", spectra={}
        )
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "UV-Vis" not in app._ana_available

    def test_navigate_button_visible_when_panel_activates(
        self, tmp_path, app, tddft_result, tddft_spectra
    ):
        saved = self._save(tmp_path, tddft_result, tddft_spectra)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert app._to_analysis_btn.layout.display == ""


# ---------------------------------------------------------------------------
# Part 4: _do_run end-to-end (PySCF-gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _PYSCF_AVAILABLE, reason="PySCF not available on this platform")
class TestTDDFTDoRunEndToEnd:
    """Full pipeline: real RHF/STO-3G TD-DFT (3 states) → disk → history → panel."""

    def _run_tddft(self, app, tmp_dir, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_dir))
        app.calc_type_dd.value = "UV-Vis (TD-DFT)"
        app.method_dd.value = (
            "B3LYP"  # RHF lacks TDDFT; B3LYP is the canonical TD-DFT choice
        )
        app.nstates_si.value = 3
        app._do_run()
        return list_results(tmp_dir)

    @pytest.fixture
    def running_app(self):
        a = QuantUIApp()
        a._set_molecule(_water())
        return a

    def test_do_run_saves_calc_type_tddft(self, tmp_path, running_app, monkeypatch):
        saved = self._run_tddft(running_app, tmp_path, monkeypatch)
        assert saved, "No result saved to disk"
        data = load_result(saved[0])
        assert data["calc_type"] == "tddft"

    def test_do_run_saves_excitation_energies(self, tmp_path, running_app, monkeypatch):
        saved = self._run_tddft(running_app, tmp_path, monkeypatch)
        assert saved
        data = load_result(saved[0])
        energies = (
            data.get("spectra", {}).get("uv_vis", {}).get("excitation_energies_ev")
        )
        assert (
            energies is not None and len(energies) >= 1
        ), "at least 1 excitation energy must be saved"

    def test_history_load_activates_uv_vis_panel(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_tddft(running_app, tmp_path, monkeypatch)
        assert saved
        running_app._deactivate_all_ana_panels()
        running_app._history_load_analysis(saved[0])
        assert "UV-Vis" in running_app._ana_available
