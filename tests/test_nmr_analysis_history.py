"""Integration tests for the NMR shielding analysis history roundtrip.

Covers the complete path from "calculation finishes" to "history panels activate":

  (1) Spectra structure  — save_result stores result.json with calc_type "nmr"
                           and the correct nmr spectra keys.
  (2) History context    — _build_history_context loads the spectra correctly.
  (3) Panel activation   — _apply_analysis_context activates the NMR panel
                           from a history context.
  (4) _do_run end-to-end — Full path: real RHF/STO-3G NMR → disk write →
                           _history_load_analysis → panel activates.
                           (PySCF-gated; requires pyscf-properties.)
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from quantui.app import QuantUIApp
from quantui.molecule import Molecule
from quantui.results_storage import list_results, load_result, save_result

try:
    from quantui.app import _PYSCF_AVAILABLE
except ImportError:
    _PYSCF_AVAILABLE = False

# NMR additionally requires pyscf-properties (pip install pyscf-properties).
# Attempt a lightweight import check without running a calculation.
try:
    from quantui.nmr_calc import run_nmr_calc  # noqa: F401

    _NMR_AVAILABLE = _PYSCF_AVAILABLE
except ImportError:
    _NMR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _water():
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )


def _make_nmr_result():
    """Minimal namespace mirroring a real RHF/STO-3G NMR result on water.

    Water has 3 atoms (1 O + 2 H) so shielding values should have length 3.
    Chemical shift keys use string indices (as _do_run serialises them).
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
        atom_symbols=["O", "H", "H"],
        shielding_iso_ppm=[320.5, 28.1, 28.1],
        chemical_shifts_ppm={1: -1.5, 2: -1.5},  # int keys, serialised to str
        reference_compound="TMS",
    )


def _make_nmr_spectra(result=None):
    """Build the spectra dict as _do_run does for an NMR calculation."""
    if result is None:
        result = _make_nmr_result()
    return {
        "nmr": {
            "atom_symbols": list(result.atom_symbols),
            "shielding_iso_ppm": list(result.shielding_iso_ppm),
            "chemical_shifts_ppm": {
                str(k): v for k, v in result.chemical_shifts_ppm.items()
            },
            "reference_compound": result.reference_compound,
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
def nmr_result():
    return _make_nmr_result()


@pytest.fixture
def nmr_spectra(nmr_result):
    return _make_nmr_spectra(nmr_result)


# ---------------------------------------------------------------------------
# Part 1: save_result stores the correct JSON structure for NMR
# ---------------------------------------------------------------------------


class TestNMRSpectraStructure:
    def test_result_json_has_nmr_calc_type(self, tmp_path, nmr_result, nmr_spectra):
        saved = save_result(
            nmr_result, results_dir=tmp_path, calc_type="nmr", spectra=nmr_spectra
        )
        data = json.loads((saved / "result.json").read_text())
        assert data["calc_type"] == "nmr"

    def test_atom_symbols_and_shielding_present(
        self, tmp_path, nmr_result, nmr_spectra
    ):
        saved = save_result(
            nmr_result, results_dir=tmp_path, calc_type="nmr", spectra=nmr_spectra
        )
        data = json.loads((saved / "result.json").read_text())
        nmr = data["spectra"]["nmr"]
        assert "atom_symbols" in nmr
        assert "shielding_iso_ppm" in nmr

    def test_atom_symbols_and_shielding_same_length(
        self, tmp_path, nmr_result, nmr_spectra
    ):
        saved = save_result(
            nmr_result, results_dir=tmp_path, calc_type="nmr", spectra=nmr_spectra
        )
        data = json.loads((saved / "result.json").read_text())
        nmr = data["spectra"]["nmr"]
        assert len(nmr["atom_symbols"]) == len(
            nmr["shielding_iso_ppm"]
        ), "atom_symbols and shielding_iso_ppm must have the same length"


# ---------------------------------------------------------------------------
# Part 2: _build_history_context reconstructs the context correctly
# ---------------------------------------------------------------------------


class TestNMRHistoryContext:
    def _save(self, tmp_path, nmr_result, nmr_spectra):
        return save_result(
            nmr_result, results_dir=tmp_path, calc_type="nmr", spectra=nmr_spectra
        )

    def test_context_has_correct_calc_type(
        self, tmp_path, app, nmr_result, nmr_spectra
    ):
        saved = self._save(tmp_path, nmr_result, nmr_spectra)
        ctx = app._build_history_context(saved)
        assert ctx is not None
        assert ctx.calc_type == "nmr"

    def test_context_spectra_data_has_nmr_keys(
        self, tmp_path, app, nmr_result, nmr_spectra
    ):
        saved = self._save(tmp_path, nmr_result, nmr_spectra)
        ctx = app._build_history_context(saved)
        nmr = ctx.spectra_data.get("nmr", {})
        assert nmr.get("atom_symbols"), "spectra_data must have nmr.atom_symbols"
        assert nmr.get(
            "shielding_iso_ppm"
        ), "spectra_data must have nmr.shielding_iso_ppm"


# ---------------------------------------------------------------------------
# Part 3: _apply_analysis_context activates the correct panels
# ---------------------------------------------------------------------------


class TestNMRPanelActivation:
    def _save(self, tmp_path, nmr_result, spectra):
        return save_result(
            nmr_result, results_dir=tmp_path, calc_type="nmr", spectra=spectra
        )

    def test_nmr_panel_activates_with_data(
        self, tmp_path, app, nmr_result, nmr_spectra
    ):
        saved = self._save(tmp_path, nmr_result, nmr_spectra)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "NMR" in app._ana_available

    def test_nmr_absent_when_spectra_missing(self, tmp_path, app, nmr_result):
        saved = save_result(
            nmr_result, results_dir=tmp_path, calc_type="nmr", spectra={}
        )
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "NMR" not in app._ana_available

    def test_navigate_button_visible_when_panel_activates(
        self, tmp_path, app, nmr_result, nmr_spectra
    ):
        saved = self._save(tmp_path, nmr_result, nmr_spectra)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert app._to_analysis_btn.layout.display == ""


# ---------------------------------------------------------------------------
# Part 4: _do_run end-to-end (PySCF + pyscf-properties gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _NMR_AVAILABLE,
    reason="PySCF or pyscf-properties not available on this platform",
)
class TestNMRDoRunEndToEnd:
    """Full pipeline: real RHF/STO-3G NMR → disk → history → panel."""

    def _run_nmr(self, app, tmp_dir, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_dir))
        app.calc_type_dd.value = "NMR Shielding"
        app._do_run()
        return list_results(tmp_dir)

    @pytest.fixture
    def running_app(self):
        a = QuantUIApp()
        a._set_molecule(_water())
        return a

    def test_do_run_saves_calc_type_nmr(self, tmp_path, running_app, monkeypatch):
        saved = self._run_nmr(running_app, tmp_path, monkeypatch)
        assert saved, "No result saved to disk"
        data = load_result(saved[0])
        assert data["calc_type"] == "nmr"

    def test_do_run_saves_shieldings_for_all_atoms(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_nmr(running_app, tmp_path, monkeypatch)
        assert saved
        data = load_result(saved[0])
        nmr = data.get("spectra", {}).get("nmr", {})
        symbols = nmr.get("atom_symbols", [])
        shieldings = nmr.get("shielding_iso_ppm", [])
        assert symbols == ["O", "H", "H"], "water must have 3 atoms: O, H, H"
        assert len(shieldings) == 3, "must have one shielding value per atom"
        assert all(
            math.isfinite(s) for s in shieldings
        ), "all shielding values must be finite floats"

    def test_history_load_activates_nmr_panel(self, tmp_path, running_app, monkeypatch):
        saved = self._run_nmr(running_app, tmp_path, monkeypatch)
        assert saved
        running_app._deactivate_all_ana_panels()
        running_app._history_load_analysis(saved[0])
        assert "NMR" in running_app._ana_available
