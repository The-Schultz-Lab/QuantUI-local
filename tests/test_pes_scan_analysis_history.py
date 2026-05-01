"""Integration tests for the PES Scan analysis history roundtrip.

Covers the complete path from "calculation finishes" to "history panels activate":

  (1) Spectra structure  — save_result stores result.json with calc_type "pes_scan"
                           and the correct pes_scan spectra keys.
  (2) History context    — _build_history_context loads the spectra correctly.
  (3) Panel activation   — _apply_analysis_context activates the PES Scan panel
                           from a history context; Trajectory activates when
                           trajectory.json is present.
  (4) _do_run end-to-end — Full path: real RHF/STO-3G bond scan on H2 →
                           disk write → _history_load_analysis → panels activate.
                           (PySCF-gated; skipped on Windows.)
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from quantui.app import QuantUIApp
from quantui.molecule import Molecule
from quantui.results_storage import (
    list_results,
    load_result,
    save_result,
    save_trajectory,
)

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


def _h2():
    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


def _make_pes_result():
    """Minimal namespace mirroring a real RHF/STO-3G bond scan on H2."""
    scan_values = [0.60, 0.70, 0.74, 0.80, 0.90]
    energies = [-1.060, -1.115, -1.117, -1.100, -1.060]
    mol_at_0 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.60]])
    mol_at_1 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.70]])
    mol_at_2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    mol_at_3 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.80]])
    mol_at_4 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.90]])
    return SimpleNamespace(
        formula="H2",
        method="RHF",
        basis="STO-3G",
        energy_hartree=min(energies),
        energy_ev=min(energies) * 27.211,
        homo_lumo_gap_ev=8.0,
        converged=True,
        n_iterations=10,
        scan_type="bond",
        atom_indices=[0, 1],
        scan_parameter_values=scan_values,
        energies_hartree=energies,
        coordinates_list=[mol_at_0, mol_at_1, mol_at_2, mol_at_3, mol_at_4],
        converged_all=True,
    )


def _make_pes_spectra(result=None):
    """Build the spectra dict as _do_run does for a PES scan."""
    if result is None:
        result = _make_pes_result()
    return {
        "pes_scan": {
            "scan_type": result.scan_type,
            "atom_indices": result.atom_indices,
            "scan_parameter_values": result.scan_parameter_values,
            "energies_hartree": result.energies_hartree,
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
def pes_result():
    return _make_pes_result()


@pytest.fixture
def pes_spectra(pes_result):
    return _make_pes_spectra(pes_result)


# ---------------------------------------------------------------------------
# Part 1: save_result stores the correct JSON structure for pes_scan
# ---------------------------------------------------------------------------


class TestPESScanSpectraStructure:
    def test_result_json_has_pes_scan_calc_type(
        self, tmp_path, pes_result, pes_spectra
    ):
        saved = save_result(
            pes_result, results_dir=tmp_path, calc_type="pes_scan", spectra=pes_spectra
        )
        data = json.loads((saved / "result.json").read_text())
        assert data["calc_type"] == "pes_scan"

    def test_pes_scan_keys_present(self, tmp_path, pes_result, pes_spectra):
        saved = save_result(
            pes_result, results_dir=tmp_path, calc_type="pes_scan", spectra=pes_spectra
        )
        data = json.loads((saved / "result.json").read_text())
        scan = data["spectra"]["pes_scan"]
        assert "scan_type" in scan
        assert "atom_indices" in scan
        assert "scan_parameter_values" in scan
        assert "energies_hartree" in scan

    def test_scan_values_non_empty(self, tmp_path, pes_result, pes_spectra):
        saved = save_result(
            pes_result, results_dir=tmp_path, calc_type="pes_scan", spectra=pes_spectra
        )
        data = json.loads((saved / "result.json").read_text())
        scan = data["spectra"]["pes_scan"]
        assert len(scan["scan_parameter_values"]) >= 2
        assert len(scan["energies_hartree"]) == len(scan["scan_parameter_values"])


# ---------------------------------------------------------------------------
# Part 2: _build_history_context reconstructs the context correctly
# ---------------------------------------------------------------------------


class TestPESScanHistoryContext:
    def _save(self, tmp_path, pes_result, pes_spectra):
        return save_result(
            pes_result, results_dir=tmp_path, calc_type="pes_scan", spectra=pes_spectra
        )

    def test_context_has_correct_calc_type(
        self, tmp_path, app, pes_result, pes_spectra
    ):
        saved = self._save(tmp_path, pes_result, pes_spectra)
        ctx = app._build_history_context(saved)
        assert ctx is not None
        assert ctx.calc_type == "pes_scan"

    def test_context_spectra_data_has_pes_scan(
        self, tmp_path, app, pes_result, pes_spectra
    ):
        saved = self._save(tmp_path, pes_result, pes_spectra)
        ctx = app._build_history_context(saved)
        scan = ctx.spectra_data.get("pes_scan", {})
        assert scan.get(
            "energies_hartree"
        ), "spectra_data must have pes_scan.energies_hartree for panel dispatch"
        assert scan.get(
            "scan_parameter_values"
        ), "spectra_data must have pes_scan.scan_parameter_values"


# ---------------------------------------------------------------------------
# Part 3: _apply_analysis_context activates the correct panels
# ---------------------------------------------------------------------------


class TestPESScanPanelActivation:
    def _save(self, tmp_path, pes_result, pes_spectra):
        return save_result(
            pes_result, results_dir=tmp_path, calc_type="pes_scan", spectra=pes_spectra
        )

    def test_pes_scan_panel_activates_with_data(
        self, tmp_path, app, pes_result, pes_spectra
    ):
        saved = self._save(tmp_path, pes_result, pes_spectra)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "PES Scan" in app._ana_available

    def test_pes_scan_absent_when_spectra_empty(self, tmp_path, app, pes_result):
        saved = save_result(
            pes_result, results_dir=tmp_path, calc_type="pes_scan", spectra={}
        )
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "PES Scan" not in app._ana_available

    def test_trajectory_panel_activates_with_trajectory_json(
        self, tmp_path, app, pes_result, pes_spectra
    ):
        saved = self._save(tmp_path, pes_result, pes_spectra)
        save_trajectory(saved, pes_result.coordinates_list, pes_result.energies_hartree)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Trajectory" in app._ana_available

    def test_trajectory_absent_when_no_trajectory_json(
        self, tmp_path, app, pes_result, pes_spectra
    ):
        saved = self._save(tmp_path, pes_result, pes_spectra)
        assert not (saved / "trajectory.json").exists()
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Trajectory" not in app._ana_available

    def test_navigate_button_visible_when_panel_activates(
        self, tmp_path, app, pes_result, pes_spectra
    ):
        saved = self._save(tmp_path, pes_result, pes_spectra)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert app._to_analysis_btn.layout.display == ""


# ---------------------------------------------------------------------------
# Part 4: _do_run end-to-end (PySCF-gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _PYSCF_AVAILABLE or sys.platform == "win32",
    reason="PySCF not available or not supported on native Windows",
)
class TestPESScanDoRunEndToEnd:
    """Full pipeline: real RHF/STO-3G H2 bond scan → disk → history → panels."""

    def _run_pes_scan(self, app, tmp_dir, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_dir))
        app.calc_type_dd.value = "PES Scan"
        app._scan_type_dd.value = "Bond"
        app._scan_atom1.value = 1
        app._scan_atom2.value = 2
        app._scan_start.value = 0.6
        app._scan_stop.value = 0.9
        app._scan_steps.value = 3
        app._do_run()
        return list_results(tmp_dir)

    @pytest.fixture
    def running_app(self):
        a = QuantUIApp()
        a._set_molecule(_h2())
        return a

    def test_do_run_saves_calc_type_pes_scan(self, tmp_path, running_app, monkeypatch):
        saved = self._run_pes_scan(running_app, tmp_path, monkeypatch)
        assert saved, "No result saved to disk"
        data = load_result(saved[0])
        assert data["calc_type"] == "pes_scan"

    def test_do_run_saves_scan_parameter_values(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_pes_scan(running_app, tmp_path, monkeypatch)
        assert saved
        data = load_result(saved[0])
        vals = data.get("spectra", {}).get("pes_scan", {}).get("scan_parameter_values")
        assert (
            vals is not None and len(vals) >= 1
        ), "at least 1 scan point must be saved"

    def test_do_run_saves_energies_hartree(self, tmp_path, running_app, monkeypatch):
        saved = self._run_pes_scan(running_app, tmp_path, monkeypatch)
        assert saved
        data = load_result(saved[0])
        energies = data.get("spectra", {}).get("pes_scan", {}).get("energies_hartree")
        assert energies is not None and len(energies) >= 1

    def test_history_load_activates_pes_scan_panel(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_pes_scan(running_app, tmp_path, monkeypatch)
        assert saved
        running_app._deactivate_all_ana_panels()
        running_app._history_load_analysis(saved[0])
        assert "PES Scan" in running_app._ana_available

    def test_history_load_activates_trajectory_panel(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_pes_scan(running_app, tmp_path, monkeypatch)
        assert saved
        running_app._deactivate_all_ana_panels()
        running_app._history_load_analysis(saved[0])
        assert "Trajectory" in running_app._ana_available
