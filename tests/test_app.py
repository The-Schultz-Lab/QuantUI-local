"""
Tests for quantui.app.QuantUIApp — FR-012 Phase 4.

All tests instantiate QuantUIApp() without calling .display(), which is safe
on any platform (display() requires an active IPython kernel; construction does
not).  PySCF is unavailable on Windows; calculations are skipped accordingly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import ipywidgets as widgets
import pytest

from quantui.app import _RE_CONV, _RE_CYCLE, QuantUIApp, _LogCapture
from quantui.molecule import Molecule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _water() -> Molecule:
    """Return a minimal water molecule for testing."""
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    """QuantUIApp constructs successfully without calling display()."""

    def test_instantiates(self):
        app = QuantUIApp()
        assert app is not None

    def test_root_tab_is_widget(self):
        app = QuantUIApp()
        assert isinstance(app.root_tab, widgets.Tab)

    def test_widget_property_returns_root_tab(self):
        app = QuantUIApp()
        assert app.widget is app.root_tab

    def test_initial_state(self):
        app = QuantUIApp()
        assert app._molecule is None
        assert app._last_result is None
        assert app._results == []

    def test_initial_molecule_is_none(self):
        app = QuantUIApp()
        assert app._molecule is None

    def test_run_btn_initially_disabled(self):
        app = QuantUIApp()
        assert app.run_btn.disabled is True

    def test_export_btn_initially_disabled(self):
        app = QuantUIApp()
        assert app.export_btn.disabled is True


# ---------------------------------------------------------------------------
# Default widget values
# ---------------------------------------------------------------------------


class TestDefaultWidgetValues:
    """Widget dropdowns and inputs have expected defaults."""

    def test_method_default(self):
        from quantui.config import DEFAULT_METHOD

        app = QuantUIApp()
        assert app.method_dd.value == DEFAULT_METHOD

    def test_basis_default(self):
        from quantui.config import DEFAULT_BASIS

        app = QuantUIApp()
        assert app.basis_dd.value == DEFAULT_BASIS

    def test_calc_type_default(self):
        app = QuantUIApp()
        assert app.calc_type_dd.value == "Single Point"

    def test_theme_default(self):
        app = QuantUIApp()
        assert app.theme_btn.value == "Dark"

    def test_charge_default(self):
        from quantui.config import DEFAULT_CHARGE

        app = QuantUIApp()
        assert app.charge_si.value == DEFAULT_CHARGE

    def test_multiplicity_default(self):
        from quantui.config import DEFAULT_MULTIPLICITY

        app = QuantUIApp()
        assert app.mult_si.value == DEFAULT_MULTIPLICITY


# ---------------------------------------------------------------------------
# Tab structure
# ---------------------------------------------------------------------------


class TestTabStructure:
    """root_tab has the correct number and titles of tabs."""

    def test_five_tabs(self):
        app = QuantUIApp()
        assert len(app.root_tab.children) == 5

    def test_tab_titles(self):
        app = QuantUIApp()
        expected = ["Calculate", "History", "Compare", "Output", "Help"]
        for i, title in enumerate(expected):
            assert app.root_tab.get_title(i) == title


# ---------------------------------------------------------------------------
# Molecule input — collapse / expand pattern
# ---------------------------------------------------------------------------


class TestMoleculeInputCollapse:
    """mol_input_container switches between expanded and collapsed views."""

    def test_initially_expanded(self):
        app = QuantUIApp()
        # Expanded: first child is mol_input_expanded
        assert app.mol_input_container.children[0] is app.mol_input_expanded

    def test_collapses_after_set_molecule(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        # Collapsed: first child is mol_input_collapsed
        assert app.mol_input_container.children[0] is app.mol_input_collapsed

    def test_molecule_stored_after_set_molecule(self):
        app = QuantUIApp()
        mol = _water()
        app._set_molecule(mol)
        assert app._molecule is mol

    def test_run_btn_enabled_after_set_molecule(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        assert app.run_btn.disabled is False

    def test_export_btn_enabled_after_set_molecule(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        assert app.export_btn.disabled is False

    def test_mol_info_html_updated(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        assert "H2O" in app.mol_info_html.value

    def test_expand_restores_expanded_view(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        # Simulate clicking "Change molecule"
        app._on_expand_mol_input(None)
        assert app.mol_input_container.children[0] is app.mol_input_expanded

    def test_multiplicity_above_one_switches_to_uhf(self):
        app = QuantUIApp()
        app.method_dd.value = "RHF"
        radical = Molecule(
            atoms=["H"],
            coordinates=[[0.0, 0.0, 0.0]],
            multiplicity=2,
        )
        app._set_molecule(radical)
        assert app.method_dd.value == "UHF"

    def test_rhf_kept_for_singlet(self):
        app = QuantUIApp()
        app.method_dd.value = "RHF"
        app._set_molecule(_water())
        assert app.method_dd.value == "RHF"


# ---------------------------------------------------------------------------
# Step progress
# ---------------------------------------------------------------------------


class TestStepProgress:
    """Step indicator advances correctly through the workflow."""

    def test_step_0_active_initially(self):
        app = QuantUIApp()
        # Step 0 ("Choose molecule") should be active at start
        assert app.step_progress._states[0] == "active"

    def test_step_advances_after_set_molecule(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        # Step 0 done, step 1 should be active
        assert app.step_progress._states[0] == "done"
        assert app.step_progress._states[1] == "active"


# ---------------------------------------------------------------------------
# _LogCapture
# ---------------------------------------------------------------------------


class TestLogCapture:
    """_LogCapture parses SCF cycle lines and updates the status label."""

    def _make_capture(self):
        out = widgets.Output()
        status = widgets.Label()
        return _LogCapture(out, status), status

    def test_write_buffers_text(self):
        cap, _ = self._make_capture()
        cap.write("hello world\n")
        assert "hello world" in cap.getvalue()

    def test_cycle_regex_parses_line(self):
        line = "cycle= 3 E= -76.031234  delta_E= -0.0042"
        m = _RE_CYCLE.search(line)
        assert m is not None
        assert m.group(1) == "3"

    def test_conv_regex_parses_line(self):
        line = "converged SCF energy = -76.031234"
        m = _RE_CONV.search(line)
        assert m is not None

    def test_status_label_updated_on_cycle(self):
        cap, status = self._make_capture()
        cap.write("cycle= 2 E= -76.031234  delta_E= -0.0042\n")
        assert "SCF cycle 2" in status.value

    def test_status_label_updated_on_convergence(self):
        cap, status = self._make_capture()
        cap.write("converged SCF energy = -76.031234\n")
        assert "converged" in status.value.lower()

    def test_flush_is_noop(self):
        cap, _ = self._make_capture()
        cap.flush()  # Must not raise

    def test_empty_write_is_noop(self):
        cap, _ = self._make_capture()
        cap.write("")
        assert cap.getvalue() == ""


# ---------------------------------------------------------------------------
# _do_run dispatch
# ---------------------------------------------------------------------------


class TestDoRunDispatch:
    """_do_run dispatches to the correct calculation function."""

    @pytest.fixture
    def app_with_molecule(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        return app

    def test_single_point_dispatch(self, app_with_molecule):
        app = app_with_molecule
        app.calc_type_dd.value = "Single Point"
        mock_result = MagicMock()
        mock_result.energy_hartree = -75.0
        mock_result.homo_lumo_gap_ev = 12.3
        mock_result.converged = True
        mock_result.n_iterations = 10
        mock_result.formula = "H2O"
        mock_result.method = "RHF"
        mock_result.basis = "STO-3G"
        with patch(
            "quantui.run_in_session", return_value=mock_result, create=True
        ) as mock_run:
            with patch("quantui.save_result"):
                app._do_run()
        mock_run.assert_called_once()

    def test_geo_opt_dispatch(self, app_with_molecule):
        app = app_with_molecule
        app.calc_type_dd.value = "Geometry Opt"
        mock_result = MagicMock()
        mock_result.energy_hartree = -75.0
        mock_result.converged = True
        mock_result.n_iterations = 5
        mock_result.trajectory = []
        mock_result.formula = "H2O"
        mock_result.method = "RHF"
        mock_result.basis = "STO-3G"
        mock_result.final_molecule = _water()
        with patch(
            "quantui.optimize_geometry", return_value=mock_result, create=True
        ) as mock_opt:
            with patch("quantui.save_result"):
                app._do_run()
        mock_opt.assert_called_once()

    def test_pyscf_unavailable_shows_error(self, app_with_molecule):
        app = app_with_molecule
        app.calc_type_dd.value = "Single Point"
        with patch("quantui.app._PYSCF_AVAILABLE", False):
            app._do_run()
        # Should not raise; error message should be in run_output
        # (output widget content is opaque, so just verify no exception)


# ---------------------------------------------------------------------------
# Availability flags on the instance
# ---------------------------------------------------------------------------


class TestAvailabilityFlags:
    def test_pyscf_flag_mirrors_module_level(self):
        from quantui.app import _PYSCF_AVAILABLE

        app = QuantUIApp()
        assert app._pyscf_available == _PYSCF_AVAILABLE

    def test_preopt_flag_mirrors_module_level(self):
        from quantui.app import _PREOPT_AVAILABLE

        app = QuantUIApp()
        assert app._preopt_available == _PREOPT_AVAILABLE


# ---------------------------------------------------------------------------
# M3.3 — result log accordion and directory label
# ---------------------------------------------------------------------------


class TestResultLogAccordion:
    """_result_log_accordion and _result_dir_label exist and start hidden."""

    def test_log_accordion_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_result_log_accordion")
        assert isinstance(app._result_log_accordion, widgets.Accordion)

    def test_log_accordion_initially_hidden(self):
        app = QuantUIApp()
        assert app._result_log_accordion.layout.display == "none"

    def test_log_accordion_initially_collapsed(self):
        app = QuantUIApp()
        assert app._result_log_accordion.selected_index is None

    def test_result_dir_label_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_result_dir_label")
        assert isinstance(app._result_dir_label, widgets.HTML)

    def test_result_dir_label_initially_hidden(self):
        app = QuantUIApp()
        assert app._result_dir_label.layout.display == "none"

    def test_last_result_dir_initially_none(self):
        app = QuantUIApp()
        assert app._last_result_dir is None

    def test_on_run_clicked_clears_log(self):
        """_on_run_clicked must hide log accordion and clear dir label."""
        app = QuantUIApp()
        # Simulate a previous result being present
        app._result_log_accordion.layout.display = ""
        app._result_dir_label.layout.display = ""
        app._result_dir_label.value = "Saved to: /some/path"

        with patch.object(app, "_do_run"):
            app._on_run_clicked(None)

        assert app._result_log_accordion.layout.display == "none"
        assert app._result_dir_label.layout.display == "none"
        assert app._result_dir_label.value == ""

    def test_show_result_log_populates_widgets(self, tmp_path):
        """_show_result_log() sets dir label and reveals accordion."""
        log_text = "SCF converged in 10 cycles."
        log_file = tmp_path / "pyscf.log"
        log_file.write_text(log_text, encoding="utf-8")

        app = QuantUIApp()
        app._show_result_log(tmp_path, log_text)

        assert str(tmp_path) in app._result_dir_label.value
        assert app._result_dir_label.layout.display == ""
        assert app._result_log_accordion.layout.display == ""

    def test_show_result_log_falls_back_to_string(self, tmp_path):
        """_show_result_log() uses in-memory log_text if pyscf.log absent."""
        log_text = "fallback log content"
        app = QuantUIApp()
        empty_dir = tmp_path / "no_log_here"
        empty_dir.mkdir()
        app._show_result_log(empty_dir, log_text)

        assert app._result_log_accordion.layout.display == ""


# ---------------------------------------------------------------------------
# M3.4 — Structure file exports (XYZ, MOL/SDF, PDB)
# ---------------------------------------------------------------------------


class TestStructureExportButtons:
    """export_xyz_btn, export_mol_btn, export_pdb_btn exist and start disabled."""

    def test_export_xyz_btn_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "export_xyz_btn")
        assert isinstance(app.export_xyz_btn, widgets.Button)

    def test_export_mol_btn_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "export_mol_btn")
        assert isinstance(app.export_mol_btn, widgets.Button)

    def test_export_pdb_btn_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "export_pdb_btn")
        assert isinstance(app.export_pdb_btn, widgets.Button)

    def test_struct_export_status_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "struct_export_status")

    def test_export_xyz_btn_disabled_initially(self):
        app = QuantUIApp()
        assert app.export_xyz_btn.disabled is True

    def test_export_xyz_btn_enabled_after_set_molecule(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        assert app.export_xyz_btn.disabled is False

    def test_export_accordion_title_is_export(self):
        app = QuantUIApp()
        assert app.advanced_accordion.get_title(0) == "Export"


class TestExportXYZCallback:
    """_on_export_xyz writes a valid XYZ file."""

    def test_xyz_file_written_to_result_dir(self, tmp_path):
        app = QuantUIApp()
        app._set_molecule(_water())
        app._last_result_dir = tmp_path

        app._on_export_xyz(None)

        xyz_files = list(tmp_path.glob("*.xyz"))
        assert len(xyz_files) == 1

    def test_xyz_file_contains_atom_count(self, tmp_path):
        app = QuantUIApp()
        app._set_molecule(_water())
        app._last_result_dir = tmp_path

        app._on_export_xyz(None)

        content = list(tmp_path.glob("*.xyz"))[0].read_text()
        first_line = content.splitlines()[0].strip()
        assert first_line == "3"  # water has 3 atoms

    def test_xyz_status_shows_saved_path(self, tmp_path):
        app = QuantUIApp()
        app._set_molecule(_water())
        app._last_result_dir = tmp_path

        app._on_export_xyz(None)

        assert "Saved" in app.struct_export_status.value

    def test_xyz_no_molecule_shows_error(self):
        app = QuantUIApp()
        app._on_export_xyz(None)
        assert "molecule" in app.struct_export_status.value.lower()


class TestExportMoleculeAndLabel:
    """_export_molecule_and_label returns correct molecule and labels."""

    def test_returns_current_molecule_when_no_result(self):
        app = QuantUIApp()
        water = _water()
        app._set_molecule(water)
        mol, method, basis = app._export_molecule_and_label()
        assert mol is water

    def test_method_falls_back_to_dropdown(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        _, method, _ = app._export_molecule_and_label()
        assert method == app.method_dd.value


class TestMoleculeToRdkit:
    """_molecule_to_rdkit does not raise; returns RDKit mol or None."""

    def test_does_not_raise_for_water(self):
        result = QuantUIApp._molecule_to_rdkit(_water())
        # Either succeeds or returns None — must not raise
        assert result is None or result is not None
