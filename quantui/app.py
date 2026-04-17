"""
QuantUI-local application class.

All widget creation, state management, callbacks, and tab wiring live here.
The notebook is a thin launcher::

    from quantui.app import QuantUIApp
    QuantUIApp().display()

CSS is injected inside ``display()`` — not on import — so importing this
module in tests or tutorials does not pollute the IPython display.
"""

from __future__ import annotations

import asyncio
import io
import re
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

import ipywidgets as widgets
from IPython import get_ipython
from IPython.display import HTML, Javascript, display

import quantui
import quantui.calc_log as _calc_log

# Import directly from submodules to avoid circular-import issues.
# quantui/__init__.py imports this module (app.py), so using
# `from quantui import X` at module load time would see a partially-
# initialised package namespace (symbols defined after the app import
# in __init__.py would not yet exist).
from quantui.config import (
    DEFAULT_BASIS,
    DEFAULT_CHARGE,
    DEFAULT_FMAX,
    DEFAULT_METHOD,
    DEFAULT_MULTIPLICITY,
    DEFAULT_OPT_STEPS,
    MOLECULE_LIBRARY,
    SUPPORTED_BASIS_SETS,
    SUPPORTED_METHODS,
)
from quantui.help_content import HELP_TOPICS
from quantui.molecule import Molecule, parse_xyz_input
from quantui.progress import StepProgress
from quantui.utils import get_session_resources

# ── Availability flags (computed once at import, not per-instantiation) ───────
try:
    from quantui.ase_bridge import ASE_AVAILABLE
except ImportError:
    ASE_AVAILABLE = False

try:
    from quantui.visualization_py3dmol import display_molecule as _display_molecule

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    _display_molecule = None  # type: ignore[assignment]

try:
    from quantui.pubchem import (
        RDKIT_AVAILABLE as _PUBCHEM_RDKIT_AVAILABLE,
    )
    from quantui.pubchem import (
        student_friendly_fetch as _student_friendly_fetch,
    )

    PUBCHEM_AVAILABLE = _PUBCHEM_RDKIT_AVAILABLE
except ImportError:
    PUBCHEM_AVAILABLE = False
    _student_friendly_fetch = None  # type: ignore[assignment]

try:
    from quantui.session_calc import SessionResult, run_in_session  # noqa: F401

    _PYSCF_AVAILABLE = True
except (ImportError, AttributeError):
    _PYSCF_AVAILABLE = False

try:
    from quantui.preopt import preoptimize

    _PREOPT_AVAILABLE = True
except (ImportError, AttributeError):
    _PREOPT_AVAILABLE = False

# ── Module-level constants ────────────────────────────────────────────────────
_THEME_HUE: dict = {"Dark": 180}

_APP_CSS: str = """<style>
/* System font stack ---------------------------------------------------- */
body, p, span, li, td, th, label, input, select, textarea, blockquote,
.jp-OutputArea-output, .widget-html-content, .widget-label-basic,
.widget-label {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
                 Roboto, "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* App title (h1 in the markdown cell) ---------------------------------- */
h1 {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    letter-spacing: -0.01em !important;
    margin: 10px 0 4px !important;
    border-bottom: none !important;
}

/* Section headers ------------------------------------------------------- */
h3 {
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
    margin: 24px 0 10px !important;
    padding-bottom: 5px !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

/* Rounded corners on inputs, dropdowns, and buttons -------------------- */
.widget-text input, .widget-textarea textarea {
    border-color: #d1d5db !important;
    border-radius: 5px !important;
}
.widget-dropdown select { border-radius: 5px !important; }
.widget-button, .widget-toggle-button { border-radius: 5px !important; }
</style>"""


# ── SCF regex (module-level so _LogCapture can use them) ─────────────────────
_RE_CYCLE = re.compile(
    r"cycle=\s*(\d+)\s+E=\s*([\-\d\.]+)\s+delta_E=\s*([\-\d\.Ee+\-]+)"
)
_RE_CONV = re.compile(r"converged SCF energy\s*=\s*([\-\d\.]+)")


# ══ LOG CAPTURE ══════════════════════════════════════════════════════════════


class _LogCapture:
    """Write PySCF output to an Output widget and capture it to a buffer."""

    def __init__(
        self,
        output_widget: widgets.Output,
        status_label: Optional[widgets.Label] = None,
    ) -> None:
        self._w = output_widget
        self._buf = io.StringIO()
        self._line_buf = ""
        self._status = status_label

    def write(self, text: str) -> None:
        if not text:
            return
        self._w.append_stdout(text)
        self._buf.write(text)
        self._line_buf += text
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            m = _RE_CYCLE.search(line)
            if m and self._status is not None:
                n, delta = m.group(1), m.group(3)
                try:
                    self._status.value = f"SCF cycle {n}  ·  ΔE = {float(delta):.4g} Ha"
                except Exception:
                    self._status.value = f"SCF cycle {n}"
                continue
            m = _RE_CONV.search(line)
            if m and self._status is not None:
                self._status.value = "SCF converged ✓"

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return self._buf.getvalue()


# ══ APP CLASS ════════════════════════════════════════════════════════════════


class QuantUIApp:
    """
    Self-contained QuantUI-local application widget.

    Instantiate once; call ``display()`` to inject CSS and show the UI::

        app = QuantUIApp()
        app.display()
    """

    def __init__(self) -> None:
        # ── Instance state ────────────────────────────────────────────────
        self._molecule: Optional[Molecule] = None
        self._last_result: Any = None
        self._results: List = []

        # Availability (copied from module-level flags)
        self._pyscf_available: bool = _PYSCF_AVAILABLE
        self._preopt_available: bool = _PREOPT_AVAILABLE

        # ── Build → wire → assemble ───────────────────────────────────────
        self._build_widgets()
        self._wire_callbacks()
        self._assemble_tabs()

        # Log startup, but never let optional logging I/O break app startup.
        try:
            _calc_log.log_event(
                "startup", f"QuantUI-local {quantui.__version__} started"
            )
        except OSError:
            pass

    def display(self) -> None:
        """Inject global CSS and render the application widget."""
        display(HTML(_APP_CSS))
        display(
            widgets.VBox(
                [
                    widgets.HBox(
                        [self.theme_btn],
                        layout=widgets.Layout(
                            justify_content="flex-end", margin="0 0 4px"
                        ),
                    ),
                    self._theme_style,
                    self._status_html,
                    self.root_tab,
                ]
            )
        )

    @property
    def widget(self) -> widgets.Tab:
        """The root tab widget (for callers that want the widget object)."""
        return self.root_tab

    # ══ BUILD METHODS ════════════════════════════════════════════════════════

    def _build_widgets(self) -> None:
        self._build_theme_selector()
        self._build_status_panel()
        self._build_shared_widgets()
        self._build_molecule_section()
        self._build_calc_setup()
        self._build_run_section()
        self._build_results_section()
        self._build_history_section()
        self._build_compare_section()
        self._build_output_tab()
        self._build_help_section()

    # ── Theme selector ────────────────────────────────────────────────────

    def _build_theme_selector(self) -> None:
        self._theme_style = widgets.Output(
            layout=widgets.Layout(
                height="0px", overflow="hidden", margin="0", padding="0"
            )
        )
        self.theme_btn = widgets.ToggleButtons(
            options=["Light", "Dark"],
            value="Dark",
            description="Theme:",
            style={"description_width": "48px", "button_width": "90px"},
            layout=widgets.Layout(margin="0"),
        )
        # Apply Dark theme immediately
        with self._theme_style:
            display(HTML(self._theme_css("Dark")))

    def _theme_css(self, theme: str) -> str:
        """Return the CSS filter block for *theme*, or '' for Light."""
        if theme not in _THEME_HUE:
            return ""
        deg = _THEME_HUE[theme]
        return (
            "<style>"
            f"html {{ filter: invert(1) hue-rotate({deg}deg) !important; }}"
            "canvas, img, iframe, video "
            f"{{ filter: invert(1) hue-rotate({deg}deg) !important; }}"
            "</style>"
        )

    # ── Status panel ──────────────────────────────────────────────────────

    def _build_status_panel(self) -> None:
        _cores, _mem_gb = get_session_resources()
        _mem = f"{_mem_gb} GB" if _mem_gb is not None else "unknown"

        def _ok(flag: bool, extra: str = "") -> str:
            tick = '<span style="color:#22c55e">&#10003;</span>'
            cross = '<span style="color:#ef4444">&#10007;</span>'
            return (tick if flag else cross) + (" " + extra if extra else "")

        _items = [
            (
                "PySCF (calculations)",
                _ok(
                    _PYSCF_AVAILABLE,
                    "" if _PYSCF_AVAILABLE else "&mdash; Linux / macOS / WSL required",
                ),
            ),
            ("ASE (structure I/O, opt.)", _ok(ASE_AVAILABLE)),
            ("PubChem search", _ok(PUBCHEM_AVAILABLE)),
            ("3D viewer (py3Dmol)", _ok(VISUALIZATION_AVAILABLE)),
            ("CPU cores / Memory", f"<b>{_cores}</b> cores / <b>{_mem}</b>"),
        ]
        _rows = "".join(
            f'<tr><td style="padding:3px 16px 3px 0;color:#64748b;font-size:13px">{k}</td>'
            f'<td style="font-size:13px">{v}</td></tr>'
            for k, v in _items
        )
        self._status_html = widgets.HTML(
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid #3b82f6;'
            f'padding:12px 16px;border-radius:6px;margin:4px 0 8px">'
            f'<span style="font-weight:600;font-size:14px;color:#1e293b">'
            f"QuantUI-local {quantui.__version__}</span>"
            f'<table style="margin-top:8px;border-collapse:collapse">{_rows}</table>'
            f"</div>"
        )

    # ── Shared widgets (Cell 3) ───────────────────────────────────────────

    def _build_shared_widgets(self) -> None:
        # Output widgets
        self.mol_info_html = widgets.HTML(
            value='<i style="color:#888">No molecule loaded yet.</i>'
        )
        self.mol_summary_compact = widgets.HTML(value="")
        self.viz_output = widgets.Output(layout=widgets.Layout(min_height="50px"))
        self.run_output = widgets.Output(
            layout=widgets.Layout(
                border="1px solid #c0ccd8",
                min_height="80px",
                max_height="400px",
                padding="8px",
                overflow_y="auto",
            )
        )
        with self.run_output:
            display(
                HTML(
                    '<p style="color:#999;font-style:italic;font-size:13px;margin:2px 0">'
                    "No calculation run yet. PySCF output and any errors will appear here."
                    "</p>"
                )
            )
        self.result_output = widgets.Output()
        self.result_viz_output = widgets.Output()
        self.comparison_output = widgets.Output()
        self.notes_output = widgets.Output()
        self.perf_estimate_html = widgets.HTML()

        # Step indicator
        self.step_progress = StepProgress(
            ["Choose molecule", "Set method", "Run", "Results"]
        )
        self.step_progress.start(0)

        # Calculation setup dropdowns
        self.method_dd = widgets.Dropdown(
            options=SUPPORTED_METHODS,
            value=DEFAULT_METHOD,
            description="Method:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="260px"),
        )
        self.basis_dd = widgets.Dropdown(
            options=SUPPORTED_BASIS_SETS,
            value=DEFAULT_BASIS,
            description="Basis Set:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="260px"),
        )
        self.charge_si = widgets.BoundedIntText(
            value=DEFAULT_CHARGE,
            min=-10,
            max=10,
            description="Charge:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="190px"),
        )
        self.mult_si = widgets.BoundedIntText(
            value=DEFAULT_MULTIPLICITY,
            min=1,
            max=10,
            description="Multiplicity:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="190px"),
        )
        self.preopt_cb = widgets.Checkbox(
            value=False,
            description="Pre-optimize geometry (fast LJ force-field)",
            disabled=not _PREOPT_AVAILABLE,
            layout=widgets.Layout(width="400px"),
        )

        # Calculation type + extra options
        self.calc_type_dd = widgets.Dropdown(
            options=["Single Point", "Geometry Opt", "Frequency", "UV-Vis (TD-DFT)"],
            value="Single Point",
            description="Calc. Type:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="310px"),
        )
        self.fmax_fi = widgets.BoundedFloatText(
            value=DEFAULT_FMAX,
            min=0.001,
            max=1.0,
            step=0.005,
            description="Force thr. (eV/Å):",
            style={"description_width": "130px"},
            layout=widgets.Layout(width="270px"),
        )
        self.max_steps_si = widgets.BoundedIntText(
            value=DEFAULT_OPT_STEPS,
            min=10,
            max=1000,
            description="Max steps:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="200px"),
        )
        self.nstates_si = widgets.BoundedIntText(
            value=10,
            min=1,
            max=50,
            description="# states:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="180px"),
        )
        self.calc_extra_opts = widgets.VBox([])

        # Context-help buttons
        self.method_help_btn = widgets.Button(
            description="?",
            button_style="",
            layout=widgets.Layout(width="28px", height="28px"),
            tooltip="RHF vs UHF — opens Help tab",
        )
        self.basis_help_btn = widgets.Button(
            description="?",
            button_style="",
            layout=widgets.Layout(width="28px", height="28px"),
            tooltip="Choosing a basis set — opens Help tab",
        )

        # Run widgets
        self.run_btn = widgets.Button(
            description="Run Calculation",
            button_style="success",
            icon="play",
            disabled=True,
            layout=widgets.Layout(width="200px", height="36px"),
        )
        self.run_status = widgets.Label()

        # Log clear button (in run panel)
        self.log_clear_btn = widgets.Button(
            description="Clear",
            button_style="",
            icon="times",
            layout=widgets.Layout(width="90px", height="26px"),
            tooltip="Clear calculation output",
        )

        # Comparison / export widgets
        self.accumulate_btn = widgets.Button(
            description="Add to Comparison",
            button_style="info",
            icon="plus",
            disabled=True,
            layout=widgets.Layout(width="190px"),
        )
        self.clear_btn = widgets.Button(
            description="Clear",
            button_style="warning",
            icon="trash",
            layout=widgets.Layout(width="100px"),
        )
        self.export_btn = widgets.Button(
            description="Export Script",
            button_style="",
            icon="download",
            disabled=True,
            layout=widgets.Layout(width="160px"),
        )
        self.export_status = widgets.Label()

    # ── Molecule section (Cell 4) ─────────────────────────────────────────

    def _build_molecule_section(self) -> None:
        # Preset dropdown
        _preset_opts = ["(select a molecule)"] + list(MOLECULE_LIBRARY.keys())
        self.preset_dd = widgets.Dropdown(
            options=_preset_opts,
            value="(select a molecule)",
            description="Molecule:",
            style={"description_width": "90px"},
            layout=widgets.Layout(width="320px"),
        )

        # XYZ input
        self.xyz_area = widgets.Textarea(
            placeholder=(
                "Paste XYZ coordinates (symbol  x  y  z):\n"
                "O  0.000  0.000  0.000\n"
                "H  0.757  0.587  0.000\n"
                "H -0.757  0.587  0.000"
            ),
            layout=widgets.Layout(width="440px", height="130px"),
        )
        self.xyz_btn = widgets.Button(
            description="Load XYZ", button_style="info", icon="upload"
        )
        self.xyz_msg = widgets.Label()

        # PubChem search
        self.pubchem_txt = widgets.Text(
            placeholder="name or SMILES  (e.g. aspirin, caffeine, CC(=O)O)",
            layout=widgets.Layout(width="380px"),
        )
        self.pubchem_btn = widgets.Button(
            description="Search",
            button_style="info",
            icon="search",
            disabled=not PUBCHEM_AVAILABLE,
            layout=widgets.Layout(width="100px"),
        )
        self.pubchem_msg = widgets.Label(
            value=(
                ""
                if PUBCHEM_AVAILABLE
                else "PubChem unavailable — check internet connection"
            )
        )

        # Assemble input tab
        _hint = '<p style="margin:4px 0 8px;color:#666;font-size:13px">'
        tab_preset = widgets.VBox(
            [
                widgets.HTML(
                    _hint + "Choose from 20+ curated educational molecules.</p>"
                ),
                self.preset_dd,
            ]
        )
        tab_xyz = widgets.VBox(
            [
                widgets.HTML(
                    _hint
                    + "Paste XYZ coordinates (element x y z, one atom per line).</p>"
                ),
                self.xyz_area,
                widgets.HBox([self.xyz_btn, self.xyz_msg]),
            ]
        )
        tab_pubchem = widgets.VBox(
            [
                widgets.HTML(
                    _hint
                    + "Search by name or SMILES. Requires internet connection.</p>"
                ),
                widgets.HBox([self.pubchem_txt, self.pubchem_btn]),
                self.pubchem_msg,
            ]
        )
        input_tab = widgets.Tab(children=[tab_preset, tab_xyz, tab_pubchem])
        for _i, _t in enumerate(["Preset Library", "XYZ Input", "PubChem Search"]):
            input_tab.set_title(_i, _t)

        # Collapsible container
        self.mol_input_expanded = widgets.VBox(
            [
                widgets.HTML('<h3 style="margin:8px 0 6px">Molecule Input</h3>'),
                input_tab,
            ]
        )
        self.change_mol_btn = widgets.Button(
            description="Change",
            button_style="",
            icon="pencil",
            layout=widgets.Layout(width="100px", height="32px"),
            tooltip="Re-expand the molecule input panel",
        )
        self.mol_input_collapsed = widgets.HBox(
            [self.mol_summary_compact, self.change_mol_btn],
            layout=widgets.Layout(align_items="center", gap="12px", padding="6px 0"),
        )
        self.mol_input_container = widgets.VBox(
            [self.mol_input_expanded, self.mol_info_html, self.viz_output],
            layout=widgets.Layout(margin="0 0 4px 0"),
        )

    # ── Calculation setup panel (Cell 5) ──────────────────────────────────

    def _build_calc_setup(self) -> None:
        self.calc_setup_panel = widgets.VBox(
            [
                widgets.HTML('<h3 style="margin:14px 0 6px">Calculation Setup</h3>'),
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.HBox(
                                    [self.method_dd, self.method_help_btn],
                                    layout=widgets.Layout(
                                        align_items="center", gap="4px"
                                    ),
                                ),
                                widgets.HBox(
                                    [self.basis_dd, self.basis_help_btn],
                                    layout=widgets.Layout(
                                        align_items="center", gap="4px"
                                    ),
                                ),
                            ]
                        ),
                        widgets.HTML("&ensp;&ensp;"),
                        widgets.VBox([self.charge_si, self.mult_si]),
                    ]
                ),
                self.calc_type_dd,
                self.calc_extra_opts,
                self.preopt_cb,
                self.notes_output,
            ]
        )

    # ── Run panel (Cell 6) ────────────────────────────────────────────────

    def _build_run_section(self) -> None:
        self.run_panel = widgets.VBox(
            [
                widgets.HTML(
                    '<h3 style="margin:14px 0 6px">Run Calculation</h3>'
                    '<p style="color:#555;font-size:13px;margin:0 0 8px">PySCF runs in this '
                    "kernel. Output appears live below. Large molecules or high-accuracy basis "
                    "sets may take several minutes on a laptop.</p>"
                ),
                self.perf_estimate_html,
                widgets.HBox([self.run_btn, self.run_status]),
                widgets.HBox(
                    [
                        widgets.HTML(
                            '<span style="font-size:13px;font-weight:600;color:#444">'
                            "Calculation Output</span>"
                        ),
                        self.log_clear_btn,
                    ],
                    layout=widgets.Layout(
                        align_items="center",
                        justify_content="space-between",
                        margin="10px 0 4px",
                        max_width="460px",
                    ),
                ),
                self.run_output,
            ]
        )

    # ── Results panel (Cell 7) ────────────────────────────────────────────

    def _build_results_section(self) -> None:
        # Trajectory accordion (Geo Opt only — hidden until a Geo Opt completes)
        self.traj_output = widgets.Output()
        self.traj_accordion = widgets.Accordion(
            children=[self.traj_output],
            layout=widgets.Layout(display="none", margin="8px 0"),
        )
        self.traj_accordion.set_title(0, "Trajectory Viewer")
        self.traj_accordion.selected_index = None  # collapsed by default

        # Vibrational animation accordion (Frequency only — hidden until Freq completes)
        self.vib_mode_dd = widgets.Dropdown(
            description="Mode:",
            options=[],
            style={"description_width": "50px"},
            layout=widgets.Layout(width="360px"),
        )
        self.vib_output = widgets.Output()
        self.vib_accordion = widgets.Accordion(
            children=[
                widgets.VBox(
                    [self.vib_mode_dd, self.vib_output],
                    layout=widgets.Layout(padding="8px"),
                )
            ],
            layout=widgets.Layout(display="none", margin="8px 0"),
        )
        self.vib_accordion.set_title(0, "Vibrational Mode Viewer")
        self.vib_accordion.selected_index = None  # collapsed by default

        self.results_panel = widgets.VBox(
            [
                widgets.HTML('<h3 style="margin:14px 0 6px">Results</h3>'),
                self.result_output,
                self.result_viz_output,
                self.traj_accordion,
                self.vib_accordion,
            ]
        )

    # ── History panel (Cell 8) ────────────────────────────────────────────

    def _build_history_section(self) -> None:
        self.past_dd = widgets.Dropdown(
            description="Load:",
            options=[("(no saved results)", "")],
            style={"description_width": "50px"},
            layout=widgets.Layout(width="500px"),
        )
        self.past_refresh_btn = widgets.Button(
            description="Refresh",
            button_style="",
            icon="refresh",
            layout=widgets.Layout(width="100px"),
            tooltip="Rescan the results directory",
        )
        self.copy_path_btn = widgets.Button(
            description="Copy path",
            button_style="",
            icon="clipboard",
            layout=widgets.Layout(width="120px"),
            tooltip="Copy the results directory path to clipboard",
        )
        self.results_path_lbl = widgets.HTML()
        self.past_output = widgets.Output()
        self.view_log_btn = widgets.Button(
            description="View log",
            button_style="",
            icon="file-text-o",
            layout=widgets.Layout(width="110px"),
            tooltip="Open the full PySCF output log in the Output tab",
        )

        # Performance stats widgets
        self._perf_stats_html = widgets.HTML()
        self._perf_events_html = widgets.HTML()
        self._reset_btn = widgets.Button(
            description="Reset performance database",
            button_style="danger",
            icon="trash",
            layout=widgets.Layout(width="230px"),
        )
        self._reset_confirm_html = widgets.HTML(
            '<span style="color:#dc2626;font-size:13px">'
            "<b>Warning:</b> This will permanently delete all performance records. "
            "Time estimates will reset to &ldquo;no data&rdquo;.</span>"
        )
        self._reset_confirm_yes = widgets.Button(
            description="Yes, delete all records",
            button_style="danger",
            icon="check",
            layout=widgets.Layout(width="190px"),
        )
        self._reset_confirm_no = widgets.Button(
            description="Cancel",
            button_style="",
            icon="times",
            layout=widgets.Layout(width="90px"),
        )
        self._reset_confirm_box = widgets.VBox(
            [
                self._reset_confirm_html,
                widgets.HBox(
                    [self._reset_confirm_yes, self._reset_confirm_no],
                    layout=widgets.Layout(gap="8px", margin="4px 0 0"),
                ),
            ],
            layout=widgets.Layout(
                display="none",
                border="1px solid #fca5a5",
                padding="8px 10px",
                margin="6px 0 0",
            ),
        )

        _perf_stats_panel = widgets.VBox(
            [
                self._perf_stats_html,
                widgets.HTML(
                    '<p style="margin:10px 0 4px;color:#475569;font-size:13px;font-weight:600">'
                    "Recent events (last 20)</p>"
                ),
                self._perf_events_html,
                widgets.HBox(
                    [self._reset_btn],
                    layout=widgets.Layout(margin="14px 0 4px"),
                ),
                self._reset_confirm_box,
            ]
        )
        self._perf_accordion = widgets.Accordion(
            children=[_perf_stats_panel], selected_index=None
        )
        self._perf_accordion.set_title(0, "Performance stats")

        self.history_panel = widgets.VBox(
            [
                widgets.HTML(
                    '<p style="color:#555;font-size:13px;margin:0 0 8px">'
                    "Calculations are saved automatically. Select one below to view its results.</p>"
                ),
                widgets.HBox(
                    [
                        self.past_dd,
                        self.past_refresh_btn,
                        self.copy_path_btn,
                        self.view_log_btn,
                    ],
                    layout=widgets.Layout(align_items="center", gap="8px"),
                ),
                self.results_path_lbl,
                self.past_output,
                self._perf_accordion,
            ]
        )

        # Populate on startup
        self._refresh_results_browser()
        self._refresh_perf_stats()

    # ── Compare panel (Cell 9) ────────────────────────────────────────────

    def _build_compare_section(self) -> None:
        self.compare_select = widgets.SelectMultiple(
            options=[("(no saved results)", "")],
            rows=8,
            description="",
            layout=widgets.Layout(width="100%"),
        )
        self.compare_refresh_btn = widgets.Button(
            description="Refresh",
            button_style="",
            icon="refresh",
            layout=widgets.Layout(width="100px"),
        )
        self.compare_btn = widgets.Button(
            description="Compare selected",
            button_style="primary",
            icon="bar-chart",
            disabled=True,
            layout=widgets.Layout(width="180px"),
        )
        self.compare_clear_btn = widgets.Button(
            description="Clear",
            button_style="warning",
            icon="times",
            layout=widgets.Layout(width="90px"),
        )
        self.compare_output = widgets.Output()

        self.compare_panel = widgets.VBox(
            [
                widgets.HTML(
                    '<h3 style="margin:14px 0 6px">Compare Calculations</h3>'
                    '<p style="color:#555;font-size:13px;margin:0 0 8px">'
                    "Select two or more saved calculations to compare side-by-side. "
                    "Hold Ctrl (or ⌘) to select multiple entries.</p>"
                ),
                widgets.HBox([self.compare_refresh_btn]),
                self.compare_select,
                widgets.HBox(
                    [self.compare_btn, self.compare_clear_btn],
                    layout=widgets.Layout(gap="8px", margin="6px 0"),
                ),
                self.compare_output,
            ],
            layout=widgets.Layout(padding="8px 0"),
        )

        # Export accordion (Advanced)
        _export_content = widgets.VBox(
            [
                widgets.HTML(
                    '<p style="color:#555;font-size:13px;margin:0 0 8px">'
                    "Download a self-contained PySCF script you can study or run outside the notebook.</p>"
                ),
                widgets.HBox([self.export_btn, self.export_status]),
            ]
        )
        self.advanced_accordion = widgets.Accordion(children=[_export_content])
        self.advanced_accordion.set_title(0, "Export Script")
        self.advanced_accordion.selected_index = None

        # Populate on startup
        self._populate_compare_list()

    # ── Output log tab (Cell 10) ──────────────────────────────────────────

    def _build_output_tab(self) -> None:
        self._log_output_html = widgets.HTML(
            '<span style="color:#94a3b8;font-size:13px">'
            "No log yet — run a calculation first, or use "
            "<b>View log</b> in the History tab.</span>"
        )
        self._log_source_lbl = widgets.HTML()
        self._log_clear_btn = widgets.Button(
            description="Clear",
            button_style="",
            icon="times",
            layout=widgets.Layout(width="80px"),
        )
        self.log_tab_panel = widgets.VBox(
            [
                widgets.HTML(
                    '<p style="color:#555;font-size:13px;margin:4px 0 8px">'
                    "Full PySCF output for the most recent calculation. "
                    "Use <b>View log</b> in the History tab to load a saved result's log.</p>"
                ),
                widgets.HBox(
                    [self._log_clear_btn],
                    layout=widgets.Layout(margin="0 0 8px"),
                ),
                self._log_source_lbl,
                self._log_output_html,
            ],
            layout=widgets.Layout(padding="8px 0"),
        )

    # ── Help section (Cell 10) ────────────────────────────────────────────

    def _build_help_section(self) -> None:
        _help_keys = list(HELP_TOPICS.keys())
        _help_labels = [HELP_TOPICS[k]["title"] for k in _help_keys]
        self.help_topic_dd = widgets.Dropdown(
            options=list(zip(_help_labels, _help_keys)),
            description="Topic:",
            style={"description_width": "60px"},
            layout=widgets.Layout(width="460px"),
        )
        self.help_content_html = widgets.HTML()
        self._render_help_topic()  # render first topic immediately

        self.help_tab_panel = widgets.VBox(
            [
                widgets.HTML(
                    '<p style="color:#555;font-size:13px;margin:4px 0 12px">'
                    "Browse help topics below. Click <b>?</b> next to the Method or Basis Set "
                    "dropdown in the Calculate tab to jump directly to a relevant topic.</p>"
                ),
                self.help_topic_dd,
                self.help_content_html,
            ],
            layout=widgets.Layout(padding="8px 0"),
        )

    # ── Tab assembly (Cell 10) ────────────────────────────────────────────

    def _assemble_tabs(self) -> None:
        _calculate_content = widgets.VBox(
            [
                self.step_progress.widget,
                self.mol_input_container,
                self.calc_setup_panel,
                self.run_panel,
                self.results_panel,
                self.advanced_accordion,
            ],
            layout=widgets.Layout(padding="8px 0"),
        )

        self.root_tab = widgets.Tab(
            children=[
                _calculate_content,
                self.history_panel,
                self.compare_panel,
                self.log_tab_panel,
                self.help_tab_panel,
            ]
        )
        self.root_tab.set_title(0, "Calculate")
        self.root_tab.set_title(1, "History")
        self.root_tab.set_title(2, "Compare")
        self.root_tab.set_title(3, "Output")
        self.root_tab.set_title(4, "Help")

    # ══ CALLBACK WIRING ══════════════════════════════════════════════════════

    def _wire_callbacks(self) -> None:
        # Theme
        self.theme_btn.observe(self._on_theme_changed, names="value")
        # Molecule input
        self.preset_dd.observe(self._on_load_preset, names="value")
        self.xyz_btn.on_click(self._on_load_xyz)
        self.pubchem_btn.on_click(self._on_search_pubchem)
        self.change_mol_btn.on_click(self._on_expand_mol_input)
        # Calc type
        self.calc_type_dd.observe(self._on_calc_type_changed, names="value")
        # Notes + estimate
        self.method_dd.observe(self._update_notes, names="value")
        self.basis_dd.observe(self._update_notes, names="value")
        self.method_dd.observe(self._update_estimate, names="value")
        self.basis_dd.observe(self._update_estimate, names="value")
        # Help buttons
        self.method_help_btn.on_click(self._on_method_help)
        self.basis_help_btn.on_click(self._on_basis_help)
        # Run
        self.run_btn.on_click(self._on_run_clicked)
        self.log_clear_btn.on_click(self._on_clear_log)
        # Accumulate / export
        self.accumulate_btn.on_click(self._on_accumulate)
        self.clear_btn.on_click(self._on_clear)
        self.export_btn.on_click(self._on_export)
        # History
        self.past_dd.observe(self._on_past_dd_changed, names="value")
        self.past_refresh_btn.on_click(self._on_past_refresh)
        self.copy_path_btn.on_click(self._on_copy_results_path)
        self.view_log_btn.on_click(self._on_view_log)
        # Perf stats reset
        self._reset_btn.on_click(self._on_reset_click)
        self._reset_confirm_yes.on_click(self._on_confirm_yes)
        self._reset_confirm_no.on_click(self._on_confirm_no)
        # Compare
        self.compare_refresh_btn.on_click(self._on_compare_refresh)
        self.compare_btn.on_click(self._on_compare)
        self.compare_clear_btn.on_click(self._on_compare_clear)
        # Output log
        self._log_clear_btn.on_click(self._on_log_clear)
        # Help
        self.help_topic_dd.observe(self._on_help_topic_changed, names="value")
        # Vibrational mode selector
        self.vib_mode_dd.observe(self._on_vib_mode_changed, names="value")

    # ══ CALLBACK METHODS ═════════════════════════════════════════════════════

    # ── Theme ─────────────────────────────────────────────────────────────

    def _on_theme_changed(self, change) -> None:
        self._theme_style.clear_output()
        css = self._theme_css(change["new"])
        if css:
            with self._theme_style:
                display(HTML(css))

    # ── Molecule input ────────────────────────────────────────────────────

    def _on_load_preset(self, change) -> None:
        name = change["new"]
        if name.startswith("("):
            return
        d = MOLECULE_LIBRARY[name]
        self._set_molecule(
            Molecule(
                atoms=d["atoms"],
                coordinates=d["coordinates"],
                charge=d["charge"],
                multiplicity=d["multiplicity"],
            ),
            d["description"],
        )

    def _on_load_xyz(self, btn) -> None:
        try:
            atoms, coords = parse_xyz_input(self.xyz_area.value.strip())
            mol = Molecule(atoms=atoms, coordinates=coords)
            self._set_molecule(mol, "Loaded from XYZ input")
            self.xyz_msg.value = ""
        except Exception as exc:
            self.xyz_msg.value = f"Parse error: {exc}"

    def _apply_pubchem_search_result(
        self,
        query: str,
        mol: Optional[Molecule] = None,
        error: Optional[Exception] = None,
    ) -> None:
        if error is None and mol is not None:
            self._set_molecule(mol, f"PubChem: {query}")
            self.pubchem_msg.value = f"Loaded {mol.get_formula()} from PubChem."
        else:
            self.pubchem_msg.value = f"Not found: {error}"
        self.pubchem_btn.disabled = False

    def _on_search_pubchem(self, btn) -> None:
        query = self.pubchem_txt.value.strip()
        if not query:
            self.pubchem_msg.value = "Enter a molecule name or SMILES."
            return
        if _student_friendly_fetch is None:
            self.pubchem_msg.value = "PubChem module not available."
            return
        self.pubchem_msg.value = f'Searching for "{query}"...'
        self.pubchem_btn.disabled = True

        loop = asyncio.get_running_loop()

        def _do():
            try:
                xyz_str, _msg = _student_friendly_fetch(query)
                if xyz_str is None:
                    raise ValueError(_msg)
                atoms, coords = parse_xyz_input(xyz_str)
                mol = Molecule(atoms=atoms, coordinates=coords)
                loop.call_soon_threadsafe(
                    self._apply_pubchem_search_result,
                    query,
                    mol,
                    None,
                )
            except Exception as exc:
                loop.call_soon_threadsafe(
                    self._apply_pubchem_search_result,
                    query,
                    None,
                    exc,
                )

        threading.Thread(target=_do, daemon=True).start()

    def _on_expand_mol_input(self, btn) -> None:
        self.mol_input_container.children = [
            self.mol_input_expanded,
            self.mol_info_html,
            self.viz_output,
        ]

    # ── Calc type ─────────────────────────────────────────────────────────

    def _on_calc_type_changed(self, change) -> None:
        ct = change["new"]
        if ct == "Geometry Opt":
            self.calc_extra_opts.children = [
                widgets.HBox(
                    [self.fmax_fi, self.max_steps_si],
                    layout=widgets.Layout(gap="8px"),
                ),
            ]
        elif ct == "UV-Vis (TD-DFT)":
            self.calc_extra_opts.children = [
                self.nstates_si,
                widgets.HTML(
                    '<span style="color:#b45309;font-size:12px">⚠ Requires a DFT '
                    "functional (e.g. B3LYP, PBE0). RHF/UHF will run TDHF (CIS) "
                    "instead.</span>"
                ),
            ]
        else:
            self.calc_extra_opts.children = []

    # ── Help buttons ──────────────────────────────────────────────────────

    def _on_method_help(self, btn) -> None:
        self._show_help_topic("method")

    def _on_basis_help(self, btn) -> None:
        self._show_help_topic("basis_set")

    # ── Run ───────────────────────────────────────────────────────────────

    def _on_run_clicked(self, btn) -> None:
        self.run_output.clear_output()
        self.result_output.clear_output()
        self.result_viz_output.clear_output()
        self.traj_accordion.layout.display = "none"
        self.vib_accordion.layout.display = "none"
        threading.Thread(target=self._do_run, daemon=True).start()

    def _on_clear_log(self, btn) -> None:
        self.run_output.clear_output()

    # ── Accumulate / export ───────────────────────────────────────────────

    def _on_accumulate(self, btn) -> None:
        r = self._last_result
        if r is None:
            return
        self._results.append(r)
        self._refresh_comparison()

    def _on_clear(self, btn) -> None:
        self._results.clear()
        self.comparison_output.clear_output()

    def _on_export(self, btn) -> None:
        if self._molecule is None:
            self.export_status.value = "Load a molecule first."
            return
        try:
            from quantui import PySCFCalculation

            calc = PySCFCalculation(
                self._molecule,
                method=self.method_dd.value,
                basis=self.basis_dd.value,
            )
            fname = (
                f"{self._molecule.get_formula()}"
                f"_{self.method_dd.value}_{self.basis_dd.value}.py"
            )
            calc.generate_calculation_script(Path(fname))
            self.export_status.value = f"Saved: {fname}"
        except Exception as exc:
            self.export_status.value = f"Error: {exc}"

    # ── Compare ───────────────────────────────────────────────────────────

    def _on_compare_refresh(self, btn) -> None:
        self._populate_compare_list()

    def _on_compare(self, btn) -> None:
        selected = self.compare_select.value
        if not selected or selected == ("",):
            return
        self.compare_output.clear_output(wait=True)
        from quantui import (
            comparison_table_html,
            plot_comparison,
            summary_from_saved_result,
        )
        from quantui.results_storage import load_result

        summaries = []
        for path_str in selected:
            if not path_str:
                continue
            try:
                data = load_result(Path(path_str))
                summaries.append(summary_from_saved_result(data))
            except Exception as exc:
                with self.compare_output:
                    display(
                        HTML(
                            f'<p style="color:#ef4444">Error loading result: {exc}</p>'
                        )
                    )
        if not summaries:
            return
        with self.compare_output:
            display(HTML(comparison_table_html(summaries)))
            if len(summaries) > 1:
                try:
                    import matplotlib.pyplot as plt

                    fig = plot_comparison(summaries)
                    display(fig)
                    plt.close(fig)
                except Exception:
                    pass

    def _on_compare_clear(self, btn) -> None:
        self.compare_select.value = ()
        self.compare_output.clear_output()

    # ── History ───────────────────────────────────────────────────────────

    def _on_past_dd_changed(self, change) -> None:
        path_str = change["new"]
        if not path_str:
            self.past_output.clear_output()
            return
        self.past_output.clear_output(wait=True)
        try:
            from quantui import load_result

            data = load_result(Path(path_str))
            self.past_output.append_display_data(HTML(self._format_past_result(data)))
        except Exception as exc:
            self.past_output.append_stdout(f"Could not load result: {exc}\n")

    def _on_past_refresh(self, btn) -> None:
        self._refresh_results_browser()

    def _on_copy_results_path(self, btn) -> None:
        p = self._get_results_dir()
        p.mkdir(parents=True, exist_ok=True)
        path_str = str(p).replace("\\", "\\\\").replace("'", "\\'")
        display(Javascript(f"navigator.clipboard.writeText('{path_str}')"))
        self.results_path_lbl.value = (
            f'<span style="color:#22c55e;font-size:13px">Copied: {p}</span>'
        )

        def _reset():
            time.sleep(3)
            self.results_path_lbl.value = (
                f'<span style="font-size:13px;color:#64748b">{p}</span>'
            )

        threading.Thread(target=_reset, daemon=True).start()

    def _on_view_log(self, btn) -> None:
        path_str = self.past_dd.value
        if not path_str:
            return
        log_path = Path(path_str) / "pyscf.log"
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            label = Path(path_str).name
        else:
            text = "(No pyscf.log found for this result.)"
            label = ""
        self._update_log_panel(text, label)
        self._goto_output_tab()

    # ── Perf stats reset ──────────────────────────────────────────────────

    def _on_reset_click(self, btn) -> None:
        self._reset_confirm_box.layout.display = ""

    def _on_confirm_yes(self, btn) -> None:
        from quantui.calc_log import reset_perf_log

        reset_perf_log()
        self._reset_confirm_box.layout.display = "none"
        self._refresh_perf_stats()

    def _on_confirm_no(self, btn) -> None:
        self._reset_confirm_box.layout.display = "none"

    # ── Output log ────────────────────────────────────────────────────────

    def _on_log_clear(self, btn) -> None:
        self._log_output_html.value = (
            '<span style="color:#94a3b8;font-size:13px">Log cleared.</span>'
        )
        self._log_source_lbl.value = ""

    # ── Help ──────────────────────────────────────────────────────────────

    def _on_help_topic_changed(self, change=None) -> None:
        self._render_help_topic()

    # ══ LOGIC METHODS ════════════════════════════════════════════════════════

    def _set_molecule(self, mol: Molecule, label: str = "") -> None:
        """Update shared state and refresh dependent widgets."""
        self._molecule = mol
        self.run_btn.disabled = False
        self.export_btn.disabled = False

        try:
            n_e = mol.get_electron_count()
            e_str = f"{n_e} electrons"
        except Exception:
            e_str = ""

        _lbl = f'<br><small style="color:#777">{label}</small>' if label else ""
        _summary = (
            f'<b style="font-size:15px">{mol.get_formula()}</b>'
            f'&ensp;<span style="color:#555;font-size:13px">'
            f"{len(mol.atoms)} atoms"
            + (f" &bull; {e_str}" if e_str else "")
            + f" &bull; charge {mol.charge} &bull; mult {mol.multiplicity}"
            + f"</span>{_lbl}"
        )
        self.mol_info_html.value = _summary
        self.mol_summary_compact.value = (
            f'<div style="background:#f0f9ff;border:1px solid #bae6fd;'
            f'border-radius:6px;padding:7px 14px;font-size:14px;display:inline-block">'
            f"{_summary}</div>"
        )

        self.charge_si.value = mol.charge
        self.mult_si.value = mol.multiplicity
        if mol.multiplicity > 1 and self.method_dd.value == "RHF":
            self.method_dd.value = "UHF"

        self.viz_output.clear_output()
        if _display_molecule is not None:
            with self.viz_output:
                _display_molecule(mol)

        self._update_notes()

        # Advance step indicator
        if self.step_progress._states[2] != "active":
            if self.step_progress._states[2] in ("done", "fail"):
                self.step_progress.reset()
            self.step_progress.complete(0)
            self.step_progress.start(1)

        self._update_estimate()

        # Collapse molecule input to compact view
        self.mol_input_container.children = [self.mol_input_collapsed, self.viz_output]

    def _queue_main_thread_callback(self, callback, *args, **kwargs) -> None:
        """Run a callback on the notebook/kernel thread when possible."""
        if threading.current_thread() is threading.main_thread():
            callback(*args, **kwargs)
            return

        ip = get_ipython()
        io_loop = getattr(getattr(ip, "kernel", None), "io_loop", None)
        if io_loop is not None:
            io_loop.add_callback(callback, *args, **kwargs)
            return

        # Best-effort fallback for non-notebook contexts where no kernel loop
        # is available. This preserves existing behaviour, but the normal
        # notebook path above keeps rendering off the worker thread.
        callback(*args, **kwargs)

    def _set_molecule_state_only(self, mol) -> None:
        """Apply only thread-safe molecule state updates."""
        self._molecule = mol

    def _set_molecule_threadsafe(self, mol, status_message: str) -> None:
        """Update molecule state safely and render on the main thread only."""
        if threading.current_thread() is threading.main_thread():
            self._set_molecule(mol, status_message)
            return

        self._set_molecule_state_only(mol)
        self._queue_main_thread_callback(self._set_molecule, mol, status_message)

    def _show_result_3d(self, molecule) -> None:
        """Render molecule 3D structure in the result visualization panel.

        Safe to call from a background thread — uses ``with output:`` context.
        """
        if _display_molecule is None or molecule is None:
            return
        self.result_viz_output.clear_output()
        with self.result_viz_output:
            _display_molecule(molecule)

    def _show_opt_trajectory(self, opt_result) -> None:
        """Render geo-opt trajectory animation and energy chart in the trajectory panel.

        Uses plotlyMol's ``create_trajectory_animation``. Safe to call from a
        background thread — uses ``with output:`` context.  No-op if plotlyMol
        is not installed.
        """
        traj = opt_result.trajectory
        energies = opt_result.energies_hartree
        if len(traj) < 2:
            return

        try:
            import plotly.graph_objects as go
            from IPython.display import display as _ipy_display
            from plotlymol3d import create_trajectory_animation
        except ImportError:
            return

        # Build full XYZ blocks (count + title + coords).
        xyzblocks = [
            f"{len(m.atoms)}\n{m.get_formula()}\n{m.to_xyz_string()}" for m in traj
        ]

        anim_fig = create_trajectory_animation(
            xyzblocks=xyzblocks,
            energies_hartree=energies if energies else None,
            charge=traj[0].charge,
            mode="ball+stick",
            resolution=12,
            title=f"Geo Opt: {opt_result.formula}",
        )
        anim_fig.update_layout(height=420)

        # Energy convergence chart (relative energies in kcal/mol).
        _HARTREE_TO_KCAL = 627.5094740631
        e0 = energies[0] if energies else 0.0
        rel_e = [(e - e0) * _HARTREE_TO_KCAL for e in energies] if energies else []
        energy_fig = go.Figure(
            go.Scatter(
                x=list(range(len(rel_e))),
                y=rel_e,
                mode="lines+markers",
                name="ΔE",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=6),
            )
        )
        energy_fig.update_layout(
            title="Energy Convergence",
            xaxis_title="Optimization Step",
            yaxis_title="ΔE (kcal/mol)",
            height=280,
            margin=dict(l=60, r=20, t=40, b=40),
        )

        self.traj_output.clear_output()
        with self.traj_output:
            _ipy_display(anim_fig)
            _ipy_display(energy_fig)

        # Reveal the accordion (collapsed).
        self.traj_accordion.selected_index = None
        self.traj_accordion.layout.display = ""

    def _build_vib_data_from_freq_result(self, freq_result, molecule):
        """Construct a ``plotlymol3d.VibrationalData`` from a FreqResult.

        Args:
            freq_result: ``FreqResult`` with ``displacements`` populated.
            molecule: The ``Molecule`` used for the frequency calculation.

        Returns:
            ``VibrationalData`` or ``None`` if prerequisites are missing.
        """
        try:
            import numpy as np
            from plotlymol3d import VibrationalData, VibrationalMode
        except ImportError:
            return None

        displacements = getattr(freq_result, "displacements", None)
        if displacements is None:
            return None

        freqs = freq_result.frequencies_cm1
        intensities = freq_result.ir_intensities
        n_modes = len(freqs)

        coords = np.array(molecule.coordinates, dtype=float)

        # Map element symbols to atomic numbers using a common-elements table.
        # ASE is not required — this covers all elements students will encounter.
        _Z = {
            "H": 1,
            "He": 2,
            "Li": 3,
            "Be": 4,
            "B": 5,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
            "Ne": 10,
            "Na": 11,
            "Mg": 12,
            "Al": 13,
            "Si": 14,
            "P": 15,
            "S": 16,
            "Cl": 17,
            "Ar": 18,
            "K": 19,
            "Ca": 20,
            "Br": 35,
            "I": 53,
        }
        atomic_numbers: List[int] = [_Z.get(sym, 0) for sym in molecule.atoms]

        modes = []
        for i in range(n_modes):
            freq = freqs[i]
            ir_inten = intensities[i] if i < len(intensities) else None
            displ = np.array(displacements[i], dtype=float)
            modes.append(
                VibrationalMode(
                    mode_number=i + 1,
                    frequency=float(freq),
                    ir_intensity=ir_inten,
                    displacement_vectors=displ,
                    is_imaginary=freq < 0,
                )
            )

        return VibrationalData(
            coordinates=coords,
            atomic_numbers=atomic_numbers,
            modes=modes,
            source_file="quantui_freq_calc",
            program="pyscf",
        )

    def _show_vib_animation(self, freq_result, molecule) -> None:
        """Populate the vibrational animation accordion after a Frequency result.

        Builds a ``VibrationalData`` from the result, populates the mode selector
        dropdown, and renders the animation for the first non-trivial mode.
        No-op if plotlyMol is unavailable or displacements are missing.
        """
        vib_data = self._build_vib_data_from_freq_result(freq_result, molecule)
        if vib_data is None:
            return

        freqs = freq_result.frequencies_cm1
        if not freqs:
            return

        # Build dropdown options: one entry per mode with frequency label.
        # Skip near-zero translation/rotation modes (|ν| < 10 cm⁻¹).
        options = []
        for m in vib_data.modes:
            freq_val = m.frequency
            if abs(freq_val) < 10:
                continue
            label = (
                f"Mode {m.mode_number}: {freq_val:.1f} cm⁻¹"
                if freq_val >= 0
                else f"Mode {m.mode_number}: {freq_val:.1f} cm⁻¹ (imaginary, TS?)"
            )
            options.append((label, m.mode_number))

        if not options:
            return

        self.vib_mode_dd.options = options
        self.vib_mode_dd.value = options[0][1]

        # Store vib_data for callback use.
        self._last_vib_data = vib_data
        self._last_vib_molecule = molecule

        # Render the first selected mode.
        self._render_vib_mode(vib_data, molecule, options[0][1])

        # Reveal the accordion (collapsed by default).
        self.vib_accordion.selected_index = None
        self.vib_accordion.layout.display = ""

    def _render_vib_mode(self, vib_data, molecule, mode_number: int) -> None:
        """Render vibrational animation for the given mode into ``vib_output``.

        Safe to call from background thread via ``with output:`` context.
        """
        try:
            from IPython.display import display as _ipy_display
            from plotlymol3d import create_vibration_animation, xyzblock_to_rdkitmol
        except ImportError:
            return

        # Build an RDKit mol for bond connectivity (required by animation function).
        xyzblock = (
            f"{len(molecule.atoms)}\n{molecule.get_formula()}\n"
            f"{molecule.to_xyz_string()}"
        )
        try:
            rdmol = xyzblock_to_rdkitmol(xyzblock, charge=molecule.charge)
        except Exception:
            return

        try:
            anim_fig = create_vibration_animation(
                vib_data=vib_data,
                mode_number=mode_number,
                mol=rdmol,
                amplitude=0.4,
                n_frames=20,
                mode="ball+stick",
                resolution=12,
            )
            anim_fig.update_layout(height=420)
        except Exception:
            return

        self.vib_output.clear_output()
        with self.vib_output:
            _ipy_display(anim_fig)

    def _on_vib_mode_changed(self, change) -> None:
        """Re-render vib animation when the mode dropdown changes."""
        mode_number = change["new"]
        vib_data = getattr(self, "_last_vib_data", None)
        molecule = getattr(self, "_last_vib_molecule", None)
        if vib_data is None or molecule is None:
            return
        threading.Thread(
            target=self._render_vib_mode,
            args=(vib_data, molecule, mode_number),
            daemon=True,
        ).start()

    def _do_run(self) -> None:
        """Main calculation dispatch — runs in a background thread."""
        mol = self._molecule
        if mol is None:
            self.run_status.value = "Load a molecule first."
            return
        self.run_btn.disabled = True
        self.run_status.value = "Starting..."

        self.step_progress.complete(1)
        self.step_progress.start(2)

        _calc_log.log_event(
            "calc_start",
            f"{self.method_dd.value}/{self.basis_dd.value} on {mol.get_formula()}",
            n_atoms=len(mol.atoms),
        )
        _run_wall_t = time.perf_counter()
        log = _LogCapture(self.run_output, self.run_status)

        try:
            calc_mol = mol
            if self.preopt_cb.value and _PREOPT_AVAILABLE:
                self.run_status.value = "Pre-optimizing..."
                calc_mol, _rmsd = preoptimize(mol)
                self._set_molecule_threadsafe(
                    calc_mol,
                    f"Geometry pre-optimized (LJ, RMSD={_rmsd:.3f} Å)",
                )

            ct = self.calc_type_dd.value
            result: Any = None
            result_html: str = ""
            save_spectra: dict = {}
            save_type: str = "single_point"
            if ct == "Geometry Opt":
                self.run_status.value = "Optimizing geometry..."
                from quantui import optimize_geometry

                result = optimize_geometry(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    fmax=self.fmax_fi.value,
                    steps=self.max_steps_si.value,
                    progress_stream=log,  # type: ignore[arg-type]
                )
                result_html = self._format_opt_result(result)
                save_spectra, save_type = {}, "geometry_opt"
            elif ct == "Frequency":
                self.run_status.value = "Computing frequencies (SCF + Hessian)..."
                from quantui.freq_calc import run_freq_calc

                result = run_freq_calc(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    progress_stream=log,  # type: ignore[arg-type]
                )
                result_html = self._format_freq_result(result)
                save_spectra = {
                    "ir": {
                        "frequencies_cm1": result.frequencies_cm1,
                        "ir_intensities": result.ir_intensities,
                        "zpve_hartree": result.zpve_hartree,
                    }
                }
                save_type = "frequency"
            elif ct == "UV-Vis (TD-DFT)":
                self.run_status.value = "Running TD-DFT excited states..."
                from quantui.tddft_calc import run_tddft_calc

                result = run_tddft_calc(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    nstates=self.nstates_si.value,
                    progress_stream=log,  # type: ignore[arg-type]
                )
                result_html = self._format_tddft_result(result)
                save_spectra = {
                    "uv_vis": {
                        "excitation_energies_ev": result.excitation_energies_ev,
                        "oscillator_strengths": result.oscillator_strengths,
                        "wavelengths_nm": result.wavelengths_nm(),
                    }
                }
                save_type = "tddft"
            else:  # Single Point
                self.run_status.value = "Calculating..."
                from quantui import run_in_session

                result = run_in_session(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    progress_stream=log,  # type: ignore[arg-type]
                )
                result_html = self._format_result(result)
                save_spectra, save_type = {}, "single_point"

            _elapsed = time.perf_counter() - _run_wall_t
            self._last_result = result
            self.accumulate_btn.disabled = False

            self.result_output.append_display_data(HTML(result_html))
            self.run_status.value = f"Done in {_elapsed:.1f} s."

            # Show 3D structure in the result panel
            _viz_mol = result.molecule if ct == "Geometry Opt" else calc_mol
            self._show_result_3d(_viz_mol)

            # Show calc-type-specific extra panels
            if ct == "Geometry Opt":
                self._show_opt_trajectory(result)
            elif ct == "Frequency":
                self._show_vib_animation(result, calc_mol)

            self.step_progress.complete(2)
            self.step_progress.complete(3)

            # Persist to disk
            try:
                from quantui import save_result

                save_result(
                    result,
                    pyscf_log=log.getvalue(),
                    calc_type=save_type,
                    spectra=save_spectra,
                )
                self._refresh_results_browser()
                self._populate_compare_list()
                self._update_log_panel(
                    log.getvalue(),
                    f"{result.formula}  {self.method_dd.value}/{self.basis_dd.value}",
                )
            except Exception:
                pass

            # Log performance
            try:
                _calc_log.log_calculation(
                    formula=result.formula,
                    n_atoms=len(calc_mol.atoms),
                    n_electrons=calc_mol.get_electron_count(),
                    method=result.method,
                    basis=result.basis,
                    n_iterations=getattr(result, "n_iterations", -1),
                    elapsed_s=_elapsed,
                    converged=result.converged,
                )
                _calc_log.log_event(
                    "calc_done",
                    f"{result.method}/{result.basis} on {result.formula}",
                    elapsed_s=round(_elapsed, 2),
                    converged=result.converged,
                )
                self._update_estimate()
            except Exception:
                pass

        except ImportError as _import_err:
            _err_detail = str(_import_err)
            _msg = (
                f"Import error: {_err_detail}\n\n"
                "A required calculation dependency could not be loaded.\n"
                "On Windows: use the Apptainer container.\n"
                "  apptainer run quantui-local.sif\n"
            )
            log.write(_msg)
            _err_html = (
                '<div style="background:#fef2f2;border:1px solid #fca5a5;'
                'border-radius:8px;padding:16px;margin:8px 0">'
                '<b style="color:#b91c1c">&#9888; Dependency Not Available</b><br>'
                f'<span style="color:#7f1d1d">{_err_detail}</span><br><br>'
                '<small style="color:#991b1b">On Windows, use the Apptainer container: '
                "<code>apptainer run quantui-local.sif</code>. "
                "Full details are in the <b>Output</b> tab.</small>"
                "</div>"
            )
            self.result_output.append_display_data(HTML(_err_html))
            self.run_status.value = "Dependency unavailable."
            self.step_progress.fail(2, _err_detail[:60])
            _calc_log.log_event("calc_error", _err_detail[:200])

        except Exception as exc:
            import traceback as _tb

            _elapsed = time.perf_counter() - _run_wall_t
            _tb_str = _tb.format_exc()
            # Full details → Output tab (for debugging/instructors)
            log.write(f"\n--- Calculation Error ---\n{exc}\n\n{_tb_str}")
            # Write to persistent error log
            try:
                import datetime as _dt
                import os as _os

                _log_dir = Path(
                    _os.environ.get(
                        "QUANTUI_LOG_DIR",
                        Path.home() / ".quantui" / "logs",
                    )
                )
                _log_dir.mkdir(parents=True, exist_ok=True)
                _ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _formula = mol.get_formula() if mol is not None else "unknown"
                _method = self.method_dd.value
                _basis = self.basis_dd.value
                with open(_log_dir / "error_log.txt", "a") as _ef:
                    _ef.write(
                        f"\n{'='*60}\n"
                        f"{_ts}  {_formula}  {_method}/{_basis}\n"
                        f"{_tb_str}"
                    )
            except Exception:
                pass
            # Clean summary → result panel (student-facing)
            _err_html = (
                '<div style="background:#fef2f2;border:1px solid #fca5a5;'
                'border-radius:8px;padding:16px;margin:8px 0">'
                '<b style="color:#b91c1c">&#9888; Calculation Failed</b><br>'
                f'<code style="color:#7f1d1d">{exc}</code><br><br>'
                '<small style="color:#991b1b">'
                "Tips: try a smaller basis set (STO-3G), use a geometry-optimized "
                "structure first, or check for unusually long/short bonds in your "
                "XYZ input. Full error details are in the <b>Output</b> tab.</small>"
                "</div>"
            )
            self.result_output.append_display_data(HTML(_err_html))
            self.run_status.value = "Calculation failed."
            self.step_progress.fail(2, str(exc)[:60])
            _calc_log.log_event(
                "calc_error", str(exc)[:200], elapsed_s=round(_elapsed, 2)
            )

        finally:
            self.run_btn.disabled = False

    def _update_notes(self, change=None) -> None:
        self.notes_output.clear_output(wait=True)
        if self._molecule is None:
            return
        try:
            from quantui import PySCFCalculation

            calc = PySCFCalculation(
                self._molecule,
                method=self.method_dd.value,
                basis=self.basis_dd.value,
            )
            notes = calc.get_educational_notes()
            if notes:
                safe = (
                    notes.replace("**", "<b>", 1)
                    .replace("**", "</b>", 1)
                    .replace("\n\n", "<br><br>")
                )
                with self.notes_output:
                    display(
                        HTML(
                            '<div style="background:#fffbf0;padding:8px 12px;'
                            'border-radius:4px;font-size:13px;margin-top:6px">'
                            + safe
                            + "</div>"
                        )
                    )
        except Exception:
            pass

    def _update_estimate(self, change=None) -> None:
        if self._molecule is None:
            self.perf_estimate_html.value = ""
            return
        try:
            est = _calc_log.estimate_time(
                n_atoms=len(self._molecule.atoms),
                n_electrons=self._molecule.get_electron_count(),
                method=self.method_dd.value,
                basis=self.basis_dd.value,
            )
            self.perf_estimate_html.value = _calc_log.format_estimate(est)
        except Exception:
            self.perf_estimate_html.value = ""

    def _refresh_results_browser(self) -> None:
        try:
            from quantui import list_results, load_result
        except ImportError:
            return
        self.results_path_lbl.value = (
            f'<span style="font-size:13px;color:#64748b">'
            f"{self._get_results_dir()}</span>"
        )
        dirs = list_results()
        if not dirs:
            self.past_dd.options = [("(no saved results)", "")]
            return
        options = []
        for d in dirs:
            try:
                data = load_result(d)
                ts = data.get("timestamp", d.name)
                label = f"{ts}  ·  {data['formula']}  {data['method']}/{data['basis']}"
                options.append((label, str(d)))
            except Exception:
                pass
        self.past_dd.options = options if options else [("(no saved results)", "")]

    def _refresh_comparison(self) -> None:
        from quantui import comparison_table_html, summary_from_session_result

        self.comparison_output.clear_output(wait=True)
        if not self._results:
            return
        summaries = [summary_from_session_result(r) for r in self._results]
        with self.comparison_output:
            display(HTML(comparison_table_html(summaries)))
            if len(summaries) > 1:
                try:
                    from quantui import plot_comparison

                    plot_comparison(summaries)
                except Exception:
                    pass

    def _populate_compare_list(self) -> None:
        from quantui.results_storage import list_results, load_result

        dirs = list_results()
        if not dirs:
            self.compare_select.options = [("(no saved results)", "")]
            self.compare_btn.disabled = True
            return
        options = []
        for d in dirs:
            try:
                data = load_result(d)
                ts = data.get("timestamp", d.name[:19])
                label = f"{ts}  {data['formula']}  {data['method']}/{data['basis']}"
                options.append((label, str(d)))
            except Exception:
                options.append((d.name, str(d)))
        self.compare_select.options = options
        self.compare_btn.disabled = False

    def _show_help_topic(self, topic: str) -> None:
        if topic in HELP_TOPICS:
            self.help_topic_dd.value = topic
        self.root_tab.selected_index = 4

    def _update_log_panel(self, log_text: str, label: str = "") -> None:
        self._render_log(log_text, label)

    def _goto_output_tab(self) -> None:
        self.root_tab.selected_index = 3

    def _render_log(self, text: str, source_label: str = "") -> None:
        import html as _html_mod

        lines = text.splitlines()
        rows = []
        for line in lines:
            esc = _html_mod.escape(line)
            if "converged SCF energy" in line or "SCF converged" in line:
                style = "color:#16a34a;font-weight:600"
            elif "cycle=" in line and "E=" in line:
                style = "color:#475569"
            elif "HOMO" in line or "LUMO" in line:
                style = "color:#2563eb"
            elif "Warning" in line or "warning" in line:
                style = "color:#d97706"
            elif "Error" in line or "error" in line or "failed" in line:
                style = "color:#dc2626"
            else:
                style = "color:#1e293b"
            rows.append(f'<div style="{style}">{esc}</div>')
        self._log_output_html.value = (
            '<div style="font-family:monospace;font-size:12px;line-height:1.4;'
            "padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;"
            'border-radius:4px;overflow-x:auto;max-height:550px;overflow-y:auto">'
            + "".join(rows)
            + "</div>"
        )
        self._log_source_lbl.value = (
            f'<span style="font-size:12px;color:#64748b">Source: {source_label}</span>'
            if source_label
            else ""
        )

    def _render_help_topic(self, change=None) -> None:
        key = self.help_topic_dd.value
        if key and key in HELP_TOPICS:
            entry = HELP_TOPICS[key]
            self.help_content_html.value = (
                f'<div style="border:1px solid #e2e8f0;border-radius:6px;'
                f'padding:14px 18px;margin:8px 0;background:#f8fafc;max-width:700px">'
                f'<h4 style="margin:0 0 10px;color:#1e293b;font-size:15px;font-weight:700">'
                f'{entry["title"]}</h4>'
                f'<div style="font-size:14px;color:#334155;line-height:1.6">'
                f'{entry["body"]}</div>'
                f"</div>"
            )

    def _refresh_perf_stats(self) -> None:
        self._perf_stats_html.value = self._build_perf_stats_html()
        self._perf_events_html.value = self._build_events_html()

    def _build_perf_stats_html(self) -> str:
        from quantui.calc_log import get_perf_history

        records = get_perf_history()
        if not records:
            return (
                '<span style="color:#94a3b8;font-size:13px">'
                "No performance data recorded yet.</span>"
            )
        groups: dict = {}
        for r in records:
            key = (r.get("method", "?"), r.get("basis", "?"))
            groups.setdefault(key, []).append(r)
        rows = ""
        for (meth, bas), recs in sorted(groups.items()):
            times = [r["elapsed_s"] for r in recs if "elapsed_s" in r]
            n = len(recs)
            if times:
                avg = sum(times) / len(times)
                rows += (
                    "<tr>"
                    f'<td style="padding:2px 12px 2px 0">{meth}</td>'
                    f'<td style="padding:2px 12px 2px 0">{bas}</td>'
                    f'<td style="padding:2px 12px 2px 0;text-align:right">{n}</td>'
                    f'<td style="padding:2px 12px 2px 0;text-align:right">{avg:.1f} s</td>'
                    f'<td style="padding:2px 12px 2px 0;text-align:right">{min(times):.1f} s</td>'
                    f'<td style="padding:2px 12px 2px 0;text-align:right">{max(times):.1f} s</td>'
                    "</tr>"
                )
        header = (
            "<tr>"
            '<th style="text-align:left;padding:2px 12px 2px 0;color:#64748b">Method</th>'
            '<th style="text-align:left;padding:2px 12px 2px 0;color:#64748b">Basis</th>'
            '<th style="text-align:right;padding:2px 12px 2px 0;color:#64748b">Runs</th>'
            '<th style="text-align:right;padding:2px 12px 2px 0;color:#64748b">Avg</th>'
            '<th style="text-align:right;padding:2px 12px 2px 0;color:#64748b">Min</th>'
            '<th style="text-align:right;padding:2px 12px 2px 0;color:#64748b">Max</th>'
            "</tr>"
        )
        return (
            '<table style="font-size:13px;border-collapse:collapse;width:100%">'
            f"{header}{rows}</table>"
        )

    def _build_events_html(self) -> str:
        from quantui.calc_log import get_recent_events

        events = get_recent_events(20)
        if not events:
            return (
                '<span style="color:#94a3b8;font-size:13px">'
                "No events recorded yet.</span>"
            )
        rows = ""
        for e in reversed(events):
            ts = e.get("timestamp", "")[:19].replace("T", " ")
            evt = e.get("event", "")
            msg = e.get("message", "")
            rows += (
                "<tr>"
                f'<td style="padding:1px 10px 1px 0;color:#94a3b8;font-size:11px;white-space:nowrap">{ts}</td>'
                f'<td style="padding:1px 10px 1px 0;color:#475569;font-size:12px">{evt}</td>'
                f'<td style="padding:1px 0;color:#334155;font-size:12px">{msg}</td>'
                "</tr>"
            )
        return (
            '<table style="font-size:13px;border-collapse:collapse;width:100%">'
            f"{rows}</table>"
        )

    # ══ RESULT FORMATTERS ════════════════════════════════════════════════════

    def _format_result(self, r) -> str:
        _conv = "Yes" if r.converged else "No (treat results with caution)"
        _cc = "green" if r.converged else "#c00"
        _gap = (
            f"{r.homo_lumo_gap_ev:.4f} eV" if r.homo_lumo_gap_ev is not None else "N/A"
        )
        _rows = "".join(
            f"<tr>"
            f'<td style="padding:3px 18px 3px 0;color:#444">{k}</td>'
            f'<td style="color:{vc}">{v}</td>'
            f"</tr>"
            for k, v, vc in [
                (
                    "Total energy",
                    f"{r.energy_hartree:.8f} Ha &ensp;({r.energy_ev:.4f} eV)",
                    "#000",
                ),
                ("HOMO-LUMO gap", _gap, "#000"),
                ("SCF converged", _conv, _cc),
                ("SCF iterations", str(r.n_iterations), "#000"),
            ]
        )
        return (
            f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
            f'padding:10px 14px;border-radius:4px;margin:6px 0">'
            f"<b>{r.formula} &mdash; {r.method}/{r.basis}</b>"
            f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
            f"{_rows}</table></div>"
        )

    def _format_opt_result(self, r) -> str:
        _conv = "Yes" if r.converged else "No (max steps reached)"
        _cc = "green" if r.converged else "#c00"
        _rows = "".join(
            f"<tr>"
            f'<td style="padding:3px 18px 3px 0;color:#444">{k}</td>'
            f'<td style="color:{vc}">{v}</td>'
            f"</tr>"
            for k, v, vc in [
                ("Final energy", f"{r.energy_hartree:.8f} Ha", "#000"),
                ("Energy change", f"{r.energy_change_hartree:+.6f} Ha", "#000"),
                ("Opt converged", _conv, _cc),
                ("Steps taken", str(r.n_steps), "#000"),
                ("Geometry RMSD", f"{r.rmsd_angstrom:.4f} Å", "#000"),
            ]
        )
        return (
            f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
            f'padding:10px 14px;border-radius:4px;margin:6px 0">'
            f"<b>Geometry Optimisation &mdash; {r.formula} ({r.method}/{r.basis})</b>"
            f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
            f"{_rows}</table></div>"
        )

    def _format_freq_result(self, r) -> str:
        _conv = "Yes" if r.converged else "No (treat with caution)"
        _cc = "green" if r.converged else "#c00"
        n_real = r.n_real_modes()
        n_imag = r.n_imaginary_modes()
        real_freqs = sorted(f for f in r.frequencies_cm1 if f > 0)[:6]
        freq_str = "  ".join(f"{f:.1f}" for f in real_freqs)
        if len([f for f in r.frequencies_cm1 if f > 0]) > 6:
            freq_str += " …"
        imag_note = ""
        if n_imag > 0:
            imag_note = (
                f'<tr><td style="padding:3px 18px 3px 0;color:#444">Imaginary modes</td>'
                f'<td style="color:#c00">{n_imag} — geometry may not be a minimum</td></tr>'
            )
        _rows = (
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">SCF energy</td>'
            f'<td style="color:#000">{r.energy_hartree:.8f} Ha</td></tr>'
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">SCF converged</td>'
            f'<td style="color:{_cc}">{_conv}</td></tr>'
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">Real modes</td>'
            f'<td style="color:#000">{n_real}</td></tr>'
            + imag_note
            + (
                f'<tr><td style="padding:3px 18px 3px 0;color:#444">Frequencies (cm⁻¹)</td>'
                f'<td style="color:#000;font-family:monospace">{freq_str or "none"}</td></tr>'
                if real_freqs
                else ""
            )
            + f'<tr><td style="padding:3px 18px 3px 0;color:#444">ZPVE</td>'
            f'<td style="color:#000">{r.zpve_hartree:.6f} Ha '
            f"({r.zpve_hartree * 27.211386245988:.4f} eV)</td></tr>"
        )
        return (
            f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
            f'padding:10px 14px;border-radius:4px;margin:6px 0">'
            f"<b>Frequency Analysis &mdash; {r.formula} ({r.method}/{r.basis})</b>"
            f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
            f"{_rows}</table></div>"
        )

    def _format_tddft_result(self, r) -> str:
        _conv = "Yes" if r.converged else "No (treat with caution)"
        _cc = "green" if r.converged else "#c00"
        header_rows = (
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">Ground-state energy</td>'
            f'<td style="color:#000">{r.energy_hartree:.8f} Ha</td></tr>'
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">SCF converged</td>'
            f'<td style="color:{_cc}">{_conv}</td></tr>'
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">States computed</td>'
            f'<td style="color:#000">{len(r.excitation_energies_ev)}</td></tr>'
        )
        exc_table = ""
        if r.excitation_energies_ev:
            wl = r.wavelengths_nm()
            exc_rows = []
            for i, (e_ev, f_osc) in enumerate(
                zip(r.excitation_energies_ev[:8], r.oscillator_strengths[:8]), 1
            ):
                bold = "font-weight:bold" if f_osc > 0.05 else ""
                exc_rows.append(
                    f'<tr style="{bold}">'
                    f'<td style="padding:2px 12px 2px 0;color:#555">S{i}</td>'
                    f'<td style="padding:2px 12px 2px 0;color:#000">{e_ev:.3f} eV</td>'
                    f'<td style="padding:2px 12px 2px 0;color:#000">{wl[i - 1]:.1f} nm</td>'
                    f'<td style="padding:2px 4px 2px 0;color:#000">f = {f_osc:.4f}</td>'
                    f"</tr>"
                )
            if len(r.excitation_energies_ev) > 8:
                exc_rows.append(
                    f'<tr><td colspan="4" style="color:#888;font-size:12px">… '
                    f"and {len(r.excitation_energies_ev) - 8} more states</td></tr>"
                )
            exc_table = (
                '<tr><td colspan="2" style="padding:8px 0 2px;color:#444;font-weight:bold">'
                "Vertical excitations:</td></tr>"
                "<tr>"
                '<th style="text-align:left;color:#555;font-size:12px;padding:2px 12px 2px 0">State</th>'
                '<th style="text-align:left;color:#555;font-size:12px;padding:2px 12px 2px 0">Energy</th>'
                '<th style="text-align:left;color:#555;font-size:12px;padding:2px 12px 2px 0">λ</th>'
                '<th style="text-align:left;color:#555;font-size:12px">Osc. str.</th></tr>'
                + "".join(exc_rows)
            )
        return (
            f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
            f'padding:10px 14px;border-radius:4px;margin:6px 0">'
            f"<b>TD-DFT / UV-Vis &mdash; {r.formula} ({r.method}/{r.basis})</b>"
            f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
            f"{header_rows}{exc_table}</table></div>"
        )

    def _format_past_result(self, data: dict) -> str:
        _conv = "Yes" if data.get("converged") else "No (treat results with caution)"
        _cc = "green" if data.get("converged") else "#c00"
        _gap = (
            f"{data['homo_lumo_gap_ev']:.4f} eV"
            if data.get("homo_lumo_gap_ev") is not None
            else "N/A"
        )
        _rows = "".join(
            f"<tr>"
            f'<td style="padding:3px 18px 3px 0;color:#444">{k}</td>'
            f'<td style="color:{vc}">{v}</td>'
            f"</tr>"
            for k, v, vc in [
                (
                    "Total energy",
                    f"{data['energy_hartree']:.8f} Ha &ensp;({data['energy_ev']:.4f} eV)",
                    "#000",
                ),
                ("HOMO-LUMO gap", _gap, "#000"),
                ("SCF converged", _conv, _cc),
                ("SCF iterations", str(data.get("n_iterations", "?")), "#000"),
            ]
        )
        ts = data.get("timestamp", "")
        return (
            f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
            f'padding:10px 14px;border-radius:4px;margin:6px 0">'
            f'<b>{data["formula"]} &mdash; {data["method"]}/{data["basis"]}</b>'
            f'&ensp;<small style="color:#777">{ts}</small>'
            f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
            f"{_rows}</table></div>"
        )

    # ══ HELPERS ══════════════════════════════════════════════════════════════

    def _get_results_dir(self) -> Path:
        from quantui.results_storage import _default_results_dir

        return _default_results_dir().resolve()
