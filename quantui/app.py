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
from typing import Any, List, Literal, Optional

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
    from quantui.visualization_py3dmol import (
        DEFAULT_LIGHTING as _DEFAULT_LIGHTING,
    )
    from quantui.visualization_py3dmol import (
        DEFAULT_STYLE as _DEFAULT_VIZ_STYLE,
    )
    from quantui.visualization_py3dmol import (
        LIGHTING_OPTIONS as _LIGHTING_OPTIONS,
    )
    from quantui.visualization_py3dmol import (
        PLOTLYMOL_AVAILABLE as _PLOTLYMOL_VIZ,
    )
    from quantui.visualization_py3dmol import (
        PY3DMOL_AVAILABLE as _PY3DMOL_VIZ,
    )
    from quantui.visualization_py3dmol import (
        VIZ_STYLE_OPTIONS as _VIZ_STYLE_OPTIONS,
    )
    from quantui.visualization_py3dmol import (
        display_molecule as _display_molecule,
    )

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    _display_molecule = None  # type: ignore[assignment]
    _PLOTLYMOL_VIZ = False
    _PY3DMOL_VIZ = False
    _DEFAULT_VIZ_STYLE = "ball+stick"
    _DEFAULT_LIGHTING = "soft"
    _VIZ_STYLE_OPTIONS = [
        ("Ball & Stick", "ball+stick"),
        ("Stick", "stick"),
        ("Sphere (VDW)", "sphere"),
        ("Line", "line"),
    ]
    _LIGHTING_OPTIONS = [
        ("Soft", "soft"),
        ("Default", "default"),
        ("Bright", "bright"),
        ("Metallic", "metallic"),
        ("Dramatic", "dramatic"),
    ]

_VizBackend = Literal["auto", "py3dmol", "plotlymol"]
_BOTH_VIZ_AVAILABLE: bool = _PLOTLYMOL_VIZ and _PY3DMOL_VIZ
_DEFAULT_VIZ_BACKEND: _VizBackend = "plotlymol" if _PLOTLYMOL_VIZ else "py3dmol"

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

_RDKIT_AVAILABLE: bool = bool(PUBCHEM_AVAILABLE)

from quantui.benchmarks import (  # noqa: E402
    BENCHMARK_SUITE as _BENCHMARK_SUITE,
)
from quantui.benchmarks import (  # noqa: E402
    load_last_calibration as _load_last_calibration_raw,
)


def _load_last_calibration_label() -> str:
    """Return a human-readable timestamp of the last calibration, or ''."""
    data = _load_last_calibration_raw()
    if data is None:
        return ""
    ts = str(data.get("timestamp", ""))
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(ts).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return ts[:19] if ts else ""


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
        self._pending_traj_result: Any = None
        self.root_tab: widgets.Tab

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
                    self._welcome_html,
                    widgets.HBox(
                        [self.theme_btn, self._help_btn],
                        layout=widgets.Layout(
                            justify_content="flex-end", margin="0 0 4px"
                        ),
                    ),
                    self._theme_style,
                    self.help_tab_panel,
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
        self._build_welcome_header()
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
        self._status_tab_panel = widgets.VBox(
            [
                widgets.HTML(
                    '<p style="color:#555;font-size:13px;margin:4px 0 12px">'
                    "System capabilities and resource availability for this session.</p>"
                ),
                self._status_html,
            ],
            layout=widgets.Layout(padding="8px 0"),
        )

    # ── Welcome header ────────────────────────────────────────────────────

    def _build_welcome_header(self) -> None:
        _logo_svg = (
            '<svg width="72" height="72" viewBox="0 0 280 280"'
            ' xmlns="http://www.w3.org/2000/svg">'
            "<defs>"
            '<filter id="q-glow" x="-50%" y="-50%" width="200%" height="200%">'
            '<feGaussianBlur stdDeviation="7" result="blur"/>'
            "<feMerge>"
            '<feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/>'
            "</feMerge></filter>"
            '<filter id="q-halo" x="-80%" y="-80%" width="260%" height="260%">'
            '<feGaussianBlur stdDeviation="22" result="blur"/>'
            "<feMerge>"
            '<feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/>'
            "</feMerge></filter>"
            "</defs>"
            '<circle cx="140" cy="140" r="48"'
            ' fill="rgba(37,99,235,0.20)" filter="url(#q-halo)"/>'
            '<g transform="rotate(0,140,140)">'
            '<ellipse cx="140" cy="140" rx="115" ry="33" fill="none"'
            ' stroke="#0891b2" stroke-width="1.4" opacity="0.70"/>'
            '<circle cx="255" cy="140" r="5.5" fill="#67e8f9"/>'
            "</g>"
            '<g transform="rotate(60,140,140)">'
            '<ellipse cx="140" cy="140" rx="115" ry="33" fill="none"'
            ' stroke="#0891b2" stroke-width="1.4" opacity="0.55"/>'
            '<circle cx="255" cy="140" r="4.5" fill="#93c5fd"/>'
            "</g>"
            '<g transform="rotate(120,140,140)">'
            '<ellipse cx="140" cy="140" rx="115" ry="33" fill="none"'
            ' stroke="#3b82f6" stroke-width="1.4" opacity="0.42"/>'
            '<circle cx="255" cy="140" r="4" fill="#60a5fa"/>'
            "</g>"
            '<circle cx="140" cy="140" r="20"'
            ' fill="rgba(37,99,235,0.25)" filter="url(#q-glow)"/>'
            '<circle cx="140" cy="140" r="14"'
            ' fill="#2563eb" filter="url(#q-glow)"/>'
            '<circle cx="140" cy="140" r="8" fill="#60a5fa"/>'
            '<circle cx="137" cy="137" r="3" fill="rgba(255,255,255,0.45)"/>'
            "</svg>"
        )
        _html = (
            f'<div style="display:flex;align-items:center;gap:20px;'
            f"padding:18px 4px 14px;margin-bottom:4px;"
            f'border-bottom:1px solid #e2e8f0">'
            f"{_logo_svg}"
            f"<div>"
            f'<div style="font-size:30px;font-weight:700;letter-spacing:-0.4px;'
            f'color:#0f172a;line-height:1.1">QuantUI (local)</div>'
            f'<div style="font-size:15px;color:#475569;margin-top:5px">'
            f"Quantum chemistry right on your device</div>"
            f'<div style="font-size:12px;color:#94a3b8;margin-top:3px">'
            f"v{quantui.__version__} &nbsp;&middot;&nbsp; "
            f"open the <b>Help</b> tab for instructions</div>"
            f"</div>"
            f"</div>"
        )
        self._welcome_html = widgets.HTML(value=_html)

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
        self._last_result_dir: Optional[Path] = None

        # 3D viewer backend selector — shown only when both backends are installed
        self._viz_backend: _VizBackend = _DEFAULT_VIZ_BACKEND
        if _BOTH_VIZ_AVAILABLE:
            self.viz_backend_toggle = widgets.ToggleButtons(
                options=[("PlotlyMol", "plotlymol"), ("py3Dmol", "py3dmol")],
                value=_DEFAULT_VIZ_BACKEND,
                tooltips=["Plotly-based interactive viewer", "WebGL viewer (py3Dmol)"],
                style={"button_width": "90px"},
                layout=widgets.Layout(margin="2px 0 0 0"),
            )
        else:
            self.viz_backend_toggle = None  # type: ignore[assignment]

        # 3D viewer style and lighting controls
        self._viz_style: str = _DEFAULT_VIZ_STYLE
        self._viz_lighting: str = _DEFAULT_LIGHTING
        self.viz_style_dd = widgets.Dropdown(
            options=_VIZ_STYLE_OPTIONS,
            value=_DEFAULT_VIZ_STYLE,
            description="Style:",
            style={"description_width": "40px"},
            layout=widgets.Layout(width="180px"),
            disabled=not VISUALIZATION_AVAILABLE,
        )
        # Lighting only applies to the PlotlyMol backend
        _lighting_available = VISUALIZATION_AVAILABLE and _PLOTLYMOL_VIZ
        self.viz_lighting_dd = widgets.Dropdown(
            options=_LIGHTING_OPTIONS,
            value=_DEFAULT_LIGHTING,
            description="Lighting:",
            style={"description_width": "58px"},
            layout=widgets.Layout(width="170px"),
            disabled=not _lighting_available,
        )
        if not _lighting_available:
            self.viz_lighting_dd.layout.visibility = "hidden"
        self.viz_controls_box = widgets.HBox(
            [self.viz_style_dd, self.viz_lighting_dd],
            layout=widgets.Layout(gap="8px", margin="2px 0 0 0", align_items="center"),
        )
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

        # Implicit solvent (PCM)
        from quantui.config import SOLVENT_OPTIONS as _SOLVENT_OPTS

        self.solvent_cb = widgets.Checkbox(
            value=False,
            description="Implicit solvent (PCM)",
            layout=widgets.Layout(width="240px"),
        )
        self.solvent_dd = widgets.Dropdown(
            options=list(_SOLVENT_OPTS.keys()),
            value="Water",
            description="Solvent:",
            style={"description_width": "70px"},
            layout=widgets.Layout(width="200px", display="none"),
        )

        # Calculation type + extra options
        self.calc_type_dd = widgets.Dropdown(
            options=[
                "Single Point",
                "Geometry Opt",
                "Frequency",
                "UV-Vis (TD-DFT)",
                "NMR Shielding",
                "PES Scan",
            ],
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

        # ── Frequency calc extra widgets ──────────────────────────────────────
        self._freq_seed_dd = widgets.Dropdown(
            options=[("(use current molecule)", "")],
            description="Seed geometry:",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="420px"),
            tooltip="Optionally load the final optimised geometry from a previous Geo Opt result",
        )
        self._freq_seed_refresh_btn = widgets.Button(
            description="",
            icon="refresh",
            layout=widgets.Layout(width="32px", height="32px"),
            tooltip="Refresh the list of saved geometry optimisations",
        )
        self._freq_preopt_cb = widgets.Checkbox(
            value=False,
            description="Pre-optimize geometry first (recommended for unoptimised inputs)",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="100%"),
        )
        self._freq_seed_note = widgets.HTML("")

        # ── PES scan extra widgets ────────────────────────────────────────────
        self._scan_type_dd = widgets.Dropdown(
            options=["Bond", "Angle", "Dihedral"],
            value="Bond",
            description="Scan type:",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="220px"),
        )
        _atom_idx_layout = widgets.Layout(width="95px")
        _atom_idx_style = {"description_width": "50px"}
        self._scan_atom1 = widgets.BoundedIntText(
            value=1,
            min=1,
            max=999,
            description="Atom 1:",
            style=_atom_idx_style,
            layout=_atom_idx_layout,
        )
        self._scan_atom2 = widgets.BoundedIntText(
            value=2,
            min=1,
            max=999,
            description="Atom 2:",
            style=_atom_idx_style,
            layout=_atom_idx_layout,
        )
        self._scan_atom3 = widgets.BoundedIntText(
            value=3,
            min=1,
            max=999,
            description="Atom 3:",
            style=_atom_idx_style,
            layout=_atom_idx_layout,
        )
        self._scan_atom4 = widgets.BoundedIntText(
            value=4,
            min=1,
            max=999,
            description="Atom 4:",
            style=_atom_idx_style,
            layout=_atom_idx_layout,
        )
        self._scan_atom34_box = widgets.HBox(
            [self._scan_atom3, self._scan_atom4],
            layout=widgets.Layout(gap="4px"),
        )
        self._scan_start = widgets.BoundedFloatText(
            value=0.5,
            min=0.01,
            max=1000.0,
            step=0.1,
            description="Start:",
            style={"description_width": "40px"},
            layout=widgets.Layout(width="140px"),
        )
        self._scan_stop = widgets.BoundedFloatText(
            value=2.0,
            min=0.01,
            max=1000.0,
            step=0.1,
            description="Stop:",
            style={"description_width": "40px"},
            layout=widgets.Layout(width="140px"),
        )
        self._scan_steps = widgets.BoundedIntText(
            value=10,
            min=2,
            max=100,
            description="Points:",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="120px"),
        )
        self._scan_unit_lbl = widgets.HTML(
            '<span style="font-size:12px;color:#555">Å</span>'
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
        _rdkit_tip = (
            ""
            if _RDKIT_AVAILABLE
            else "Requires RDKit (conda install -c conda-forge rdkit)"
        )
        self.export_xyz_btn = widgets.Button(
            description="Export XYZ",
            icon="download",
            disabled=True,
            layout=widgets.Layout(width="130px"),
        )
        self.export_mol_btn = widgets.Button(
            description="Export MOL",
            icon="download",
            disabled=True,
            tooltip=_rdkit_tip,
            layout=widgets.Layout(width="130px"),
        )
        self.export_pdb_btn = widgets.Button(
            description="Export PDB",
            icon="download",
            disabled=True,
            tooltip=_rdkit_tip,
            layout=widgets.Layout(width="130px"),
        )
        self.struct_export_status = widgets.Label()

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
        _mol_container_children = [
            self.mol_input_expanded,
            self.mol_info_html,
            self.viz_output,
        ]
        if self.viz_backend_toggle is not None:
            _mol_container_children.append(self.viz_backend_toggle)
        if VISUALIZATION_AVAILABLE:
            _mol_container_children.append(self.viz_controls_box)
        self.mol_input_container = widgets.VBox(
            _mol_container_children,
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
                widgets.HBox(
                    [self.solvent_cb, self.solvent_dd],
                    layout=widgets.Layout(align_items="center", gap="4px"),
                ),
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
        # PES scan energy plot accordion (hidden until a PES Scan completes)
        self._pes_plot_html = widgets.HTML(
            value="", layout=widgets.Layout(width="100%")
        )
        self._pes_scan_accordion = widgets.Accordion(
            children=[
                widgets.VBox(
                    [self._pes_plot_html],
                    layout=widgets.Layout(padding="8px"),
                )
            ],
            layout=widgets.Layout(display="none", margin="8px 0"),
        )
        self._pes_scan_accordion.set_title(0, "PES Energy Profile")
        self._pes_scan_accordion.selected_index = None

        # Trajectory accordion (Geo Opt / PES Scan — hidden until result completes)
        self.traj_output = widgets.Output()
        self.traj_accordion = widgets.Accordion(
            children=[self.traj_output],
            layout=widgets.Layout(display="none", margin="8px 0"),
        )
        self.traj_accordion.set_title(0, "Trajectory Viewer")
        self.traj_accordion.selected_index = None  # collapsed by default
        self.traj_accordion.observe(self._on_traj_expand, names=["selected_index"])

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

        # IR Spectrum accordion (hidden until a Frequency result is available)
        self._ir_mode_toggle = widgets.ToggleButtons(
            options=["Stick", "Broadened"],
            value="Stick",
            style={"button_width": "80px", "font_size": "12px"},
            layout=widgets.Layout(margin="0 8px 0 0"),
        )
        self._ir_fwhm_slider = widgets.FloatSlider(
            value=20.0,
            min=5.0,
            max=100.0,
            step=5.0,
            description="Line width:",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="260px", display="none"),
        )
        self._ir_fig = widgets.HTML(value="", layout=widgets.Layout(width="100%"))

        _ir_controls = widgets.HBox(
            [self._ir_mode_toggle, self._ir_fwhm_slider],
            layout=widgets.Layout(align_items="center", margin="0 0 6px 0"),
        )
        _ir_body_children = [_ir_controls, self._ir_fig]
        self._ir_accordion = widgets.Accordion(
            children=[
                widgets.VBox(
                    _ir_body_children,
                    layout=widgets.Layout(padding="8px"),
                )
            ],
            layout=widgets.Layout(display="none", margin="8px 0"),
        )
        self._ir_accordion.set_title(0, "IR Spectrum")
        self._ir_accordion.selected_index = None

        # Orbital energy diagram + isosurface accordion (Single Point / Geo Opt)
        # Use plotly.io.to_html so FigureWidget / anywidget dependency is not needed.

        self._orb_ymin_input = widgets.BoundedFloatText(
            value=-30.0,
            min=-500.0,
            max=200.0,
            step=1.0,
            description="Y min:",
            layout=widgets.Layout(width="140px"),
            style={"description_width": "45px"},
        )
        self._orb_ymax_input = widgets.BoundedFloatText(
            value=5.0,
            min=-500.0,
            max=500.0,
            step=1.0,
            description="Y max:",
            layout=widgets.Layout(width="140px"),
            style={"description_width": "45px"},
        )
        self._orb_n_orb_input = widgets.BoundedIntText(
            value=20,
            min=4,
            max=200,
            step=2,
            description="Show N:",
            layout=widgets.Layout(width="120px"),
            style={"description_width": "50px"},
        )
        _orb_controls_row = widgets.HBox(
            [
                widgets.HTML(
                    '<span style="font-size:11px;color:#555;font-weight:600">Y range:</span>'
                ),
                self._orb_ymin_input,
                self._orb_ymax_input,
                widgets.HTML(
                    '<span style="font-size:11px;color:#555;font-weight:600;margin-left:8px">'
                    "Orbitals shown:</span>"
                ),
                self._orb_n_orb_input,
            ],
            layout=widgets.Layout(
                align_items="center",
                flex_wrap="wrap",
                gap="4px",
                margin="0 0 6px 0",
            ),
        )
        self._orb_diagram_html = widgets.HTML(
            value="", layout=widgets.Layout(width="100%")
        )
        _orb_diagram_content: list = [_orb_controls_row, self._orb_diagram_html]
        self._orb_diagram_box = widgets.VBox(
            _orb_diagram_content,
            layout=widgets.Layout(width="100%"),
        )
        self._orb_toggle = widgets.ToggleButtons(
            options=["HOMO-1", "HOMO", "LUMO", "LUMO+1"],
            value="HOMO",
            style={"button_width": "70px", "font_size": "12px"},
            layout=widgets.Layout(margin="8px 0 4px 0"),
        )
        self._orb_iso_output = widgets.Output()
        self._orb_iso_controls = widgets.VBox(
            [
                widgets.HTML(
                    '<span style="font-size:12px;color:#555;font-weight:bold">'
                    "Orbital isosurface:</span>"
                ),
                self._orb_toggle,
                self._orb_iso_output,
            ],
            layout=widgets.Layout(display="none", margin="8px 0 0 0"),
        )
        self._orb_accordion = widgets.Accordion(
            children=[
                widgets.VBox(
                    [self._orb_diagram_box],
                    layout=widgets.Layout(padding="8px"),
                )
            ],
            layout=widgets.Layout(display="none", margin="8px 0"),
        )
        self._orb_accordion.set_title(0, "Orbital Diagram")
        self._orb_accordion.selected_index = None

        # Post-calculate panel — isosurface and other heavy on-demand analyses
        self._iso_generate_btn = widgets.Button(
            description="Generate Isosurface",
            button_style="primary",
            icon="flask",
            disabled=True,
            tooltip=(
                "Generate a 3D orbital isosurface. "
                "Available after running or loading a Single Point or Geometry Optimization."
            ),
            layout=widgets.Layout(width="200px", margin="8px 0 4px 0"),
        )
        _iso_body = widgets.VBox(
            [
                widgets.HTML(
                    '<p style="color:#555;font-size:12px;margin:0 0 8px">'
                    "Visualise a molecular orbital as a 3D isosurface (Linux / WSL only — "
                    "requires PySCF and RDKit). Run or load a Single Point or Geometry "
                    "Optimization first, then click <b>Generate</b>.</p>"
                ),
                self._orb_iso_controls,
                self._iso_generate_btn,
            ],
            layout=widgets.Layout(padding="8px"),
        )
        self._iso_accordion = widgets.Accordion(
            children=[_iso_body],
            layout=widgets.Layout(display="none", margin="8px 0"),
        )
        self._iso_accordion.set_title(0, "Orbital Isosurface")
        self._iso_accordion.selected_index = None

        # ── Result directory path label (hidden until a calculation saves) ──
        self._result_dir_label = widgets.HTML(
            value="",
            layout=widgets.Layout(display="none", margin="4px 0 0 0"),
        )

        # ── Full output log accordion (hidden until a calculation saves) ────
        self._result_log_output = widgets.Output()
        self._result_log_accordion = widgets.Accordion(
            children=[self._result_log_output],
            layout=widgets.Layout(display="none", margin="8px 0 0 0"),
        )
        self._result_log_accordion.set_title(0, "Full output log (pyscf.log)")
        self._result_log_accordion.selected_index = None

        # ── Completion banner (Calculate tab — hidden until run finishes) ───
        self._go_results_btn = widgets.Button(
            description="→ View Results",
            button_style="success",
            layout=widgets.Layout(width="130px"),
        )
        self._go_analysis_btn = widgets.Button(
            description="→ View Analysis",
            button_style="info",
            layout=widgets.Layout(width="140px"),
        )
        self._completion_mol_lbl = widgets.HTML(value="")
        self._completion_banner = widgets.HBox(
            [
                widgets.HTML(
                    '<span style="color:#22c55e;font-weight:600;font-size:13px">'
                    "✓ Calculation complete — </span>"
                ),
                self._completion_mol_lbl,
                self._go_results_btn,
                self._go_analysis_btn,
            ],
            layout=widgets.Layout(
                display="none",
                align_items="center",
                gap="8px",
                padding="10px 12px",
                border="1px solid #bbf7d0",
                border_radius="6px",
                background_color="#f0fdf4",
                margin="8px 0",
            ),
        )

        # ── Results tab panel (Tab 1) ─────────────────────────────────────
        self._to_analysis_btn = widgets.Button(
            description="→ View Analysis",
            button_style="",
            icon="bar-chart",
            layout=widgets.Layout(display="none", width="160px", margin="8px 0 0 0"),
        )
        # Label above the 3D viewer — updated by _do_run to say "Optimized geometry"
        # for Geometry Opt, or hidden for other calc types that don't change geometry.
        self._viz_label = widgets.HTML(
            value="",
            layout=widgets.Layout(display="none"),
        )
        self.results_tab_panel = widgets.VBox(
            [
                widgets.HTML('<h3 style="margin:14px 0 6px">Results</h3>'),
                self.result_output,
                self._viz_label,
                self.result_viz_output,
                self._result_dir_label,
                # advanced_accordion appended in _assemble_tabs (built later in
                # _build_compare_section — must run before it can be referenced here)
                self._to_analysis_btn,
            ],
            layout=widgets.Layout(padding="8px 0"),
        )
        # Backward-compat alias — existing methods that reference results_panel still work
        self.results_panel = self.results_tab_panel

        # ── Analysis tab: molecule viewer (shown for all calc types) ─────
        self._analysis_mol_output = widgets.Output()

        # ── Analysis tab panel (Tab 2) ────────────────────────────────────
        self._analysis_context_lbl = widgets.HTML(
            value=(
                '<p style="color:#555;font-size:13px;margin:4px 0 12px">'
                "No result loaded yet. Run a calculation or load one from History.</p>"
            )
        )
        self._analysis_empty_html = widgets.HTML(
            value=(
                '<p style="color:#888;font-size:13px;font-style:italic;margin:8px 0">'
                "No interactive analysis is available for this calculation type.<br>"
                "Run a Single Point, Geo Opt, or Frequency calculation to see "
                "orbital diagrams, trajectory animations, and spectra here.</p>"
            ),
            layout=widgets.Layout(display="none"),
        )
        self._build_ana_switcher()
        self.analysis_tab_panel = widgets.VBox(
            [
                self._analysis_context_lbl,
                self._analysis_mol_output,
                self._analysis_empty_html,
                self._ana_switcher_box,
                self._ana_unavail_html,
                self._orb_accordion,
                self._pes_scan_accordion,
                self.traj_accordion,
                self.vib_accordion,
                self._ir_accordion,
                self._iso_accordion,
            ],
            layout=widgets.Layout(padding="8px 0"),
        )
        # Backward-compat alias for post_calc_panel references in tests
        self.post_calc_panel = self.analysis_tab_panel

    # ── Analysis panel switcher ───────────────────────────────────────────

    def _build_ana_switcher(self) -> None:
        """Build the always-visible panel switcher strip for the Analysis tab."""
        _PANEL_META = [
            ("Orbitals", self._orb_accordion, "Single Point / UV-Vis"),
            ("Trajectory", self.traj_accordion, "Geometry Opt / PES Scan"),
            ("Vibrational", self.vib_accordion, "Frequency"),
            ("IR Spectrum", self._ir_accordion, "Frequency"),
            ("PES Scan", self._pes_scan_accordion, "PES Scan"),
            ("Isosurface", self._iso_accordion, "Single Point (Linux/WSL only)"),
        ]
        self._ana_panel_names: list = [m[0] for m in _PANEL_META]
        self._ana_accordions: list = [m[1] for m in _PANEL_META]
        self._ana_available: set = set()
        self._ana_active: str = ""
        self._ana_unavail_html = widgets.HTML(
            value="",
            layout=widgets.Layout(display="none", margin="4px 0 8px"),
        )
        self._ana_btns: list = []
        for name, _acc, when in _PANEL_META:
            btn = widgets.Button(
                description=name,
                button_style="",
                tooltip=f"Available after: {when}",
                layout=widgets.Layout(margin="0"),
            )
            btn.layout.opacity = "0.35"
            btn.on_click(lambda _b, n=name: self._on_ana_panel_click(n))
            self._ana_btns.append(btn)
        self._ana_switcher_box = widgets.HBox(
            self._ana_btns,
            layout=widgets.Layout(
                flex_wrap="wrap",
                gap="4px",
                margin="0 0 6px 0",
                padding="6px 4px",
                border="1px solid #e2e8f0",
                border_radius="6px",
            ),
        )

    def _on_ana_panel_click(self, name: str) -> None:
        if name in self._ana_available:
            self._select_ana_panel(name)
        else:
            # Grey out all buttons except clicked; show "not available" note
            for btn in self._ana_btns:
                btn.button_style = ""
            for btn, pname in zip(self._ana_btns, self._ana_panel_names):
                if pname == name:
                    btn.button_style = "warning"
            for acc in self._ana_accordions:
                acc.layout.display = "none"
            self._ana_unavail_html.value = (
                f'<p style="color:#92400e;background:#fffbeb;border:1px solid #fde68a;'
                f'border-radius:4px;padding:6px 12px;margin:0;font-size:13px">'
                f"<b>{name}</b> is not available for this calculation type.</p>"
            )
            self._ana_unavail_html.layout.display = ""
            self._ana_active = ""

    def _select_ana_panel(self, name: str) -> None:
        """Show the named panel; hide all others and update button styles."""
        self._ana_active = name
        self._ana_unavail_html.layout.display = "none"
        for pname, acc, btn in zip(
            self._ana_panel_names, self._ana_accordions, self._ana_btns
        ):
            if pname == name:
                acc.layout.display = ""
                acc.selected_index = 0
                btn.button_style = "primary"
            else:
                acc.layout.display = "none"
                btn.button_style = ""

    def _activate_ana_panel(self, name: str, auto_select: bool = True) -> None:
        """Mark a panel as available (full opacity) and optionally select it."""
        self._ana_available.add(name)
        for btn, pname in zip(self._ana_btns, self._ana_panel_names):
            if pname == name:
                btn.layout.opacity = "1.0"
                btn.tooltip = name
        if auto_select:
            self._select_ana_panel(name)

    def _deactivate_all_ana_panels(self) -> None:
        """Reset all panels to hidden/unavailable; used at start of each new run."""
        self._ana_available.clear()
        self._ana_active = ""
        self._ana_unavail_html.layout.display = "none"
        for acc, btn, _name, meta in zip(
            self._ana_accordions,
            self._ana_btns,
            self._ana_panel_names,
            # Re-read tooltips from scratch
            [
                "Single Point / UV-Vis",
                "Geometry Opt / PES Scan",
                "Frequency",
                "Frequency",
                "PES Scan",
                "Single Point (Linux/WSL only)",
            ],
        ):
            acc.layout.display = "none"
            acc.selected_index = None
            btn.button_style = ""
            btn.layout.opacity = "0.35"
            btn.tooltip = f"Available after: {meta}"

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

        # Calibration widgets
        self._cal_run_btn = widgets.Button(
            description="Run Calibration",
            button_style="primary",
            icon="play",
            disabled=not _PYSCF_AVAILABLE,
            tooltip=(
                "Run a short benchmark suite to calibrate time estimates"
                if _PYSCF_AVAILABLE
                else "PySCF required (Linux / macOS / WSL)"
            ),
            layout=widgets.Layout(width="180px"),
        )
        self._cal_stop_btn = widgets.Button(
            description="Stop",
            button_style="warning",
            icon="stop",
            layout=widgets.Layout(width="90px", display="none"),
        )
        self._cal_progress = widgets.IntProgress(
            min=0,
            max=len(_BENCHMARK_SUITE),
            value=0,
            description="",
            bar_style="info",
            layout=widgets.Layout(width="300px", display="none"),
        )
        self._cal_step_label = widgets.HTML(
            value="",
            layout=widgets.Layout(display="none"),
        )
        self._cal_results_html = widgets.HTML(value="")

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

        # Calibration accordion
        _cal_last = _load_last_calibration_label()
        _cal_note = (
            f'<p style="color:#64748b;font-size:12px;margin:0 0 6px">'
            f"Last run: {_cal_last}</p>"
            if _cal_last
            else ""
        )
        _cal_panel = widgets.VBox(
            [
                widgets.HTML(
                    f'<p style="color:#555;font-size:13px;margin:0 0 6px">'
                    f"Run a short benchmark suite ({len(_BENCHMARK_SUITE)} calculations) "
                    f"to give the time estimator a real baseline for this machine.</p>"
                    + _cal_note
                ),
                widgets.HBox(
                    [self._cal_run_btn, self._cal_stop_btn],
                    layout=widgets.Layout(gap="6px", align_items="center"),
                ),
                self._cal_progress,
                self._cal_step_label,
                self._cal_results_html,
            ],
            layout=widgets.Layout(padding="4px 0"),
        )
        self._cal_accordion = widgets.Accordion(
            children=[_cal_panel], selected_index=None
        )
        self._cal_accordion.set_title(0, "Calibrate time estimates")

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
                self._cal_accordion,
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
        _rdkit_note = (
            ""
            if _RDKIT_AVAILABLE
            else '<p style="color:#888;font-size:12px;margin:4px 0 0">MOL/PDB export requires RDKit '
            "(<code>conda install -c conda-forge rdkit</code>).</p>"
        )
        _export_content = widgets.VBox(
            [
                widgets.HTML(
                    '<p style="color:#555;font-size:13px;margin:0 0 8px">'
                    "Download a self-contained PySCF script you can study or run outside the notebook.</p>"
                ),
                widgets.HBox([self.export_btn, self.export_status]),
                widgets.HTML('<hr style="margin:10px 0 8px">'),
                widgets.HTML(
                    '<p style="color:#555;font-size:13px;margin:0 0 6px">'
                    "Download the molecular structure in a standard chemistry file format.</p>"
                    + _rdkit_note
                ),
                widgets.HBox(
                    [self.export_xyz_btn, self.export_mol_btn, self.export_pdb_btn],
                    layout=widgets.Layout(flex_flow="row wrap", gap="6px"),
                ),
                self.struct_export_status,
            ]
        )
        self.advanced_accordion = widgets.Accordion(children=[_export_content])
        self.advanced_accordion.set_title(0, "Export")
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
                    "Raw PySCF output for the most recent calculation. "
                    "Use <b>View log</b> in the History tab to load a saved result's log. "
                    "Orbital diagrams, trajectories, and spectra are in the "
                    "<b>Analysis</b> tab.</p>"
                ),
                widgets.HBox(
                    [self._log_clear_btn],
                    layout=widgets.Layout(margin="0 0 8px"),
                ),
                self._log_source_lbl,
                self._log_output_html,
                self._result_log_accordion,
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

        # [?] toggle button shown in the top bar
        self._help_btn = widgets.Button(
            description="?",
            button_style="",
            tooltip="Help topics",
            layout=widgets.Layout(width="34px", margin="0 0 0 8px"),
        )

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
            layout=widgets.Layout(
                display="none",
                padding="8px 0",
                border="1px solid #e2e8f0",
                border_radius="6px",
                padding_left="12px",
                margin="0 0 8px",
            ),
        )

    # ── Tab assembly (Cell 10) ────────────────────────────────────────────

    def _assemble_tabs(self) -> None:
        _calculate_content = widgets.VBox(
            [
                self.step_progress.widget,
                self.mol_input_container,
                self.calc_setup_panel,
                self.run_panel,
                self._completion_banner,
            ],
            layout=widgets.Layout(padding="8px 0"),
        )

        # Splice advanced_accordion into results_tab_panel before _to_analysis_btn.
        # It cannot be referenced in _build_results_section because it is built later
        # in _build_compare_section.
        _rtp = list(self.results_tab_panel.children)
        _rtp.insert(_rtp.index(self._to_analysis_btn), self.advanced_accordion)
        self.results_tab_panel.children = tuple(_rtp)

        self.root_tab = widgets.Tab(
            children=[
                _calculate_content,
                self.results_tab_panel,
                self.analysis_tab_panel,
                self.history_panel,
                self.compare_panel,
                self.log_tab_panel,
                self._status_tab_panel,
            ]
        )
        self.root_tab.set_title(0, "Calculate")
        self.root_tab.set_title(1, "Results")
        self.root_tab.set_title(2, "Analysis")
        self.root_tab.set_title(3, "History")
        self.root_tab.set_title(4, "Compare")
        self.root_tab.set_title(5, "Log")
        self.root_tab.set_title(6, "Status")

    # ══ CALLBACK WIRING ══════════════════════════════════════════════════════

    def _wire_callbacks(self) -> None:
        # 3D viewer backend toggle (only wired when both backends are available)
        if self.viz_backend_toggle is not None:
            self.viz_backend_toggle.observe(self._on_viz_backend_changed, names="value")
        # 3D viewer style and lighting controls
        if VISUALIZATION_AVAILABLE:
            self.viz_style_dd.observe(self._on_viz_style_changed, names="value")
            self.viz_lighting_dd.observe(self._on_viz_lighting_changed, names="value")
        # Theme
        self.theme_btn.observe(self._on_theme_changed, names="value")
        # Molecule input
        self.preset_dd.observe(self._on_load_preset, names="value")
        self.xyz_btn.on_click(self._on_load_xyz)
        self.pubchem_btn.on_click(self._on_search_pubchem)
        self.change_mol_btn.on_click(self._on_expand_mol_input)
        # Calc type
        self.calc_type_dd.observe(self._on_calc_type_changed, names="value")
        self._freq_seed_dd.observe(self._on_freq_seed_changed, names="value")
        self._scan_type_dd.observe(self._update_scan_widgets, names="value")
        self._freq_seed_refresh_btn.on_click(
            lambda _btn: self._refresh_freq_seed_options()
        )
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
        self.solvent_cb.observe(self._on_solvent_cb_changed, names="value")
        self._cal_run_btn.on_click(self._on_cal_run)
        self._cal_stop_btn.on_click(self._on_cal_stop)
        self.export_btn.on_click(self._on_export)
        self.export_xyz_btn.on_click(self._on_export_xyz)
        self.export_mol_btn.on_click(self._on_export_mol)
        self.export_pdb_btn.on_click(self._on_export_pdb)
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
        # Help [?] toggle
        self._help_btn.on_click(self._on_help_toggle)
        self.help_topic_dd.observe(self._on_help_topic_changed, names="value")
        # Tab navigation buttons
        self._go_results_btn.on_click(
            lambda _: setattr(self.root_tab, "selected_index", 1)
        )
        self._go_analysis_btn.on_click(
            lambda _: setattr(self.root_tab, "selected_index", 2)
        )
        self._to_analysis_btn.on_click(
            lambda _: setattr(self.root_tab, "selected_index", 2)
        )
        # Vibrational mode selector
        self.vib_mode_dd.observe(self._on_vib_mode_changed, names="value")
        # Orbital diagram axis controls
        self._orb_ymin_input.observe(self._on_orb_range_changed, names="value")
        self._orb_ymax_input.observe(self._on_orb_range_changed, names="value")
        self._orb_n_orb_input.observe(self._on_orb_range_changed, names="value")
        # Orbital isosurface generate button
        self._iso_generate_btn.on_click(self._on_iso_generate)

    # ══ CALLBACK METHODS ═════════════════════════════════════════════════════

    # ── Theme ─────────────────────────────────────────────────────────────

    def _on_theme_changed(self, change) -> None:
        self._theme_style.clear_output()
        css = self._theme_css(change["new"])
        if css:
            with self._theme_style:
                display(HTML(css))
        self._rerender_plotly_theme()

    def _plotly_theme_colors(self) -> dict:
        """Return plot background, text, and grid colors for the current theme."""
        if self.theme_btn.value == "Dark":
            return {
                "plot_bgcolor": "#1e1e1e",
                "paper_bgcolor": "#1e1e1e",
                "font_color": "#e4e4e7",
                "grid_color": "#3f3f46",
            }
        return {
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "font_color": "#111827",
            "grid_color": "#e5e7eb",
        }

    def _apply_plotly_theme(self, fig) -> None:
        """Apply current theme colors to a plotly Figure in-place."""
        tc = self._plotly_theme_colors()
        fig.update_layout(
            plot_bgcolor=tc["plot_bgcolor"],
            paper_bgcolor=tc["paper_bgcolor"],
            font=dict(color=tc["font_color"]),
            xaxis=dict(gridcolor=tc["grid_color"]),
            yaxis=dict(gridcolor=tc["grid_color"]),
        )

    def _rerender_plotly_theme(self) -> None:
        """Re-render all visible Plotly charts in the updated theme."""
        if getattr(self, "_last_orb_info", None) is not None:
            self._on_orb_range_changed()
        if getattr(self, "_last_ir_freqs", None) is not None:
            self._update_ir_figure(
                self._ir_mode_toggle.value,
                self._ir_fwhm_slider.value,
            )
        if getattr(self, "_last_pes_result", None) is not None:
            self._show_pes_scan_result(self._last_pes_result)

    def _on_viz_backend_changed(self, change) -> None:
        self._viz_backend = change["new"]  # type: ignore[assignment]
        # Lighting only works with the PlotlyMol backend
        _lighting_usable = _PLOTLYMOL_VIZ and self._viz_backend == "plotlymol"
        self.viz_lighting_dd.disabled = not _lighting_usable
        self.viz_lighting_dd.layout.visibility = (
            "visible" if _lighting_usable else "hidden"
        )
        if self._molecule is not None and _display_molecule is not None:
            self.viz_output.clear_output()
            with self.viz_output:
                _display_molecule(
                    self._molecule,
                    backend=self._viz_backend,
                    style=self._viz_style,
                    lighting=self._viz_lighting,
                )

    def _on_viz_style_changed(self, change) -> None:
        self._viz_style = change["new"]
        if self._molecule is not None and _display_molecule is not None:
            self.viz_output.clear_output()
            with self.viz_output:
                _display_molecule(
                    self._molecule,
                    backend=self._viz_backend,
                    style=self._viz_style,
                    lighting=self._viz_lighting,
                )

    def _on_viz_lighting_changed(self, change) -> None:
        self._viz_lighting = change["new"]
        if self._molecule is not None and _display_molecule is not None:
            self.viz_output.clear_output()
            with self.viz_output:
                _display_molecule(
                    self._molecule,
                    backend=self._viz_backend,
                    style=self._viz_style,
                    lighting=self._viz_lighting,
                )

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
        _children = [self.mol_input_expanded, self.mol_info_html, self.viz_output]
        if self.viz_backend_toggle is not None:
            _children.append(self.viz_backend_toggle)
        if VISUALIZATION_AVAILABLE:
            _children.append(self.viz_controls_box)
        self.mol_input_container.children = _children

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
        elif ct == "Frequency":
            self._refresh_freq_seed_options()
            self.calc_extra_opts.children = [
                widgets.HBox(
                    [self._freq_seed_dd, self._freq_seed_refresh_btn],
                    layout=widgets.Layout(align_items="center", gap="6px"),
                ),
                self._freq_preopt_cb,
                self._freq_seed_note,
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
        elif ct == "NMR Shielding":
            self.calc_extra_opts.children = [
                widgets.HTML(
                    '<span style="color:#b45309;font-size:12px">'
                    "⚠ Recommended: B3LYP/6-31G* or better. "
                    "STO-3G and 3-21G give qualitative results only. "
                    "Start from an optimised geometry for best accuracy.</span>"
                ),
            ]
        elif ct == "PES Scan":
            self._update_scan_widgets()
            self.calc_extra_opts.children = [
                widgets.HBox(
                    [self._scan_type_dd],
                    layout=widgets.Layout(margin="0 0 4px 0"),
                ),
                widgets.HBox(
                    [self._scan_atom1, self._scan_atom2],
                    layout=widgets.Layout(gap="4px"),
                ),
                self._scan_atom34_box,
                widgets.HBox(
                    [
                        self._scan_start,
                        self._scan_stop,
                        self._scan_steps,
                        self._scan_unit_lbl,
                    ],
                    layout=widgets.Layout(gap="4px", align_items="center"),
                ),
            ]
        else:
            self.calc_extra_opts.children = []

    def _update_scan_widgets(self, _change=None) -> None:
        """Show/hide atom3/4 inputs and update unit label based on scan type."""
        st = self._scan_type_dd.value
        if st == "Bond":
            self._scan_atom34_box.layout.display = "none"
            self._scan_unit_lbl.value = (
                '<span style="font-size:12px;color:#555">Å</span>'
            )
        elif st == "Angle":
            self._scan_atom4.layout.display = "none"
            self._scan_atom3.layout.display = ""
            self._scan_atom34_box.layout.display = ""
            self._scan_unit_lbl.value = (
                '<span style="font-size:12px;color:#555">°</span>'
            )
        else:  # Dihedral
            self._scan_atom3.layout.display = ""
            self._scan_atom4.layout.display = ""
            self._scan_atom34_box.layout.display = ""
            self._scan_unit_lbl.value = (
                '<span style="font-size:12px;color:#555">°</span>'
            )

    def _refresh_freq_seed_options(self) -> None:
        """Populate _freq_seed_dd with saved geometry-opt results."""
        from quantui.results_storage import list_results, load_result

        options = [("(use current molecule)", "")]
        for d in list_results():
            try:
                data = load_result(d)
                if data.get("calc_type") != "geometry_opt":
                    continue
                traj_file = d / "trajectory.json"
                if not traj_file.exists():
                    continue
                ts = data.get("timestamp", d.name[:19])
                label = (
                    f"{data['formula']}  {data['method']}/{data['basis']}" f"  —  {ts}"
                )
                options.append((label, str(d)))
            except Exception:
                continue
        self._freq_seed_dd.options = options

    def _on_freq_seed_changed(self, change) -> None:
        """Enable/disable pre-opt checkbox and update the seed note."""
        path_str = change["new"]
        if path_str:
            # A history geometry is selected — pre-optimize makes no sense.
            self._freq_preopt_cb.value = False
            self._freq_preopt_cb.disabled = True
            self._freq_seed_note.value = (
                '<span style="font-size:12px;color:#16a34a">'
                "✓ Final optimised geometry will be loaded from the selected result."
                "</span>"
            )
        else:
            self._freq_preopt_cb.disabled = False
            self._freq_seed_note.value = ""

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
        self._analysis_mol_output.clear_output()
        self._viz_label.layout.display = "none"
        self._viz_label.value = ""
        self._deactivate_all_ana_panels()
        self._pes_plot_html.value = ""
        self._result_dir_label.value = ""
        self._result_dir_label.layout.display = "none"
        self._result_log_accordion.layout.display = "none"
        self._result_log_accordion.selected_index = None
        self._result_log_output.clear_output()
        self._completion_banner.layout.display = "none"
        self._to_analysis_btn.layout.display = "none"
        self._analysis_empty_html.layout.display = "none"
        threading.Thread(target=self._do_run, daemon=True).start()

    def _on_solvent_cb_changed(self, change) -> None:
        self.solvent_dd.layout.display = "" if change["new"] else "none"

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

    def _on_export_xyz(self, btn) -> None:
        if self._molecule is None:
            self.struct_export_status.value = "Load a molecule first."
            return
        try:
            mol, method, basis = self._export_molecule_and_label()
            fname = f"{mol.get_formula()}_{method}_{basis}.xyz"
            xyz_body = mol.to_xyz_string()
            full_xyz = (
                f"{len(mol.atoms)}\n{mol.get_formula()} {method}/{basis}\n{xyz_body}\n"
            )
            dest = (
                (self._last_result_dir / fname)
                if self._last_result_dir
                else Path(fname)
            )
            dest.write_text(full_xyz, encoding="utf-8")
            self.struct_export_status.value = f"Saved: {dest}"
        except Exception as exc:
            self.struct_export_status.value = f"Error: {exc}"

    def _on_export_mol(self, btn) -> None:
        if self._molecule is None:
            self.struct_export_status.value = "Load a molecule first."
            return
        try:
            from rdkit import Chem

            mol, method, basis = self._export_molecule_and_label()
            fname = f"{mol.get_formula()}_{method}_{basis}.mol"
            rdmol = self._molecule_to_rdkit(mol)
            if rdmol is None:
                self.struct_export_status.value = "RDKit could not parse the structure."
                return
            mol_block = Chem.MolToMolBlock(rdmol)
            dest = (
                (self._last_result_dir / fname)
                if self._last_result_dir
                else Path(fname)
            )
            dest.write_text(mol_block, encoding="utf-8")
            self.struct_export_status.value = f"Saved: {dest}"
        except Exception as exc:
            self.struct_export_status.value = f"Error: {exc}"

    def _on_export_pdb(self, btn) -> None:
        if self._molecule is None:
            self.struct_export_status.value = "Load a molecule first."
            return
        try:
            from rdkit import Chem

            mol, method, basis = self._export_molecule_and_label()
            fname = f"{mol.get_formula()}_{method}_{basis}.pdb"
            rdmol = self._molecule_to_rdkit(mol)
            if rdmol is None:
                self.struct_export_status.value = "RDKit could not parse the structure."
                return
            pdb_block = Chem.MolToPDBBlock(rdmol)
            dest = (
                (self._last_result_dir / fname)
                if self._last_result_dir
                else Path(fname)
            )
            dest.write_text(pdb_block, encoding="utf-8")
            self.struct_export_status.value = f"Saved: {dest}"
        except Exception as exc:
            self.struct_export_status.value = f"Error: {exc}"

    def _export_molecule_and_label(self):
        """Return (molecule, method, basis) for structure export.

        For geo opt results, returns the final optimised geometry.
        Falls back to the currently loaded molecule for all other calc types.
        """
        from quantui.optimizer import OptimizationResult

        r = self._last_result
        if isinstance(r, OptimizationResult):
            mol = r.molecule
        else:
            assert self._molecule is not None
            mol = self._molecule
        method = (
            getattr(r, "method", self.method_dd.value)
            if r is not None
            else self.method_dd.value
        )
        basis = (
            getattr(r, "basis", self.basis_dd.value)
            if r is not None
            else self.basis_dd.value
        )
        return mol, method, basis

    @staticmethod
    def _molecule_to_rdkit(mol):
        """Convert a Molecule to an RDKit Mol with inferred bonds (best-effort)."""
        try:
            from rdkit import Chem

            xyz_block = (
                f"{len(mol.atoms)}\n{mol.get_formula()}\n{mol.to_xyz_string()}\n"
            )
            rdmol = Chem.MolFromXYZBlock(xyz_block)
            if rdmol is None:
                return None
            try:
                from rdkit.Chem import rdDetermineBonds

                rdDetermineBonds.DetermineBonds(rdmol, charge=mol.charge)
            except Exception:
                pass
            return rdmol
        except Exception:
            return None

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
        valid_dirs: list = []
        for path_str in selected:
            if not path_str:
                continue
            try:
                data = load_result(Path(path_str))
                summaries.append(summary_from_saved_result(data))
                valid_dirs.append(Path(path_str))
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
            # Per-row → Analyse buttons
            if valid_dirs:
                _btns = []
                for s, rdir in zip(summaries, valid_dirs):
                    _short = f"{s.formula} {s.method}/{s.basis}"
                    _btn = widgets.Button(
                        description=f"→ Analyse  {_short}"[:48],
                        button_style="info",
                        layout=widgets.Layout(width="auto", max_width="340px"),
                        tooltip=f"Load {_short} into the Analysis tab",
                    )
                    _btn.on_click(lambda _, rd=rdir: self._history_load_analysis(rd))
                    _btns.append(_btn)
                display(
                    widgets.HTML(
                        '<p style="margin:12px 0 4px;color:#475569;'
                        'font-size:13px;font-weight:600">Analyse a result:</p>'
                    )
                )
                display(widgets.VBox(_btns, layout=widgets.Layout(gap="4px")))

    def _on_compare_clear(self, btn) -> None:
        self.compare_select.value = ()
        self.compare_output.clear_output()

    # ── History ───────────────────────────────────────────────────────────

    def _on_past_dd_changed(self, change) -> None:
        path_str = change["new"]
        # Hide result-specific panels whenever the selection changes so stale
        # content from a previous "View log" click doesn't persist.
        self._deactivate_all_ana_panels()
        self._pending_traj_result = None
        self._result_log_accordion.layout.display = "none"
        self._result_dir_label.layout.display = "none"
        self._iso_generate_btn.disabled = True
        if not path_str:
            self.past_output.clear_output()
            return
        self.past_output.clear_output()
        with self.past_output:
            try:
                from quantui import load_result

                _result_dir = Path(path_str)
                data = load_result(_result_dir)
                display(HTML(self._format_past_result(data, result_dir=_result_dir)))
                _btn_res = widgets.Button(
                    description="→ View Results",
                    button_style="success",
                    layout=widgets.Layout(width="130px"),
                    tooltip="Show this result in the Results tab",
                )
                _btn_ana = widgets.Button(
                    description="→ View Analysis",
                    button_style="info",
                    layout=widgets.Layout(width="140px"),
                    tooltip="Load analysis panels and navigate to the Analysis tab",
                )
                _btn_res.on_click(
                    lambda _, d=data, rd=_result_dir: self._history_load_results(d, rd)
                )
                _btn_ana.on_click(
                    lambda _, rd=_result_dir: self._history_load_analysis(rd)
                )
                display(
                    widgets.HBox(
                        [_btn_res, _btn_ana],
                        layout=widgets.Layout(gap="8px", margin="6px 0 0"),
                    )
                )
            except Exception as exc:
                print(f"Could not load result: {exc}")

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
        import types

        path_str = self.past_dd.value
        if not path_str:
            return
        result_dir = Path(path_str)

        # Read log text
        log_path = result_dir / "pyscf.log"
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            label = result_dir.name
        else:
            text = "(No pyscf.log found for this result.)"
            label = ""

        self._deactivate_all_ana_panels()
        self._pending_traj_result = None

        # Populate inline log view and navigate to the Output tab
        self._update_log_panel(text, label)
        # Full Output Log accordion — always available
        self._show_result_log(result_dir, text)

        # Load result.json to reconstruct calc-type-specific panels
        try:
            from quantui import load_result
            from quantui.results_storage import load_orbitals, load_trajectory

            data = load_result(result_dir)
            calc_type = data.get("calc_type", "")
            formula = data.get("formula", "")

            # Orbital diagram — available for SP and Geo Opt (orbitals.npz)
            if calc_type in ("single_point", "geometry_opt"):
                try:
                    orb = load_orbitals(result_dir)
                    orb.formula = formula
                    self._show_orbital_diagram(orb)
                except Exception:
                    pass

            # Geometry opt → trajectory carousel (if trajectory.json was saved)
            if calc_type == "geometry_opt":
                traj_file = result_dir / "trajectory.json"
                if traj_file.exists():
                    try:
                        traj, energies = load_trajectory(result_dir)
                        if len(traj) >= 2:
                            stub = types.SimpleNamespace(
                                trajectory=traj,
                                energies_hartree=energies,
                                formula=formula,
                            )
                            self._pending_traj_result = stub
                            self._activate_ana_panel("Trajectory")
                    except Exception:
                        pass

            # Frequency → IR spectrum + vibrational mode viewer
            elif calc_type == "frequency":
                ir = data.get("spectra", {}).get("ir", {})
                mol_data = data.get("spectra", {}).get("molecule", {})
                freqs = ir.get("frequencies_cm1")
                ints = ir.get("ir_intensities")
                displacements = ir.get("displacements")

                if freqs and ints:
                    freq_stub = types.SimpleNamespace(
                        frequencies_cm1=freqs,
                        ir_intensities=ints,
                        displacements=displacements,
                    )
                    self._show_ir_spectrum(freq_stub)

                    # Vibrational mode viewer needs displacements + molecule geometry
                    if displacements and mol_data.get("atoms"):
                        from quantui.molecule import Molecule as _Mol

                        hist_mol = _Mol(
                            atoms=mol_data["atoms"],
                            coordinates=mol_data["coords"],
                            charge=mol_data.get("charge", 0),
                            multiplicity=mol_data.get("multiplicity", 1),
                        )
                        self._show_vib_animation(freq_stub, hist_mol)
        except Exception:
            pass

        # Update Analysis tab context for the loaded result
        try:
            _hist_formula = data.get("formula", result_dir.name)  # type: ignore[possibly-undefined]
            _hist_method = data.get("method", "")
            _hist_basis = data.get("basis", "")
            _hist_ct = data.get("calc_type", "")
            _hist_label = (
                f"{_hist_formula}  {_hist_method}/{_hist_basis}"
                if _hist_method
                else _hist_formula
            )
            self._analysis_context_lbl.value = (
                f'<p style="color:#555;font-size:13px;margin:4px 0 12px">'
                f"Analysing: {_hist_label} (from History)</p>"
            )
            _hist_has_analysis = _hist_ct in (
                "single_point",
                "geometry_opt",
                "frequency",
            )
            self._to_analysis_btn.layout.display = "" if _hist_has_analysis else "none"
            self._analysis_empty_html.layout.display = (
                "none" if _hist_has_analysis else ""
            )
            # Mirror structure into Analysis tab viewer
            try:
                _mol = self._mol_from_result_dir(result_dir, data)
                if _mol is not None:
                    self._show_result_3d(_mol, extra_output=self._analysis_mol_output)
                else:
                    self._analysis_mol_output.clear_output()
            except Exception:
                pass
        except Exception:
            pass

        self._goto_output_tab()

    def _mol_from_result_dir(self, result_dir: Path, data: dict):
        """Try to reconstruct a displayable Molecule from a saved result directory.

        Returns a Molecule or None if geometry data is not available.
        Tries sources in order: frequency spectra → orbitals_meta → trajectory.
        """
        import json as _json

        from quantui.molecule import Molecule

        ct = data.get("calc_type", "")

        # Frequency: geometry stored inside spectra.molecule
        if ct == "frequency":
            mol_data = data.get("spectra", {}).get("molecule", {})
            if mol_data.get("atoms") and mol_data.get("coords"):
                try:
                    return Molecule(
                        atoms=mol_data["atoms"],
                        coordinates=mol_data["coords"],
                        charge=mol_data.get("charge", 0),
                        multiplicity=mol_data.get("multiplicity", 1),
                    )
                except Exception:
                    pass

        # Single point / Geo opt: atom list from orbitals_meta.json
        meta_path = result_dir / "orbitals_meta.json"
        if meta_path.exists():
            try:
                meta = _json.loads(meta_path.read_text())
                mol_atom = meta.get("mol_atom")
                if mol_atom:
                    atoms = [sym for sym, _ in mol_atom]
                    coords = [c for _, c in mol_atom]
                    return Molecule(atoms=atoms, coordinates=coords)
            except Exception:
                pass

        # Geo opt fallback: last step of trajectory.json
        if ct == "geometry_opt":
            traj_path = result_dir / "trajectory.json"
            if traj_path.exists():
                try:
                    traj_data = _json.loads(traj_path.read_text())
                    steps = traj_data.get("steps", [])
                    if steps:
                        return Molecule(
                            atoms=traj_data["atoms"],
                            coordinates=steps[-1]["coords"],
                            charge=traj_data.get("charge", 0),
                            multiplicity=traj_data.get("multiplicity", 1),
                        )
                except Exception:
                    pass

        return None

    def _history_load_results(self, data: dict, result_dir: Path) -> None:
        """Display a history result card in the Results tab and navigate there."""
        self.result_output.clear_output()
        with self.result_output:
            display(HTML(self._format_past_result(data, result_dir=result_dir)))
        self._result_dir_label.layout.display = "none"
        # Also show 3D structure if geometry is recoverable
        mol = self._mol_from_result_dir(result_dir, data)
        if mol is not None:
            self._show_result_3d(mol)
        self.root_tab.selected_index = 1

    def _history_load_analysis(self, result_dir: Path) -> None:
        """Load analysis panels for a history result and navigate to Analysis tab."""
        # Reuse _on_view_log machinery but navigate to Analysis instead of Log.
        import types

        log_path = result_dir / "pyscf.log"
        text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else "(No pyscf.log found for this result.)"
        )
        label = result_dir.name if log_path.exists() else ""

        self._deactivate_all_ana_panels()
        self._pending_traj_result = None
        self._update_log_panel(text, label)
        self._show_result_log(result_dir, text)

        try:
            from quantui import load_result
            from quantui.results_storage import load_orbitals, load_trajectory

            data = load_result(result_dir)
            calc_type = data.get("calc_type", "")
            formula = data.get("formula", "")

            if calc_type in ("single_point", "geometry_opt"):
                try:
                    orb = load_orbitals(result_dir)
                    orb.formula = formula
                    self._show_orbital_diagram(orb)
                except Exception:
                    pass

            if calc_type == "geometry_opt":
                traj_file = result_dir / "trajectory.json"
                if traj_file.exists():
                    try:
                        traj, energies = load_trajectory(result_dir)
                        if len(traj) >= 2:
                            stub = types.SimpleNamespace(
                                trajectory=traj,
                                energies_hartree=energies,
                                formula=formula,
                            )
                            self._pending_traj_result = stub
                            self._activate_ana_panel("Trajectory")
                    except Exception:
                        pass

            elif calc_type == "frequency":
                ir = data.get("spectra", {}).get("ir", {})
                mol_data = data.get("spectra", {}).get("molecule", {})
                freqs = ir.get("frequencies_cm1")
                ints = ir.get("ir_intensities")
                displacements = ir.get("displacements")
                if freqs and ints:
                    freq_stub = types.SimpleNamespace(
                        frequencies_cm1=freqs,
                        ir_intensities=ints,
                        displacements=displacements,
                    )
                    self._show_ir_spectrum(freq_stub)
                    if displacements and mol_data.get("atoms"):
                        from quantui.molecule import Molecule as _Mol

                        hist_mol = _Mol(
                            atoms=mol_data["atoms"],
                            coordinates=mol_data["coords"],
                            charge=mol_data.get("charge", 0),
                            multiplicity=mol_data.get("multiplicity", 1),
                        )
                        self._show_vib_animation(freq_stub, hist_mol)

            _has = calc_type in ("single_point", "geometry_opt", "frequency")
            _label = (
                f'{formula}  {data.get("method","")}/{data.get("basis","")}'
                if data.get("method")
                else formula
            )
            self._analysis_context_lbl.value = (
                f'<p style="color:#555;font-size:13px;margin:4px 0 12px">'
                f"Analysing: {_label} (from History)</p>"
            )
            self._to_analysis_btn.layout.display = "" if _has else "none"
            self._analysis_empty_html.layout.display = "none" if _has else ""
            # Mirror structure into Analysis tab viewer
            try:
                _mol = self._mol_from_result_dir(result_dir, data)
                if _mol is not None:
                    self._show_result_3d(_mol, extra_output=self._analysis_mol_output)
                else:
                    self._analysis_mol_output.clear_output()
            except Exception:
                pass
        except Exception:
            pass

        self.root_tab.selected_index = 2

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

    # ── Calibration ───────────────────────────────────────────────────────

    def _on_cal_run(self, btn) -> None:
        import threading as _threading

        self._cal_stop_event = _threading.Event()
        self._cal_run_btn.disabled = True
        self._cal_stop_btn.layout.display = ""
        self._cal_progress.value = 0
        self._cal_progress.layout.display = ""
        self._cal_step_label.layout.display = ""
        self._cal_step_label.value = (
            '<span style="font-size:12px;color:#475569">Starting…</span>'
        )
        self._cal_results_html.value = ""

        _threading.Thread(target=self._do_calibration, daemon=True).start()

    def _on_cal_stop(self, btn) -> None:
        if hasattr(self, "_cal_stop_event"):
            self._cal_stop_event.set()

    def _do_calibration(self) -> None:
        from quantui.benchmarks import run_calibration

        def _progress(
            step_n: int, total: int, label: str, status: str, elapsed: float
        ) -> None:
            _icon = {"ok": "✓", "timed_out": "⏱", "stopped": "⛔", "error": "✗"}.get(
                status, "?"
            )
            self._cal_progress.value = step_n
            self._cal_step_label.value = (
                f'<span style="font-size:12px;color:#475569">'
                f"Step {step_n} / {total} — {label} "
                f"[{_icon} {elapsed:.1f} s]</span>"
            )

        result = run_calibration(
            progress_cb=_progress,
            stop_event=self._cal_stop_event,
            timeout_per_step=120.0,
        )

        # Render results table
        _rows = "".join(
            f"<tr>"
            f'<td style="padding:2px 12px 2px 0;font-size:12px">{s.label}</td>'
            f'<td style="padding:2px 12px 2px 0;font-size:12px;text-align:right">'
            f"{s.n_electrons}</td>"
            f'<td style="padding:2px 12px 2px 0;font-size:12px;text-align:right">'
            f"{s.elapsed_s:.2f} s</td>"
            f'<td style="padding:2px 0;font-size:12px">'
            f'{"✓" if s.status == "ok" else ("⏱ timed out" if s.status == "timed_out" else ("⛔ stopped" if s.status == "stopped" else "✗ error"))}'
            f"</td>"
            f"</tr>"
            for s in result.steps
        )
        _summary = f"Completed {result.n_completed} / {result.n_total} steps." + (
            " (stopped early)" if result.stopped_early else ""
        )
        self._cal_results_html.value = (
            f'<div style="margin-top:8px">'
            f'<p style="font-size:13px;color:#374151;margin:0 0 6px">{_summary}</p>'
            f'<table style="border-collapse:collapse">'
            f"<tr>"
            f'<th style="padding:2px 12px 2px 0;font-size:12px;text-align:left">Calculation</th>'
            f'<th style="padding:2px 12px 2px 0;font-size:12px;text-align:right">Electrons</th>'
            f'<th style="padding:2px 12px 2px 0;font-size:12px;text-align:right">Wall time</th>'
            f'<th style="padding:2px 0;font-size:12px">Status</th>'
            f"</tr>"
            f"{_rows}</table></div>"
        )

        self._cal_step_label.value = (
            '<span style="font-size:12px;color:#16a34a"><b>Calibration complete.</b> '
            "Time estimates are now active.</span>"
            if result.n_completed > 0
            else '<span style="font-size:12px;color:#dc2626">No steps completed.</span>'
        )
        self._cal_stop_btn.layout.display = "none"
        self._cal_run_btn.disabled = not _PYSCF_AVAILABLE
        self._refresh_perf_stats()

    # ── Output log ────────────────────────────────────────────────────────

    def _on_log_clear(self, btn) -> None:
        self._log_output_html.value = (
            '<span style="color:#94a3b8;font-size:13px">Log cleared.</span>'
        )
        self._log_source_lbl.value = ""

    # ── Help ──────────────────────────────────────────────────────────────

    def _on_help_toggle(self, _=None) -> None:
        visible = self.help_tab_panel.layout.display != "none"
        self.help_tab_panel.layout.display = "none" if visible else ""

    def _on_help_topic_changed(self, change=None) -> None:
        self._render_help_topic()

    # ══ LOGIC METHODS ════════════════════════════════════════════════════════

    def _set_molecule(self, mol: Molecule, label: str = "") -> None:
        """Update shared state and refresh dependent widgets."""
        self._molecule = mol
        self.run_btn.disabled = False
        self.export_btn.disabled = False
        self.export_xyz_btn.disabled = False
        self.export_mol_btn.disabled = not _RDKIT_AVAILABLE
        self.export_pdb_btn.disabled = not _RDKIT_AVAILABLE

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
                _display_molecule(
                    mol,
                    backend=self._viz_backend,
                    style=self._viz_style,
                    lighting=self._viz_lighting,
                )

        self._update_notes()

        # Advance step indicator
        if self.step_progress._states[2] != "active":
            if self.step_progress._states[2] in ("done", "fail"):
                self.step_progress.reset()
            self.step_progress.complete(0)
            self.step_progress.start(1)

        self._update_estimate()

        # Collapse molecule input to compact view
        _collapsed_children = [self.mol_input_collapsed, self.viz_output]
        if self.viz_backend_toggle is not None:
            _collapsed_children.append(self.viz_backend_toggle)
        if VISUALIZATION_AVAILABLE:
            _collapsed_children.append(self.viz_controls_box)
        self.mol_input_container.children = _collapsed_children

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

    def _show_result_3d(self, molecule, extra_output=None) -> None:
        """Render molecule 3D structure in the result visualization panel.

        Renders into ``result_viz_output`` and, if supplied, into *extra_output*
        as well (used to mirror the structure into the Analysis tab viewer).
        Safe to call from a background thread — uses ``with output:`` context.
        """
        if _display_molecule is None or molecule is None:
            return
        for _out in [self.result_viz_output, extra_output]:
            if _out is None:
                continue
            _out.clear_output()
            with _out:
                _display_molecule(
                    molecule,
                    backend=self._viz_backend,
                    style=self._viz_style,
                    lighting=self._viz_lighting,
                )

    def _show_result_log(self, saved_dir: Path, log_text: str) -> None:
        """Populate the result-directory label and output-log accordion.

        Safe to call from a background thread.
        """
        # Path label
        self._result_dir_label.value = (
            f'<span style="font-size:12px;color:#555;font-family:monospace">'
            f"Saved to: {saved_dir}</span>"
        )
        self._result_dir_label.layout.display = ""

        # Log accordion — prefer on-disk file (written by save_result) over in-memory string
        import html as _html_mod

        _log_path = saved_dir / "pyscf.log"
        try:
            log_content = _log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_content = log_text

        if not log_content.strip():
            log_content = "(No output captured for this calculation.)"

        self._result_log_output.clear_output()
        with self._result_log_output:
            display(
                HTML(
                    f'<pre style="font-size:11px;max-height:400px;overflow-y:auto;'
                    f'white-space:pre-wrap;word-break:break-all;margin:0;padding:6px">'
                    f"{_html_mod.escape(log_content)}</pre>"
                )
            )
        self._result_log_accordion.layout.display = ""

    def _on_traj_expand(self, change) -> None:
        """Lazily generate the trajectory animation when the accordion is first opened."""
        if change["new"] != 0:
            return
        result = self._pending_traj_result
        if result is None:
            return
        self._pending_traj_result = None

        from IPython.display import HTML as _H
        from IPython.display import display as _d

        self.traj_output.clear_output()
        with self.traj_output:
            _d(
                _H(
                    '<p style="color:#555;font-style:italic;padding:8px">Loading trajectory viewer…</p>'
                )
            )

        def _render():
            try:
                self._show_opt_trajectory(result)
            except Exception as exc:
                from IPython.display import HTML as _H2
                from IPython.display import display as _d2

                self.traj_output.clear_output()
                with self.traj_output:
                    _d2(
                        _H2(
                            f'<p style="color:#b91c1c;padding:8px">⚠ Trajectory rendering failed: {exc}</p>'
                        )
                    )

        threading.Thread(target=_render, daemon=True).start()

    def _show_opt_trajectory(self, opt_result) -> None:
        """Build the trajectory carousel and energy chart in the trajectory panel.

        Shows a step slider for flipping through frames and an energy-convergence
        chart.  An Export button generates a standalone HTML animation file on demand.
        Safe to call from a background thread.

        When plotlymol is available:
        - Bond perception runs once on frame 0 (RDKit DetermineConnectivity is slow).
        - All remaining frames are pre-rendered in a background thread pool so
          slider navigation is instant after a few seconds.
        """
        import concurrent.futures

        from IPython.display import display as _ipy_display

        # Support both OptimizationResult (.trajectory) and PESScanResult (.coordinates_list)
        traj = getattr(opt_result, "trajectory", None) or getattr(
            opt_result, "coordinates_list", []
        )
        energies = opt_result.energies_hartree
        n = len(traj)
        if n < 2:
            return

        _HARTREE_TO_KCAL = 627.5094740631
        e0 = energies[0] if energies else 0.0
        rel_e = [(e - e0) * _HARTREE_TO_KCAL for e in energies] if energies else []

        # --- Energy convergence chart ---
        _has_plotly = False
        try:
            import plotly.graph_objects as go

            energy_fig = go.Figure(
                go.Scatter(
                    x=list(range(n)),
                    y=rel_e,
                    mode="lines+markers",
                    name="ΔE",
                    line=dict(color="#2563eb", width=2),
                    marker=dict(size=6),
                )
            )
            energy_fig.update_layout(
                title="Energy Convergence",
                xaxis_title="Step",
                yaxis_title="ΔE (kcal/mol)",
                height=220,
                margin=dict(l=60, r=20, t=40, b=40),
            )
            _has_plotly = True
        except ImportError:
            pass

        # --- Pre-build XYZ blocks (reused by carousel, fast path, and export) ---
        _charge = traj[0].charge
        _xyzblocks = [
            f"{len(m.atoms)}\n{m.get_formula()}\n{m.to_xyz_string()}" for m in traj
        ]
        _FRAME_W, _FRAME_H, _FRAME_RES = 460, 340, 8

        # --- Attempt to set up fast-path: bond perception once on frame 0 ---
        # draw_3D_mol accepts a pre-parsed RDKit mol and skips bond perception,
        # so we only pay that cost for the first frame instead of every frame.
        _ref_mol = None
        _plotlymol_fast = False
        try:
            from plotlymol3d import (
                draw_3D_mol as _draw_3D_mol,
            )
            from plotlymol3d import (
                format_figure as _fmt_fig,
            )
            from plotlymol3d import (
                format_lighting as _fmt_light,
            )
            from plotlymol3d import (
                make_subplots as _make_subplots,
            )
            from plotlymol3d import (
                xyzblock_to_rdkitmol as _xyz_to_rdkit,
            )
            from rdkit import Chem as _Chem

            from quantui.visualization_py3dmol import LIGHTING_PRESETS as _LP

            _ref_mol = _xyz_to_rdkit(_xyzblocks[0], charge=_charge)
            _plotlymol_fast = _ref_mol is not None
        except Exception:
            pass

        def _build_fig_fast(idx: int):
            """Reuse frame-0 bond topology; only swap in new atom positions."""
            mol_xyz = _Chem.MolFromXYZBlock(_xyzblocks[idx] + "\n")
            if mol_xyz is None:
                return None
            rw = _Chem.RWMol(_ref_mol)
            conf_src = mol_xyz.GetConformer()
            conf_dst = rw.GetConformer()
            for atom_idx in range(rw.GetNumAtoms()):
                conf_dst.SetAtomPosition(atom_idx, conf_src.GetAtomPosition(atom_idx))
            fig = _make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
            _draw_3D_mol(fig, rw.GetMol(), _FRAME_RES, "ball+stick")
            fig = _fmt_fig(fig)
            fig = _fmt_light(fig, **_LP.get("soft", _LP["soft"]))
            _bg = self._plotly_theme_colors()["paper_bgcolor"]
            fig.update_layout(
                width=_FRAME_W,
                height=_FRAME_H,
                paper_bgcolor=_bg,
                scene=dict(bgcolor=_bg),
                margin=dict(l=0, r=0, t=0, b=0),
            )
            return fig

        def _build_fig(idx: int):
            """Return (kind, obj) for frame idx; fast path when bonds are cached."""
            if _plotlymol_fast:
                try:
                    fig = _build_fig_fast(idx)
                    if fig is not None:
                        return ("plotly", fig)
                except Exception:
                    pass
            # Slow fallback: full plotlymol pipeline
            try:
                from quantui.visualization_py3dmol import visualize_molecule_plotlymol

                fig = visualize_molecule_plotlymol(
                    traj[idx],
                    mode="ball+stick",
                    resolution=_FRAME_RES,
                    width=_FRAME_W,
                    height=_FRAME_H,
                )
                _bg = self._plotly_theme_colors()["paper_bgcolor"]
                fig.update_layout(paper_bgcolor=_bg, scene=dict(bgcolor=_bg))
                return ("plotly", fig)
            except ImportError:
                pass
            # Last resort: py3Dmol
            try:
                import py3Dmol as _p3d

                view = _p3d.view(width=_FRAME_W, height=_FRAME_H)
                view.addModel(_xyzblocks[idx], "xyz")
                view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
                view.setBackgroundColor(
                    "white" if self.theme_btn.value == "Light" else "#1e1e1e"
                )
                view.zoomTo()
                return ("py3dmol", view)
            except Exception as exc:
                return ("error", str(exc))

        _frame_cache: dict = {}

        # --- Carousel controls ---
        _step_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=n - 1,
            description="Step:",
            continuous_update=False,
            style={"description_width": "40px"},
            layout=widgets.Layout(width="360px"),
        )
        _step_info = widgets.HTML(value=self._traj_step_html(0, traj, energies, rel_e))
        _frame_out = widgets.Output(layout=widgets.Layout(min_height="340px"))
        _cache_label = widgets.HTML(
            value=f'<span style="color:#888;font-size:11px;font-style:italic">'
            f"Pre-rendering frames… 0 / {n}</span>"
        )

        def _display_frame(idx: int) -> None:
            kind, obj = _frame_cache[idx]
            _frame_out.clear_output()
            with _frame_out:
                if kind == "error":
                    _ipy_display(
                        HTML(
                            f'<p style="color:#b91c1c;padding:8px">Frame render failed: {obj}</p>'
                        )
                    )
                else:
                    _ipy_display(obj)

        def _update_frame(change) -> None:
            idx = change["new"]
            _step_info.value = self._traj_step_html(idx, traj, energies, rel_e)
            if idx in _frame_cache:
                _display_frame(idx)
                return
            _frame_out.clear_output()
            with _frame_out:
                _ipy_display(
                    HTML(
                        '<p style="color:#555;font-style:italic;padding:8px">Rendering…</p>'
                    )
                )

            def _on_demand():
                try:
                    _frame_cache[idx] = _build_fig(idx)
                    _display_frame(idx)
                except Exception as exc:
                    _frame_out.clear_output()
                    with _frame_out:
                        _ipy_display(
                            HTML(
                                f'<p style="color:#b91c1c;padding:8px">Frame render failed: {exc}</p>'
                            )
                        )

            threading.Thread(target=_on_demand, daemon=True).start()

        _step_slider.observe(_update_frame, names="value")

        # --- Export button ---
        _export_btn = widgets.Button(
            description="Export Animation",
            icon="download",
            layout=widgets.Layout(width="160px", margin="0 0 0 12px"),
            tooltip="Generate a standalone HTML animation file (may take a minute)",
        )
        _export_status = widgets.HTML()

        def _on_export(_btn):
            _btn.disabled = True
            _export_status.value = (
                f'<span style="color:#555;font-style:italic">'
                f"Generating {n}-frame animation, please wait…</span>"
            )

            def _do_export():
                try:
                    from plotlymol3d import create_trajectory_animation

                    anim_fig = create_trajectory_animation(
                        xyzblocks=_xyzblocks,
                        energies_hartree=energies if energies else None,
                        charge=_charge,
                        mode="ball+stick",
                        resolution=12,
                        title=f"Geo Opt: {opt_result.formula}",
                    )
                    _result_dir = getattr(self, "_last_result_dir", None)
                    out_path = (
                        _result_dir / "trajectory_animation.html"
                        if _result_dir is not None
                        else Path.home() / f"{opt_result.formula}_trajectory.html"
                    )
                    anim_fig.write_html(str(out_path))
                    _export_status.value = (
                        f'<span style="color:#16a34a;font-size:12px">'
                        f"✓ Saved: {out_path}</span>"
                    )
                except Exception as exc:
                    _export_status.value = (
                        f'<span style="color:#b91c1c">Export failed: {exc}</span>'
                    )
                finally:
                    _btn.disabled = False

            threading.Thread(target=_do_export, daemon=True).start()

        _export_btn.on_click(_on_export)

        # --- Assemble layout ---
        _header = widgets.HBox(
            [_step_slider, _export_btn],
            layout=widgets.Layout(align_items="center", margin="4px 0"),
        )
        _panel = widgets.VBox(
            [_header, _step_info, _cache_label, _frame_out, _export_status]
        )

        # Display panel immediately — clears the “Loading…” message right away.
        self.traj_output.clear_output()
        with self.traj_output:
            if _has_plotly and rel_e:
                _ipy_display(energy_fig)
            _ipy_display(_panel)

        # Show placeholder while frame 0 renders in the background.
        _frame_out.clear_output()
        with _frame_out:
            _ipy_display(
                HTML(
                    '<p style="color:#555;font-style:italic;padding:8px">'
                    "Rendering frame 0…</p>"
                )
            )

        # Render all frames (0 first, then 1+) in a background thread.
        def _prerender_all() -> None:
            try:
                _frame_cache[0] = _build_fig(0)
                _display_frame(0)
                _cache_label.value = (
                    f'<span style="color:#888;font-size:11px;font-style:italic">'
                    f"Pre-rendering frames… 1 / {n}</span>"
                )
                if n > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        futures = {pool.submit(_build_fig, i): i for i in range(1, n)}
                        done = 1
                        for fut in concurrent.futures.as_completed(futures):
                            i = futures[fut]
                            try:
                                _frame_cache[i] = fut.result()
                            except Exception:
                                pass
                            done += 1
                            _cache_label.value = (
                                f'<span style="color:#888;font-size:11px;font-style:italic">'
                                f"Pre-rendering frames… {done} / {n}</span>"
                            )
            except Exception:
                pass
            _cache_label.value = (
                f'<span style="color:#16a34a;font-size:11px">'
                f"✓ All {n} frames ready</span>"
            )

        threading.Thread(target=_prerender_all, daemon=True).start()

    def _traj_step_html(self, step: int, traj, energies, rel_e) -> str:
        """One-line info label for the given trajectory step index."""
        n = len(traj)
        mol = traj[step]
        e_abs = f"{energies[step]:.8f} Ha" if energies and step < len(energies) else "—"
        delta = (
            f" &nbsp;·&nbsp; ΔE = {rel_e[step]:+.3f} kcal/mol"
            if rel_e and step < len(rel_e)
            else ""
        )
        return (
            f'<span style="font-size:12px;color:#666">'
            f"Step {step} / {n - 1} &nbsp;·&nbsp; {mol.get_formula()}"
            f" &nbsp;·&nbsp; E = {e_abs}{delta}</span>"
        )

    def _render_traj_frame(self, molecule, output_widget) -> None:
        """Render a single trajectory frame into output_widget (thread-safe).

        Tries plotlymol first, falls back to py3Dmol.
        """
        try:
            from quantui.visualization_py3dmol import visualize_molecule_plotlymol

            fig = visualize_molecule_plotlymol(
                molecule, mode="ball+stick", resolution=8, width=460, height=340
            )
            _bg = self._plotly_theme_colors()["paper_bgcolor"]
            fig.update_layout(paper_bgcolor=_bg, scene=dict(bgcolor=_bg))
            output_widget.clear_output()
            with output_widget:
                display(fig)
            return
        except ImportError:
            pass

        # Fallback: py3Dmol
        try:
            import py3Dmol as _p3d

            xyz = (
                f"{len(molecule.atoms)}\n"
                f"{molecule.get_formula()}\n"
                f"{molecule.to_xyz_string()}"
            )
            view = _p3d.view(width=460, height=340)
            view.addModel(xyz, "xyz")
            view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
            view.setBackgroundColor("white")
            view.zoomTo()
            output_widget.clear_output()
            with output_widget:
                display(view)
        except Exception as exc:
            output_widget.clear_output()
            with output_widget:
                display(
                    HTML(
                        f'<p style="color:#b91c1c;padding:8px">Frame render failed: {exc}</p>'
                    )
                )

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

        # Show loading indicator and render in a background thread so _do_run
        # is not blocked while the animation is generated (can take several seconds).
        _first_label, _first_mode = options[0]
        self.vib_output.clear_output()
        with self.vib_output:
            display(
                HTML(
                    f'<p style="color:#555;font-style:italic;padding:8px">'
                    f"⏳ Rendering vibrational animation ({_first_label})…</p>"
                )
            )
        threading.Thread(
            target=self._render_vib_mode,
            args=(vib_data, molecule, _first_mode),
            daemon=True,
        ).start()

        # Reveal the accordion (auto-open so the animation is visible).
        self._activate_ana_panel("Vibrational")

    def _show_ir_spectrum(self, freq_result) -> None:
        """Populate and reveal the IR Spectrum accordion after a Frequency result."""
        freqs = list(freq_result.frequencies_cm1 or [])
        ints = list(freq_result.ir_intensities or [])
        if not freqs or not ints:
            return

        # Store for callbacks
        self._last_ir_freqs = freqs
        self._last_ir_ints = ints

        self._update_ir_figure("Stick", 20.0)

        # Wire callbacks (replace any prior bindings)
        self._ir_mode_toggle.unobserve_all()
        self._ir_fwhm_slider.unobserve_all()

        def _on_mode(change) -> None:
            mode = change["new"]
            self._ir_fwhm_slider.layout.display = "" if mode == "Broadened" else "none"
            self._update_ir_figure(mode, self._ir_fwhm_slider.value)

        def _on_fwhm(change) -> None:
            if self._ir_mode_toggle.value == "Broadened":
                self._update_ir_figure("Broadened", change["new"])

        self._ir_mode_toggle.observe(_on_mode, names="value")
        self._ir_fwhm_slider.observe(_on_fwhm, names="value")

        # Reset toggle/slider to defaults
        self._ir_mode_toggle.value = "Stick"
        self._ir_fwhm_slider.value = 20.0
        self._ir_fwhm_slider.layout.display = "none"

        self._activate_ana_panel("IR Spectrum")

    def _update_ir_figure(self, mode: str, fwhm: float) -> None:
        """Re-render the IR spectrum chart for the given mode and FWHM."""
        try:
            import plotly.io as _pio

            from quantui.ir_plot import plot_ir_spectrum

            fig = plot_ir_spectrum(
                self._last_ir_freqs,
                self._last_ir_ints,
                mode=mode.lower(),
                fwhm=fwhm,
            )
            self._apply_plotly_theme(fig)
            self._ir_fig.value = _pio.to_html(
                fig, include_plotlyjs="cdn", full_html=False
            )
        except Exception:
            pass

    def _show_orbital_diagram(self, result) -> None:
        """Build and reveal the interactive orbital diagram accordion."""
        mo_energy = getattr(result, "mo_energy_hartree", None)
        mo_occ = getattr(result, "mo_occ", None)
        if mo_energy is None or mo_occ is None:
            return

        try:
            from quantui.orbital_visualization import orbital_info_from_arrays

            info = orbital_info_from_arrays(mo_energy, mo_occ, formula=result.formula)
        except Exception:
            return

        self._last_orb_info = info
        self._last_orb_mo_coeff = getattr(result, "mo_coeff", None)
        self._last_orb_mol_atom = getattr(result, "pyscf_mol_atom", None)
        self._last_orb_mol_basis = getattr(result, "pyscf_mol_basis", None)

        _plotly_rendered = False
        try:
            import plotly.io as _pio

            from quantui.orbital_visualization import plot_orbital_diagram_plotly

            fig = plot_orbital_diagram_plotly(
                info, max_orbitals=self._orb_n_orb_input.value
            )
            # Sync axis limit controls to auto-computed range
            yr = fig.layout.yaxis.range
            if yr is not None:
                self._orb_ymin_input.value = round(float(yr[0]), 2)
                self._orb_ymax_input.value = round(float(yr[1]), 2)
            self._apply_plotly_theme(fig)
            html_str = _pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
            self._orb_diagram_html.value = html_str
            _plotly_rendered = True
        except Exception:
            pass

        if not _plotly_rendered:
            # Fallback: static matplotlib PNG (plotly not installed)
            import base64
            import io as _io

            try:
                from matplotlib.backends.backend_agg import (
                    FigureCanvasAgg as _AggCanvas,
                )

                from quantui.orbital_visualization import plot_orbital_diagram

                mpl_fig = plot_orbital_diagram(info)
                _AggCanvas(mpl_fig)
                buf = _io.BytesIO()
                mpl_fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode()
                self._orb_diagram_html.value = (
                    f'<img src="data:image/png;base64,{img_b64}" '
                    'style="max-width:100%;height:auto" />'
                )
            except Exception:
                pass

        if (
            self._last_orb_mo_coeff is not None
            and self._last_orb_mol_atom is not None
            and self._last_orb_mol_basis is not None
        ):
            self._orb_iso_output.clear_output()
            self._orb_toggle.value = "HOMO"
            self._orb_iso_controls.layout.display = ""
            self._iso_generate_btn.disabled = False
        else:
            self._orb_iso_controls.layout.display = "none"
            self._iso_generate_btn.disabled = True

        self._activate_ana_panel("Orbitals")
        if not self._iso_generate_btn.disabled:
            self._activate_ana_panel("Isosurface", auto_select=False)

    def _on_iso_generate(self, btn) -> None:
        """Generate an orbital isosurface for the currently selected orbital."""
        orbital_label = self._orb_toggle.value
        btn.disabled = True
        btn.description = "Generating…"
        self._orb_iso_output.clear_output()
        with self._orb_iso_output:
            display(
                HTML(
                    f'<p style="color:#555;font-style:italic;padding:4px 0">'
                    f"⏳ Generating {orbital_label} cube file and rendering isosurface"
                    f" — this may take 15–30 s…</p>"
                )
            )

        def _run():
            try:
                self._render_orbital_isosurface(orbital_label)
            finally:
                btn.disabled = False
                btn.description = "Generate Isosurface"

        threading.Thread(target=_run, daemon=True).start()

    def _on_orb_range_changed(self, _change=None) -> None:
        """Live-update the orbital diagram when axis limits or orbital count changes."""
        info = getattr(self, "_last_orb_info", None)
        if info is None:
            return
        ymin = self._orb_ymin_input.value
        ymax = self._orb_ymax_input.value
        if ymin >= ymax:
            return
        try:
            import plotly.io as _pio

            from quantui.orbital_visualization import plot_orbital_diagram_plotly

            fig = plot_orbital_diagram_plotly(
                info,
                max_orbitals=self._orb_n_orb_input.value,
                yrange=(ymin, ymax),
            )
            self._apply_plotly_theme(fig)
            self._orb_diagram_html.value = _pio.to_html(
                fig, include_plotlyjs="cdn", full_html=False
            )
        except Exception:
            pass

    def _render_orbital_isosurface(self, orbital_label: str) -> None:
        """Generate a cube file and render an orbital isosurface (Linux/WSL only)."""
        import tempfile

        orb_info = getattr(self, "_last_orb_info", None)
        if orb_info is None:
            return

        n_occ = orb_info.n_occupied
        n_total = len(orb_info.mo_energies_ev)
        _idx_map = {
            "HOMO-1": n_occ - 2,
            "HOMO": n_occ - 1,
            "LUMO": n_occ,
            "LUMO+1": n_occ + 1,
        }
        orb_idx = _idx_map.get(orbital_label)
        if orb_idx is None or orb_idx < 0 or orb_idx >= n_total:
            return

        mo_coeff = getattr(self, "_last_orb_mo_coeff", None)
        mol_atom = getattr(self, "_last_orb_mol_atom", None)
        mol_basis = getattr(self, "_last_orb_mol_basis", None)
        if mo_coeff is None or mol_atom is None or mol_basis is None:
            return

        try:
            from quantui.orbital_visualization import (
                generate_cube_from_arrays,
                plot_cube_isosurface,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                cube_path = Path(tmpdir) / f"orbital_{orbital_label}.cube"
                generate_cube_from_arrays(
                    mol_atom, mol_basis, mo_coeff, orb_idx, cube_path
                )
                fig = plot_cube_isosurface(
                    cube_path, title=f"{orbital_label} Isosurface"
                )
        except Exception as _exc:
            from IPython.display import HTML as _H
            from IPython.display import display as _d

            self._orb_iso_output.clear_output()
            with self._orb_iso_output:
                _d(
                    _H(
                        f'<p style="color:#b91c1c;padding:8px">⚠ Orbital isosurface failed: {_exc}</p>'
                    )
                )
            return

        from IPython.display import display as _ipy_display

        self._orb_iso_output.clear_output()
        with self._orb_iso_output:
            _ipy_display(fig)

    def _render_vib_mode(self, vib_data, molecule, mode_number: int) -> None:
        """Render vibrational animation for the given mode into ``vib_output``.

        Safe to call from background thread via ``with output:`` context.
        """
        from IPython.display import HTML as _H
        from IPython.display import display as _ipy_display

        def _err(msg: str) -> None:
            self.vib_output.clear_output()
            with self.vib_output:
                _ipy_display(_H(f'<p style="color:#b91c1c;padding:8px">⚠ {msg}</p>'))

        try:
            from plotlymol3d import create_vibration_animation, xyzblock_to_rdkitmol
        except ImportError as exc:
            _err(
                f"Vibrational animation requires plotlymol3d "
                f"(<code>pip install plotlymol3d</code>): {exc}"
            )
            return

        # Build an RDKit mol for bond connectivity (required by animation function).
        xyzblock = (
            f"{len(molecule.atoms)}\n{molecule.get_formula()}\n"
            f"{molecule.to_xyz_string()}"
        )
        try:
            rdmol = xyzblock_to_rdkitmol(xyzblock, charge=molecule.charge)
        except Exception as exc:
            _err(f"Could not parse molecule for bond connectivity: {exc}")
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
        except Exception as exc:
            _err(f"Animation generation failed: {exc}")
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
        # Show a loading indicator immediately so the user gets feedback while
        # the animation generates in the background.
        _label = next(
            (lbl for lbl, num in self.vib_mode_dd.options if num == mode_number),
            f"mode {mode_number}",
        )
        self.vib_output.clear_output()
        with self.vib_output:
            display(
                HTML(
                    f'<p style="color:#555;font-style:italic;padding:8px">'
                    f"⏳ Rendering vibrational animation ({_label})…</p>"
                )
            )
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
        _run_cpu_t = time.process_time()
        log = _LogCapture(self.run_output, self.run_status)

        # Write structured log header immediately so it appears at the top of output
        try:
            from quantui.log_utils import format_log_header as _fmt_log_hdr

            _hdr_calc_type = {
                "Geometry Opt": "geometry_opt",
                "Frequency": "frequency",
                "UV-Vis (TD-DFT)": "tddft",
                "NMR Shielding": "nmr",
                "PES Scan": "pes_scan",
            }.get(self.calc_type_dd.value, "single_point")
            log.write(
                _fmt_log_hdr(
                    formula=mol.get_formula(),
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    calc_type=_hdr_calc_type,
                )
            )
        except Exception:
            pass

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
                from quantui.freq_calc import run_freq_calc

                # ── Step 1: resolve seed geometry ─────────────────────────────
                _seed_path = self._freq_seed_dd.value
                if _seed_path:
                    from quantui.results_storage import load_trajectory

                    self.run_status.value = "Loading seed geometry from history…"
                    _seed_traj, _ = load_trajectory(Path(_seed_path))
                    calc_mol = _seed_traj[-1]
                    log.write(
                        f"\nSeed geometry loaded from: {Path(_seed_path).name}\n"
                        f"  Formula: {calc_mol.get_formula()}  "
                        f"Atoms: {len(calc_mol.atoms)}\n\n"
                    )

                # ── Step 2: optional geometry pre-optimisation ────────────────
                if self._freq_preopt_cb.value:
                    from quantui import optimize_geometry

                    self.run_status.value = "Pre-optimizing geometry before frequency…"
                    log.write(
                        "\n── Pre-optimisation (before frequency analysis) ──────────────────\n"
                    )
                    _pre_opt = optimize_geometry(
                        molecule=calc_mol,
                        method=self.method_dd.value,
                        basis=self.basis_dd.value,
                        progress_stream=log,  # type: ignore[arg-type]
                    )
                    calc_mol = _pre_opt.molecule
                    _conv_str = (
                        "converged" if _pre_opt.converged else "did NOT fully converge"
                    )
                    log.write(
                        f"\nPre-optimisation {_conv_str} in {_pre_opt.n_steps} steps."
                        f"  E = {_pre_opt.energies_hartree[-1]:.8f} Ha\n\n"
                    )
                    if not _pre_opt.converged:
                        log.write(
                            "⚠ Pre-optimisation did not fully converge — "
                            "proceeding with best available geometry.\n\n"
                        )

                # ── Step 3: frequency analysis ────────────────────────────────
                self.run_status.value = "Computing frequencies (SCF + Hessian)…"
                result = run_freq_calc(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    progress_stream=log,  # type: ignore[arg-type]
                )
                result_html = self._format_freq_result(result)
                _displacements_serialized = None
                if result.displacements is not None:
                    try:
                        import numpy as _np_d

                        _displacements_serialized = _np_d.asarray(
                            result.displacements
                        ).tolist()
                    except Exception:
                        pass
                save_spectra = {
                    "ir": {
                        "frequencies_cm1": result.frequencies_cm1,
                        "ir_intensities": result.ir_intensities,
                        "zpve_hartree": result.zpve_hartree,
                        "displacements": _displacements_serialized,
                    },
                    "molecule": {
                        "atoms": list(calc_mol.atoms),
                        "coords": [
                            list(map(float, row)) for row in calc_mol.coordinates
                        ],
                        "charge": calc_mol.charge,
                        "multiplicity": calc_mol.multiplicity,
                    },
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
            elif ct == "NMR Shielding":
                self.run_status.value = "Running NMR shielding (SCF + GIAO)..."
                from quantui.nmr_calc import run_nmr_calc

                result = run_nmr_calc(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    progress_stream=log,  # type: ignore[arg-type]
                )
                result_html = self._format_nmr_result(result)
                save_spectra, save_type = {}, "nmr"
            elif ct == "PES Scan":
                self.run_status.value = "Running PES scan…"
                from quantui.pes_scan import run_pes_scan

                _st = self._scan_type_dd.value.lower()
                _atom_idx: list = [
                    self._scan_atom1.value - 1,
                    self._scan_atom2.value - 1,
                ]
                if _st in ("angle", "dihedral"):
                    _atom_idx.append(self._scan_atom3.value - 1)
                if _st == "dihedral":
                    _atom_idx.append(self._scan_atom4.value - 1)

                result = run_pes_scan(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    scan_type=_st,
                    atom_indices=_atom_idx,
                    start=self._scan_start.value,
                    stop=self._scan_stop.value,
                    steps=self._scan_steps.value,
                    progress_stream=log,  # type: ignore[arg-type]
                )
                result_html = self._format_pes_scan_result(result)
                save_spectra, save_type = {}, "pes_scan"
            else:  # Single Point
                self.run_status.value = "Calculating..."
                from quantui import run_in_session

                # MP2 heavy-atom warning
                if self.method_dd.value.upper() == "MP2":
                    _n_heavy = sum(1 for a in calc_mol.atoms if a != "H")
                    if _n_heavy > 20:
                        self.result_output.append_display_data(
                            HTML(
                                '<div style="background:#fffbe6;border-left:4px solid #f59e0b;'
                                'padding:8px 12px;border-radius:4px;margin:4px 0;font-size:13px">'
                                f"⚠️ MP2 scales as O(N⁵) — this molecule has {_n_heavy} heavy atoms "
                                "and may be slow. Consider using DFT instead.</div>"
                            )
                        )

                _solvent = self.solvent_dd.value if self.solvent_cb.value else None
                result = run_in_session(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    progress_stream=log,  # type: ignore[arg-type]
                    solvent=_solvent,
                )
                result_html = self._format_result(result)
                save_spectra, save_type = {}, "single_point"

            _elapsed = time.perf_counter() - _run_wall_t
            _elapsed_cpu = time.process_time() - _run_cpu_t
            self._last_result = result
            self.accumulate_btn.disabled = False

            self.result_output.append_display_data(HTML(result_html))
            self.run_status.value = f"Done in {_elapsed:.1f} s."

            # Show 3D structure in the result panel and mirrored in Analysis tab
            _viz_mol = result.molecule if ct == "Geometry Opt" else calc_mol
            if ct == "Geometry Opt":
                self._viz_label.value = (
                    '<p style="color:#555;font-size:12px;font-weight:600;'
                    'margin:6px 0 2px">Optimized geometry</p>'
                )
                self._viz_label.layout.display = ""
            self._show_result_3d(_viz_mol, extra_output=self._analysis_mol_output)

            # Show calc-type-specific extra panels
            if ct == "Geometry Opt":
                # Stash trajectory and open accordion immediately to start rendering.
                self._pending_traj_result = result
                self._activate_ana_panel("Trajectory")
                self._show_orbital_diagram(result)
            elif ct == "Frequency":
                self._show_vib_animation(result, calc_mol)
                self._show_ir_spectrum(result)
            elif ct == "PES Scan":
                self._show_pes_scan_result(result)
            elif ct == "Single Point":
                self._show_orbital_diagram(result)

            self.step_progress.complete(2)
            self.step_progress.complete(3)

            # Update completion banner and Analysis tab context
            _mol_label = (
                f"{result.formula}  {self.method_dd.value}/{self.basis_dd.value}"
            )
            self._completion_mol_lbl.value = (
                f'<span style="color:#1e293b;font-size:13px;font-weight:500">'
                f"{_mol_label}</span>"
            )
            self._completion_banner.layout.display = ""
            self._analysis_context_lbl.value = (
                f'<p style="color:#555;font-size:13px;margin:4px 0 12px">'
                f"Analysing: {_mol_label}</p>"
            )
            _has_analysis = ct in (
                "Single Point",
                "Geometry Opt",
                "Frequency",
                "PES Scan",
            )
            self._to_analysis_btn.layout.display = "" if _has_analysis else "none"
            self._analysis_empty_html.layout.display = "none" if _has_analysis else ""

            # Write structured log footer
            try:
                from quantui.log_utils import format_log_footer as _fmt_log_ftr

                log.write(
                    _fmt_log_ftr(
                        result=result,
                        wall_time=_elapsed,
                        cpu_time=_elapsed_cpu,
                        log_text=log.getvalue(),
                        success=True,
                    )
                )
            except Exception:
                pass

            # Persist to disk
            try:
                from quantui import load_result, save_result
                from quantui.results_storage import (
                    save_orbitals,
                    save_thumbnail,
                    save_trajectory,
                )

                _saved_dir = save_result(
                    result,
                    pyscf_log=log.getvalue(),
                    calc_type=save_type,
                    spectra=save_spectra,
                )
                self._last_result_dir = _saved_dir
                save_thumbnail(_saved_dir, load_result(_saved_dir))
                # Persist trajectory so history viewer can replay it.
                if ct in ("Geometry Opt", "PES Scan"):
                    _traj = getattr(
                        result,
                        "trajectory" if ct == "Geometry Opt" else "coordinates_list",
                        None,
                    )
                    _e_list = getattr(result, "energies_hartree", [])
                    if _traj:
                        save_trajectory(_saved_dir, _traj, _e_list or [])
                # Persist MO data for orbital diagram + isosurface replay.
                if ct in ("Single Point", "Geometry Opt"):
                    save_orbitals(_saved_dir, result)
                self._refresh_results_browser()
                self._populate_compare_list()
                self._update_log_panel(
                    log.getvalue(),
                    f"{result.formula}  {self.method_dd.value}/{self.basis_dd.value}",
                )
                self._show_result_log(_saved_dir, log.getvalue())
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
            _elapsed_cpu = time.process_time() - _run_cpu_t
            _tb_str = _tb.format_exc()
            # Full details → Output tab (for debugging/instructors)
            log.write(f"\n--- Calculation Error ---\n{exc}\n\n{_tb_str}")
            # Structured failure footer
            try:
                from quantui.log_utils import format_log_footer as _fmt_log_ftr

                log.write(
                    _fmt_log_ftr(
                        result=None,
                        wall_time=_elapsed,
                        cpu_time=_elapsed_cpu,
                        log_text=log.getvalue(),
                        success=False,
                    )
                )
            except Exception:
                pass
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
        # Keep frequency seed dropdown in sync if it's currently visible.
        if self.calc_type_dd.value == "Frequency":
            self._refresh_freq_seed_options()

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
        self.help_tab_panel.layout.display = ""

    def _update_log_panel(self, log_text: str, label: str = "") -> None:
        self._render_log(log_text, label)

    def _goto_output_tab(self) -> None:
        self.root_tab.selected_index = 5

    def _render_log(self, text: str, source_label: str = "") -> None:
        import html as _html_mod
        import re as _re

        _bfgs_re = _re.compile(r"^BFGS:\s+(\d+)\s+\S+\s+([-\d.]+)\s+([\d.]+)")

        lines = text.splitlines()
        rows = []
        for line in lines:
            esc = _html_mod.escape(line)
            # ── Log header / footer structure ─────────────────────────────────
            if len(line) >= 40 and line == "=" * len(line):
                style = "color:#1e3a5f;font-weight:700"
            elif "QuantUI — Quantum Chemistry Interface" in line:
                style = "color:#6d28d9;font-weight:700"
            elif line.startswith("  ── "):
                style = "color:#334155;font-weight:700"
            elif line.startswith("  ✓"):
                style = "color:#16a34a;font-weight:700"
            elif line.startswith("  ✗"):
                style = "color:#dc2626;font-weight:700"
            elif (
                line.startswith("  Machine:")
                or line.startswith("  GPU:")
                or line.startswith("  Threads:")
            ):
                style = "color:#475569"
            elif (
                line.startswith("  Molecule:")
                or line.startswith("  Method/Basis:")
                or line.startswith("  Calc type:")
                or line.startswith("  Started:")
            ):
                style = "color:#1d4ed8"
            elif (
                line.startswith("    Energy:")
                or line.startswith("    HOMO-LUMO gap:")
                or line.startswith("    ZPVE:")
            ):
                style = "color:#0f766e;font-weight:600"
            elif line.startswith("    Wall time:"):
                style = "color:#64748b"
            elif line.startswith("    ✔") or line.startswith("    ⚠"):
                style = "color:#d97706"
            # ── Geometry optimisation (ASE BFGS) ──────────────────────────────
            elif line.startswith("BFGS:"):
                m = _bfgs_re.match(line)
                if m:
                    fmax = float(m.group(3))
                    # Colour by convergence: green when nearly converged, teal otherwise
                    style = (
                        "color:#16a34a;font-weight:600"
                        if fmax < 0.1
                        else "color:#0d9488"
                    )
                else:
                    style = "color:#0d9488"
            elif line.strip() == "Step Time Energy fmax":
                style = "color:#334155;font-weight:700"
            # ── Post-optimisation summary ──────────────────────────────────────
            elif line.startswith("── Final SCF"):
                style = "color:#6d28d9;font-weight:600"
            elif "HOMO-LUMO gap:" in line:
                style = "color:#6d28d9;font-weight:600"
            # ── SCF convergence ────────────────────────────────────────────────
            elif "converged SCF energy" in line or "SCF converged" in line:
                style = "color:#16a34a;font-weight:600"
            elif line.lstrip().startswith("cycle=") and "E=" in line:
                style = "color:#64748b"
            # ── MO / orbital info (verbose=4) ──────────────────────────────────
            elif "MO energies" in line or "** MO" in line:
                style = "color:#1d4ed8;font-weight:600"
            elif "HOMO" in line or "LUMO" in line or "All MO energies" in line:
                style = "color:#2563eb"
            elif line.lstrip().startswith("occupied:") or line.lstrip().startswith(
                "virtual:"
            ):
                style = "color:#3b82f6"
            # ── Thermo / properties ────────────────────────────────────────────
            elif "Mulliken" in line or "mulliken" in line:
                style = "color:#7c3aed"
            elif "dipole" in line.lower() or "Dipole" in line:
                style = "color:#7c3aed"
            elif "nuclear repulsion" in line.lower() or "Nuclear repulsion" in line:
                style = "color:#94a3b8"
            elif "E(MP2)" in line or "MP2 correlation" in line:
                style = "color:#0891b2"
            # ── Warnings / errors ──────────────────────────────────────────────
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
        _extra = ""
        # MP2: show HF reference energy separately
        _mp2_corr = getattr(r, "mp2_correlation_hartree", None)
        if _mp2_corr is not None:
            _hf_e = r.energy_hartree - _mp2_corr
            _extra += (
                f'<tr><td style="padding:3px 18px 3px 0;color:#444">HF reference</td>'
                f'<td style="color:#000">{_hf_e:.8f} Ha</td></tr>'
                f'<tr><td style="padding:3px 18px 3px 0;color:#444">MP2 correlation</td>'
                f'<td style="color:#000">{_mp2_corr:.8f} Ha</td></tr>'
            )
        _solvent = getattr(r, "solvent", None)
        if _solvent is not None:
            _extra += (
                f'<tr><td style="padding:3px 18px 3px 0;color:#444">Solvent (PCM)</td>'
                f'<td style="color:#000">{_solvent}</td></tr>'
            )
        _dip = getattr(r, "dipole_moment_debye", None)
        if _dip is not None:
            _extra += (
                f'<tr><td style="padding:3px 18px 3px 0;color:#444">Dipole moment</td>'
                f'<td style="color:#000">{_dip:.4f} D</td></tr>'
            )
        _chg = getattr(r, "mulliken_charges", None)
        _syms = getattr(r, "atom_symbols", None)
        if _chg is not None and _syms is not None:
            _charge_str = "  ".join(f"{sym}:{c:+.3f}" for sym, c in zip(_syms, _chg))
            _extra += (
                f'<tr><td style="padding:3px 18px 3px 0;color:#444;vertical-align:top">'
                f"Mulliken charges</td>"
                f'<td style="color:#000;font-family:monospace;font-size:12px;'
                f'word-break:break-all">{_charge_str}</td></tr>'
            )
        return (
            f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
            f'padding:10px 14px;border-radius:4px;margin:6px 0">'
            f"<b>{r.formula} &mdash; {r.method}/{r.basis}</b>"
            f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
            f"{_rows}{_extra}</table></div>"
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
        _thermo_rows = ""
        _thermo = getattr(r, "thermo", None)
        if _thermo is not None:
            _kj = 2625.5  # kJ/mol per Hartree
            _thermo_rows = (
                f'<tr><td colspan="2" style="padding:6px 0 2px 0;color:#666;'
                f'font-size:12px;font-style:italic">'
                f"&#8212; Thermochemistry at {_thermo.temperature_k:.0f} K / 1 atm &#8212;"
                f"</td></tr>"
                f'<tr><td style="padding:3px 18px 3px 0;color:#444">H (298 K)</td>'
                f'<td style="color:#000">{_thermo.H_hartree:.6f} Ha</td></tr>'
                f'<tr><td style="padding:3px 18px 3px 0;color:#444">S (298 K)</td>'
                f'<td style="color:#000">{_thermo.S_jmol:.2f} J/(mol·K)</td></tr>'
                f'<tr><td style="padding:3px 18px 3px 0;color:#444">G (298 K)</td>'
                f'<td style="color:#000">{_thermo.G_hartree:.6f} Ha'
                f" ({_thermo.G_hartree * _kj:.2f} kJ/mol)</td></tr>"
            )
        return (
            f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
            f'padding:10px 14px;border-radius:4px;margin:6px 0">'
            f"<b>Frequency Analysis &mdash; {r.formula} ({r.method}/{r.basis})</b>"
            f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
            f"{_rows}{_thermo_rows}</table></div>"
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

    def _format_nmr_result(self, r) -> str:
        _conv = "Yes" if r.converged else "No (treat with caution)"
        _cc = "green" if r.converged else "#c00"
        header_rows = (
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">SCF converged</td>'
            f'<td style="color:{_cc}">{_conv}</td></tr>'
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">Reference</td>'
            f'<td style="color:#000">{r.reference_compound} ({r.method}/{r.basis})</td></tr>'
        )

        def _nmr_table(label: str, shifts: list, sym: str) -> str:
            if not shifts:
                return ""
            rows = "".join(
                f"<tr>"
                f'<td style="padding:2px 14px 2px 0;color:#555">{sym}-{n}</td>'
                f'<td style="color:#000">{d:.2f} ppm</td>'
                f"</tr>"
                for n, (_i, d) in enumerate(shifts, 1)
            )
            return (
                f'<tr><td colspan="2" style="padding:8px 0 2px;color:#444;font-weight:bold">'
                f"{label} shifts (vs. TMS):</td></tr>"
                f"<tr>"
                f'<th style="text-align:left;color:#555;font-size:12px;padding:2px 14px 2px 0">Atom</th>'
                f'<th style="text-align:left;color:#555;font-size:12px">δ (ppm)</th></tr>'
                + rows
            )

        h_table = _nmr_table("¹H", r.h_shifts(), "H")
        c_table = _nmr_table("¹³C", r.c_shifts(), "C")

        _basis_warn = ""
        if r.basis.upper() in ("STO-3G", "3-21G"):
            _basis_warn = (
                '<tr><td colspan="2" style="padding:6px 0 0">'
                '<span style="color:#b45309;font-size:12px">'
                f"⚠ {r.basis} gives qualitative NMR only — use 6-31G* or better.</span>"
                "</td></tr>"
            )

        _empty = ""
        if not r.h_shifts() and not r.c_shifts():
            _empty = (
                '<tr><td colspan="2" style="color:#888;font-size:12px">'
                "No ¹H or ¹³C atoms found in this molecule.</td></tr>"
            )

        return (
            f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
            f'padding:10px 14px;border-radius:4px;margin:6px 0">'
            f"<b>NMR Shielding &mdash; {r.formula} ({r.method}/{r.basis})</b>"
            f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
            f"{header_rows}{h_table}{c_table}{_empty}{_basis_warn}</table></div>"
        )

    def _format_pes_scan_result(self, r) -> str:
        """Format a PESScanResult as an HTML result card."""
        _conv = "Yes" if r.converged_all else "No (some points did not converge)"
        _cc = "green" if r.converged_all else "#c00"
        if r.energies_hartree:
            e_min = min(r.energies_hartree)
            e_max = max(r.energies_hartree)
            barrier_kcal = (e_max - e_min) * 627.509474
            _e_row = (
                f'<tr><td style="padding:3px 18px 3px 0;color:#444">Min energy</td>'
                f'<td style="color:#000">{e_min:.8f} Ha</td></tr>'
                f'<tr><td style="padding:3px 18px 3px 0;color:#444">Energy range</td>'
                f'<td style="color:#000">{barrier_kcal:.2f} kcal/mol</td></tr>'
            )
        else:
            _e_row = ""
        _idx_str = "–".join(str(i + 1) for i in r.atom_indices)
        return (
            f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
            f'padding:10px 14px;border-radius:4px;margin:6px 0">'
            f"<b>PES Scan &mdash; {r.formula} ({r.method}/{r.basis})</b>"
            f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">Scan type</td>'
            f'<td style="color:#000">{r.scan_type.capitalize()} ({_idx_str})</td></tr>'
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">Range</td>'
            f'<td style="color:#000">{r.scan_parameter_values[0]:.3f} → '
            f"{r.scan_parameter_values[-1]:.3f} {r.scan_unit} "
            f"({r.n_steps} points)</td></tr>"
            f"{_e_row}"
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">All converged</td>'
            f'<td style="color:{_cc}">{_conv}</td></tr>'
            f"</table></div>"
        )

    def _show_pes_scan_result(self, result) -> None:
        """Render the PES energy profile chart and trajectory for a PES scan result."""
        self._last_pes_result = result
        try:
            import plotly.graph_objects as go
            import plotly.io as pio

            e_rel = result.energies_relative_kcal
            x_vals = result.scan_parameter_values

            hover_text = [
                f"{result.scan_coordinate_label}: {x:.4f}<br>"
                f"ΔE = {de:.3f} kcal/mol<br>"
                f"E = {e:.8f} Ha"
                for x, de, e in zip(x_vals, e_rel, result.energies_hartree)
            ]

            fig = go.Figure(
                go.Scatter(
                    x=x_vals,
                    y=e_rel,
                    mode="lines+markers",
                    line=dict(color="#2563eb", width=2),
                    marker=dict(size=8, color="#2563eb"),
                    hovertext=hover_text,
                    hoverinfo="text",
                )
            )
            tc = self._plotly_theme_colors()
            fig.update_layout(
                xaxis_title=result.scan_coordinate_label,
                yaxis_title="Relative energy / kcal mol⁻¹",
                height=380,
                margin=dict(l=60, r=20, t=30, b=50),
                plot_bgcolor=tc["plot_bgcolor"],
                paper_bgcolor=tc["paper_bgcolor"],
                font=dict(color=tc["font_color"]),
                xaxis=dict(showgrid=True, gridcolor=tc["grid_color"]),
                yaxis=dict(showgrid=True, gridcolor=tc["grid_color"]),
                hovermode="closest",
            )
            self._pes_plot_html.value = pio.to_html(
                fig, include_plotlyjs="cdn", full_html=False
            )
        except Exception:
            pass

        self._activate_ana_panel("PES Scan")

        # Reuse trajectory accordion for the scan geometry sequence
        if result.coordinates_list:
            self._pending_traj_result = result
            self.traj_accordion.set_title(0, "Geometry at Each Scan Point")
            self._activate_ana_panel("Trajectory", auto_select=False)

    def _format_past_result(self, data: dict, result_dir: Optional[Path] = None) -> str:
        import base64 as _b64

        _ct_labels = {
            "single_point": ("Single Point", "#2563eb", "#dbeafe"),
            "geometry_opt": ("Geometry Optimization", "#7c3aed", "#ede9fe"),
            "frequency": ("Frequency Analysis", "#15803d", "#dcfce7"),
            "tddft": ("TD-DFT", "#b45309", "#fef3c7"),
            "nmr": ("NMR", "#0d9488", "#ccfbf1"),
            "pes_scan": ("PES Scan", "#c2410c", "#ffedd5"),
        }
        ct = data.get("calc_type", "")
        _ct_label, _ct_fg, _ct_bg = _ct_labels.get(
            ct, (ct.replace("_", " ").title(), "#555", "#f3f4f6")
        )
        _ct_badge = (
            f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
            f"background:{_ct_bg};color:{_ct_fg};font-size:12px;font-weight:700;"
            f'letter-spacing:0.03em;margin-bottom:6px">{_ct_label}</span>'
        )
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

        # Embed thumbnail if saved
        _thumb_html = ""
        if result_dir is not None:
            _thumb_path = Path(result_dir) / "thumbnail.png"
            if _thumb_path.exists():
                _img_b64 = _b64.b64encode(_thumb_path.read_bytes()).decode()
                _thumb_html = (
                    f'<img src="data:image/png;base64,{_img_b64}" '
                    f'style="float:right;margin:0 0 6px 14px;border-radius:4px;'
                    f'border:1px solid #e2e8f0" width="173" height="108" />'
                )

        return (
            f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
            f'padding:10px 14px;border-radius:4px;margin:6px 0;overflow:hidden">'
            f"{_thumb_html}"
            f"{_ct_badge}<br>"
            f'<b>{data["formula"]} &mdash; {data["method"]}/{data["basis"]}</b>'
            f'&ensp;<small style="color:#777">{ts}</small>'
            f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
            f"{_rows}</table></div>"
        )

    # ══ HELPERS ══════════════════════════════════════════════════════════════

    def _get_results_dir(self) -> Path:
        from quantui.results_storage import _default_results_dir

        return _default_results_dir().resolve()
