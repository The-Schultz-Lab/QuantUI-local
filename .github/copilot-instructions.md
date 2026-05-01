# QuantUI — AI Assistant Context

> Stable project context for GitHub Copilot, Claude, and other AI coding assistants.
> Describes what the project IS and how it is built — not where development currently
> stands (see your project STATUS.md for current session state). Update this file when
> architecture or conventions change, not every session.

---

## Overview

QuantUI is an interactive Jupyter/Voilà interface for running PySCF quantum
chemistry calculations locally — no cluster account, no SLURM, no queueing. Students
design molecules, launch RHF/UHF/DFT calculations in their own Python kernel, and
visualize results in minutes. It is a downstream port of the cluster-focused
`QuantUI` repo with all SLURM infrastructure removed.

**Target audience:** Undergraduate chemistry students at North Carolina Central
University. The UI runs as a Voilà app — students never see code.

---

## Repository Structure

```
QuantUI/
├── quantui/                  ← Main Python package (imports as `quantui`)
│   ├── app.py                ← QuantUIApp class — all widgets, callbacks, state
│   ├── molecule.py           ← Molecule dataclass + XYZ/SMILES parsing
│   ├── session_calc.py       ← In-session PySCF runner (run_in_session)
│   ├── optimizer.py          ← QM geometry optimization (ASE-BFGS + PySCF)
│   ├── freq_calc.py          ← Frequency / vibrational analysis
│   ├── tddft_calc.py         ← TD-DFT UV-Vis excited states
│   ├── nmr_calc.py           ← NMR shielding (GIAO), returns NMRResult
│   ├── pes_scan.py           ← Potential energy surface (bond/angle/dihedral)
│   ├── calculator.py         ← PySCFCalculation abstraction
│   ├── comparison.py         ← Side-by-side result comparison table
│   ├── results_storage.py    ← Persist/reload calculation results (JSON + log)
│   ├── calc_log.py           ← Performance + event logging (JSONL)
│   ├── pubchem.py            ← PubChem molecule search
│   ├── visualization_py3dmol.py ← 3D molecular viewer (py3Dmol / plotlyMol)
│   ├── ase_bridge.py         ← ASE structure I/O + molecule library
│   ├── preopt.py             ← ASE force-field pre-optimisation (fast, no PySCF)
│   ├── progress.py           ← StepProgress widget
│   ├── help_content.py       ← HELP_TOPICS dict — in-app educational text
│   ├── orbital_visualization.py ← Orbital energy diagrams, cube file viewer
│   ├── ir_plot.py            ← IR spectrum Plotly figure builder
│   ├── config.py             ← All constants/defaults (methods, basis sets, etc.)
│   ├── utils.py              ← Session resource checks, sanitize_filename, etc.
│   ├── issue_tracker.py      ← In-app issue/bug logging to issues.db
│   ├── log_utils.py          ← Shared logging helpers
│   ├── benchmarks.py         ← Performance benchmarking utilities
│   └── security.py           ← SecurityError exception class
├── notebooks/
│   ├── molecule_computations.ipynb  ← Student-facing Voilà app (thin launcher)
│   └── tutorials/                   ← 01–05 step-by-step tutorial notebooks
├── tests/                    ← pytest suite (~700 tests)
├── .github/
│   └── copilot-instructions.md  ← This file
├── apptainer/
│   ├── quantui.def     ← Apptainer container definition
│   └── build.sh              ← Build script
├── local-setup/              ← Conda environment YAMLs
├── launch-app.bat            ← Windows double-click launcher (Voilà app mode)
├── launch-dev.bat            ← Windows double-click launcher (JupyterLab mode)
├── pyproject.toml            ← Package config (name: quantui, imports as quantui)
└── pytest.ini                ← pytest configuration
```

---

## Architecture

```
notebooks/molecule_computations.ipynb
    Cell 0: Markdown title
    Cell 1: Conda env check (skip-execution, remove-input)
    Cell 2: from quantui.app import QuantUIApp; QuantUIApp().display()
                │
                ▼
        quantui/app.py — QuantUIApp
        ┌──────────────────────────────────────────────────────────┐
        │  _build_shared_widgets()    → StepProgress, run_output   │
        │  _build_molecule_section()  → mol_input_container        │
        │  _build_calc_setup()        → method_dd, basis_dd, etc.  │
        │  _build_run_section()       → run_btn, run_panel         │
        │  _build_results_section()   → all panel accordions       │
        │  _build_history_section()   → history_panel              │
        │  _build_compare_section()   → compare_panel              │
        │  _build_output_tab()        → log viewer                 │
        │  _build_help_section()      → help panel                 │
        │  _build_ana_switcher()      → Analysis tab always-visible panels │
        │  _assemble_tabs()           → root_tab (Tab widget)      │
        └──────────────────────────────────────────────────────────┘
                │  _do_run() dispatches by calc_type_dd.value:
                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Single Point  → session_calc.run_in_session()              │
    │  Geometry Opt  → optimizer.optimize_geometry()              │
    │  Frequency     → freq_calc.run_freq_calc()                  │
    │  UV-Vis        → tddft_calc.run_tddft_calc()                │
    │  NMR Shielding → nmr_calc.run_nmr_calc()                    │
    │  PES Scan      → pes_scan.run_pes_scan()                    │
    └──────────────────────────────────────────────────────────────┘
                │
                ▼  _apply_analysis_context(_AnalysisContext)
    ┌──────────────────────────────────────────────────────────────┐
    │  _PANEL_REGISTRY[calc_type] → [(_panel_name, _pop_fn, auto)]│
    │  Each _pop_xxx() returns bool → _activate_ana_panel()       │
    └──────────────────────────────────────────────────────────────┘
                │
                ▼
    results_storage.save_result()   calc_log.log_perf_record()
```

**Tab order in the app:** Calculate (0) → History (1) → Compare (2) → Output (3) → Help (4)

---

## The Panel Registry Pattern — CRITICAL

This is the core architecture for the Analysis tab. **Any work touching analysis panels
must understand this pattern.**

### How it works

After a calculation completes (live or history), `_apply_analysis_context(ctx)` is the
single entry point for all panel population. It:

1. Calls `_deactivate_all_ana_panels()` to reset all 8 panels — collapses accordions
   and restores "Not available" placeholders (panels remain in the DOM — never hidden)
2. Looks up `_PANEL_REGISTRY[ctx.calc_type]` for an ordered list of
   `(panel_name, populate_method_name, auto_select)` tuples
3. Calls each populate method (e.g. `_pop_energies(ctx)`)
4. If it returns `True`, calls `_activate_ana_panel(panel_name)` to make the panel
   available and optionally auto-select it
5. Updates the Analysis tab context label and navigation button visibility

### `_AnalysisContext` dataclass

```python
@dataclass
class _AnalysisContext:
    calc_type: str           # "single_point" | "geometry_opt" | "frequency" | etc.
    formula: str
    method: str
    basis: str
    live_result: Any = None  # result object from _do_run; None for history
    result_dir: Optional[Path] = None  # saved result dir
    molecule: Optional[Any] = None
    spectra_data: dict = field(default_factory=dict)
    preopt_result: Optional[Any] = None  # OptimizationResult from Frequency pre-opt
    source: str = "live"     # "live" | "history"
```

### `_PANEL_REGISTRY`

```python
_PANEL_REGISTRY = {
    "single_point": [("Energies", "_pop_energies", True), ("Isosurface", "_pop_isosurface", False)],
    "geometry_opt": [("Trajectory", "_pop_geo_trajectory", True), ("Energies", "_pop_energies", False), ...],
    "frequency":    [("Trajectory", "_pop_preopt_trajectory", False), ("Vibrational", "_pop_vibrational", True), ("IR Spectrum", "_pop_ir_spectrum", False)],
    "tddft":        [("UV-Vis", "_pop_uv_vis", True)],
    "nmr":          [("NMR", "_pop_nmr_shielding", True)],
    "pes_scan":     [("PES Scan", "_pop_pes_plot", True), ("Trajectory", "_pop_pes_trajectory", False)],
}
```

### Rules for all panel populate methods (`_pop_xxx`)

- **Must return `bool`** — `True` if data was populated, `False` if data is missing/unavailable
- **Must NOT call `_activate_ana_panel()`** — the registry loop handles activation
- Support both live (use `ctx.live_result`) and history (use `ctx.result_dir` / `ctx.spectra_data`)
- Must be side-effect safe: if data is missing, clear/no-op the widget and return `False`

### Adding a new panel — exactly these steps

1. Create the accordion widget in `_build_results_section()` (follow existing pattern)
2. Add it to the `analysis_tab_panel` VBox in `_build_analysis_section()`
3. Add `(panel_name, accordion_attr_name, "available after X")` to `_PANEL_META`
   (class-level `ClassVar` — this is the single source of truth for placeholder text
   and accordion attribute lookup; `_build_ana_switcher()` reads it at init time)
4. Write `_pop_xxx(self, ctx: _AnalysisContext) -> bool`
5. Add the entry to `_PANEL_REGISTRY`
6. If history replay needs data: ensure `save_spectra` in `_do_run` saves it

### Live run vs history replay — same code path

Both `_do_run()` and the history loaders (`_on_view_log`, `_history_load_analysis`)
build an `_AnalysisContext` and call `_apply_analysis_context()`. They are guaranteed
to show identical panels for the same result data. Use `_build_history_context(result_dir)`
to build the context from disk.

---

## Analysis Tab — 8 Panels

All 8 panels are **always in the DOM** (`layout.display=""`, `selected_index=None`).
Unavailable panels show a "Not available — run a X calculation first" placeholder.
`_activate_ana_panel()` swaps the placeholder for real content and expands the accordion.
`_deactivate_all_ana_panels()` restores placeholders and collapses — never hides.

| Panel | Accordion attr | Activated by calc types |
|---|---|---|
| Energies | `_orb_accordion` | Single Point, Geometry Opt |
| Trajectory | `traj_accordion` | Geometry Opt, PES Scan, Frequency (pre-opt only) |
| Vibrational | `vib_accordion` | Frequency |
| IR Spectrum | `_ir_accordion` | Frequency |
| PES Scan | `_pes_scan_accordion` | PES Scan |
| Isosurface | `_iso_accordion` | Single Point (Linux/WSL) |
| UV-Vis | `_tddft_accordion` | UV-Vis (TD-DFT) |
| NMR | `_nmr_accordion` | NMR Shielding |

---

## Key Files

| File | Purpose |
| ---- | ------- |
| `quantui/app.py` | **Primary development target.** All widget logic lives here. |
| `quantui/session_calc.py` | `run_in_session()` — PySCF SCF runner; returns `SessionResult` |
| `quantui/molecule.py` | `Molecule` dataclass, `parse_xyz_input()` |
| `quantui/config.py` | `SUPPORTED_METHODS`, `SUPPORTED_BASIS_SETS`, `MOLECULE_LIBRARY`, widget layout constants |
| `quantui/results_storage.py` | `save_result()`, `load_result()`, `list_results()` — result.json schema v2 |
| `quantui/calc_log.py` | `perf_log.jsonl` + `event_log.jsonl` under `~/.quantui/logs/` |
| `quantui/optimizer.py` | `optimize_geometry()` — ASE-BFGS + custom `_QuantUIPySCFCalc` |
| `quantui/freq_calc.py` | `run_freq_calc()` — vibrational analysis via `pyscf.hessian` |
| `quantui/tddft_calc.py` | `run_tddft_calc()` — excited states via `pyscf.tddft` |
| `quantui/nmr_calc.py` | `run_nmr_calc()` — GIAO shielding; returns `NMRResult` |
| `quantui/pes_scan.py` | `run_pes_scan()` — bond/angle/dihedral PES; returns `PESScanResult` |
| `quantui/ir_plot.py` | `plot_ir_spectrum()` — Plotly stick/broadened IR chart |
| `notebooks/molecule_computations.ipynb` | Thin launcher — 3 cells only (do not add logic here) |
| your project `STATUS.md` | **Read this first every session** — current state, git log, open tasks |
| your project `feature-requests.md` | FR backlog |

---

## Critical Constraints

> These are hard rules that must be respected in all implementation work.

1. **PySCF is Linux/macOS/WSL only.** Never assume PySCF is available. All
   PySCF-dependent code must be guarded with `try/except ImportError`. The
   availability flags in `app.py` (`_PYSCF_AVAILABLE`, `_PREOPT_AVAILABLE`,
   `ASE_AVAILABLE`, `VISUALIZATION_AVAILABLE`) are computed once at module
   import — check these instead of importing inline.

2. **No notebook cell logic.** `notebooks/molecule_computations.ipynb` is a
   three-cell thin launcher. Never add widget creation, callbacks, or business
   logic to notebook cells. All logic belongs in `quantui/app.py`.

3. **Thread-safe widget updates only.** `_do_run()` runs in a background thread.
   Widget updates from threads must use `.value =` assignment,
   `.append_stdout()`, or `.append_display_data()`. Never call `display()` inside
   `with output_widget:` from a background thread except via `with output_widget:` context.

4. **No new top-level dependencies** without updating both `pyproject.toml`
   and the Apptainer container `apptainer/quantui.def`.

5. **All constants in `config.py`.** Method names, basis sets, layout widths,
   and other shared literals must be defined in `config.py` and imported from
   there — never hard-coded in `app.py` or other modules.

6. **Result schema versioning.** `results_storage.py` uses `_SCHEMA_VERSION = 2`.
   Any new fields must be additive (never remove or rename existing keys). Bump
   `_SCHEMA_VERSION` only when a breaking change is unavoidable.

7. **Panel populate methods must not call `_activate_ana_panel()`.** Only the
   registry loop in `_apply_analysis_context()` activates panels. Helper methods
   (`_show_orbital_diagram`, `_show_vib_animation`, `_show_ir_spectrum`,
   `_show_pes_scan_result`) return `bool` and must not call `_activate_ana_panel()`.

8. **Never use `include_plotlyjs="cdn"` in widget HTML.** CDN requests fail
   silently in offline classroom deployments — the figure div stays blank with no
   error anywhere. Use `include_plotlyjs="require"` (Voilà/RequireJS path) or
   `include_plotlyjs=True` (inline bundle, works everywhere). This rule is enforced
   by `tests/test_code_quality.py::test_no_cdn_plotlyjs`.

9. **All `.observe()` callbacks must be wrapped with `_safe_cb`.** Exceptions in
   raw `.observe()` handlers disappear into the kernel console — invisible in Voilà.
   Use `widget.observe(self._safe_cb(self._on_x), names="value")` so exceptions are
   routed to the Log tab instead. See `_safe_cb()` in `app.py`.

---

## Supported Calculations

| Calc type | Module | Key function | Returns | save_type |
| --- | --- | --- | --- | --- |
| Single Point | `session_calc` | `run_in_session()` | `SessionResult` | `"single_point"` |
| Geometry Opt | `optimizer` | `optimize_geometry()` | `OptResult` | `"geometry_opt"` |
| Frequency | `freq_calc` | `run_freq_calc()` | `FreqResult` | `"frequency"` |
| UV-Vis TD-DFT | `tddft_calc` | `run_tddft_calc()` | `TDDFTResult` | `"tddft"` |
| NMR Shielding | `nmr_calc` | `run_nmr_calc()` | `NMRResult` | `"nmr"` |
| PES Scan | `pes_scan` | `run_pes_scan()` | `PESScanResult` | `"pes_scan"` |

**Supported methods:** RHF, UHF, B3LYP, PBE, PBE0, M06-2X, MP2, CCSD, CAM-B3LYP,
M06-L, wB97X-D, PBE-D3 (defined in `config.SUPPORTED_METHODS`).

**Supported basis sets:** STO-3G, 3-21G, 6-31G, 6-31G\*, 6-31G\*\*, cc-pVDZ,
cc-pVTZ, def2-SVP, def2-TZVP (defined in `config.SUPPORTED_BASIS_SETS`).

---

## `QuantUIApp` Class (`quantui/app.py`)

### Construction order

```
__init__()
  _build_shared_widgets()
  _build_molecule_section()
  _build_calc_setup()
  _build_run_section()        # uses self.calc_type_dd from _build_calc_setup
  _build_results_section()    # builds all panel accordions incl. tddft + nmr
  _build_history_section()
  _build_compare_section()
  _build_output_tab()
  _build_help_section()
  _build_ana_switcher()        # builds 8-panel switcher strip
  _assemble_tabs()             # builds self.root_tab
  _wire_callbacks()            # all .observe() and .on_click() wiring
```

### Key instance state — Analysis tab

| Attribute | Type | Purpose |
| --- | --- | --- |
| `self._ana_available` | `set[str]` | Names of panels with data; populated by `_activate_ana_panel()` |
| `self._ana_active` | `str` | Currently expanded panel name |
| `self._ana_panel_names` | `list[str]` | Ordered panel names derived from `_PANEL_META` |
| `self._ana_accordions` | `list[Accordion]` | Ordered accordions derived from `_PANEL_META` |
| `self._ana_unavail_msgs` | `dict[str, HTML]` | Panel name → "Not available" placeholder widget |
| `self._ana_content_boxes` | `dict[str, VBox]` | Panel name → real content VBox (hidden until activated) |
| `self._ana_btns` | `list[Button]` | Switcher buttons — **pending removal in M-PLOT-FIX.3** |
| `self._pending_traj_result` | `Any` | Trajectory stub; rendered lazily on accordion expand |
| `self._last_orb_info` | `Optional[...]` | Orbital info from last successful `_show_orbital_diagram` |
| `self._last_ir_freqs` | `list[float]` | IR frequencies stored by `_show_ir_spectrum` |
| `self._last_ir_ints` | `list[float]` | IR intensities stored by `_show_ir_spectrum` |
| `self._last_vib_data` | `Optional[VibrationalData]` | plotlyMol vib data for mode selector |
| `self._last_pes_result` | `Optional[...]` | PES result stored by `_show_pes_scan_result` |

### Key instance state — general

| Attribute | Type | Purpose |
| --- | --- | --- |
| `self._molecule` | `Optional[Molecule]` | Currently loaded molecule |
| `self._last_result` | `Optional[...]` | Most recent calculation result |
| `self._results` | `list` | All results from this session |
| `self._pyscf_available` | `bool` | Mirrors module-level `_PYSCF_AVAILABLE` |
| `self.root_tab` | `widgets.Tab` | Top-level displayed widget |
| `self.method_dd` | `widgets.Dropdown` | Selected QC method |
| `self.basis_dd` | `widgets.Dropdown` | Selected basis set |
| `self.calc_type_dd` | `widgets.Dropdown` | Calc type selector |
| `self.result_viz_output` | `widgets.Output` | 3D molecule rendered after every calc |
| `self._last_result_dir` | `Optional[Path]` | Saved result directory from most recent run |

---

## Result Storage (`quantui/results_storage.py`)

Results are saved to timestamped subdirectories:
```
<QUANTUI_RESULTS_DIR>/<timestamp>_<formula>_<method>_<basis>/
    result.json    ← schema v2 (see below)
    pyscf.log      ← raw PySCF stdout (may be absent)
    trajectory.json ← Geo Opt / PES Scan trajectories
    orbitals.npz   ← MO arrays for Single Point / Geo Opt
    thumbnail.png  ← auto-generated molecule thumbnail
```

Default results dir: `Path("results")` relative to cwd, or `$QUANTUI_RESULTS_DIR`.

### result.json schema (version 2)

```json
{
  "_schema_version": 2,
  "timestamp": "YYYY-MM-DD_HH-MM-SS-ffffff",
  "calc_type": "single_point | geometry_opt | frequency | tddft | nmr | pes_scan",
  "formula": "H2O",
  "method": "RHF",
  "basis": "STO-3G",
  "energy_hartree": -75.0,
  "energy_ev": -2040.8,
  "homo_lumo_gap_ev": 12.3,
  "converged": true,
  "n_iterations": 12,
  "spectra": {
    "ir":     {"frequencies_cm1": [...], "ir_intensities": [...], "zpve_hartree": 0.021, "displacements": [...]},
    "uv_vis": {"excitation_energies_ev": [...], "oscillator_strengths": [...], "wavelengths_nm": [...]},
    "nmr":    {"atom_symbols": [...], "shielding_iso_ppm": [...], "chemical_shifts_ppm": {"0": 1.2, ...}, "reference_compound": "TMS"},
    "molecule": {"atoms": [...], "coords": [...], "charge": 0, "multiplicity": 1}
  }
}
```

Schema rules: new fields must be additive only. Never remove or rename existing keys.

---

## Performance Logging (`quantui/calc_log.py`)

Two JSONL files under `~/.quantui/logs/` (override with `$QUANTUI_LOG_DIR`):

| File | Contents |
| --- | --- |
| `perf_log.jsonl` | One record per converged run: formula, n_atoms, n_electrons, method, basis, elapsed_s, n_basis |
| `event_log.jsonl` | Startup / calc_start / calc_done / calc_error events (7-day auto-prune) |

Key API: `log_calculation()`, `get_perf_history()`, `estimate_time()`, `reset_perf_log()`.
Performance estimation uses a 4-strategy priority chain: N_basis-normalised efficiency →
electron-count scaling → cross-method N_basis → cross-method electron-count.

---

## Naming Conventions

- **Functions:** `verb_noun()` — e.g., `parse_xyz_input()`, `run_in_session()`, `save_result()`
- **Classes:** `PascalCase` — e.g., `QuantUIApp`, `SessionResult`, `FreqResult`
- **Private methods/helpers:** leading underscore — e.g., `_do_run()`, `_set_molecule()`
- **Builder methods in `QuantUIApp`:** `_build_<section>()` — e.g., `_build_calc_setup()`
- **Callback methods:** `_on_<widget>_<event>()` — e.g., `_on_run_clicked()`, `_on_theme_changed()`
- **Panel populate methods:** `_pop_<panel_name>()` — e.g., `_pop_energies()`, `_pop_ir_spectrum()`
- **Module-level availability flags:** `_PYSCF_AVAILABLE`, `_PREOPT_AVAILABLE`, `ASE_AVAILABLE`
- **Config constants:** `ALL_CAPS_SNAKE_CASE` in `config.py`
- **Section banners in `app.py`:** `# ══ SECTION NAME ══` delimiters for VS Code outline navigation

---

## How to Run

```powershell
# Activate environment (Windows PowerShell)
& "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate quantui

# Voilà app mode (student-facing — no code visible)
voila notebooks/molecule_computations.ipynb

# JupyterLab (development)
jupyter lab notebooks/molecule_computations.ipynb

# Run tests
python -m pytest --tb=short -q

# Install/update package
pip install -e ".[dev]"

# Verify app.py import
python -c "from quantui.app import QuantUIApp; print('OK')"
```

**WSL note:** Always `cd` to the repo root before running Python in WSL. The `''`
entry in `sys.path` resolves to the current directory; running from the wrong directory
can shadow the editable install with a different repo's `quantui/` package.

**Python executable:** `C:\Users\schul\miniconda3\envs\quantui\python.exe`

Note: PySCF calculations will show "unavailable" on Windows — this is expected.
All UI, molecule, visualization, and PubChem features work natively on Windows.

---

## Testing

Test files in `tests/`:

| File | What it covers |
| --- | --- |
| `test_app.py` | QuantUIApp — widgets, panel registry, callbacks, always-visible panels |
| `test_molecule.py` | Molecule parsing, validation, formula |
| `test_session_calc.py` | `run_in_session()` — PySCF-gated |
| `test_notebook_workflows.py` | End-to-end HF/DFT/preopt/thread-safety — PySCF-gated |
| `test_optimizer.py` | `optimize_geometry()` — PySCF + ASE required |
| `test_freq_calc.py` | `run_freq_calc()` — PySCF-gated |
| `test_tddft_calc.py` | `run_tddft_calc()` — PySCF-gated |
| `test_nmr_calc.py` | `run_nmr_calc()` — PySCF + pyscf.prop required |
| `test_pes_scan.py` | `run_pes_scan()` + PES analysis panel |
| `test_ir_plot.py` | `plot_ir_spectrum()` — stick and broadened modes |
| `test_comparison.py` | Result comparison tables |
| `test_results_storage.py` | Save/load/list round-trip |
| `test_security.py` | `SecurityError`, `sanitize_filename()` |
| `test_code_quality.py` | Static analysis — bans CDN plotlyjs, bare except/pass *(pending M-PLOT-FIX.5)* |
| `test_sp/` | Single-point analysis history replay (end-to-end) |
| `test_geo_opt/` | Geometry opt analysis history replay |
| `test_tddft/` | TD-DFT / UV-Vis analysis history replay |
| `test_nmr/` | NMR analysis history replay |
| `test_freq_analysis_history.py` | Frequency analysis history replay (Vibrational + IR) |
| `test_pes_scan_analysis_history.py` | PES scan analysis history replay |

**PySCF-gated tests** use `@pytest.mark.skipif(not _PYSCF_AVAILABLE, ...)`.
On Windows, these become skips — not failures.

**Baseline (WSL, 2026-05-01 session 29):** 865 passed, 15 skipped.

---

## Optional Dependencies

| Extra | Packages | Gated by |
| --- | --- | --- |
| `pyscf` | `pyscf>=2.3.0` | `_PYSCF_AVAILABLE` flag |
| `ase` | `ase>=3.22.0` | `ASE_AVAILABLE` flag |
| `plotly` | `plotly` | checked inline at render time |
| `plotlymol` | `plotlymol3d` (local) | checked inline at render time |
| `app` | `voila, jupyterlab` | Always present in the conda env |

Install all: `pip install -e ".[pyscf,ase,app,dev]"`

---

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `QUANTUI_RESULTS_DIR` | `./results` | Where calculation results are saved |
| `QUANTUI_LOG_DIR` | `~/.quantui/logs` | Where perf_log and event_log live |

---

## Apptainer Container

The container at `apptainer/quantui.def` bundles Python + PySCF + ASE +
py3Dmol + Voilà into a single portable `.sif` file. This is the supported path
for Windows users.

- Build: `bash apptainer/build.sh --clean` (requires Linux/WSL with Apptainer ≥ 1.0)
- Run (app mode): `apptainer run quantui.sif app`
- Run (JupyterLab): `apptainer run quantui.sif`
- Verify: `apptainer test quantui.sif`

The container sets `QUANTUI_RESULTS_DIR=$HOME/.quantui/results` so results survive
across kernel restarts and are accessible from the host (home dir is bind-mounted).

---

## Relationship to Source Repo

QuantUI is a downstream port of `NCCU-Schultz-Lab/QuantUI` (the cluster version).
Bug fixes and module updates originate in `QuantUI` and are ported here.

| Removed from source | Reason |
| --- | --- |
| `job_manager.py` | SLURM batch submission |
| `storage.py` | SLURM job metadata |
| `slurm_errors.py` | SLURM error translation |
| `visualization.py` | PlotlyMol fallback (excluded here) |
| SLURM templates in `config.py` | No cluster |
