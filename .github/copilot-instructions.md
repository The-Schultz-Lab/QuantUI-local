# QuantUI-local — AI Assistant Context

> Stable project context for GitHub Copilot, Claude, and other AI coding assistants.
> Describes what the project IS and how it is built — not where development currently
> stands (see `planning/TODO/STATUS.md` for that). Update this file when
> architecture or conventions change, not every session.

---

## Overview

QuantUI-local is an interactive Jupyter/Voilà interface for running PySCF quantum
chemistry calculations locally — no cluster account, no SLURM, no queueing. Students
design molecules, launch RHF/UHF/DFT calculations in their own Python kernel, and
visualize results in minutes. It is a downstream port of the cluster-focused
`QuantUI` repo with all SLURM infrastructure removed.

**Target audience:** Undergraduate chemistry students at North Carolina Central
University. The UI runs as a Voilà app — students never see code.

---

## Repository Structure

```
QuantUI-local/
├── quantui/                  ← Main Python package (imports as `quantui`)
│   ├── app.py                ← QuantUIApp class — all widgets, callbacks, state
│   ├── molecule.py           ← Molecule dataclass + XYZ/SMILES parsing
│   ├── session_calc.py       ← In-session PySCF runner (run_in_session)
│   ├── optimizer.py          ← QM geometry optimization (ASE-BFGS + PySCF)
│   ├── freq_calc.py          ← Frequency / vibrational analysis
│   ├── tddft_calc.py         ← TD-DFT UV-Vis excited states
│   ├── calculator.py         ← PySCFCalculation abstraction
│   ├── comparison.py         ← Side-by-side result comparison table
│   ├── results_storage.py    ← Persist/reload calculation results (JSON + log)
│   ├── calc_log.py           ← Performance + event logging (JSONL)
│   ├── pubchem.py            ← PubChem molecule search
│   ├── visualization_py3dmol.py ← 3D molecular viewer (py3Dmol)
│   ├── ase_bridge.py         ← ASE structure I/O + molecule library
│   ├── preopt.py             ← ASE force-field pre-optimisation (fast, no PySCF)
│   ├── progress.py           ← StepProgress widget
│   ├── help_content.py       ← HELP_TOPICS dict — in-app educational text
│   ├── orbital_visualization.py ← Orbital energy diagrams, cube file viewer
│   ├── config.py             ← All constants/defaults (methods, basis sets, etc.)
│   ├── utils.py              ← Session resource checks, sanitize_filename, etc.
│   └── security.py           ← SecurityError exception class
├── notebooks/
│   ├── molecule_computations.ipynb  ← Student-facing Voilà app (thin launcher)
│   └── tutorials/                   ← 01–05 step-by-step tutorial notebooks
├── tests/                    ← pytest suite (~440 tests)
├── planning/                 ← Planning docs (not committed to git)
│   ├── TODO/
│   │   ├── STATUS.md         ← Start here each session — current state
│   │   ├── TODO.md           ← Milestone task list with acceptance criteria
│   │   ├── DECISIONS.md      ← Resolved design decisions
│   │   └── GOTCHAS.md        ← Known pitfalls and deliberate deferrals
│   ├── archive/              ← Old SESSION-HANDOFF and FR specs
│   └── feature-requests.md   ← FR backlog
├── apptainer/
│   ├── quantui-local.def     ← Apptainer container definition
│   └── build.sh              ← Build script
├── local-setup/              ← Conda environment YAMLs
├── launch-app.bat            ← Windows double-click launcher (Voilà app mode)
├── launch-dev.bat            ← Windows double-click launcher (JupyterLab mode)
├── pyproject.toml            ← Package config (name: quantui-local, imports as quantui)
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
        │  _build_results_section()   → results_panel              │
        │  _build_history_section()   → history_panel              │
        │  _build_compare_section()   → compare_panel              │
        │  _build_output_tab()        → log viewer                 │
        │  _build_help_section()      → help panel                 │
        │  _assemble_tabs()           → root_tab (Tab widget)      │
        └──────────────────────────────────────────────────────────┘
                │  _do_run() dispatches by calc_type_dd.value:
                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Single Point  → session_calc.run_in_session()              │
    │  Geometry Opt  → optimizer.optimize_geometry()              │
    │  Frequency     → freq_calc.run_freq_calc()                  │
    │  UV-Vis        → tddft_calc.run_tddft_calc()                │
    └──────────────────────────────────────────────────────────────┘
                │
                ▼
    results_storage.save_result()   calc_log.log_perf_record()
```

**Tab order in the app:** Calculate (0) → History (1) → Compare (2) → Output (3) → Help (4)

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
| `notebooks/molecule_computations.ipynb` | Thin launcher — 3 cells only (do not add logic here) |
| `planning/TODO/STATUS.md` | **Read this first every session** — current state, git log, open tasks |
| `planning/feature-requests.md` | FR backlog |

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
   `with output_widget:` from a background thread.

4. **No new top-level dependencies** without updating both `pyproject.toml`
   and the Apptainer container `apptainer/quantui-local.def`.

5. **All constants in `config.py`.** Method names, basis sets, layout widths,
   and other shared literals must be defined in `config.py` and imported from
   there — never hard-coded in `app.py` or other modules.

6. **Result schema versioning.** `results_storage.py` uses `_SCHEMA_VERSION = 2`.
   Any new fields must be additive (never remove or rename existing keys). Bump
   `_SCHEMA_VERSION` only when a breaking change is unavoidable.

---

## Supported Calculations

| Calc type | Module | Key function | Returns |
| --- | --- | --- | --- |
| Single Point | `session_calc` | `run_in_session()` | `SessionResult` |
| Geometry Opt | `optimizer` | `optimize_geometry()` | `OptResult` |
| Frequency | `freq_calc` | `run_freq_calc()` | `FreqResult` |
| UV-Vis TD-DFT | `tddft_calc` | `run_tddft_calc()` | `TDDFTResult` |

**Supported methods:** RHF, UHF, B3LYP, PBE, PBE0, M06-2X (defined in
`config.SUPPORTED_METHODS`).

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
  _build_results_section()
  _build_history_section()
  _build_compare_section()
  _build_output_tab()
  _build_help_section()
  _assemble_tabs()             # builds self.root_tab
  _wire_callbacks()            # all .observe() and .on_click() wiring
```

### Key instance state

| Attribute | Type | Purpose |
| --- | --- | --- |
| `self._molecule` | `Optional[Molecule]` | Currently loaded molecule |
| `self._last_result` | `Optional[...]` | Most recent calculation result |
| `self._results` | `list` | All results from this session |
| `self._pyscf_available` | `bool` | Mirrors module-level `_PYSCF_AVAILABLE` |
| `self.root_tab` | `widgets.Tab` | Top-level displayed widget |
| `self.method_dd` | `widgets.Dropdown` | Selected QC method |
| `self.basis_dd` | `widgets.Dropdown` | Selected basis set |
| `self.calc_type_dd` | `widgets.Dropdown` | Single Point / Geo Opt / Frequency / UV-Vis |

### Molecule collapse/expand pattern

`mol_input_container` is a `widgets.VBox` whose `.children` is swapped:
- **Expanded** (initial): `[mol_input_expanded, mol_info_html, viz_output]`
- **Collapsed** (after `_set_molecule()`): `[mol_input_collapsed, viz_output]`

Clicking "Change molecule" re-expands.

### CSS injection

`display(HTML(_APP_CSS))` fires inside `display()` before `display(self.root_tab)`.
Never import this module in a context where IPython display is not available without
catching the resulting error — or just don't call `.display()` (instantiation is safe).

---

## Result Storage (`quantui/results_storage.py`)

Results are saved to timestamped subdirectories:
```
<QUANTUI_RESULTS_DIR>/<timestamp>_<formula>_<method>_<basis>/
    result.json    ← schema v2 (see below)
    pyscf.log      ← raw PySCF stdout (may be absent)
```

Default results dir: `Path("results")` relative to cwd, or `$QUANTUI_RESULTS_DIR`.
In the Apptainer container: `$HOME/.quantui/results`.

### result.json schema (version 2)

```json
{
  "_schema_version": 2,
  "timestamp": "YYYY-MM-DD_HH-MM-SS-ffffff",
  "calc_type": "single_point | geometry_opt | frequency | tddft",
  "formula": "H2O",
  "method": "RHF",
  "basis": "STO-3G",
  "energy_hartree": -75.0,
  "energy_ev": -2040.8,
  "homo_lumo_gap_ev": 12.3,
  "converged": true,
  "n_iterations": 12,
  "spectra": {
    "ir": {"frequencies_cm1": [...], "ir_intensities": [...], "zpve_hartree": 0.021},
    "uv_vis": {"excitation_energies_ev": [...], "oscillator_strengths": [...], "wavelengths_nm": [...]}
  }
}
```

Timestamp includes microseconds (`-%f`) to prevent same-second directory collisions.

---

## Performance Logging (`quantui/calc_log.py`)

Two JSONL files under `~/.quantui/logs/` (override with `$QUANTUI_LOG_DIR`):

| File | Contents | Retention |
| --- | --- | --- |
| `perf_log.jsonl` | One record per converged run: formula, n_atoms, n_electrons, method, basis, elapsed_s | Permanent |
| `event_log.jsonl` | Startup / calc_start / calc_done / calc_error events | 7-day auto-prune |

Key API: `log_perf_record()`, `get_perf_history()`, `get_recent_events(n)`,
`reset_perf_log()`, `estimate_time(n_atoms, n_electrons, method, basis)`.

---

## Naming Conventions

- **Functions:** `verb_noun()` — e.g., `parse_xyz_input()`, `run_in_session()`, `save_result()`
- **Classes:** `PascalCase` — e.g., `QuantUIApp`, `SessionResult`, `FreqResult`
- **Private methods/helpers:** leading underscore — e.g., `_do_run()`, `_set_molecule()`
- **Builder methods in `QuantUIApp`:** `_build_<section>()` — e.g., `_build_calc_setup()`
- **Callback methods:** `_on_<widget>_<event>()` — e.g., `_on_run_clicked()`, `_on_theme_changed()`
- **Module-level availability flags:** `_PYSCF_AVAILABLE`, `_PREOPT_AVAILABLE`, `ASE_AVAILABLE`
- **Config constants:** `ALL_CAPS_SNAKE_CASE` in `config.py`
- **Section banners in `app.py`:** `# ══ SECTION NAME ══` delimiters for VS Code outline navigation

---

## How to Run

```powershell
# Activate environment (Windows PowerShell)
& "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate quantui-local

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

**Python executable:** `C:\Users\schul\miniconda3\envs\quantui-local\python.exe`

Note: PySCF calculations will show "unavailable" on Windows — this is expected.
All UI, molecule, visualization, and PubChem features work natively on Windows.

---

## Testing

Test files in `tests/`:

| File | What it covers |
| --- | --- |
| `test_molecule.py` | Molecule parsing, validation, formula |
| `test_session_calc.py` | `run_in_session()` — PySCF-gated with `pyscf_only` marker |
| `test_notebook_workflows.py` | End-to-end HF/DFT/preopt/thread-safety — PySCF-gated |
| `test_optimizer.py` | `optimize_geometry()` — PySCF + ASE required |
| `test_comparison.py` | Result comparison tables |
| `test_results_storage.py` | Save/load/list round-trip |
| `test_security.py` | `SecurityError`, `sanitize_filename()` |
| `test_phase1.py` | `QuantUIApp` instantiation (no display) |

**PySCF-gated tests** use `@pytest.mark.skipif(not _PYSCF_AVAILABLE, ...)`.
On Windows, these become skips — not failures.

---

## Optional Dependencies

| Extra | Packages | Gated by |
| --- | --- | --- |
| `pyscf` | `pyscf>=2.3.0` | `_PYSCF_AVAILABLE` flag |
| `ase` | `ase>=3.22.0` | `ASE_AVAILABLE` flag |
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

The container at `apptainer/quantui-local.def` bundles Python + PySCF + ASE +
py3Dmol + Voilà into a single portable `.sif` file. This is the supported path
for Windows users.

- Build: `bash apptainer/build.sh --clean` (requires Linux/WSL with Apptainer ≥ 1.0)
- Run (app mode): `apptainer run quantui-local.sif app`
- Run (JupyterLab): `apptainer run quantui-local.sif`
- Verify: `apptainer test quantui-local.sif`

The container sets `QUANTUI_RESULTS_DIR=$HOME/.quantui/results` so results survive
across kernel restarts and are accessible from the host (home dir is bind-mounted).

---

## Relationship to Source Repo

QuantUI-local is a downstream port of `NCCU-Schultz-Lab/QuantUI` (the cluster version).
Bug fixes and module updates originate in `QuantUI` and are ported here.
Never make independent architectural changes in this repo — propose them in `QuantUI` first.

| Removed from source | Reason |
| --- | --- |
| `job_manager.py` | SLURM batch submission |
| `storage.py` | SLURM job metadata |
| `slurm_errors.py` | SLURM error translation |
| `visualization.py` | PlotlyMol fallback (excluded here) |
| SLURM templates in `config.py` | No cluster |

---

## Active Development Branch

Branch: `app-restructure` — FR-012 App Module Refactor in progress.
See `planning/TODO/STATUS.md` for current phase and uncommitted changes.
