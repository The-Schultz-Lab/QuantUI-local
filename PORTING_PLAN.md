# QuantUI-local — Porting Plan

**Source repo:** `repos-DEVS/QuantUI`
**Target repo:** `repos-DEVS/QuantUI-local`
**Scope:** Teaching-only, PySCF in-session, no SLURM, no cluster

---

## What This Repo Will Be

A lightweight Jupyter interface for running PySCF quantum chemistry calculations
locally (or on any Linux/Mac/WSL machine). No cluster account, no SLURM, no
Apptainer. Students install via conda/pip and open a single notebook.

**Single execution path:** molecule input → `session_calc.run_in_session()` → results

---

## Files to NOT Port (SLURM infrastructure — delete entirely)

| File | Reason |
|------|--------|
| `quantui/job_manager.py` | Entire module is SLURM: sbatch, squeue, job lifecycle |
| `quantui/storage.py` | Job metadata persistence keyed to SLURM job IDs |
| `quantui/slurm_errors.py` | Translates SLURM error codes to student messages |
| `quantui/visualization.py` | PlotlyMol fallback — PlotlyMol excluded; py3Dmol only |
| `apptainer/quantui-jupyter.def` | Older/experimental definition |
| `apptainer/quantui-jupyter-conda.def` | Older/experimental definition |
| `apptainer/quantui-jupyter-locked.def` | Older/experimental definition |
| `apptainer/quantui-jupyter-with-plotlymol.def` | PlotlyMol not in this repo |
| `apptainer/quantui-jupyter-dual-viz.def` | Older/experimental definition |
| `apptainer/test-minimal.def` | Dev testing only |
| `apptainer/test-minimal-compatible.def` | Dev testing only |
| `apptainer/build-on-compute-node.sh` | HPC-specific build script |
| `scripts/` | Build/maintenance utilities for cluster deployment |

---

## Phase 0 — Apptainer Container (port with minor edits)

Source: `apptainer/quantui-jupyter-py3dmol.def` — already clean, no SLURM anywhere.

**Why keep it:** PySCF requires Linux. Windows students can't install it natively.
The container is the "just works" path: build once, run on any machine with Apptainer
(Linux/Mac/WSL). Also gives a clean Voilà app entry point for classroom demos.

**File to create:** `apptainer/quantui-local.def`

**Changes from source:**

| Change | Detail |
| --- | --- |
| `%files` path | `'/hpc/home/jschultz1/QuantUI /opt/quantui'` → `'. /opt/quantui'` (build from repo root) |
| `%labels` | Update Version, Purpose to reflect local teaching use |
| `%help` | Update description |
| `pip install rdkit` | Remove — PlotlyMol not in this repo |
| Runscript notebook path | `/opt/quantui/notebooks/molecule_computations.ipynb` — same, no change needed |

**Build & run story for students:**

```bash
# Build (once, takes ~5 min)
apptainer build quantui-local.sif apptainer/quantui-local.def

# Run Voilà app (clean UI, no code visible)
apptainer run quantui-local.sif app

# Run JupyterLab (dev/exploration mode)
apptainer run quantui-local.sif
```

Also port `apptainer/BUILD_QUANTUI_CONTAINER.md` → `apptainer/BUILD.md` with updated paths.

---

## Phase 1 — Scaffold (new files, written from scratch)

### 1a. `pyproject.toml`

New simplified version. Key changes from source:
- Name: `quantui-local` (PyPI name); import name stays `quantui`
- Remove optional groups: `docs` (Sphinx)
- Keep optional groups: `pyscf`, `ase`, `app` (Voilà), `dev`, `notebook`
- Remove from core deps: `kaleido` (only needed for static export in SLURM outputs)
- Add install note about Linux/WSL requirement for PySCF
- Simpler entry points (no CLI subcommands needed)

**Why keep Voilà:** locally it lets you serve the notebook as a clean widget-only
app (`voila notebooks/molecule_computations.ipynb`) — students see only the UI,
no code. Ideal for classroom demos or guided lab sessions via JupyterHub.
Install via `pip install "quantui-local[app]"`.

```toml
# Rough shape:
[project]
name = "quantui-local"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "jupyter>=1.0.0",
    "ipywidgets>=8.0.0",
    "notebook>=7.0.0",
    "numpy>=1.24.0",
    "requests>=2.28.0",
    "py3Dmol>=2.0.0",
    "matplotlib>=3.7.0",
    "plotly>=5.0.0",
]

[project.optional-dependencies]
pyscf = ["pyscf>=2.3.0"]
ase   = ["ase>=3.22.0"]
app   = ["voila>=0.5.0", "ipykernel"]
dev   = ["pytest>=7.0", "pytest-cov", "pytest-mock", "mypy"]
```

### 1b. `.gitignore`

Standard Python gitignore plus:
- `__pycache__/`, `*.pyc`, `*.pyo`
- `.venv/`, `*.egg-info/`
- `config.local.py`
- `CLAUDE.md` (local AI context, not for public repo)
- `.ipynb_checkpoints/`
- `*.npz`, `*.out`, `*.err` (calculation output files)

### 1c. `CLAUDE.md` (git-ignored)

Local AI working context file. Should document:
- What this repo is and is not
- The source repo relationship (QuantUI → QuantUI-local)
- PySCF Linux/WSL constraint
- How to run: `conda activate quantui && jupyter lab`

### 1d. `local-setup/environment.yml`

Conda environment definition for students. Mirror source with SLURM-specific
packages removed. Include a note that PySCF requires Linux/Mac/WSL.

---

## Phase 2 — Direct Copies (no changes needed)

Copy these files verbatim. They have no SLURM dependencies:

| Source file | Notes |
|-------------|-------|
| `quantui/molecule.py` | Pure data: atoms, coordinates, validation, presets |
| `quantui/pubchem.py` | PubChem API; pure internet I/O |
| `quantui/visualization_py3dmol.py` | 3D viewer; no cluster deps |
| `quantui/orbital_visualization.py` | Orbital energy diagrams, cube file parsing |
| `quantui/comparison.py` | Side-by-side comparison tables/plots |
| `quantui/progress.py` | Progress bar widget |
| `quantui/ase_bridge.py` | ASE structure I/O; molecule ↔ ASE.Atoms |
| `quantui/preopt.py` | LJ force-field pre-optimization |
| `quantui/optimizer.py` | QM geometry optimization + trajectory |
| `quantui/session_calc.py` | **Core runner** — this is the whole point |

### Verify after copying
Run `python -c "from quantui import Molecule, run_in_session"` and confirm
no import errors on SLURM-free modules.

---

## Phase 3 — Port with Edits

### 3a. `quantui/config.py`

**Strip:**
- `SLURM_SCRIPT_TEMPLATE` (entire string)
- `SLURM_SCRIPT_OPTIONAL_DIRECTIVES_TEMPLATE`
- SLURM resource defaults and limits block:
  `DEFAULT_CORES`, `DEFAULT_MEMORY_GB`, `DEFAULT_WALLTIME`, `MAX_CORES`,
  `MAX_MEMORY_GB`, `VALID_WALLTIME_OPTIONS`, `DEFAULT_PARTITION`
- Container settings: `USE_APPTAINER_BATCH`, `APPTAINER_BATCH_IMAGE`
- Email notification settings: `ALLOWED_MAIL_EVENTS`, `DEFAULT_MAIL_EVENTS`
- `JOB_HISTORY_DIR`, `CALCULATIONS_DIR` (no persistent job storage)
- `JOB_ID_FORMAT`, `TIMESTAMP_FORMAT`
- Local config override block (lines loading `config.local.py`) — simplify
- `STATUS_REFRESH_INTERVAL` (no job polling)

**Keep:**
- `SUPPORTED_METHODS`, `SUPPORTED_BASIS_SETS`, `DEFAULT_METHOD`, `DEFAULT_BASIS`
- `DEFAULT_CHARGE`, `DEFAULT_MULTIPLICITY`
- `DEFAULT_FMAX`, `DEFAULT_OPT_STEPS` (geometry optimization)
- All 20+ `MOLECULE_LIBRARY` presets — these are the teaching core
- `VALID_ATOMS` set
- `QUICK_START_TEMPLATES` — keep the 4 templates, remove resource fields from each
- `WIDGET_LAYOUT`, `DESCRIPTION_WIDTH`
- `PYSCF_SCRIPT_TEMPLATE` — keep as an "export/save script" feature
- Logging config

**Edit:**
- `PROJECT_ROOT` — still valid; just points to repo root
- Remove cluster-specific override variables from local config block

### 3b. `quantui/utils.py`

Read the full file before editing. Expected strips:
- `parse_job_timestamp()` — SLURM job ID parsing (remove)
- `sanitize_filename()` — may be used for job dirs only (check usage; if yes, remove)
- Any SLURM path utilities

**Keep:**
- `get_session_resources()` — tells students if their machine can handle a calc
- `session_can_handle()` — same
- `validate_atom_symbol()`, `validate_coordinates()`, `validate_charge()`, `validate_multiplicity()`
- Logging helpers

### 3c. `quantui/security.py`

**Strip:**
- `check_concurrent_job_limit()` — SLURM concurrent job enforcement
- `validate_email()` — SLURM notification email validation
- `validate_mail_events()` — SLURM mail event validation
- `validate_user_path()` / `safe_join()` if they reference SLURM job directories

**Keep:**
- `SecurityError` exception class
- `validate_resources()` — still useful to cap local resource requests
- Any general path safety utilities

### 3d. `quantui/calculator.py`

**Repurpose** rather than remove. In a local context, the calculator has one
remaining use: generating a standalone `.py` script a student can download and
run independently (educational "look under the hood" feature).

- Remove `estimate_resources()` method (SLURM-oriented resource sizing)
- Remove `get_educational_notes()` if it references cluster wait times
- Rename class to `PySCFScript` or keep as `PySCFCalculation` — either works
- Keep `generate_calculation_script(output_path)` — useful as "Export Script" button

### 3e. `quantui/__init__.py`

**Strip these exports:**
```python
# Remove entirely:
from .job_manager import JobMetadata, JobStorage, SLURMJobManager, create_job_metadata
from .slurm_errors import translate_slurm_error, format_error_for_student, ...
from .storage import ...
```

**Simplify visualization imports:** Remove the dual-backend fallback block entirely.
Keep only the py3Dmol imports; drop `PLOTLYMOL_AVAILABLE`, `visualization.py` imports,
and the `get_available_backends()` / `get_installation_message()` dispatch logic.
py3Dmol is the one and only visualization backend.

**Keep all other exports** (molecule, pubchem, visualization_py3dmol, session_calc,
ase_bridge, preopt, optimizer, orbital_visualization, comparison, security,
help_content, progress).

### 3f. `quantui/help_content.py`

Scan for SLURM-specific help topics (`HELP_TOPICS` dict). Remove any entries
covering job submission, job status, SLURM errors, partitions, walltime.
Keep: molecule input help, basis set guide, method guide, results interpretation.

---

## Phase 4 — New Notebooks

### 4a. `notebooks/molecule_computations.ipynb` (primary — rewrite)

**Not a port — write fresh.** The existing notebook is deeply wired to the
SLURM submission UI. Structure of the new notebook:

```
Section 1: Setup & Imports
  - conda env check cell (tagged skip-execution, remove-input)
  - Import quantui; print dependency status

Section 2: Molecule Input
  - Tab widget: [Preset Library | XYZ Input | PubChem Search | File Upload]
  - 3D visualization output (auto-updates on molecule change)
  - Optional: pre-optimization checkbox (preopt.py)

Section 3: Calculation Setup
  - Method dropdown (RHF / UHF)
  - Basis set dropdown (STO-3G → cc-pVTZ)
  - Charge + multiplicity inputs
  - "Estimate compute time" helper (session_can_handle())

Section 4: Run
  - Single "Run Calculation" button
  - Live SCF output stream (stdout captured from PySCF)
  - Progress indicator

Section 5: Results
  - Energy (Hartree + eV)
  - HOMO-LUMO gap
  - Convergence status + iterations
  - 3D optimized geometry (if opt was requested)

Section 6: Export (optional)
  - "Save Results" → writes results.npz + summary.txt
  - "Export PySCF Script" → generates standalone .py file (uses calculator.py)

Section 7: Comparison (optional)
  - Accumulate multiple SessionResult objects
  - comparison_table_html() display
```

### 4b. `notebooks/tutorials/` (port with edits)

Port all 5 tutorials. For each:
- Remove any SLURM submission cells
- Replace "submit job and wait" narrative with "run in session"
- Update timing expectations (local calc vs. cluster job)
- Keep all chemistry content unchanged

| Tutorial | Notes |
|----------|-------|
| `01_first_calculation.ipynb` | Minimal edits — mainly narrative |
| `02_basis_set_effects.ipynb` | Core content unchanged; remove cluster timing notes |
| `03_multiplicity_radicals.ipynb` | Core content unchanged |
| `04_charged_species.ipynb` | Core content unchanged |
| `05_comparison_studies.ipynb` | Replace SLURM batch workflow with loop + list |

### 4c. `notebooks/analysis.ipynb` (port with edits)

Remove sections that load from SLURM job history directories.
Replace with: load from `results.npz` files or a list of `SessionResult` objects.

---

## Phase 5 — Tests

### What to port
- `tests/test_molecule.py` — copy verbatim
- `tests/test_pubchem.py` — copy verbatim (skip network-marked tests in CI)
- `tests/test_visualization.py` — copy verbatim
- `tests/test_session_calc.py` — copy; mark as `slow` + `integration` (needs PySCF)
- `tests/test_ase_bridge.py` — copy verbatim
- `tests/test_config.py` — port with edits; remove SLURM config assertions
- `tests/test_security.py` — port with edits; remove SLURM security test cases
- `tests/test_calculator.py` — port with edits; adapt for repurposed module

### What to drop
- `tests/test_job_manager.py`
- `tests/test_storage.py`
- `tests/test_slurm_errors.py`
- Any test that mocks `sbatch` or `squeue`

### `pytest.ini` / `pyproject.toml` test config
Copy markers section. Remove any SLURM-specific fixture paths.

---

## Work Order (Recommended Sequence)

```
1. Phase 1  — Scaffold (pyproject.toml, .gitignore, CLAUDE.md, environment.yml)
2. Phase 2  — Direct copies (12 files; just cp)
3. Phase 3a — config.py port (everything else depends on this)
3. Phase 3e — __init__.py port (controls what's importable)
4. Phase 3b — utils.py port
5. Phase 3c — security.py port
6. Phase 3d — calculator.py repurpose
7. Phase 3f — help_content.py edit
8. Smoke test: `python -c "import quantui; print(quantui.MOLECULE_LIBRARY.keys())"`
9. Phase 4a — new molecule_computations.ipynb
10. Phase 4b — port tutorials
11. Phase 4c — port analysis.ipynb
12. Phase 5  — tests
13. Final: run pytest, fix anything that breaks
```

---

## Known Gotchas

- **PySCF is Linux/Mac/WSL only.** Tests that call `run_in_session()` must be
  skipped on Windows (`pytest.mark.skipif(sys.platform == 'win32', ...)`).
  `session_calc.py` already raises `ImportError` gracefully; tests should
  handle that too.

- **`config.py` imports.** After stripping SLURM constants, anything that
  imported them from config will break. Grep for each removed name before
  deleting: `grep -r "DEFAULT_PARTITION\|MAX_CORES\|JOB_HISTORY_DIR" quantui/`.

- **`comparison.py` uses `summary_from_job_metadata()`.** That function takes
  a `JobMetadata` object (from storage.py). In the local version, only
  `summary_from_session_result()` is relevant. Either strip the job_metadata
  variant or keep it with a stub type to avoid the storage import.

- **`orbital_visualization.py` may reference output file paths** from SLURM
  job directories. Verify it can accept a path argument pointing to any
  `results.npz`, not just cluster job dirs.

- **`help_content.py` uses `ipywidgets.Tab`.** No changes needed there — just
  audit the topic content for SLURM references.
