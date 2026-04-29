# QuantUI

[![CI](https://github.com/The-Schultz-Lab/QuantUI/actions/workflows/ci.yml/badge.svg)](https://github.com/The-Schultz-Lab/QuantUI/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://the-schultz-lab.github.io/QuantUI/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org)

An interactive Jupyter interface for running quantum chemistry calculations
locally — no cluster account, no SLURM, no queueing. Students design
molecules, launch PySCF calculations in their own Python session, and
visualize results in minutes.

Built for classroom teaching at the
[Schultz Lab, North Carolina Central University](https://github.com/The-Schultz-Lab).

---

## What it does

- **Molecule input** — paste XYZ coordinates, draw from a 20+ preset library,
  or search PubChem by name or SMILES
- **3D visualization** — interactive py3Dmol or PlotlyMol viewer with a live
  backend toggle when both are installed; post-calculation structure rendered
  automatically in the results panel
- **In-session calculations** — RHF, UHF, 9 DFT functionals, MP2, and NMR
  shielding via PySCF, running in your Python kernel (no batch submission)
- **Implicit solvent** — PCM solvation (Water, Ethanol, THF, DMSO,
  Acetonitrile) via a single checkbox
- **Rich results** — total energy, HOMO-LUMO gap, Mulliken charges, dipole
  moment, thermochemistry (H, S, G at 298 K), IR spectrum chart (stick and
  Lorentzian-broadened), ¹H/¹³C NMR chemical shifts, orbital energy-level
  diagram, HOMO/LUMO isosurface (cube-file rendering with toggle for HOMO-1,
  HOMO, LUMO, LUMO+1), and a side-by-side comparison table for multiple
  calculations
- **Geometry optimization** — BFGS optimizer with step-by-step trajectory
  animation; vibrational frequency analysis with animated normal modes
- **Results persistence** — every calculation is saved automatically to a
  timestamped directory; a built-in browser lets students reload past results
  after a kernel restart; the full `pyscf.log` is shown inline
- **Structure exports** — download XYZ, MOL/SDF, or PDB files alongside the
  saved results; script export for a standalone `.py` file
- **Timing calibration** — one-click benchmark suite populates the time
  estimator with real machine data so predictions are accurate from the first run
- **Voilà app mode** — serve the notebook as a polished widget-only UI (no code
  visible) for classroom demos, with dark mode toggle and dedicated output log

---

## Platform requirements

| Platform | Works? | Notes |
| --- | --- | --- |
| Linux / macOS | Full | PySCF installs natively |
| WSL (Windows) | Full | Use an Ubuntu WSL environment |
| Windows (native) | Partial | All UI and visualization features work; PySCF calculations require the Apptainer container |

### Windows users: Apptainer container

PySCF does not install on Windows natively. The
[`apptainer/quantui.def`](apptainer/quantui.def) container bundles
the complete environment and runs anywhere Apptainer/Singularity is available.
See [`apptainer/README.md`](apptainer/README.md) for build and run instructions.

---

## Installation

### Option A — conda (recommended for Linux/macOS/WSL)

```bash
# Create a dedicated environment
conda create -n quantui python=3.11
conda activate quantui

# Install with PySCF and ASE
pip install -e ".[pyscf,ase,app]"
```

### Option B — pip only

```bash
python -m pip install quantui[pyscf,ase,app]
```

### Option C — Apptainer container (Windows / reproducible deployment)

See [apptainer/README.md](apptainer/README.md).

---

## Quick start

```bash
# Activate your environment
conda activate quantui

# JupyterLab (full IDE — shows code)
jupyter lab notebooks/molecule_computations.ipynb

# Voilà app mode (widget-only — for classroom demos)
voila notebooks/molecule_computations.ipynb
```

Open the notebook, pick a molecule, choose a method and basis set, and click
**Run Calculation**. Results appear directly in the notebook.

---

## Tutorials

Five step-by-step notebooks in [`notebooks/tutorials/`](notebooks/tutorials/):

| Notebook | Topic |
| --- | --- |
| [01_first_calculation.ipynb](notebooks/tutorials/01_first_calculation.ipynb) | Your first RHF calculation |
| [02_basis_set_study.ipynb](notebooks/tutorials/02_basis_set_study.ipynb) | Comparing STO-3G, 6-31G, cc-pVDZ |
| [03_multiplicity_radicals.ipynb](notebooks/tutorials/03_multiplicity_radicals.ipynb) | Open-shell molecules and UHF |
| [04_charged_species.ipynb](notebooks/tutorials/04_charged_species.ipynb) | Ions and charged systems |
| [05_comparing_results.ipynb](notebooks/tutorials/05_comparing_results.ipynb) | Side-by-side result analysis |

---

## Supported calculations

### Methods

| Method | Type | Best for |
| --- | --- | --- |
| RHF | Hartree-Fock | Closed-shell molecules; baseline reference |
| UHF | Hartree-Fock | Radicals and open-shell systems |
| B3LYP | DFT hybrid | General organic chemistry (default DFT choice) |
| PBE | DFT GGA | Large molecules; metals; when speed matters |
| PBE0 | DFT hybrid | Charge-transfer, band gaps |
| M06-2X | DFT meta-hybrid | Thermochemistry, barrier heights |
| wB97X-D | DFT range-sep. + D3 | Non-covalent interactions, excited states |
| CAM-B3LYP | DFT range-sep. | Charge-transfer UV-Vis, Rydberg states |
| M06-L | DFT local meta-GGA | Large molecules; transition metals |
| HSE06 | DFT screened hybrid | Band gaps, large molecules |
| PBE-D3 | DFT GGA + dispersion | Van der Waals complexes, stacking |
| MP2 | Post-HF | Accurate energetics for small molecules (O(N⁵)) |

### Calculation types

| Type | Output |
| --- | --- |
| Single Point | Energy, HOMO-LUMO gap, Mulliken charges, dipole moment |
| Geometry Opt | Optimised structure, trajectory animation |
| Frequency | Vibrational frequencies, ZPVE, IR intensities, thermochemistry (H/S/G at 298 K), animated normal modes, IR spectrum chart (stick / Lorentzian broadened) |
| UV-Vis (TD-DFT) | Excitation energies, oscillator strengths, UV-Vis spectrum plot |
| NMR Shielding | ¹H and ¹³C chemical shifts relative to TMS via GIAO; tabulated by element |

### Basis sets

STO-3G (fast, good for learning) → 3-21G → 6-31G / 6-31G\* / 6-31G\*\* →
cc-pVDZ / cc-pVTZ → def2-SVP / def2-TZVP

---

## Running tests

```bash
pip install -e ".[dev]"

# All tests (Linux/macOS — PySCF available)
pytest -m "not network"

# Skip PySCF-dependent tests (Windows without container)
pytest -m "not network" \
  --ignore=tests/test_session_calc.py \
  --ignore=tests/test_optimizer.py \
  --ignore=tests/test_preopt.py
```

---

## Project structure

```text
quantui/                  Main package
  app.py                  QuantUIApp widget class (all tabs, UI logic)
  molecule.py             Molecule input and validation
  session_calc.py         In-session PySCF runner (RHF/UHF/DFT/MP2/PCM)
  freq_calc.py            Vibrational frequency + thermochemistry analysis
  ir_plot.py              IR spectrum chart (stick and Lorentzian broadened)
  tddft_calc.py           TD-DFT UV-Vis excited-state calculations
  nmr_calc.py             NMR shielding + ¹H/¹³C chemical shift prediction
  optimizer.py            QM geometry optimization with trajectory
  visualization_py3dmol.py  3D viewer (py3Dmol + PlotlyMol backends)
  pubchem.py              PubChem molecule search
  comparison.py           Side-by-side result tables
  results_storage.py      Timestamped result persistence
  calc_log.py             Performance logging and time estimation
  benchmarks.py           Timing calibration benchmark suite
  config.py               Methods, basis sets, solvent/NMR options, presets
  ase_bridge.py           ASE structure I/O
  preopt.py               LJ force-field pre-optimization
notebooks/
  molecule_computations.ipynb   Main student-facing interface
  tutorials/                    Step-by-step guided notebooks (01–05)
tests/                    pytest test suite (575+ tests)
apptainer/                Container definition for reproducible deployment
local-setup/              Conda environment definition
pyproject.toml            Package metadata and tool config
```

---

## Relationship to the cluster version

QuantUI (this repo) is a downstream port of the cluster-based
[QuantUI-cluster](https://github.com/The-Schultz-Lab/QuantUI) repository. All SLURM
infrastructure (job manager, job storage, batch templates) has been removed.
Bug fixes flow from the cluster repo into this one, not the other way around.

---

## License

[MIT](LICENSE) — Copyright 2026 The Schultz Lab, North Carolina Central University
