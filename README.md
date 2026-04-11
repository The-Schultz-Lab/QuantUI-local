# QuantUI-local

[![CI](https://github.com/The-Schultz-Lab/QuantUI-local/actions/workflows/ci.yml/badge.svg)](https://github.com/The-Schultz-Lab/QuantUI-local/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://the-schultz-lab.github.io/QuantUI-local/)
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

- **Molecule input** — paste XYZ coordinates, draw from a preset library, or
  search PubChem by name or SMILES
- **3D visualization** — interactive py3Dmol viewer, directly in the notebook
- **In-session calculations** — RHF and UHF via PySCF, running in your Python
  kernel (no batch submission)
- **Results** — total energy, HOMO-LUMO gap, convergence status, and a
  side-by-side comparison table for multiple calculations
- **Results persistence** — every calculation is saved automatically to a
  timestamped directory; a built-in browser lets students reload past results
  after a kernel restart
- **Script export** — download a standalone `.py` file to run or study outside
  the notebook
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
[`apptainer/quantui-local.def`](apptainer/quantui-local.def) container bundles
the complete environment and runs anywhere Apptainer/Singularity is available.
See [`apptainer/README.md`](apptainer/README.md) for build and run instructions.

---

## Installation

### Option A — conda (recommended for Linux/macOS/WSL)

```bash
# Create a dedicated environment
conda create -n quantui-local python=3.11
conda activate quantui-local

# Install with PySCF and ASE
pip install -e ".[pyscf,ase,app]"
```

### Option B — pip only

```bash
python -m pip install quantui-local[pyscf,ase,app]
```

### Option C — Apptainer container (Windows / reproducible deployment)

See [apptainer/README.md](apptainer/README.md).

---

## Quick start

```bash
# Activate your environment
conda activate quantui-local

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

| Method | When to use |
| --- | --- |
| RHF | Closed-shell molecules — all electrons paired |
| UHF | Open-shell molecules — radicals or unpaired electrons |

**Basis sets:** STO-3G (fast, good for learning) → 6-31G (common research
choice) → cc-pVTZ (high accuracy)

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
  molecule.py             Molecule input and validation
  session_calc.py         In-session PySCF runner
  visualization_py3dmol.py  3D viewer
  pubchem.py              PubChem molecule search
  comparison.py           Side-by-side result tables
  ase_bridge.py           ASE structure I/O
  optimizer.py            QM geometry optimization
  ...
notebooks/
  molecule_computations.ipynb   Main student-facing interface
  tutorials/                    Step-by-step guided notebooks
tests/                    pytest test suite (439 tests)
apptainer/                Container definition for reproducible deployment
local-setup/              Conda environment definition
pyproject.toml            Package metadata and tool config
```

---

## Relationship to the cluster version

QuantUI-local is a downstream port of the cluster-based
[QuantUI](https://github.com/The-Schultz-Lab/QuantUI) repository. All SLURM
infrastructure (job manager, job storage, batch templates) has been removed.
Bug fixes flow `QuantUI → QuantUI-local`, not the other way around.

---

## License

[MIT](LICENSE) — Copyright 2026 The Schultz Lab, North Carolina Central University
