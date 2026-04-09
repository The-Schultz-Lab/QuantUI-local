# QuantUI-local — Apptainer Container

The Apptainer container packages Python, PySCF, ASE, py3Dmol, and Voilà
into a single portable `.sif` file. It is the **recommended path for Windows
users** (via WSL) and for anyone who wants a zero-installation experience —
students copy one file and run it.

---

## Contents

| File | Purpose |
| --- | --- |
| `quantui-local.def` | Container definition file (the "recipe") |
| `build.sh` | Build script with clean/test/fakeroot options |
| `README.md` (this file) | Build, run, and distribution guide |

The compiled `.sif` image is **not** committed to git — it is too large (~4–5 GB).
Build it locally (see below) or download the latest release asset from the
[GitHub Releases page](https://github.com/The-Schultz-Lab/QuantUI-local/releases).

---

## Getting the container

### Option A — Download a pre-built release (easiest)

1. Go to [Releases](https://github.com/The-Schultz-Lab/QuantUI-local/releases)
2. Download `quantui-local.sif` from the latest release
3. Run it directly — no build step needed

### Option B — Build from source

Use the provided build script — see the [Build section](#building-from-source) below.

---

## Prerequisites (for building locally)

- **Apptainer ≥ 1.0** on Linux, macOS, or WSL
- ~6 GB free disk space (build scratch + final image)
- Internet access during build (packages are downloaded from conda-forge and PyPI)

### Install Apptainer on Ubuntu / WSL

```bash
sudo apt-get update
sudo apt-get install -y apptainer
```

For other platforms see the
[official Apptainer install docs](https://apptainer.org/docs/admin/main/installation.html).

---

## Building from source

Use the provided `build.sh` script. It handles the correct working directory,
optional clean builds, and post-build testing.

```bash
# From the repo root:
cd /path/to/QuantUI-local

# Standard build
bash apptainer/build.sh

# Build + run container tests immediately after
bash apptainer/build.sh --test

# Remove old .sif first (clean rebuild)
bash apptainer/build.sh --clean

# On HPC systems without root (uses --fakeroot)
bash apptainer/build.sh --fakeroot
```

Build time: **20–40 minutes** depending on internet speed and CPU.
Final image size: **~4–5 GB**.

The script must be run from the **repo root** (not from `apptainer/`) because
the `.def` file copies the entire repo root into the container with
`%files . /opt/quantui`.

---

## Running the container

### Voilà app mode — recommended for students

Launches the notebook as a clean widget-only interface. Students see no code.

```bash
apptainer run quantui-local.sif app
```

Then open a browser at [http://localhost:8866](http://localhost:8866).

### JupyterLab mode — for exploration or development

```bash
apptainer run quantui-local.sif
```

Then open the URL printed in the terminal (contains a login token).

### Bind a local directory

By default Apptainer binds your current working directory so you can
access local files (e.g. your own XYZ files or saved results) inside the
container:

```bash
# Work from a specific project folder
cd ~/my-calculations
apptainer run /path/to/quantui-local.sif app
```

### Custom port

```bash
# Voilà on port 9000
apptainer run quantui-local.sif app --port=9000
```

---

## Verifying the container

After building (or downloading), run the built-in test to confirm all
packages loaded correctly:

```bash
# Built-in %test section
apptainer test quantui-local.sif

# Manual import check
apptainer exec quantui-local.sif python -c "
import quantui, pyscf, ase, py3Dmol
from quantui import Molecule, parse_xyz_input
atoms, coords = parse_xyz_input('O 0 0 0\nH 0.757 0.587 0\nH -0.757 0.587 0')
mol = Molecule(atoms, coords)
print('OK:', mol.get_formula())
"
```

Expected output: `OK: H2O` and all import messages.

---

## Distributing to students

The `.sif` is a single self-contained file — share it however is convenient:

```bash
# Network drive / shared folder
cp quantui-local.sif /shared/drive/

# SCP to a department server students can pull from
scp quantui-local.sif user@server.dept.edu:/shared/tools/

# USB drive
cp quantui-local.sif /media/usb/
```

Students then run:

```bash
apptainer run quantui-local.sif app
```

No Python, no conda, no pip — everything is bundled.

---

## Rebuilding after code changes

```bash
# Pull latest code
git pull origin main

# Rebuild (overwrites existing .sif) and verify
bash apptainer/build.sh --clean --test
```

---

## Troubleshooting

### "No space left on device" during build

Apptainer uses `/tmp` as scratch space. Redirect it to somewhere with more room:

```bash
export APPTAINER_TMPDIR=~/apptainer-tmp
mkdir -p ~/apptainer-tmp
apptainer build quantui-local.sif apptainer/quantui-local.def
```

### "Permission denied" or "root required"

Use `--fakeroot` if your HPC or server supports it:

```bash
apptainer build --fakeroot quantui-local.sif apptainer/quantui-local.def
```

On a personal machine or in WSL you typically have root access and don't
need this flag.

### PySCF or conda download times out

PySCF is the largest package (~500 MB). If the download keeps timing out:

1. Try building during off-peak hours

2. Pre-download the conda packages:

   ```bash
   conda create -n build-cache pyscf -c conda-forge --download-only
   ```

3. Point Apptainer at a local conda mirror by editing `%post` in the `.def` file

### Container starts but PySCF crashes

PySCF requires OpenMP. Make sure the host kernel provides it (virtually all
modern Linux kernels do). If running in a restricted environment:

```bash
export OMP_NUM_THREADS=1
apptainer run quantui-local.sif app
```

---

## What's inside the container

| Layer | Contents |
| --- | --- |
| Base | `continuumio/miniconda3:latest` (Debian + conda) |
| conda-forge | jupyter, jupyterlab, ipywidgets, pyscf, numpy, scipy, matplotlib, plotly, h5py |
| pip | voila, ase, py3dmol, requests |
| QuantUI-local | installed from `/opt/quantui` (the repo root, copied at build time) |

The `.git` directory and `__pycache__` folders are removed during build to
keep the image lean.

---

## Updating the container version

Edit `%labels` in `quantui-local.def` to bump the version string, then rebuild:

```singularity
%labels
  Version "0.2.0"
```

Tag the git commit and push so the version is traceable:

```bash
git tag v0.2.0
git push origin v0.2.0
```
