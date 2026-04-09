# Building the QuantUI-local Container

The Apptainer container bundles Python, PySCF, ASE, and Voilà in a single file.
It is the recommended path for **Windows users** (via WSL) and for anyone who
wants a zero-installation experience.

---

## Prerequisites

- Apptainer installed (Linux/macOS/WSL)
- ~6 GB free disk space for build + final image
- Internet access during build

Install Apptainer on Ubuntu/WSL:
```bash
sudo apt-get install -y apptainer
```

---

## Build

Run from the **repo root** (not from `apptainer/`):

```bash
apptainer build quantui-local.sif apptainer/quantui-local.def
```

Build time: 20–40 minutes depending on internet speed and CPU.
Final image size: ~4–5 GB.

---

## Run

```bash
# Voilà app — clean widget UI, no code visible (recommended for students)
apptainer run quantui-local.sif app

# JupyterLab — full notebook interface (for exploration or development)
apptainer run quantui-local.sif
```

Both modes bind your current directory so you can access local files inside
the container.

---

## Verify

```bash
# Quick import check
apptainer exec quantui-local.sif python -c "
import quantui, pyscf, ase, py3Dmol
print('All imports OK')
"

# Run container's built-in test suite
apptainer test quantui-local.sif
```

---

## Sharing with Students

Place the built `.sif` file somewhere students can copy it:

```bash
# Students copy and run — no installation needed
cp quantui-local.sif ~/
apptainer run ~/quantui-local.sif app
```

Or share via a network drive / USB. The `.sif` is a single self-contained file.

---

## Rebuilding After Code Changes

```bash
# Pull latest code
git pull origin main

# Rebuild (overwrites existing .sif)
apptainer build quantui-local.sif apptainer/quantui-local.def
```

> Note: `.sif` files are listed in `.gitignore` — do not commit them.

---

## Troubleshooting

**"No space left on device" during build**

Apptainer uses `/tmp` for scratch. Redirect it:
```bash
export APPTAINER_TMPDIR=~/apptainer-tmp
mkdir -p ~/apptainer-tmp
apptainer build quantui-local.sif apptainer/quantui-local.def
```

**"Permission denied" building without root**

Use `--fakeroot` if your system supports it, or build inside WSL where you have
root access:
```bash
apptainer build --fakeroot quantui-local.sif apptainer/quantui-local.def
```

**PySCF install times out**

PySCF is the largest package. Try building during off-peak hours, or
pre-download the wheel:
```bash
pip download pyscf --dest ./wheels
```
Then add a `%files` line to copy `./wheels` into the container and install
from the local copy.
