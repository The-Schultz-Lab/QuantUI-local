"""
Vibrational frequency analysis using PySCF's analytical Hessian.

Runs an SCF calculation and then computes the analytical Hessian to
obtain vibrational frequencies, zero-point vibrational energy (ZPVE),
and (where available) IR intensities.

Platform notes
--------------
Requires PySCF — Linux / macOS / WSL only.

Educational value
-----------------
* Students see which vibrational modes are IR-active and which are not.
* ZPVE correction shows how quantum mechanical zero-point motion contributes
  to molecular stability.
* Imaginary frequencies flag a transition state or saddle point — the
  geometry should be optimised first.

Typical usage
-------------
>>> from quantui.freq_calc import run_freq_calc
>>> result = run_freq_calc(molecule, method="RHF", basis="STO-3G")
>>> print(result.frequencies_cm1[:6])  # first 6 vibrational modes
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import IO, List, Optional

from .molecule import Molecule
from .session_calc import HARTREE_TO_EV

logger = logging.getLogger(__name__)

# 1 cm^-1 = h·c·100 / E_h  (NIST 2018 CODATA)
_CM1_TO_HARTREE: float = 4.556335252912e-6

# Exact: 1 Hartree = HARTREE_TO_EV * e * N_A joules/mol
_HARTREE_TO_JMOL: float = 2625499.6  # J/mol per Hartree (NIST 2018 CODATA)


# ============================================================================
# Result dataclass
# ============================================================================


@dataclass
class ThermoData:
    """Thermochemical data from the harmonic approximation at 298.15 K / 1 atm.

    All energies are in Hartrees; entropy is in J/(mol·K).
    H and G include the SCF electronic energy.
    """

    zpve_hartree: float
    H_hartree: float
    S_jmol: float
    G_hartree: float
    temperature_k: float = 298.15


@dataclass
class FreqResult:
    """Structured output from a vibrational frequency analysis.

    Attributes:
        energy_hartree: SCF energy at the input geometry in Hartrees.
        homo_lumo_gap_ev: HOMO-LUMO gap in eV, or ``None``.
        converged: ``True`` if the SCF converged.
        n_iterations: Number of SCF macro-iterations.
        method: Calculation method (e.g. ``'RHF'``, ``'B3LYP'``).
        basis: Basis set (e.g. ``'STO-3G'``).
        formula: Hill-notation molecular formula.
        frequencies_cm1: Vibrational frequencies in cm⁻¹.  Negative values
            indicate imaginary frequencies (transition-state modes).
        ir_intensities: IR intensities in km/mol per mode.  Empty list if
            the IR calculation is not available.
        zpve_hartree: Zero-point vibrational energy in Hartrees, computed as
            ½·Σ(ν_i) for all positive-frequency modes.
    """

    energy_hartree: float
    homo_lumo_gap_ev: Optional[float]
    converged: bool
    n_iterations: int
    method: str
    basis: str
    formula: str
    frequencies_cm1: List[float] = field(default_factory=list)
    ir_intensities: List[float] = field(default_factory=list)
    zpve_hartree: float = 0.0
    thermo: Optional[ThermoData] = None
    displacements: Optional[List] = None
    """Normalized displacement vectors from PySCF harmonic analysis.

    Shape: ``(n_modes, n_atoms, 3)`` stored as a nested Python list.
    ``None`` if the Hessian calculation failed or PySCF version does not
    provide ``norm_mode``.
    """
    mo_energy_hartree: Optional[List] = None
    mo_occ: Optional[List] = None
    pyscf_mol_atom: Optional[List] = None
    pyscf_mol_basis: Optional[str] = None

    @property
    def energy_ev(self) -> float:
        """SCF energy in electronvolts."""
        return self.energy_hartree * HARTREE_TO_EV

    def n_real_modes(self) -> int:
        """Number of real (positive-frequency) vibrational modes."""
        return sum(1 for f in self.frequencies_cm1 if f > 0)

    def n_imaginary_modes(self) -> int:
        """Number of imaginary (negative-frequency) modes."""
        return sum(1 for f in self.frequencies_cm1 if f < 0)


# ============================================================================
# Main function
# ============================================================================


def run_freq_calc(
    molecule: Molecule,
    method: str = "RHF",
    basis: str = "STO-3G",
    progress_stream: Optional[IO[str]] = None,
) -> FreqResult:
    """Run SCF + analytical Hessian to obtain vibrational frequencies.

    The function first converges the SCF energy, then computes the analytical
    Hessian and performs a normal-mode analysis to extract frequencies and
    (optionally) IR intensities.

    For physically meaningful frequencies, the input geometry should be at
    (or near) a local energy minimum.  Frequencies from an unoptimised
    geometry will be large and potentially imaginary.

    Args:
        molecule: Validated :class:`~quantui.molecule.Molecule`.  Should be
            an optimised geometry for best results.
        method: SCF method — ``'RHF'``, ``'UHF'``, or a DFT functional
            name (e.g. ``'B3LYP'``).  Default: ``'RHF'``.
        basis: Basis set name.  Default: ``'STO-3G'``.
        progress_stream: Optional writable text stream for live PySCF output.

    Returns:
        :class:`FreqResult` with frequencies, ZPVE, and SCF properties.

    Raises:
        ImportError: If PySCF is not installed.
        RuntimeError: If the SCF calculation fails.  If the Hessian
            computation fails, frequencies are omitted and a warning is
            written to progress_stream — no exception is raised.
    """
    try:
        from pyscf import dft, gto, scf
        from pyscf.hessian import thermo as pyscf_thermo
    except ImportError as exc:
        raise ImportError(
            "PySCF is not installed — cannot run frequency analysis.\n"
            "PySCF requires Linux, macOS, or WSL."
        ) from exc

    stream: IO[str] = progress_stream if progress_stream is not None else sys.stdout

    # ── Build Mole object ────────────────────────────────────────────────────
    mol = gto.Mole()
    mol.atom = molecule.to_pyscf_format()
    mol.basis = basis
    mol.charge = molecule.charge
    mol.spin = molecule.multiplicity - 1
    mol.verbose = 4
    mol.stdout = stream
    mol.build()

    # ── SCF ──────────────────────────────────────────────────────────────────
    method_upper = method.upper()
    if method_upper == "RHF":
        mf = scf.RHF(mol)
    elif method_upper == "UHF":
        mf = scf.UHF(mol)
    else:
        mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        mf.xc = method

    try:
        energy_hartree = float(mf.kernel())
    except Exception as exc:
        raise RuntimeError(
            f"SCF failed for {molecule.get_formula()} ({method}/{basis}): {exc}"
        ) from exc

    converged = bool(getattr(mf, "converged", False))
    n_iterations = int(getattr(mf, "cycles", -1))

    # ── HOMO-LUMO gap (non-fatal) ────────────────────────────────────────────
    homo_lumo_gap_ev: Optional[float] = None
    try:
        import numpy as _np

        mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy
        if isinstance(mo_energy, (list, _np.ndarray)) and hasattr(
            mo_energy[0], "__len__"
        ):
            mo_e_ref, mo_occ_ref = mo_energy[0], mo_occ[0]
        else:
            mo_e_ref, mo_occ_ref = mo_energy, mo_occ
        n_occ = int((_np.array(mo_occ_ref) > 0).sum())
        if 0 < n_occ < len(mo_e_ref):
            homo_lumo_gap_ev = float(
                (mo_e_ref[n_occ] - mo_e_ref[n_occ - 1]) * HARTREE_TO_EV
            )
    except Exception:
        pass

    # ── MO data for orbital energy diagram (best-effort) ─────────────────────
    mo_energy_hartree: Optional[List] = None
    mo_occ_list: Optional[List] = None
    pyscf_mol_atom: Optional[List] = None
    try:
        import numpy as _np_mo

        _moe = mf.mo_energy
        _moo = mf.mo_occ
        if isinstance(_moe, (list, _np_mo.ndarray)) and hasattr(_moe[0], "__len__"):
            _moe, _moo = _moe[0], _moo[0]
        mo_energy_hartree = _np_mo.asarray(_moe, dtype=float).tolist()
        mo_occ_list = _np_mo.asarray(_moo, dtype=float).tolist()
        pyscf_mol_atom = [(str(s), list(map(float, c))) for s, c in mol._atom]
    except Exception:
        pass

    # ── Hessian + frequency analysis ─────────────────────────────────────────
    frequencies_cm1: List[float] = []
    ir_intensities: List[float] = []
    zpve_hartree: float = 0.0
    displacements: Optional[List] = None
    thermo_data: Optional[ThermoData] = None

    try:
        hess_obj = mf.Hessian()
        hess_obj.verbose = mol.verbose
        hess_obj.stdout = stream

        # Primary: Infrared.kernel() (pyscf-properties) computes the Hessian +
        # CPHF in one pass and stores IR intensities in ir_inten.
        # Fallback: plain hess_obj.kernel() when pyscf-properties is absent.
        _ir_inten_raw = None
        try:
            from pyscf.prop.infrared import rhf as _ir_rhf_mod

            _ir_obj = _ir_rhf_mod.Infrared(mf)
            _ir_obj.verbose = mol.verbose
            _ir_obj.stdout = stream
            _ir_obj.kernel()
            h = _ir_obj.de
            _ir_inten_raw = getattr(_ir_obj, "ir_inten", None)
        except ImportError:
            h = hess_obj.kernel()
        except Exception as _ir_exc:
            logger.warning(
                "Infrared.kernel() failed (%s); using Hessian.kernel() only",
                _ir_exc,
            )
            h = hess_obj.kernel()

        freq_info = pyscf_thermo.harmonic_analysis(mol, h)

        # freq_wavenumber entries may be complex numbers when PySCF uses a
        # complex square-root convention for imaginary modes.  Map them to
        # signed real values: negative = imaginary frequency.
        raw_freqs = freq_info["freq_wavenumber"]
        frequencies_cm1 = []
        for f in raw_freqs:
            if hasattr(f, "imag") and abs(f.imag) > abs(f.real):
                frequencies_cm1.append(float(-abs(f.imag)))
            else:
                frequencies_cm1.append(float(f.real if hasattr(f, "real") else f))

        # ZPVE = ½ · Σ ν_i (positive modes only), converted cm⁻¹ → Hartree
        zpve_hartree = sum(0.5 * f * _CM1_TO_HARTREE for f in frequencies_cm1 if f > 0)

        # Normalized displacement vectors: shape (n_modes, n_atoms, 3).
        # Stored as a nested Python list for JSON-friendliness and to avoid
        # a hard numpy dependency in the dataclass.
        try:
            import numpy as _np

            norm_mode = freq_info.get("norm_mode")
            if norm_mode is not None:
                # norm_mode has shape (n_modes, n_atoms*3) or (n_modes, n_atoms, 3);
                # reshape to (n_modes, n_atoms, 3) if needed.
                nm = _np.array(norm_mode, dtype=float)
                n_modes_out = nm.shape[0]
                n_atoms = len(molecule.atoms)
                if nm.ndim == 2:
                    nm = nm.reshape(n_modes_out, n_atoms, 3)
                displacements = nm.tolist()
        except Exception:
            displacements = None

        # IR intensities — two code paths depending on which Hessian route ran.
        def _store_ir(arr):
            """Validate length and store IR intensities."""
            nonlocal ir_intensities
            import numpy as _np_ir2

            _ir_arr = _np_ir2.asarray(arr, dtype=float).real.ravel().tolist()
            if len(_ir_arr) == len(frequencies_cm1):
                ir_intensities = _ir_arr
            else:
                logger.warning(
                    "IR intensity count (%d) != frequency count (%d); discarding",
                    len(_ir_arr),
                    len(frequencies_cm1),
                )

        if _ir_inten_raw is not None:
            try:
                _store_ir(_ir_inten_raw)
            except Exception as _ir_exc:
                logger.warning("IR intensities unavailable: %s", _ir_exc)

        # Thermochemistry at 298.15 K / 1 atm — best-effort
        try:
            import numpy as _np

            _freq_au = freq_info.get("freq_au")
            if _freq_au is None:
                _freq_au = _np.array(frequencies_cm1) * _CM1_TO_HARTREE
            else:
                # PySCF may return complex freq_au for imaginary modes; take real parts.
                _freq_au = _np.array(
                    [f.real if hasattr(f, "real") else f for f in _freq_au],
                    dtype=float,
                )

            # PySCF 2.x thermo() may or may not accept the pressure argument.
            try:
                _tout = pyscf_thermo.thermo(mf, _freq_au, 298.15, 101325)
            except TypeError:
                _tout = pyscf_thermo.thermo(mf, _freq_au, 298.15)

            # PySCF 2.x returns (value, unit_string) tuples; earlier versions
            # return plain floats.  _tv() extracts the numeric value either way.
            def _tv(v):
                if isinstance(v, (tuple, list)):
                    return float(v[0])
                if hasattr(v, "item"):
                    return float(v.item())
                return float(v)

            # PySCF 2.x (>=2.6) uses "H_tot"/"S_tot"; earlier versions used "H"/"S".
            _H_raw, _S_raw, _Z_raw = None, None, None
            for _k in ("H_tot", "H", "Htot", "H_0K"):
                if _tout.get(_k) is not None:
                    _H_raw = _tout[_k]
                    break
            for _k in ("S_tot", "S", "Stot"):
                if _tout.get(_k) is not None:
                    _S_raw = _tout[_k]
                    break
            for _k in ("ZPE", "zpve", "ZPE_vib"):
                if _tout.get(_k) is not None:
                    _Z_raw = _tout[_k]
                    break
            if _H_raw is None or _S_raw is None:
                raise KeyError(
                    f"Missing H or S in thermo dict (keys: {sorted(_tout.keys())})"
                )
            _H = _tv(_H_raw)
            _S = _tv(_S_raw)  # J/(mol·K)
            _zpve = _tv(_Z_raw) if _Z_raw is not None else zpve_hartree
            _G = _H - 298.15 * _S / _HARTREE_TO_JMOL
            thermo_data = ThermoData(
                zpve_hartree=_zpve,
                H_hartree=_H,
                S_jmol=_S,
                G_hartree=_G,
            )
        except Exception as _exc:
            logger.warning("Thermochemistry failed: %s", _exc)

    except Exception as exc:
        logger.warning("Hessian/frequency computation failed: %s", exc)
        if progress_stream is not None:
            try:
                progress_stream.write(f"\n⚠ Hessian failed: {exc}\n")
            except Exception:
                pass

    return FreqResult(
        energy_hartree=energy_hartree,
        homo_lumo_gap_ev=homo_lumo_gap_ev,
        converged=converged,
        n_iterations=n_iterations,
        method=method,
        basis=basis,
        formula=molecule.get_formula(),
        frequencies_cm1=frequencies_cm1,
        ir_intensities=ir_intensities,
        zpve_hartree=zpve_hartree,
        thermo=thermo_data,
        displacements=displacements,
        mo_energy_hartree=mo_energy_hartree,
        mo_occ=mo_occ_list,
        pyscf_mol_atom=pyscf_mol_atom,
        pyscf_mol_basis=basis,
    )
