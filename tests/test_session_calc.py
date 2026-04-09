"""
Tests for quantui.session_calc — ASE-PySCF in-session calculator.

Test strategy
-------------
* Import-guard tests run on any platform (including Windows) — they verify
  that helpful ImportError messages are raised when ASE or PySCF is absent.
* Calculation tests require both ASE and PySCF, so they are marked with
  ``pyscf_only`` and skipped everywhere PySCF is unavailable (Windows, CI
  without the pyscf extra).
* SessionResult unit tests (dataclass behaviour, summary formatting) run
  unconditionally — they construct the dataclass directly without PySCF.

WSL / Linux testing
--------------------
Run the full suite on your WSL terminal with both ase and pyscf installed:
    pytest tests/test_session_calc.py -v

Run only the fast, platform-independent tests anywhere:
    pytest tests/test_session_calc.py -v -k "not pyscf_only"
"""

import io
import pytest

from quantui.ase_bridge import ASE_AVAILABLE
from quantui.molecule import Molecule
from quantui.session_calc import HARTREE_TO_EV, SessionResult

# Check for PySCF availability independently of ASE
_PYSCF_AVAILABLE = False
try:
    import pyscf as _pyscf  # noqa: F401
    _PYSCF_AVAILABLE = True
except ImportError:
    pass

_ASE_PYSCF_AVAILABLE = False
try:
    from ase.calculators.pyscf import PySCF as _check  # noqa: F401
    _ASE_PYSCF_AVAILABLE = True
except ImportError:
    pass

# Skip marker for tests that need the full ASE-PySCF stack
pyscf_only = pytest.mark.skipif(
    not (ASE_AVAILABLE and _PYSCF_AVAILABLE and _ASE_PYSCF_AVAILABLE),
    reason="ase>=3.22 and pyscf not both installed (Linux/WSL only)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h2() -> Molecule:
    """H2 with equilibrium geometry — fastest meaningful QM calculation."""
    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


def _water() -> Molecule:
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


def _make_result(**overrides) -> SessionResult:
    """Build a SessionResult with sensible defaults, allowing field overrides."""
    defaults = dict(
        energy_hartree=-1.117,
        homo_lumo_gap_ev=10.5,
        converged=True,
        n_iterations=8,
        method="RHF",
        basis="STO-3G",
        formula="H2",
    )
    defaults.update(overrides)
    return SessionResult(**defaults)


# ============================================================================
# SessionResult dataclass — no PySCF needed
# ============================================================================


class TestSessionResultDataclass:
    """Unit tests for SessionResult fields, properties, and summary."""

    def test_energy_ev_property(self):
        result = _make_result(energy_hartree=1.0)
        assert abs(result.energy_ev - HARTREE_TO_EV) < 1e-9

    def test_energy_ev_negative(self):
        result = _make_result(energy_hartree=-1.117)
        assert result.energy_ev == pytest.approx(-1.117 * HARTREE_TO_EV, rel=1e-9)

    def test_summary_contains_formula(self):
        result = _make_result(formula="H2O")
        assert "H2O" in result.summary()

    def test_summary_contains_method_basis(self):
        result = _make_result(method="UHF", basis="cc-pVDZ")
        summary = result.summary()
        assert "UHF" in summary
        assert "cc-pVDZ" in summary

    def test_summary_converged_shows_yes(self):
        result = _make_result(converged=True)
        assert "Yes" in result.summary()

    def test_summary_not_converged_shows_warning(self):
        result = _make_result(converged=False)
        summary = result.summary()
        assert "NO" in summary or "not converge" in summary.lower()

    def test_summary_contains_energy(self):
        result = _make_result(energy_hartree=-76.1234)
        assert "-76.1234" in result.summary()

    def test_summary_contains_homo_lumo_gap(self):
        result = _make_result(homo_lumo_gap_ev=8.5432)
        assert "8.5432" in result.summary()

    def test_summary_omits_gap_when_none(self):
        result = _make_result(homo_lumo_gap_ev=None)
        assert "HOMO" not in result.summary()

    def test_summary_contains_iterations(self):
        result = _make_result(n_iterations=13)
        assert "13" in result.summary()

    def test_all_fields_accessible(self):
        result = _make_result()
        assert isinstance(result.energy_hartree, float)
        assert isinstance(result.converged, bool)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.method, str)
        assert isinstance(result.basis, str)
        assert isinstance(result.formula, str)

    def test_homo_lumo_gap_can_be_none(self):
        result = _make_result(homo_lumo_gap_ev=None)
        assert result.homo_lumo_gap_ev is None


# ============================================================================
# Import-guard tests — run on all platforms
# ============================================================================


class TestRunInSessionImportGuards:
    """run_in_session() raises ImportError with helpful messages when deps absent."""

    def test_raises_when_ase_unavailable(self, monkeypatch):
        import quantui.ase_bridge as bridge
        monkeypatch.setattr(bridge, "ASE_AVAILABLE", False)

        import importlib
        import quantui.session_calc as sc
        importlib.reload(sc)
        from quantui.session_calc import run_in_session

        with pytest.raises(ImportError, match="pip install"):
            run_in_session(_h2())

    def test_import_error_message_is_actionable(self, monkeypatch):
        import quantui.ase_bridge as bridge
        monkeypatch.setattr(bridge, "ASE_AVAILABLE", False)

        import importlib
        import quantui.session_calc as sc
        importlib.reload(sc)
        from quantui.session_calc import run_in_session

        with pytest.raises(ImportError) as exc_info:
            run_in_session(_h2())
        msg = str(exc_info.value)
        assert "pip install" in msg or "conda install" in msg


# ============================================================================
# Calculation tests — Linux/WSL with ase + pyscf
# ============================================================================


class TestRunInSessionBasic:
    """Basic functional tests for run_in_session()."""

    @pyscf_only
    @pytest.mark.slow
    def test_returns_session_result(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        assert isinstance(result, SessionResult)

    @pyscf_only
    @pytest.mark.slow
    def test_h2_rhf_sto3g_converges(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        assert result.converged is True

    @pyscf_only
    @pytest.mark.slow
    def test_h2_energy_plausible(self):
        """RHF/STO-3G energy for H2 near equilibrium should be around -1.117 Ha."""
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        # Literature: ~-1.117 Ha; allow ±0.1 Ha for geometry/basis variation
        assert -1.25 < result.energy_hartree < -1.0

    @pyscf_only
    @pytest.mark.slow
    def test_result_formula_matches_molecule(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        assert result.formula == "H2"

    @pyscf_only
    @pytest.mark.slow
    def test_result_method_matches_input(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        assert result.method == "RHF"

    @pyscf_only
    @pytest.mark.slow
    def test_result_basis_matches_input(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        assert result.basis == "STO-3G"

    @pyscf_only
    @pytest.mark.slow
    def test_homo_lumo_gap_is_positive(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        if result.homo_lumo_gap_ev is not None:
            assert result.homo_lumo_gap_ev > 0

    @pyscf_only
    @pytest.mark.slow
    def test_n_iterations_is_positive(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        assert result.n_iterations > 0


class TestRunInSessionOutputStream:
    """Verify that PySCF output is routed to the progress_stream."""

    @pyscf_only
    @pytest.mark.slow
    def test_output_written_to_stream(self):
        from quantui.session_calc import run_in_session

        buf = io.StringIO()
        run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=3, progress_stream=buf)
        output = buf.getvalue()
        # With verbose=3, PySCF writes SCF cycle information
        assert len(output) > 0

    @pyscf_only
    @pytest.mark.slow
    def test_silent_when_verbose_zero(self):
        from quantui.session_calc import run_in_session

        buf = io.StringIO()
        run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0, progress_stream=buf)
        output = buf.getvalue()
        # verbose=0 should produce little or no output
        assert len(output) < 500  # allow for minimal header lines


class TestRunInSessionMetadata:
    """run_in_session() preserves charge and multiplicity into the calculation."""

    @pyscf_only
    @pytest.mark.slow
    def test_water_rhf_converges(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_water(), method="RHF", basis="STO-3G", verbose=0)
        assert result.converged is True

    @pyscf_only
    @pytest.mark.slow
    def test_water_energy_plausible(self):
        """RHF/STO-3G for H2O should be around -74.96 Ha."""
        from quantui.session_calc import run_in_session

        result = run_in_session(_water(), method="RHF", basis="STO-3G", verbose=0)
        assert -76.0 < result.energy_hartree < -73.0


# ============================================================================
# Public API surface
# ============================================================================


class TestPublicAPI:
    """SessionResult and run_in_session are importable from quantui top-level."""

    def test_session_result_importable_from_quantui(self):
        from quantui import SessionResult  # noqa: F401

    def test_run_in_session_importable_from_quantui(self):
        from quantui import run_in_session  # noqa: F401

    def test_hartree_to_ev_constant(self):
        """Sanity-check the conversion constant against a known value."""
        assert abs(HARTREE_TO_EV - 27.211) < 0.01


# ============================================================================
# Run directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
