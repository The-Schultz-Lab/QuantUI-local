"""
Tests for quantui.comparison

Covers CalcSummary construction, the builders (from JobMetadata and
SessionResult), the HTML table renderer, and the matplotlib comparison chart.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from quantui.comparison import (
    CalcSummary,
    comparison_table_html,
    plot_comparison,
    summary_from_job_metadata,
    summary_from_session_result,
    HARTREE_TO_EV,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_summaries():
    return [
        CalcSummary(
            label="H2O RHF/STO-3G",
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            energy_hartree=-75.5,
            homo_lumo_gap_ev=12.5,
            converged=True,
            n_iterations=10,
            status="COMPLETED",
        ),
        CalcSummary(
            label="H2O RHF/6-31G",
            formula="H2O",
            method="RHF",
            basis="6-31G",
            energy_hartree=-76.0,
            homo_lumo_gap_ev=13.1,
            converged=True,
            n_iterations=8,
            status="COMPLETED",
        ),
    ]


@pytest.fixture()
def mock_job_metadata(tmp_path):
    """Create a mock JobMetadata with a real results.npz."""
    mo_energy = np.array([-0.6, -0.3, 0.2, 0.8])
    mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
    np.savez(
        tmp_path / "results.npz",
        mo_energy=mo_energy,
        mo_occ=mo_occ,
        energy=-75.5,
        converged=True,
    )
    meta = MagicMock()
    meta.job_name = "H2O_RHF_STO-3G"
    meta.method = "RHF"
    meta.basis = "STO-3G"
    meta.status = "COMPLETED"
    meta.paths = {"results": str(tmp_path / "results.npz")}
    return meta


@pytest.fixture()
def mock_session_result():
    result = MagicMock()
    result.formula = "H2O"
    result.method = "RHF"
    result.basis = "6-31G"
    result.energy_hartree = -76.0
    result.homo_lumo_gap_ev = 13.1
    result.converged = True
    result.n_iterations = 8
    return result


# ---------------------------------------------------------------------------
# CalcSummary basics
# ---------------------------------------------------------------------------

class TestCalcSummary:

    def test_energy_ev_conversion(self):
        s = CalcSummary(label="x", formula="H2", method="RHF", basis="STO-3G",
                        energy_hartree=-1.0)
        assert abs(s.energy_ev - (-1.0 * HARTREE_TO_EV)) < 1e-6

    def test_energy_ev_none(self):
        s = CalcSummary(label="x", formula="H2", method="RHF", basis="STO-3G")
        assert s.energy_ev is None

    def test_defaults(self):
        s = CalcSummary(label="x", formula="H2", method="RHF", basis="STO-3G")
        assert s.converged is None
        assert s.n_iterations is None
        assert s.status == ""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

class TestSummaryFromJobMetadata:

    def test_loads_energy(self, mock_job_metadata):
        s = summary_from_job_metadata(mock_job_metadata)
        assert s.energy_hartree is not None
        assert abs(s.energy_hartree - (-75.5)) < 1e-6

    def test_loads_convergence(self, mock_job_metadata):
        s = summary_from_job_metadata(mock_job_metadata)
        assert s.converged is True

    def test_computes_gap(self, mock_job_metadata):
        s = summary_from_job_metadata(mock_job_metadata)
        assert s.homo_lumo_gap_ev is not None
        assert s.homo_lumo_gap_ev > 0

    def test_uses_label(self, mock_job_metadata):
        s = summary_from_job_metadata(mock_job_metadata, label="custom")
        assert s.label == "custom"

    def test_fallback_label(self, mock_job_metadata):
        s = summary_from_job_metadata(mock_job_metadata)
        assert s.label == "H2O_RHF_STO-3G"

    def test_missing_results_file(self, tmp_path):
        meta = MagicMock()
        meta.job_name = "NH3_RHF_STO-3G"
        meta.method = "RHF"
        meta.basis = "STO-3G"
        meta.status = "PENDING"
        meta.paths = {"results": str(tmp_path / "nonexistent.npz")}
        s = summary_from_job_metadata(meta)
        assert s.energy_hartree is None
        assert s.converged is None


class TestSummaryFromSessionResult:

    def test_copies_data(self, mock_session_result):
        s = summary_from_session_result(mock_session_result)
        assert s.energy_hartree == -76.0
        assert s.homo_lumo_gap_ev == 13.1
        assert s.converged is True
        assert s.method == "RHF"
        assert s.basis == "6-31G"

    def test_custom_label(self, mock_session_result):
        s = summary_from_session_result(mock_session_result, label="Test")
        assert s.label == "Test"

    def test_auto_label(self, mock_session_result):
        s = summary_from_session_result(mock_session_result)
        assert "H2O" in s.label
        assert "RHF" in s.label


# ---------------------------------------------------------------------------
# HTML table
# ---------------------------------------------------------------------------

class TestComparisonTableHtml:

    def test_empty_list(self):
        html = comparison_table_html([])
        assert "No calculations" in html

    def test_contains_labels(self, two_summaries):
        html = comparison_table_html(two_summaries)
        assert "H2O RHF/STO-3G" in html
        assert "H2O RHF/6-31G" in html

    def test_contains_energy(self, two_summaries):
        html = comparison_table_html(two_summaries)
        assert "-75.50000000" in html
        assert "-76.00000000" in html

    def test_contains_gap(self, two_summaries):
        html = comparison_table_html(two_summaries)
        assert "12.5000" in html
        assert "13.1000" in html

    def test_converged_icon(self, two_summaries):
        html = comparison_table_html(two_summaries)
        assert "✅" in html

    def test_not_converged_icon(self):
        s = CalcSummary(label="x", formula="H2", method="RHF", basis="STO-3G",
                        converged=False)
        html = comparison_table_html([s])
        assert "❌" in html

    def test_none_converged_dash(self):
        s = CalcSummary(label="x", formula="H2", method="RHF", basis="STO-3G")
        html = comparison_table_html([s])
        assert "—" in html

    def test_html_escaping(self):
        s = CalcSummary(label="<script>alert(1)</script>", formula="H2",
                        method="RHF", basis="STO-3G")
        html = comparison_table_html([s])
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_table_tag_present(self, two_summaries):
        html = comparison_table_html(two_summaries)
        assert "<table" in html
        assert "</table>" in html


# ---------------------------------------------------------------------------
# Matplotlib comparison chart
# ---------------------------------------------------------------------------

class TestPlotComparison:

    def test_returns_figure(self, two_summaries):
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_comparison(two_summaries)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_custom_title(self, two_summaries):
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_comparison(two_summaries, title="My Comparison")
        assert fig._suptitle.get_text() == "My Comparison"

    def test_no_data_message(self):
        import matplotlib
        matplotlib.use("Agg")
        summaries = [
            CalcSummary(label="a", formula="H2", method="RHF", basis="STO-3G"),
            CalcSummary(label="b", formula="H2", method="RHF", basis="6-31G"),
        ]
        fig = plot_comparison(summaries)
        # Should still return a figure (with a "no data" message)
        assert fig is not None

    def test_partial_data(self):
        import matplotlib
        matplotlib.use("Agg")
        summaries = [
            CalcSummary(label="a", formula="H2", method="RHF", basis="STO-3G",
                        energy_hartree=-1.0),
            CalcSummary(label="b", formula="H2", method="RHF", basis="6-31G"),
        ]
        fig = plot_comparison(summaries)
        assert fig is not None
