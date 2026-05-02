"""Runflow helpers used by QuantUIApp."""

from __future__ import annotations

import threading
from typing import Any

from IPython.display import HTML, display


def on_run_clicked(app: Any, btn: Any) -> None:
    """Reset result panes and start the background run thread."""
    app.run_output.clear_output()
    app.result_output.clear_output()
    app.result_viz_output.clear_output()
    app._analysis_mol_output.clear_output()
    app._viz_label.layout.display = "none"
    app._viz_label.value = ""
    app._deactivate_all_ana_panels()
    app._clear_output_widget(app._pes_plot_html)
    app._result_dir_label.value = ""
    app._result_dir_label.layout.display = "none"
    app._result_log_accordion.layout.display = "none"
    app._result_log_accordion.selected_index = None
    app._result_log_output.clear_output()
    app._completion_banner.layout.display = "none"
    app._to_analysis_btn.layout.display = "none"
    app._analysis_empty_html.layout.display = "none"
    threading.Thread(target=app._do_run, daemon=True).start()


def update_notes(app: Any, change: Any = None) -> None:
    """Refresh educational method notes for the active molecule/method."""
    app.notes_output.clear_output(wait=True)
    if app._molecule is None:
        return
    try:
        from quantui import PySCFCalculation

        calc = PySCFCalculation(
            app._molecule,
            method=app.method_dd.value,
            basis=app.basis_dd.value,
        )
        notes = calc.get_educational_notes()
        if notes:
            safe = (
                notes.replace("**", "<b>", 1)
                .replace("**", "</b>", 1)
                .replace("\n\n", "<br><br>")
            )
            with app.notes_output:
                display(
                    HTML(
                        '<div style="background:#fffbf0;padding:8px 12px;'
                        'border-radius:4px;font-size:13px;margin-top:6px">'
                        + safe
                        + "</div>"
                    )
                )
    except Exception:
        pass


def update_estimate(app: Any, *, calc_log_mod: Any, change: Any = None) -> None:
    """Refresh runtime estimate text from the performance model."""
    if app._molecule is None:
        app.perf_estimate_html.value = ""
        return
    try:
        n_basis = calc_log_mod.count_basis_functions(
            app._molecule.atoms, app.basis_dd.value
        )
        est = calc_log_mod.estimate_time(
            n_atoms=len(app._molecule.atoms),
            n_electrons=app._molecule.get_electron_count(),
            method=app.method_dd.value,
            basis=app.basis_dd.value,
            n_basis=n_basis,
        )
        app.perf_estimate_html.value = calc_log_mod.format_estimate(est)
    except Exception:
        app.perf_estimate_html.value = ""
