"""Runflow helpers used by QuantUIApp."""

from __future__ import annotations

import threading
from typing import Any

import ipywidgets as widgets
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


def on_calc_type_changed(app: Any, change: Any, *, layout_fn: Any) -> None:
    """Update extra options panel based on selected calculation type."""
    ct = change["new"]
    if ct == "Geometry Opt":
        app.calc_extra_opts.children = [
            widgets.HBox(
                [app.fmax_fi, app.max_steps_si],
                layout=layout_fn(gap="8px"),
            ),
        ]
    elif ct == "Frequency":
        app._refresh_freq_seed_options()
        app.calc_extra_opts.children = [
            widgets.HBox(
                [app._freq_seed_dd, app._freq_seed_refresh_btn],
                layout=layout_fn(align_items="center", gap="6px"),
            ),
            app._freq_preopt_cb,
            app._freq_seed_note,
        ]
    elif ct == "UV-Vis (TD-DFT)":
        app.calc_extra_opts.children = [
            app.nstates_si,
            widgets.HTML(
                '<span style="color:#b45309;font-size:12px">⚠ Requires a DFT '
                "functional (e.g. B3LYP, PBE0). RHF/UHF will run TDHF (CIS) "
                "instead.</span>"
            ),
        ]
    elif ct == "NMR Shielding":
        app.calc_extra_opts.children = [
            widgets.HTML(
                '<span style="color:#b45309;font-size:12px">'
                "⚠ Recommended: B3LYP/6-31G* or better. "
                "STO-3G and 3-21G give qualitative results only. "
                "Start from an optimised geometry for best accuracy.</span>"
            ),
        ]
    elif ct == "PES Scan":
        app._update_scan_widgets()
        app.calc_extra_opts.children = [
            widgets.HBox(
                [app._scan_type_dd],
                layout=layout_fn(margin="0 0 4px 0"),
            ),
            widgets.HBox(
                [app._scan_atom1, app._scan_atom2],
                layout=layout_fn(gap="4px"),
            ),
            app._scan_atom34_box,
            widgets.HBox(
                [
                    app._scan_start,
                    app._scan_stop,
                    app._scan_steps,
                    app._scan_unit_lbl,
                ],
                layout=layout_fn(gap="4px", align_items="center"),
            ),
        ]
    else:
        app.calc_extra_opts.children = []


def update_scan_widgets(app: Any, _change: Any = None) -> None:
    """Show/hide atom inputs and unit label based on scan type."""
    st = app._scan_type_dd.value
    if st == "Bond":
        app._scan_atom34_box.layout.display = "none"
        app._scan_unit_lbl.value = '<span style="font-size:12px;color:#555">Å</span>'
    elif st == "Angle":
        app._scan_atom4.layout.display = "none"
        app._scan_atom3.layout.display = ""
        app._scan_atom34_box.layout.display = ""
        app._scan_unit_lbl.value = '<span style="font-size:12px;color:#555">°</span>'
    else:  # Dihedral
        app._scan_atom3.layout.display = ""
        app._scan_atom4.layout.display = ""
        app._scan_atom34_box.layout.display = ""
        app._scan_unit_lbl.value = '<span style="font-size:12px;color:#555">°</span>'


def refresh_freq_seed_options(app: Any) -> None:
    """Populate frequency seed dropdown with saved geometry optimisations."""
    from quantui.results_storage import list_results, load_result

    options = [("(use current molecule)", "")]
    for d in list_results():
        try:
            data = load_result(d)
            if data.get("calc_type") != "geometry_opt":
                continue
            traj_file = d / "trajectory.json"
            if not traj_file.exists():
                continue
            ts = data.get("timestamp", d.name[:19])
            label = f"{data['formula']}  {data['method']}/{data['basis']}" f"  —  {ts}"
            options.append((label, str(d)))
        except Exception:
            continue
    app._freq_seed_dd.options = options


def on_freq_seed_changed(app: Any, change: Any) -> None:
    """Enable/disable pre-opt checkbox and update seed note message."""
    path_str = change["new"]
    if path_str:
        app._freq_preopt_cb.value = False
        app._freq_preopt_cb.disabled = True
        app._freq_seed_note.value = (
            '<span style="font-size:12px;color:#16a34a">'
            "✓ Final optimised geometry will be loaded from the selected result."
            "</span>"
        )
    else:
        app._freq_preopt_cb.disabled = False
        app._freq_seed_note.value = ""


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
