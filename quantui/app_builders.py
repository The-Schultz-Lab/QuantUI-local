"""UI builder helpers used by QuantUIApp."""

from __future__ import annotations

from typing import Any

import ipywidgets as widgets
from IPython.display import HTML, display

import quantui
from quantui.help_content import HELP_TOPICS


def build_theme_selector(app: Any, *, layout_fn: Any) -> None:
    """Build the theme selector widgets and apply default theme CSS."""
    app._theme_style = widgets.Output(
        layout=layout_fn(height="0px", overflow="hidden", margin="0", padding="0")
    )
    app.theme_btn = widgets.ToggleButtons(
        options=["Light", "Dark"],
        value="Dark",
        description="Theme:",
        style={"description_width": "48px", "button_width": "90px"},
        layout=layout_fn(margin="0"),
    )
    with app._theme_style:
        display(HTML(app._theme_css("Dark")))


def build_welcome_header(app: Any) -> None:
    """Build the static QuantUI welcome banner."""
    logo_svg = (
        '<svg width="120" height="120" viewBox="0 0 280 280"'
        ' xmlns="http://www.w3.org/2000/svg">'
        "<defs>"
        '<filter id="q-glow" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="7" result="blur"/>'
        "<feMerge>"
        '<feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/>'
        "</feMerge></filter>"
        '<filter id="q-halo" x="-80%" y="-80%" width="260%" height="260%">'
        '<feGaussianBlur stdDeviation="22" result="blur"/>'
        "<feMerge>"
        '<feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/>'
        "</feMerge></filter>"
        "</defs>"
        '<circle cx="140" cy="140" r="48"'
        ' fill="rgba(37,99,235,0.20)" filter="url(#q-halo)"/>'
        '<g transform="rotate(0,140,140)">'
        '<ellipse cx="140" cy="140" rx="115" ry="33" fill="none"'
        ' stroke="#0891b2" stroke-width="1.4" opacity="0.70"/>'
        '<circle cx="255" cy="140" r="5.5" fill="#67e8f9"/>'
        "</g>"
        '<g transform="rotate(60,140,140)">'
        '<ellipse cx="140" cy="140" rx="115" ry="33" fill="none"'
        ' stroke="#0891b2" stroke-width="1.4" opacity="0.55"/>'
        '<circle cx="255" cy="140" r="4.5" fill="#93c5fd"/>'
        "</g>"
        '<g transform="rotate(120,140,140)">'
        '<ellipse cx="140" cy="140" rx="115" ry="33" fill="none"'
        ' stroke="#3b82f6" stroke-width="1.4" opacity="0.42"/>'
        '<circle cx="255" cy="140" r="4" fill="#60a5fa"/>'
        "</g>"
        '<circle cx="140" cy="140" r="20"'
        ' fill="rgba(37,99,235,0.25)" filter="url(#q-glow)"/>'
        '<circle cx="140" cy="140" r="14"'
        ' fill="#2563eb" filter="url(#q-glow)"/>'
        '<circle cx="140" cy="140" r="8" fill="#60a5fa"/>'
        '<circle cx="137" cy="137" r="3" fill="rgba(255,255,255,0.45)"/>'
        "</svg>"
    )
    html = (
        f'<div style="display:flex;align-items:center;gap:28px;'
        f"padding:22px 4px 18px;margin-bottom:4px;"
        f"border-bottom:1px solid #e2e8f0"
        ">"
        f"{logo_svg}"
        f"<div>"
        f'<div style="font-size:44px;font-weight:700;letter-spacing:-0.8px;'
        f'color:#0f172a;line-height:1.05">QuantUI</div>'
        f'<div style="font-size:20px;color:#475569;margin-top:7px">'
        f"Quantum chemistry calculations, right on your device</div>"
        f'<div style="font-size:13px;color:#94a3b8;margin-top:5px">'
        f"v{quantui.__version__} &nbsp;&middot;&nbsp; "
        f"<b>Help</b> tab for instructions &nbsp;&middot;&nbsp; "
        f"<b>Status</b> tab for system info</div>"
        f"</div>"
        f"</div>"
    )
    app._welcome_html = widgets.HTML(value=html)


def build_molecule_section(
    app: Any,
    *,
    layout_fn: Any,
    molecule_library: dict[str, Any],
    pubchem_available: bool,
    visualization_available: bool,
) -> None:
    """Build molecule input widgets and collapsed summary container."""
    preset_opts = ["(select a molecule)"] + list(molecule_library.keys())
    app.preset_dd = widgets.Dropdown(
        options=preset_opts,
        value="(select a molecule)",
        description="Molecule:",
        style={"description_width": "90px"},
        layout=layout_fn(width="320px"),
    )

    app.xyz_area = widgets.Textarea(
        placeholder=(
            "Paste XYZ coordinates (symbol  x  y  z):\n"
            "O  0.000  0.000  0.000\n"
            "H  0.757  0.587  0.000\n"
            "H -0.757  0.587  0.000"
        ),
        layout=layout_fn(width="440px", height="130px"),
    )
    app.xyz_btn = widgets.Button(
        description="Load XYZ", button_style="info", icon="upload"
    )
    app.xyz_msg = widgets.Label()

    app.pubchem_txt = widgets.Text(
        placeholder="name or SMILES  (e.g. aspirin, caffeine, CC(=O)O)",
        layout=layout_fn(width="380px"),
    )
    app.pubchem_btn = widgets.Button(
        description="Search",
        button_style="info",
        icon="search",
        disabled=not pubchem_available,
        layout=layout_fn(width="100px"),
    )
    app.pubchem_msg = widgets.Label(
        value=(
            ""
            if pubchem_available
            else "PubChem unavailable — check internet connection"
        )
    )

    hint = '<p style="margin:4px 0 8px;color:#666;font-size:13px">'
    tab_preset = widgets.VBox(
        [
            widgets.HTML(hint + "Choose from 20+ curated educational molecules.</p>"),
            app.preset_dd,
        ]
    )
    tab_xyz = widgets.VBox(
        [
            widgets.HTML(
                hint + "Paste XYZ coordinates (element x y z, one atom per line).</p>"
            ),
            app.xyz_area,
            widgets.HBox([app.xyz_btn, app.xyz_msg]),
        ]
    )
    tab_pubchem = widgets.VBox(
        [
            widgets.HTML(
                hint + "Search by name or SMILES. Requires internet connection.</p>"
            ),
            widgets.HBox([app.pubchem_txt, app.pubchem_btn]),
            app.pubchem_msg,
        ]
    )
    input_tab = widgets.Tab(children=[tab_preset, tab_xyz, tab_pubchem])
    for i, title in enumerate(["Preset Library", "XYZ Input", "PubChem Search"]):
        input_tab.set_title(i, title)

    app.mol_input_expanded = widgets.VBox(
        [
            widgets.HTML('<h3 style="margin:8px 0 6px">Molecule Input</h3>'),
            input_tab,
        ]
    )
    app.change_mol_btn = widgets.Button(
        description="Change",
        button_style="",
        icon="pencil",
        layout=layout_fn(width="100px", height="32px"),
        tooltip="Re-expand the molecule input panel",
    )
    app.mol_input_collapsed = widgets.HBox(
        [app.mol_summary_compact, app.change_mol_btn],
        layout=layout_fn(align_items="center", gap="12px", padding="6px 0"),
    )
    mol_container_children = [
        app.mol_input_expanded,
        app.mol_info_html,
        app.viz_output,
    ]
    if app.viz_backend_toggle is not None:
        mol_container_children.append(app.viz_backend_toggle)
    if visualization_available:
        mol_container_children.append(app.viz_controls_box)
    app.mol_input_container = widgets.VBox(
        mol_container_children,
        layout=layout_fn(margin="0 0 4px 0"),
    )


def build_calc_setup(app: Any, *, layout_fn: Any) -> None:
    """Build the calculation setup panel."""
    app.calc_setup_panel = widgets.VBox(
        [
            widgets.HTML('<h3 style="margin:14px 0 6px">Calculation Setup</h3>'),
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            widgets.HBox(
                                [app.method_dd, app.method_help_btn],
                                layout=layout_fn(align_items="center", gap="4px"),
                            ),
                            widgets.HBox(
                                [app.basis_dd, app.basis_help_btn],
                                layout=layout_fn(align_items="center", gap="4px"),
                            ),
                        ]
                    ),
                    widgets.HTML("&ensp;&ensp;"),
                    widgets.VBox([app.charge_si, app.mult_si]),
                ]
            ),
            app.calc_type_dd,
            app.calc_extra_opts,
            app.preopt_cb,
            widgets.HBox(
                [app.solvent_cb, app.solvent_dd],
                layout=layout_fn(align_items="center", gap="4px"),
            ),
            app.notes_output,
        ]
    )


def build_run_section(app: Any, *, layout_fn: Any) -> None:
    """Build the run panel shown in the Calculate tab."""
    app.run_panel = widgets.VBox(
        [
            widgets.HTML(
                '<h3 style="margin:14px 0 6px">Run Calculation</h3>'
                '<p style="color:#555;font-size:13px;margin:0 0 8px">PySCF runs in this '
                "kernel. Output appears live below. Large molecules or high-accuracy basis "
                "sets may take several minutes on a laptop.</p>"
            ),
            app.perf_estimate_html,
            widgets.HBox([app.run_btn, app.run_status]),
            widgets.HBox(
                [
                    widgets.HTML(
                        '<span style="font-size:13px;font-weight:600;color:#444">'
                        "Calculation Output</span>"
                    ),
                    app.log_clear_btn,
                ],
                layout=layout_fn(
                    align_items="center",
                    justify_content="space-between",
                    margin="10px 0 4px",
                    max_width="460px",
                ),
            ),
            app.run_output,
        ]
    )


def build_compare_section(app: Any, *, layout_fn: Any, rdkit_available: bool) -> None:
    """Build compare tab widgets and export accordion."""
    app.compare_select = widgets.SelectMultiple(
        options=[("(no saved results)", "")],
        rows=8,
        description="",
        layout=layout_fn(width="100%"),
    )
    app.compare_refresh_btn = widgets.Button(
        description="Refresh",
        button_style="",
        icon="refresh",
        layout=layout_fn(width="100px"),
    )
    app.compare_btn = widgets.Button(
        description="Compare selected",
        button_style="primary",
        icon="bar-chart",
        disabled=True,
        layout=layout_fn(width="180px"),
    )
    app.compare_clear_btn = widgets.Button(
        description="Clear",
        button_style="warning",
        icon="times",
        layout=layout_fn(width="90px"),
    )
    app.compare_output = widgets.Output()

    app.compare_panel = widgets.VBox(
        [
            widgets.HTML(
                '<h3 style="margin:14px 0 6px">Compare Calculations</h3>'
                '<p style="color:#555;font-size:13px;margin:0 0 8px">'
                "Select two or more saved calculations to compare side-by-side. "
                "Hold Ctrl (or ⌘) to select multiple entries.</p>"
            ),
            widgets.HBox([app.compare_refresh_btn]),
            app.compare_select,
            widgets.HBox(
                [app.compare_btn, app.compare_clear_btn],
                layout=layout_fn(gap="8px", margin="6px 0"),
            ),
            app.compare_output,
        ],
        layout=layout_fn(padding="8px 0"),
    )

    rdkit_note = (
        ""
        if rdkit_available
        else '<p style="color:#888;font-size:12px;margin:4px 0 0">MOL/PDB export requires RDKit '
        "(<code>conda install -c conda-forge rdkit</code>).</p>"
    )
    export_content = widgets.VBox(
        [
            widgets.HTML(
                '<p style="color:#555;font-size:13px;margin:0 0 8px">'
                "Download a self-contained PySCF script you can study or run outside the notebook.</p>"
            ),
            widgets.HBox([app.export_btn, app.export_status]),
            widgets.HTML('<hr style="margin:10px 0 8px">'),
            widgets.HTML(
                '<p style="color:#555;font-size:13px;margin:0 0 6px">'
                "Download the molecular structure in a standard chemistry file format.</p>"
                + rdkit_note
            ),
            widgets.HBox(
                [app.export_xyz_btn, app.export_mol_btn, app.export_pdb_btn],
                layout=layout_fn(flex_wrap="wrap", gap="6px"),
            ),
            app.struct_export_status,
        ]
    )
    app.advanced_accordion = widgets.Accordion(children=[export_content])
    app.advanced_accordion.set_title(0, "Export")
    app.advanced_accordion.selected_index = None

    app._populate_compare_list()


def build_output_tab(app: Any, *, layout_fn: Any) -> None:
    """Build the Output tab panel widgets."""
    app._log_output_html = widgets.HTML(
        '<span style="color:#94a3b8;font-size:13px">'
        "No log yet — run a calculation first, or use "
        "<b>View log</b> in the History tab.</span>"
    )
    app._log_source_lbl = widgets.HTML()
    app._log_clear_btn = widgets.Button(
        description="Clear",
        button_style="",
        icon="times",
        layout=layout_fn(width="80px"),
    )
    app._clear_log_cache_btn = widgets.Button(
        description="Clear Log Cache",
        button_style="",
        icon="trash",
        tooltip=(
            "Delete the session event log (event_log.jsonl). "
            "Calculation performance data is preserved."
        ),
        layout=layout_fn(width="160px"),
    )
    app._clear_log_cache_confirm_btn = widgets.Button(
        description="Confirm clear?",
        button_style="danger",
        layout=layout_fn(width="140px", display="none"),
    )
    app.log_tab_panel = widgets.VBox(
        [
            widgets.HTML(
                '<p style="color:#555;font-size:13px;margin:4px 0 8px">'
                "Raw PySCF output for the most recent calculation. "
                "Use <b>View log</b> in the History tab to load a saved result's log. "
                "Orbital diagrams, trajectories, and spectra are in the "
                "<b>Analysis</b> tab.</p>"
            ),
            widgets.HBox(
                [app._log_clear_btn],
                layout=layout_fn(margin="0 0 8px"),
            ),
            app._log_source_lbl,
            app._log_output_html,
            app._result_log_accordion,
            widgets.HTML(
                '<hr style="border:none;border-top:1px solid #e2e8f0;margin:16px 0 10px"/>'
                '<p style="color:#94a3b8;font-size:12px;margin:0 0 6px">'
                "Session event log — records molecule loads, calculations, "
                "and issue reports across this session.</p>"
            ),
            widgets.HBox(
                [app._clear_log_cache_btn, app._clear_log_cache_confirm_btn],
                layout=layout_fn(align_items="center", gap="8px"),
            ),
        ],
        layout=layout_fn(padding="8px 0"),
    )


def build_help_section(app: Any, *, layout_fn: Any) -> None:
    """Build the floating help panel and top-bar help/exit buttons."""
    help_keys = list(HELP_TOPICS.keys())
    help_labels = [HELP_TOPICS[k]["title"] for k in help_keys]
    app.help_topic_dd = widgets.Dropdown(
        options=list(zip(help_labels, help_keys)),
        description="Topic:",
        style={"description_width": "60px"},
        layout=layout_fn(width="460px"),
    )
    app.help_content_html = widgets.HTML()
    app._render_help_topic()

    app._help_btn = widgets.Button(
        description="?",
        button_style="",
        tooltip="Help topics",
        layout=layout_fn(width="34px", margin="0 0 0 8px"),
    )

    app._exit_btn = widgets.Button(
        description="Exit",
        button_style="danger",
        tooltip="Shut down the QuantUI server and close this session",
        layout=layout_fn(width="64px", margin="0 0 0 8px"),
    )
    app._exit_output = widgets.Output(layout=layout_fn(height="0px", overflow="hidden"))

    app.help_tab_panel = widgets.VBox(
        [
            widgets.HTML(
                '<p style="color:#555;font-size:13px;margin:4px 0 12px">'
                "Browse help topics below. Click <b>?</b> next to the Method or Basis Set "
                "dropdown in the Calculate tab to jump directly to a relevant topic.</p>"
            ),
            app.help_topic_dd,
            app.help_content_html,
        ],
        layout=layout_fn(
            display="none",
            padding="8px 0",
            border="1px solid #e2e8f0",
            border_radius="6px",
            padding_left="12px",
            margin="0 0 8px",
        ),
    )


def build_issue_widgets(app: Any, *, layout_fn: Any) -> None:
    """Build issue-report widgets shown from the top toolbar."""
    app._issue_btn = widgets.Button(
        description="Report Issue",
        button_style="warning",
        icon="flag",
        tooltip="Report a bug or unexpected behaviour observed in this session",
        layout=layout_fn(width="140px", margin="0 0 0 8px"),
    )
    app._issue_textarea = widgets.Textarea(
        placeholder=(
            "Describe what you observed — what you did, what you expected, "
            "and what actually happened."
        ),
        layout=layout_fn(width="100%", height="90px"),
    )
    app._issue_submit_btn = widgets.Button(
        description="Submit",
        button_style="success",
        layout=layout_fn(width="90px"),
    )
    app._issue_cancel_btn = widgets.Button(
        description="Cancel",
        button_style="",
        layout=layout_fn(width="80px"),
    )
    app._issue_status_html = widgets.HTML()
    app._issue_overlay = widgets.VBox(
        [
            widgets.HTML(
                '<p style="font-size:13px;font-weight:600;margin:0 0 6px;color:#92400e">'
                "&#9872; Report Issue</p>"
                '<p style="font-size:12px;color:#78350f;margin:0 0 8px">'
                "Your report (and a snapshot of the current session state) will be "
                "saved to <code>issues.db</code> and the session event log.</p>"
            ),
            app._issue_textarea,
            widgets.HBox(
                [app._issue_submit_btn, app._issue_cancel_btn],
                layout=layout_fn(margin="6px 0 0", gap="8px"),
            ),
            app._issue_status_html,
        ],
        layout=layout_fn(
            display="none",
            border="1px solid #f59e0b",
            border_radius="6px",
            padding="12px 14px",
            margin="0 0 6px",
            background_color="#fffbeb",
        ),
    )
