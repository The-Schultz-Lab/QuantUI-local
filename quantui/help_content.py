"""
Educational help content for QuantUI notebook widgets.

Provides collapsible HTML help panels that explain quantum chemistry
concepts to students.  Each help topic is a short, jargon-light
explanation with concrete examples and tips.

Usage in a notebook cell::

    from quantui.help_content import help_panel
    display(help_panel("charge"))
    display(help_panel("multiplicity"))
"""

from __future__ import annotations

from typing import Dict

import ipywidgets as widgets

# ---------------------------------------------------------------------------
# Help text bank — keys used by help_panel()
# ---------------------------------------------------------------------------

HELP_TOPICS: Dict[str, Dict[str, str]] = {
    "charge": {
        "title": "What is molecular charge?",
        "body": (
            "<p>The <b>charge</b> is the total electric charge of the molecule, "
            "measured in units of the elementary charge (<i>e</i>).</p>"
            "<ul>"
            "<li><b>0</b> — neutral molecule (most common)</li>"
            "<li><b>+1</b> — cation (one electron removed), e.g. NH₄⁺</li>"
            "<li><b>−1</b> — anion (one electron added), e.g. Cl⁻</li>"
            "</ul>"
            "<p><b>Tip:</b> If you are unsure, start with charge = 0. "
            "Most stable molecules are neutral.</p>"
        ),
    },
    "multiplicity": {
        "title": "What is spin multiplicity?",
        "body": (
            "<p><b>Multiplicity</b> = 2S + 1, where S is the total electron spin. "
            "It tells the computer how many unpaired electrons the molecule has.</p>"
            "<table style='border-collapse:collapse; margin:6px 0;'>"
            "<tr style='border-bottom:1px solid #ccc;'>"
            "  <th style='padding:3px 12px; text-align:left;'>Unpaired e⁻</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Multiplicity</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Name</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Example</th></tr>"
            "<tr><td style='padding:3px 12px;'>0</td>"
            "  <td style='padding:3px 12px;'>1</td>"
            "  <td style='padding:3px 12px;'>Singlet</td>"
            "  <td style='padding:3px 12px;'>H₂O, CH₄</td></tr>"
            "<tr><td style='padding:3px 12px;'>1</td>"
            "  <td style='padding:3px 12px;'>2</td>"
            "  <td style='padding:3px 12px;'>Doublet</td>"
            "  <td style='padding:3px 12px;'>NO, OH radical</td></tr>"
            "<tr><td style='padding:3px 12px;'>2</td>"
            "  <td style='padding:3px 12px;'>3</td>"
            "  <td style='padding:3px 12px;'>Triplet</td>"
            "  <td style='padding:3px 12px;'>O₂</td></tr>"
            "</table>"
            "<p><b>Rule of thumb:</b> Use <b>1</b> (singlet) for most closed-shell "
            "molecules. Use <b>2</b> for radicals. Use <b>3</b> for O₂.</p>"
            "<p>The charge and multiplicity must be consistent with the electron count. "
            "QuantUI will warn you if the combination is impossible.</p>"
        ),
    },
    "method": {
        "title": "RHF vs UHF — which method should I use?",
        "body": (
            "<p>Both methods approximate the electronic wavefunction using "
            "Hartree–Fock theory, but they treat electron spin differently.</p>"
            "<table style='border-collapse:collapse; margin:6px 0;'>"
            "<tr style='border-bottom:1px solid #ccc;'>"
            "  <th style='padding:3px 12px; text-align:left;'>Method</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Full name</th>"
            "  <th style='padding:3px 12px; text-align:left;'>When to use</th></tr>"
            "<tr><td style='padding:3px 12px;'><b>RHF</b></td>"
            "  <td style='padding:3px 12px;'>Restricted Hartree–Fock</td>"
            "  <td style='padding:3px 12px;'>All electrons are paired → multiplicity = 1</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>UHF</b></td>"
            "  <td style='padding:3px 12px;'>Unrestricted Hartree–Fock</td>"
            "  <td style='padding:3px 12px;'>Any unpaired electrons → multiplicity ≥ 2</td></tr>"
            "</table>"
            "<p><b>Quick guide:</b></p>"
            "<ul>"
            "<li>Neutral H₂O, CH₄, NH₃ → <b>RHF</b> (singlet)</li>"
            "<li>O₂ (triplet) → <b>UHF</b></li>"
            "<li>OH radical → <b>UHF</b> (doublet)</li>"
            "</ul>"
            "<p>If you pick RHF for an open-shell molecule, the calculation will "
            "likely fail or give wrong energies.</p>"
        ),
    },
    "basis_set": {
        "title": "Choosing a basis set",
        "body": (
            "<p>A <b>basis set</b> is the mathematical toolkit used to describe "
            "electron orbitals. Larger basis sets are more accurate but slower.</p>"
            "<table style='border-collapse:collapse; margin:6px 0;'>"
            "<tr style='border-bottom:1px solid #ccc;'>"
            "  <th style='padding:3px 12px; text-align:left;'>Basis set</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Speed</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Accuracy</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Best for</th></tr>"
            "<tr><td style='padding:3px 12px;'>STO-3G</td>"
            "  <td style='padding:3px 12px;'>⚡ Very fast</td>"
            "  <td style='padding:3px 12px;'>Low</td>"
            "  <td style='padding:3px 12px;'>Learning, quick tests</td></tr>"
            "<tr><td style='padding:3px 12px;'>3-21G</td>"
            "  <td style='padding:3px 12px;'>⚡ Fast</td>"
            "  <td style='padding:3px 12px;'>Low–Medium</td>"
            "  <td style='padding:3px 12px;'>Quick estimates</td></tr>"
            "<tr><td style='padding:3px 12px;'>6-31G</td>"
            "  <td style='padding:3px 12px;'>Moderate</td>"
            "  <td style='padding:3px 12px;'>Medium</td>"
            "  <td style='padding:3px 12px;'>General purpose</td></tr>"
            "<tr><td style='padding:3px 12px;'>6-31G*</td>"
            "  <td style='padding:3px 12px;'>Moderate</td>"
            "  <td style='padding:3px 12px;'>Good</td>"
            "  <td style='padding:3px 12px;'>Research-quality (recommended)</td></tr>"
            "<tr><td style='padding:3px 12px;'>6-31G**</td>"
            "  <td style='padding:3px 12px;'>Slower</td>"
            "  <td style='padding:3px 12px;'>Good</td>"
            "  <td style='padding:3px 12px;'>Hydrogen bonding</td></tr>"
            "<tr><td style='padding:3px 12px;'>cc-pVDZ</td>"
            "  <td style='padding:3px 12px;'>Slower</td>"
            "  <td style='padding:3px 12px;'>High</td>"
            "  <td style='padding:3px 12px;'>Correlation-consistent studies</td></tr>"
            "<tr><td style='padding:3px 12px;'>cc-pVTZ</td>"
            "  <td style='padding:3px 12px;'>🐢 Slow</td>"
            "  <td style='padding:3px 12px;'>Very high</td>"
            "  <td style='padding:3px 12px;'>Benchmark / publication</td></tr>"
            "</table>"
            "<p><b>Recommendation:</b> Start with <b>STO-3G</b> for learning. "
            "Use <b>6-31G*</b> for serious work. Only use cc-pVTZ if you need "
            "high-accuracy results and have time to wait.</p>"
        ),
    },
}

# All valid topic keys (for testing / discovery)
VALID_TOPICS = frozenset(HELP_TOPICS.keys())


# ---------------------------------------------------------------------------
# Widget builder
# ---------------------------------------------------------------------------

_PANEL_CSS = (
    "border: 1px solid #ddd; border-radius: 6px; padding: 8px 12px; "
    "margin: 4px 0 8px 0; background: #f8f9fa; max-width: 620px;"
)


def help_panel(topic: str) -> widgets.HTML:
    """
    Return a collapsible HTML help widget for a given topic.

    The widget uses a ``<details>/<summary>`` element so it starts
    collapsed and can be expanded by clicking.

    Args:
        topic: One of ``'charge'``, ``'multiplicity'``, ``'method'``,
               or ``'basis_set'``.

    Returns:
        ``ipywidgets.HTML`` widget ready for ``display()``.

    Raises:
        KeyError: If topic is not in :data:`HELP_TOPICS`.
    """
    if topic not in HELP_TOPICS:
        raise KeyError(
            f"Unknown help topic '{topic}'. "
            f"Valid topics: {', '.join(sorted(HELP_TOPICS))}"
        )

    entry = HELP_TOPICS[topic]
    html = (
        f'<details style="{_PANEL_CSS}">'
        f'<summary style="cursor:pointer; font-weight:bold; color:#0366d6;">'
        f'ℹ️ {entry["title"]}</summary>'
        f'<div style="margin-top:6px; font-size:13px;">{entry["body"]}</div>'
        f'</details>'
    )
    return widgets.HTML(value=html)
