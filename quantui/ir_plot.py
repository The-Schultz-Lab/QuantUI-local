"""
IR spectrum visualization: stick chart and Lorentzian broadened lineshape.

Accepts vibrational frequencies (cm⁻¹) and IR intensities (km/mol)
from a frequency calculation and returns a Plotly Figure.

Typical usage::

    from quantui.ir_plot import plot_ir_spectrum
    fig = plot_ir_spectrum(result.frequencies_cm1, result.ir_intensities)
    fig = plot_ir_spectrum(freqs, intensities, mode="broadened", fwhm=30.0)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go

# x-axis range follows the standard IR convention: high → low wavenumber
_XRANGE = [4000, 400]
_XGRID = np.arange(400, 4001, 1.0)  # 1 cm⁻¹ resolution for broadened mode


def plot_ir_spectrum(
    frequencies: List[float],
    intensities: List[float],
    *,
    fwhm: float = 20.0,
    mode: str = "stick",
) -> go.Figure:
    """Return a Plotly figure for the IR absorption spectrum.

    Args:
        frequencies: Vibrational frequencies in cm⁻¹.
            Values ≤ 0 (imaginary / translation / rotation) are silently skipped.
        intensities: IR intensities in km/mol, same length as *frequencies*.
        fwhm: Full width at half maximum for the Lorentzian lineshape in cm⁻¹.
            Only used when ``mode="broadened"``. Default: 20.
        mode: Display mode.
            ``"stick"`` — vertical bars at each active frequency.
            ``"broadened"`` — Lorentzian convolution of all peaks.

    Returns:
        :class:`plotly.graph_objects.Figure` ready for display or wrapping
        in a :class:`~plotly.graph_objects.FigureWidget`.
    """
    real_pairs = [(f, i) for f, i in zip(frequencies, intensities) if f > 0]

    _base_layout = dict(
        xaxis=dict(
            title="Wavenumber (cm⁻¹)",
            range=_XRANGE,
            showgrid=True,
            gridcolor="#e5e7eb",
        ),
        yaxis=dict(
            title="IR Intensity (km/mol)",
            rangemode="tozero",
            showgrid=True,
            gridcolor="#e5e7eb",
        ),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=20, t=20, b=55),
        height=300,
        plot_bgcolor="#fafafa",
    )

    fig = go.Figure(layout=_base_layout)

    if not real_pairs:
        return fig

    freqs_real, ints_real = zip(*real_pairs)

    if mode == "broadened":
        half_gamma = fwhm / 2.0
        y_broad = np.zeros_like(_XGRID)
        for nu0, inten in zip(freqs_real, ints_real):
            y_broad += inten * half_gamma**2 / ((_XGRID - nu0) ** 2 + half_gamma**2)

        fig.add_trace(
            go.Scatter(
                x=_XGRID,
                y=y_broad,
                mode="lines",
                line=dict(color="#2563eb", width=1.5),
                name="IR (broadened)",
                hovertemplate="%{x:.0f} cm⁻¹ | %{y:.2f} km/mol<extra></extra>",
            )
        )
    else:  # stick
        x_stick: List[Optional[float]] = []
        y_stick: List[Optional[float]] = []
        for nu, inten in zip(freqs_real, ints_real):
            x_stick.extend([nu, nu, None])
            y_stick.extend([0.0, inten, None])

        fig.add_trace(
            go.Scatter(
                x=x_stick,
                y=y_stick,
                mode="lines",
                line=dict(color="#2563eb", width=2),
                name="IR (stick)",
                hovertemplate="%{x:.0f} cm⁻¹<extra></extra>",
            )
        )

    return fig
