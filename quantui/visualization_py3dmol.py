"""
Molecular visualization using py3Dmol (and optional PlotlyMol).

This module provides 3D molecular visualization using py3Dmol as the primary
backend (stable, widely used, already installed). PlotlyMol is supported as
an optional alternative for users who prefer Plotly-based figures.

Author: Jonathan Schultz, NCCU
Created: 2026-02-17
"""

from typing import Optional, Literal, cast
import logging

logger = logging.getLogger(__name__)

Py3DmolStyle = Literal["stick", "sphere", "line", "cartoon"]
BackendName = Literal["auto", "py3dmol", "plotlymol"]

# Check available visualization backends
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False
    logger.warning("py3Dmol not available - primary visualization disabled")

try:
    from plotlymol3d import draw_3D_rep
    PLOTLYMOL_AVAILABLE = True
except ImportError:
    PLOTLYMOL_AVAILABLE = False
    logger.info("PlotlyMol not available (optional)")


def is_visualization_available() -> bool:
    """
    Check if molecular visualization is available.

    Returns:
        True if py3Dmol OR PlotlyMol is available, False otherwise.
    """
    return PY3DMOL_AVAILABLE or PLOTLYMOL_AVAILABLE


def get_available_backends() -> list[str]:
    """
    Get list of available visualization backends.

    Returns:
        List of available backend names (e.g., ['py3dmol', 'plotlymol'])
    """
    backends = []
    if PY3DMOL_AVAILABLE:
        backends.append('py3dmol')
    if PLOTLYMOL_AVAILABLE:
        backends.append('plotlymol')
    return backends


def molecule_to_xyz_string(molecule) -> str:
    """
    Convert QuantUI Molecule to XYZ string format.

    Args:
        molecule: QuantUI Molecule object

    Returns:
        XYZ format string suitable for py3Dmol or PlotlyMol
    """
    from quantui.molecule import Molecule

    if not isinstance(molecule, Molecule):
        raise TypeError("Expected QuantUI Molecule object")

    return molecule.to_xyz_string()


def visualize_molecule_py3dmol(
    molecule,
    style: Py3DmolStyle = "stick",
    width: int = 600,
    height: int = 500,
    bgcolor: str = "white"
):
    """
    Create interactive 3D visualization using py3Dmol.

    Args:
        molecule: QuantUI Molecule object
        style: Visualization style:
            - "stick": Stick representation (default, good for small molecules)
            - "sphere": Van der Waals spheres
            - "line": Line representation
            - "cartoon": Cartoon (for proteins)
        width: Viewer width in pixels (default: 600)
        height: Viewer height in pixels (default: 500)
        bgcolor: Background color (default: "white")

    Returns:
        py3Dmol.view object (call .show() in Jupyter to display)

    Raises:
        ImportError: If py3Dmol is not installed

    Example:
        >>> mol = Molecule(['O', 'H', 'H'], [[0,0,0], [0.757,0.587,0], [-0.757,0.587,0]])
        >>> view = visualize_molecule_py3dmol(mol, style="stick")
        >>> view.show()  # In Jupyter
    """
    if not PY3DMOL_AVAILABLE:
        raise ImportError(
            "py3Dmol is not installed. To enable 3D visualization:\n"
            "  pip install py3dmol"
        )

    # Build a well-formed XYZ block: count line + title line + coordinates.
    # py3Dmol is lenient about the header in most environments, but browsers
    # running the exported HTML require the standard two-line header to parse
    # the format correctly.
    bare_xyz = molecule.to_xyz_string()
    xyz_string = f"{len(molecule.atoms)}\n{molecule.get_formula()}\n{bare_xyz}"

    logger.info(
        f"Creating py3Dmol visualization for {molecule.get_formula()} "
        f"(style={style})"
    )

    # Create viewer
    view = py3Dmol.view(width=width, height=height)

    # Add molecule
    view.addModel(xyz_string, "xyz")

    # Set style
    view.setStyle({style: {}})

    # Set background
    view.setBackgroundColor(bgcolor)

    # Zoom to fit
    view.zoomTo()

    return view


def _validate_py3dmol_style(style: str) -> Py3DmolStyle:
    valid_styles: tuple[Py3DmolStyle, ...] = ("stick", "sphere", "line", "cartoon")
    if style not in valid_styles:
        raise ValueError(f"style must be one of {list(valid_styles)}, got '{style}'")
    return cast(Py3DmolStyle, style)


def visualize_molecule_plotlymol(
    molecule,
    mode: str = "ball+stick",
    resolution: int = 32,
    width: int = 600,
    height: int = 500,
    bgcolor: str = "#ffffff"
):
    """
    Create interactive 3D visualization using PlotlyMol (optional backend).

    Args:
        molecule: QuantUI Molecule object
        mode: Visualization mode - one of:
            - "ball+stick": Full-size atoms with bonds (default)
            - "stick": Small atoms with bonds
            - "vdw": Van der Waals spheres only (no bonds)
        resolution: Sphere tessellation resolution (16-64, default: 32)
        width: Figure width in pixels (default: 600)
        height: Figure height in pixels (default: 500)
        bgcolor: Background color as hex string or name (default: "#ffffff")

    Returns:
        plotly.graph_objects.Figure object

    Raises:
        ImportError: If PlotlyMol is not installed
    """
    if not PLOTLYMOL_AVAILABLE:
        raise ImportError(
            "PlotlyMol is not installed. To enable PlotlyMol visualization:\n"
            "  pip install plotlymol"
        )

    # Validate mode
    valid_modes = ["ball+stick", "stick", "vdw"]
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

    # Convert to XYZ string
    xyz_string = molecule_to_xyz_string(molecule)

    # Get charge for RDKit processing
    charge = molecule.charge

    logger.info(
        f"Creating PlotlyMol visualization for {molecule.get_formula()} "
        f"(mode={mode}, resolution={resolution})"
    )

    # Create visualization using PlotlyMol
    fig = draw_3D_rep(
        xyzblock=xyz_string,
        charge=charge,
        mode=mode,
        resolution=resolution,
        bgcolor=bgcolor
    )

    # Set figure size and title
    fig.update_layout(
        width=width,
        height=height,
        title=f"{molecule.get_formula()} - {mode.replace('+', ' & ').title()}"
    )

    return fig


def visualize_molecule(
    molecule,
    backend: BackendName = "auto",
    style: str = "stick",
    width: int = 600,
    height: int = 500,
    bgcolor: str = "white",
    **kwargs
):
    """
    Create interactive 3D visualization (backend-agnostic).

    This is the main visualization function. It automatically selects the
    best available backend or uses the one specified.

    Args:
        molecule: QuantUI Molecule object
        backend: Visualization backend:
            - "auto": Use py3Dmol if available, else PlotlyMol (default)
            - "py3dmol": Use py3Dmol (recommended, stable)
            - "plotlymol": Use PlotlyMol (optional, Plotly-based)
        style: Visualization style (backend-dependent):
            - py3Dmol: "stick", "sphere", "line", "cartoon"
            - PlotlyMol: "ball+stick", "stick", "vdw"
        width: Viewer/figure width in pixels (default: 600)
        height: Viewer/figure height in pixels (default: 500)
        bgcolor: Background color (default: "white")
        **kwargs: Additional backend-specific arguments

    Returns:
        py3Dmol.view or plotly Figure depending on backend

    Raises:
        ImportError: If no visualization backend is available
        ValueError: If specified backend is not available

    Example:
        >>> mol = Molecule(['H', 'H'], [[0, 0, 0], [0, 0, 0.74]])
        >>> # Use default backend (py3Dmol)
        >>> view = visualize_molecule(mol)
        >>> view.show()  # In Jupyter
    """
    # Determine backend
    if backend == "auto":
        if PY3DMOL_AVAILABLE:
            backend = "py3dmol"
        elif PLOTLYMOL_AVAILABLE:
            backend = "plotlymol"
        else:
            raise ImportError(
                "No visualization backend available. Install one of:\n"
                "  pip install py3dmol  (recommended)\n"
                "  pip install plotlymol"
            )

    # Use selected backend
    if backend == "py3dmol":
        py3dmol_style = _validate_py3dmol_style(style)
        return visualize_molecule_py3dmol(
            molecule, style=py3dmol_style, width=width, height=height, bgcolor=bgcolor
        )
    elif backend == "plotlymol":
        # Map common styles to PlotlyMol modes
        mode_map = {"stick": "stick", "sphere": "vdw"}
        mode = mode_map.get(style, "ball+stick")
        return visualize_molecule_plotlymol(
            molecule, mode=mode, width=width, height=height, bgcolor=bgcolor, **kwargs
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def display_molecule(
    molecule,
    backend: Literal["auto", "py3dmol", "plotlymol"] = "auto",
    style: str = "stick",
    show_info: bool = True,
    width: int = 600,
    height: int = 500
):
    """
    Display molecule in Jupyter notebook with optional info box.

    This is the main function for notebook integration. It handles all
    available backends and provides a consistent interface.

    Args:
        molecule: QuantUI Molecule object
        backend: Visualization backend ("auto", "py3dmol", "plotlymol")
        style: Visualization style (backend-dependent)
        show_info: Whether to show molecular info box
        width: Viewer/figure width in pixels
        height: Viewer/figure height in pixels

    Example:
        >>> # In Jupyter notebook
        >>> mol = Molecule(['H', 'H'], [[0, 0, 0], [0, 0, 0.74]])
        >>> display_molecule(mol)  # Uses py3Dmol by default
    """
    from IPython.display import display, HTML

    if not is_visualization_available():
        # Fallback: show text representation
        print("⚠️  3D visualization not available")
        print("\nTo enable visualization, install one of:")
        print("  pip install py3dmol  (recommended)")
        print("  pip install plotlymol")
        print("\nMolecule Information:")
        print(f"  Formula: {molecule.get_formula()}")
        print(f"  Atoms: {len(molecule.atoms)}")
        print(f"  Electrons: {molecule.get_electron_count()}")
        print(f"  Charge: {molecule.charge}")
        print(f"  Multiplicity: {molecule.multiplicity}")
        print("\nXYZ Coordinates:")
        print(molecule.to_xyz_string())
        return

    # Show info box if requested
    if show_info:
        backends = get_available_backends()
        backend_str = ", ".join(backends)
        selected = backend if backend != "auto" else backends[0]

        info_html = f"""
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;
                    margin-bottom: 10px; border-left: 4px solid #4a90e2;">
            <strong>📊 Molecule Information</strong><br>
            <strong>Formula:</strong> {molecule.get_formula()} |
            <strong>Atoms:</strong> {len(molecule.atoms)} |
            <strong>Electrons:</strong> {molecule.get_electron_count()} |
            <strong>Charge:</strong> {molecule.charge} |
            <strong>Multiplicity:</strong> {molecule.multiplicity}<br>
            <small style="color: #666;">Using: {selected} (available: {backend_str})</small>
        </div>
        """
        display(HTML(info_html))

    # Create and display visualization
    try:
        viz = visualize_molecule(
            molecule, backend=backend, style=style, width=width, height=height
        )

        # display(viz) triggers py3Dmol's _repr_html_() method, which embeds
        # the viewer as self-contained HTML.  This works in both JupyterLab
        # and classic Notebook.  viz.show() uses IPython.display.Javascript
        # which is blocked by JupyterLab's content-security-policy and
        # returns None (causing "None" to appear in cell output).
        display(viz)

        logger.info(f"Successfully displayed {molecule.get_formula()}")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        logger.error(f"Display failed for {molecule.get_formula()}: {e}")


def get_installation_message() -> str:
    """
    Get installation instructions for visualization backends.

    Returns:
        Formatted string with installation instructions
    """
    return """
To enable 3D molecular visualization:

Option 1 (Recommended): py3Dmol
  pip install py3dmol

Option 2 (Optional): PlotlyMol
  conda install -c conda-forge rdkit plotly kaleido
  pip install plotlymol

For most users, py3Dmol is sufficient and more stable.
"""


# Module-level check and logging
available = get_available_backends()
if available:
    logger.info(f"Visualization backends available: {', '.join(available)}")
else:
    logger.warning("No visualization backends available")
