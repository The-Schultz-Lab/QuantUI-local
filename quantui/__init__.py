"""
QuantUI-local Package

Lightweight educational quantum chemistry interface for local PySCF calculations.
No cluster or SLURM required — calculations run directly in the Jupyter session.

PySCF requires Linux/macOS/WSL. Windows users should use the Apptainer container.
"""

__version__ = "0.1.0"

from .config import (
    MOLECULE_LIBRARY,
    SUPPORTED_METHODS,
    SUPPORTED_BASIS_SETS,
    DEFAULT_METHOD,
    DEFAULT_BASIS,
    DEFAULT_CHARGE,
    DEFAULT_MULTIPLICITY,
    DEFAULT_FMAX,
    DEFAULT_OPT_STEPS,
    VALID_ATOMS,
    QUICK_START_TEMPLATES,
    WIDGET_LAYOUT,
    DESCRIPTION_WIDTH,
    PYSCF_SCRIPT_TEMPLATE,
    LOG_LEVEL,
    LOG_FORMAT,
)

from .utils import (
    get_username,
    sanitize_filename,
    get_session_resources,
    session_can_handle,
)

from .molecule import Molecule, parse_xyz_input
from .calculator import PySCFCalculation, create_calculation

# Security — catchable exception for constraint violations
from .security import SecurityError

# Educational help content (requires ipywidgets at display time)
from .help_content import help_panel, HELP_TOPICS, VALID_TOPICS

# Progress indicators
from .progress import StepProgress

# Orbital visualization (matplotlib energy diagrams, cube-file viewer)
from .orbital_visualization import (
    OrbitalInfo,
    load_orbital_info,
    orbital_info_from_arrays,
    plot_orbital_diagram,
    orbital_summary_html,
    parse_cube_file,
)

# Calculation comparison (session results only — no job metadata)
from .comparison import (
    CalcSummary,
    summary_from_session_result,
    comparison_table_html,
    plot_comparison,
)

# ASE bridge (optional — requires ase>=3.22.0)
try:
    from .ase_bridge import (
        is_ase_available,
        molecule_to_atoms,
        atoms_to_molecule,
        read_structure_file,
        ase_molecule_library,
        ASE_AVAILABLE,
        ASE_MOLECULE_PRESETS,
    )
except ImportError:
    ASE_AVAILABLE = False
    ASE_MOLECULE_PRESETS: dict = {}

# ASE pre-optimization (optional — requires ase_bridge)
try:
    from .preopt import preoptimize
except ImportError:
    pass

# ASE-PySCF in-session calculator (optional — requires ase>=3.22 + pyscf, Linux/WSL)
try:
    from .session_calc import SessionResult, run_in_session
except ImportError:
    pass

# QM geometry optimizer (optional — requires ase>=3.22 + pyscf, Linux/WSL)
try:
    from .optimizer import OptimizationResult, optimize_geometry
except ImportError:
    pass

# PubChem integration (optional — requires internet)
try:
    from .pubchem import (
        fetch_molecule,
        student_friendly_fetch,
        get_common_molecules,
        check_pubchem_availability,
        PubChemError,
        MoleculeNotFoundError,
        smiles_to_xyz,
        student_friendly_smiles_to_xyz,
        generate_2d_structure_svg,
        display_2d_structure,
        get_smiles_examples,
        validate_smiles,
    )
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False

# Visualization — py3Dmol only (no PlotlyMol fallback)
try:
    from .visualization_py3dmol import (
        is_visualization_available,
        visualize_molecule,
        display_molecule,
    )
    VISUALIZATION_AVAILABLE = True
    PY3DMOL_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    PY3DMOL_AVAILABLE = False

__all__ = [
    # Config constants
    "MOLECULE_LIBRARY",
    "SUPPORTED_METHODS",
    "SUPPORTED_BASIS_SETS",
    "DEFAULT_METHOD",
    "DEFAULT_BASIS",
    "DEFAULT_CHARGE",
    "DEFAULT_MULTIPLICITY",
    "DEFAULT_FMAX",
    "DEFAULT_OPT_STEPS",
    "VALID_ATOMS",
    "QUICK_START_TEMPLATES",
    "WIDGET_LAYOUT",
    "DESCRIPTION_WIDTH",
    "PYSCF_SCRIPT_TEMPLATE",
    # Utils
    "get_username",
    "sanitize_filename",
    "get_session_resources",
    "session_can_handle",
    # Core
    "Molecule",
    "parse_xyz_input",
    "PySCFCalculation",
    "create_calculation",
    # Security
    "SecurityError",
    # UI components
    "help_panel",
    "HELP_TOPICS",
    "VALID_TOPICS",
    "StepProgress",
    # Orbital visualization
    "OrbitalInfo",
    "load_orbital_info",
    "orbital_info_from_arrays",
    "plot_orbital_diagram",
    "orbital_summary_html",
    "parse_cube_file",
    # Comparison
    "CalcSummary",
    "summary_from_session_result",
    "comparison_table_html",
    "plot_comparison",
    # ASE bridge (optional)
    "is_ase_available",
    "molecule_to_atoms",
    "atoms_to_molecule",
    "read_structure_file",
    "ase_molecule_library",
    "ASE_AVAILABLE",
    "ASE_MOLECULE_PRESETS",
    # ASE pre-optimization (optional)
    "preoptimize",
    # In-session calculator (optional — Linux/WSL)
    "SessionResult",
    "run_in_session",
    # QM geometry optimizer (optional — Linux/WSL)
    "OptimizationResult",
    "optimize_geometry",
    # PubChem (optional)
    "fetch_molecule",
    "student_friendly_fetch",
    "get_common_molecules",
    "check_pubchem_availability",
    "PubChemError",
    "MoleculeNotFoundError",
    "PUBCHEM_AVAILABLE",
    "smiles_to_xyz",
    "student_friendly_smiles_to_xyz",
    "generate_2d_structure_svg",
    "display_2d_structure",
    "get_smiles_examples",
    "validate_smiles",
    # Visualization (optional)
    "is_visualization_available",
    "visualize_molecule",
    "display_molecule",
    "VISUALIZATION_AVAILABLE",
    "PY3DMOL_AVAILABLE",
]
