"""
PubChem Integration Module

Provides functions to search and retrieve molecular structures from PubChem
for educational use in quantum chemistry calculations.
"""

import requests
import logging
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import time
from functools import lru_cache

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from . import config

logger = logging.getLogger(__name__)

# PubChem API endpoints
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_TIMEOUT = 10  # seconds


class PubChemError(Exception):
    """Base exception for PubChem-related errors."""
    pass


class MoleculeNotFoundError(PubChemError):
    """Raised when a molecule cannot be found in PubChem."""
    pass


class PubChemAPIError(PubChemError):
    """Raised when PubChem API request fails."""
    pass


def search_molecule_by_name(name: str) -> int:
    """
    Search for a molecule in PubChem by name and return its CID.

    Args:
        name: Common name or IUPAC name of the molecule

    Returns:
        int: PubChem Compound ID (CID) if found, None otherwise

    Raises:
        PubChemAPIError: If API request fails
        MoleculeNotFoundError: If molecule not found
    """
    url = f"{PUBCHEM_BASE_URL}/compound/name/{name}/cids/JSON"

    try:
        logger.debug(f"Searching PubChem for: {name}")
        response = requests.get(url, timeout=PUBCHEM_TIMEOUT)

        if response.status_code == 404:
            raise MoleculeNotFoundError(f"Molecule '{name}' not found in PubChem")

        response.raise_for_status()
        data = response.json()

        cids = data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            raise MoleculeNotFoundError(f"No CID found for '{name}'")

        cid: int = int(cids[0])  # Take first match
        logger.info(f"Found CID {cid} for '{name}'")
        return cid

    except requests.RequestException as e:
        logger.error(f"PubChem API request failed: {e}")
        raise PubChemAPIError(f"Failed to connect to PubChem: {e}")


@lru_cache(maxsize=50)
def get_molecule_sdf(cid: int, conformer_3d: bool = True) -> str:
    """
    Retrieve molecule SDF from PubChem by CID.

    Args:
        cid: PubChem Compound ID
        conformer_3d: If True, fetch 3D conformer; if False, fetch 2D structure

    Returns:
        str: SDF file content

    Raises:
        PubChemAPIError: If API request fails
        MoleculeNotFoundError: If CID not found
    """
    record_type = "3d" if conformer_3d else "2d"
    url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/record/SDF"

    params = {}
    if conformer_3d:
        params["record_type"] = "3d"

    try:
        logger.debug(f"Fetching {record_type.upper()} SDF for CID {cid}")
        response = requests.get(url, params=params, timeout=PUBCHEM_TIMEOUT)

        if response.status_code == 404:
            if conformer_3d:
                # Try falling back to 2D if 3D not available
                logger.warning(f"No 3D structure for CID {cid}, trying 2D")
                return get_molecule_sdf(cid, conformer_3d=False)
            raise MoleculeNotFoundError(f"CID {cid} not found in PubChem")

        response.raise_for_status()
        sdf_content: str = str(response.text)

        logger.info(f"Retrieved {record_type.upper()} SDF for CID {cid}")
        return sdf_content

    except requests.RequestException as e:
        logger.error(f"PubChem SDF request failed: {e}")
        raise PubChemAPIError(f"Failed to retrieve molecule: {e}")


def sdf_to_xyz(sdf_content: str) -> Tuple[str, Dict[str, Any]]:
    """
    Convert SDF content to XYZ format string.

    Args:
        sdf_content: SDF file content as string

    Returns:
        Tuple of (xyz_string, metadata_dict)
        xyz_string format: "n_atoms\\ncomment\\natom x y z\\n..."
        metadata includes: formula, molecular_weight, charge

    Raises:
        ValueError: If SDF parsing fails
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SDF to XYZ conversion")

    try:
        # Parse SDF with RDKit
        mol = Chem.MolFromMolBlock(sdf_content)

        if mol is None:
            raise ValueError("Failed to parse SDF content")

        # Add hydrogens if not present
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates if needed
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)

        # Extract coordinates and build XYZ string
        conf = mol.GetConformer()
        xyz_lines = [str(mol.GetNumAtoms())]

        # Get molecular formula
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        xyz_lines.append(f"PubChem molecule: {formula}")

        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            symbol = atom.GetSymbol()
            xyz_lines.append(f"{symbol:3s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

        xyz_string = "\n".join(xyz_lines)

        # Gather metadata
        metadata = {
            "formula": formula,
            "molecular_weight": Chem.Descriptors.MolWt(mol),
            "charge": Chem.GetFormalCharge(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_heavy_atoms": mol.GetNumHeavyAtoms()
        }

        logger.debug(f"Converted SDF to XYZ: {metadata['formula']}")
        return xyz_string, metadata

    except Exception as e:
        logger.error(f"SDF to XYZ conversion failed: {e}")
        raise ValueError(f"Failed to convert SDF to XYZ: {e}")


def fetch_molecule(name: str, conformer_3d: bool = True) -> Tuple[str, Dict[str, Any], int]:
    """
    High-level function to fetch molecule from PubChem by name.

    Performs search, retrieves SDF, and converts to XYZ in one call.

    Args:
        name: Molecule name (common or IUPAC)
        conformer_3d: If True, fetch 3D structure; if False, 2D

    Returns:
        Tuple of (xyz_string, metadata_dict, cid)

    Raises:
        PubChemError: If any step fails
    """
    logger.info(f"Fetching molecule '{name}' from PubChem")

    # Search for CID
    cid = search_molecule_by_name(name)

    # Get SDF
    sdf_content = get_molecule_sdf(cid, conformer_3d=conformer_3d)

    # Convert to XYZ
    xyz_string, metadata = sdf_to_xyz(sdf_content)

    # Add CID to metadata
    metadata["pubchem_cid"] = cid
    metadata["pubchem_name"] = name

    logger.info(f"Successfully fetched '{name}' (CID: {cid})")
    return xyz_string, metadata, cid


def get_common_molecules() -> Dict[str, str]:
    """
    Get a curated list of common molecules for educational use.

    Returns:
        Dict mapping display names to PubChem search names
    """
    return {
        # Simple molecules
        "Water (H₂O)": "water",
        "Hydrogen (H₂)": "hydrogen",
        "Oxygen (O₂)": "oxygen",
        "Nitrogen (N₂)": "nitrogen",
        "Carbon Dioxide (CO₂)": "carbon dioxide",
        "Ammonia (NH₃)": "ammonia",
        "Methane (CH₄)": "methane",

        # Organic molecules
        "Ethanol (CH₃CH₂OH)": "ethanol",
        "Acetic Acid (CH₃COOH)": "acetic acid",
        "Acetone (CH₃COCH₃)": "acetone",
        "Benzene (C₆H₆)": "benzene",
        "Toluene (C₆H₅CH₃)": "toluene",
        "Phenol (C₆H₅OH)": "phenol",

        # Biochemical molecules
        "Glucose (C₆H₁₂O₆)": "glucose",
        "Glycine (NH₂CH₂COOH)": "glycine",
        "Alanine (CH₃CH(NH₂)COOH)": "alanine",
        "Caffeine": "caffeine",
        "Aspirin": "aspirin",
        "Vitamin C": "ascorbic acid",

        # Ions (may need special handling)
        "Hydronium (H₃O⁺)": "hydronium",
        "Hydroxide (OH⁻)": "hydroxide",
        "Ammonium (NH₄⁺)": "ammonium",
    }


def student_friendly_fetch(name: str) -> Tuple[Optional[str], str]:
    """
    Fetch molecule with student-friendly error messages.

    Args:
        name: Molecule name to search

    Returns:
        Tuple of (xyz_string, message)
        xyz_string is None if fetch failed
        message describes success or failure
    """
    try:
        xyz_string, metadata, cid = fetch_molecule(name, conformer_3d=True)

        message = (
            f"✓ Found '{name}' in PubChem!\n"
            f"  CID: {cid}\n"
            f"  Formula: {metadata['formula']}\n"
            f"  Atoms: {metadata['num_atoms']} "
            f"({metadata['num_heavy_atoms']} heavy atoms)\n"
            f"  Molecular Weight: {metadata['molecular_weight']:.2f} g/mol"
        )

        return xyz_string, message

    except MoleculeNotFoundError:
        message = (
            f"❌ Could not find '{name}' in PubChem.\n"
            f"   Try:\n"
            f"   • Check spelling (e.g., 'ethanol' not 'ethonal')\n"
            f"   • Use IUPAC name (e.g., 'ethanol' not 'alcohol')\n"
            f"   • Use common name (e.g., 'water' not 'dihydrogen monoxide')\n"
            f"   • Search manually at: https://pubchem.ncbi.nlm.nih.gov/"
        )
        return None, message

    except PubChemAPIError:
        message = (
            f"❌ Connection to PubChem failed.\n"
            f"   • Check your internet connection\n"
            f"   • Try again in a moment\n"
            f"   • Use preset molecules if problem persists"
        )
        return None, message

    except Exception as e:
        message = (
            f"❌ Error fetching molecule: {str(e)}\n"
            f"   Please try a different molecule or contact your instructor."
        )
        logger.error(f"Unexpected error in student_friendly_fetch: {e}", exc_info=True)
        return None, message


def check_pubchem_availability() -> bool:
    """
    Check if PubChem API is accessible.

    Returns:
        bool: True if PubChem is accessible, False otherwise
    """
    try:
        url = f"{PUBCHEM_BASE_URL}/compound/cid/962/property/MolecularFormula/JSON"
        response = requests.get(url, timeout=5)
        return bool(response.status_code == 200)
    except:
        return False


# ============================================================================
# SMILES Input and 2D Structure Rendering
# ============================================================================

def smiles_to_xyz(smiles: str, optimize_3d: bool = True) -> Tuple[str, Dict[str, Any]]:
    """
    Convert SMILES string to XYZ coordinates with 3D structure generation.

    Args:
        smiles: SMILES string (e.g., "CCO" for ethanol)
        optimize_3d: If True, generate and optimize 3D coordinates with UFF

    Returns:
        Tuple of (xyz_string, metadata_dict)
        xyz_string format: "n_atoms\\ncomment\\natom x y z\\n..."
        metadata includes: formula, molecular_weight, charge, smiles

    Raises:
        ValueError: If SMILES parsing fails or 3D generation fails
        ImportError: If RDKit is not available
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMILES conversion")

    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        if optimize_3d:
            # Generate conformer
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result != 0:
                # Try with random coords if embedding fails
                AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)

            # Optimize with UFF force field
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except:
                logger.warning("UFF optimization failed, using unoptimized coordinates")

        # Extract coordinates
        if mol.GetNumConformers() == 0:
            raise ValueError("Failed to generate 3D coordinates")

        conf = mol.GetConformer()
        xyz_lines = [str(mol.GetNumAtoms())]

        # Get molecular formula
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        xyz_lines.append(f"Generated from SMILES: {smiles} ({formula})")

        # Build XYZ string
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            symbol = atom.GetSymbol()
            xyz_lines.append(f"{symbol:3s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

        xyz_string = "\n".join(xyz_lines)

        # Gather metadata
        metadata = {
            "formula": formula,
            "molecular_weight": Chem.Descriptors.MolWt(mol),
            "charge": Chem.GetFormalCharge(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol)
        }

        logger.info(f"Converted SMILES '{smiles}' to XYZ: {metadata['formula']}")
        return xyz_string, metadata

    except Exception as e:
        logger.error(f"SMILES to XYZ conversion failed: {e}")
        raise ValueError(f"Failed to convert SMILES to XYZ: {e}")


def student_friendly_smiles_to_xyz(smiles: str) -> Tuple[Optional[str], str]:
    """
    Convert SMILES to XYZ with student-friendly error messages.

    Args:
        smiles: SMILES string

    Returns:
        Tuple of (xyz_string, message)
        xyz_string is None if conversion failed
        message describes success or failure
    """
    try:
        xyz_string, metadata = smiles_to_xyz(smiles, optimize_3d=True)

        message = (
            f"✓ Converted SMILES to 3D structure!\n"
            f"  SMILES: {smiles}\n"
            f"  Formula: {metadata['formula']}\n"
            f"  Atoms: {metadata['num_atoms']} "
            f"({metadata['num_heavy_atoms']} heavy atoms)\n"
            f"  Molecular Weight: {metadata['molecular_weight']:.2f} g/mol\n"
            f"  Canonical SMILES: {metadata['canonical_smiles']}"
        )

        return xyz_string, message

    except ValueError as e:
        message = (
            f"❌ Invalid SMILES string: {smiles}\n"
            f"   Error: {str(e)}\n\n"
            f"   SMILES Tips:\n"
            f"   • Check syntax (use RDKit/OpenBabel style)\n"
            f"   • Ethanol: CCO or C(C)O\n"
            f"   • Benzene: c1ccccc1 or C1=CC=CC=C1\n"
            f"   • Water: O (just the atom symbol)\n"
            f"   • Methane: C\n\n"
            f"   Resources:\n"
            f"   • SMILES Tutorial: https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html\n"
            f"   • Draw structure: https://pubchem.ncbi.nlm.nih.gov/edit3/index.html"
        )
        return None, message

    except ImportError:
        message = (
            f"❌ RDKit is required for SMILES conversion.\n"
            f"   Install with: conda install -c conda-forge rdkit"
        )
        return None, message

    except Exception as e:
        message = (
            f"❌ Error converting SMILES: {str(e)}\n"
            f"   Please try a different molecule or contact your instructor."
        )
        logger.error(f"Unexpected error in student_friendly_smiles_to_xyz: {e}", exc_info=True)
        return None, message


def generate_2d_structure_svg(smiles: Optional[str] = None,
                               mol: Optional[object] = None,
                               xyz_string: Optional[str] = None,
                               width: int = 300,
                               height: int = 300) -> Optional[str]:
    """
    Generate 2D structure diagram as SVG string.

    Can accept input as SMILES, RDKit Mol object, or XYZ string.

    Args:
        smiles: SMILES string (if provided)
        mol: RDKit Mol object (if provided)
        xyz_string: XYZ coordinate string (if provided)
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        SVG string of 2D structure, or None if generation fails

    Raises:
        ImportError: If RDKit is not available
        ValueError: If no valid input provided
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for 2D structure rendering")

    try:
        from rdkit.Chem import Draw
        from io import BytesIO

        # Get RDKit molecule from input
        if mol is not None:
            rdkit_mol = mol
        elif smiles is not None:
            rdkit_mol = Chem.MolFromSmiles(smiles)
            if rdkit_mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
        elif xyz_string is not None:
            # Convert XYZ to SMILES (requires RDKit bond perception)
            # Parse XYZ
            lines = xyz_string.strip().split('\n')
            if len(lines) < 3:
                raise ValueError("Invalid XYZ format")

            # Build mol from XYZ
            from rdkit.Chem import rdDetermineBonds
            rdkit_mol = Chem.Mol()
            conf = Chem.Conformer()

            for i, line in enumerate(lines[2:]):  # Skip first 2 lines
                parts = line.split()
                if len(parts) < 4:
                    continue
                symbol = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

                atom = Chem.Atom(symbol)
                rdkit_mol.AddAtom(atom)
                conf.SetAtomPosition(i, (x, y, z))

            rdkit_mol.AddConformer(conf)

            # Determine bonds
            rdDetermineBonds.DetermineBonds(rdkit_mol)
        else:
            raise ValueError("Must provide smiles, mol, or xyz_string")

        # Generate 2D coordinates for nice layout
        AllChem.Compute2DCoords(rdkit_mol)

        # Draw molecule to SVG
        drawer = Draw.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(rdkit_mol)
        drawer.FinishDrawing()
        svg: str = str(drawer.GetDrawingText())

        logger.debug("Generated 2D structure SVG")
        return svg

    except Exception as e:
        logger.error(f"2D structure generation failed: {e}")
        return None


def display_2d_structure(smiles: Optional[str] = None,
                         mol: Optional[object] = None,
                         xyz_string: Optional[str] = None,
                         width: int = 400,
                         height: int = 300):
    """
    Display 2D structure diagram in Jupyter notebook.

    Args:
        smiles: SMILES string (if provided)
        mol: RDKit Mol object (if provided)
        xyz_string: XYZ coordinate string (if provided)
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        IPython display object or None if fails
    """
    try:
        from IPython.display import SVG, display as ipython_display

        svg = generate_2d_structure_svg(
            smiles=smiles,
            mol=mol,
            xyz_string=xyz_string,
            width=width,
            height=height
        )

        if svg:
            ipython_display(SVG(svg))
            return True
        else:
            print("⚠️  Could not generate 2D structure")
            return False

    except ImportError as e:
        logger.warning(f"Could not display 2D structure: {e}")
        print("⚠️  IPython display not available")
        return False
    except Exception as e:
        logger.error(f"2D structure display failed: {e}")
        print(f"⚠️  2D structure display failed: {e}")
        return False


def get_smiles_examples() -> Dict[str, str]:
    """
    Get example SMILES strings for educational use.

    Returns:
        Dict mapping molecule names to SMILES strings
    """
    return {
        # Simple molecules
        "Water": "O",
        "Ammonia": "N",
        "Methane": "C",
        "Ethane": "CC",
        "Propane": "CCC",

        # Functional groups
        "Methanol": "CO",
        "Ethanol": "CCO",
        "Acetic Acid": "CC(=O)O",
        "Acetone": "CC(=O)C",
        "Formaldehyde": "C=O",

        # Aromatics
        "Benzene": "c1ccccc1",
        "Toluene": "Cc1ccccc1",
        "Phenol": "Oc1ccccc1",
        "Aniline": "Nc1ccccc1",

        # Biochemical
        "Glycine": "NCC(=O)O",
        "Alanine": "CC(N)C(=O)O",
        "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",

        # Common molecules
        "Carbon Dioxide": "O=C=O",
        "Hydrogen Peroxide": "OO",
        "Ethylene": "C=C",
        "Acetylene": "C#C",
    }


def validate_smiles(smiles: str) -> Tuple[bool, str]:
    """
    Validate a SMILES string.

    Args:
        smiles: SMILES string to validate

    Returns:
        Tuple of (is_valid, message)
    """
    if not RDKIT_AVAILABLE:
        return False, "RDKit not available"

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES syntax"

        # Check if molecule is reasonable
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return False, "Molecule has no atoms"

        if num_atoms > 200:
            return False, f"Molecule too large ({num_atoms} atoms). Consider smaller molecules for calculations."

        return True, f"Valid SMILES ({num_atoms} atoms)"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
