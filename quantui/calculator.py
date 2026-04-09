"""
QuantUI-local Calculator Module

Generates standalone PySCF Python scripts that students can download and
run independently. This is an "Export Script" feature — the primary
calculation path in QuantUI-local is session_calc.run_in_session(), not
batch script submission.
"""

import logging
from pathlib import Path

from . import config
from .molecule import Molecule

logger = logging.getLogger(__name__)


class PySCFCalculation:
    """
    Generates standalone PySCF scripts for a given molecule and method.

    The primary use in QuantUI-local is the "Export Script" button in the
    notebook, which lets students download a self-contained .py file they
    can study or run outside the notebook environment.
    """

    def __init__(
        self,
        molecule: Molecule,
        method: str = "RHF",
        basis: str = "6-31G",
    ):
        """
        Initialize a PySCF calculation.

        Args:
            molecule: Molecule object to calculate
            method: Calculation method (RHF, UHF)
            basis: Basis set name

        Raises:
            ValueError: If method is not supported
        """
        self.molecule = molecule
        self.method = method.upper()
        self.basis = basis

        if self.method not in config.SUPPORTED_METHODS:
            raise ValueError(
                f"Method '{method}' not supported. "
                f"Supported methods: {', '.join(config.SUPPORTED_METHODS)}"
            )

        if self.basis not in config.SUPPORTED_BASIS_SETS:
            logger.warning(
                f"Basis set '{basis}' not in standard list. "
                f"Proceeding anyway, but it may not be available in PySCF."
            )

        logger.info(
            f"Initialized {self.method}/{self.basis} calculation for "
            f"{self.molecule.get_formula()}"
        )

    def generate_calculation_script(self, output_path: Path) -> str:
        """
        Generate a standalone Python script for the calculation.

        The script runs independently (no QuantUI required) and saves
        results to results.npz next to the script file.

        Args:
            output_path: Path where the script will be saved

        Returns:
            str: The generated script content
        """
        geometry = self.molecule.to_pyscf_format()
        spin = self.molecule.multiplicity - 1
        formula = self.molecule.get_formula()
        job_name = f"{formula}_{self.method}_{self.basis}"

        script_content = config.PYSCF_SCRIPT_TEMPLATE.format(
            job_name=job_name,
            method=self.method,
            basis=self.basis,
            geometry=geometry,
            charge=self.molecule.charge,
            spin=spin,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(script_content)

        logger.info(f"Generated calculation script: {output_path}")
        return script_content

    def get_description(self) -> str:
        """
        Get a human-readable description of the calculation.

        Returns:
            str: Description string
        """
        method_names = {
            "RHF": "Restricted Hartree-Fock",
            "UHF": "Unrestricted Hartree-Fock",
        }
        method_full = method_names.get(self.method, self.method)
        return (
            f"{method_full} calculation of {self.molecule.get_formula()} "
            f"using {self.basis} basis set"
        )

    def get_educational_notes(self) -> str:
        """
        Get educational notes about the calculation method and basis set.

        Returns:
            str: Markdown-formatted educational description
        """
        notes = []

        if self.method == "RHF":
            notes.append(
                "**Restricted Hartree-Fock (RHF)**: Used for closed-shell molecules "
                "where all electrons are paired. Assumes spatial orbitals are "
                "doubly occupied."
            )
        elif self.method == "UHF":
            notes.append(
                "**Unrestricted Hartree-Fock (UHF)**: Used for open-shell molecules "
                "with unpaired electrons. Alpha and beta electrons occupy different "
                "spatial orbitals."
            )

        if self.basis == "STO-3G":
            notes.append(
                "**STO-3G basis**: Minimal basis set using 3 Gaussian functions to "
                "approximate Slater-type orbitals. Fast but low accuracy."
            )
        elif "6-31G" in self.basis:
            notes.append(
                "**6-31G family**: Split-valence basis sets providing good balance "
                "between accuracy and computational cost. The * indicates polarization "
                "functions for better description of molecular bonding."
            )
        elif "cc-pV" in self.basis:
            notes.append(
                "**Correlation-consistent basis sets**: High-quality basis sets "
                "designed for systematic convergence to exact results. More expensive "
                "but more accurate than minimal or split-valence sets."
            )

        if self.molecule.multiplicity > 1:
            notes.append(
                f"**Spin multiplicity {self.molecule.multiplicity}**: This molecule "
                f"has {self.molecule.multiplicity - 1} unpaired electron(s). "
                "UHF is appropriate for open-shell systems."
            )

        return "\n\n".join(notes)


def create_calculation(
    molecule: Molecule,
    method: str,
    basis: str,
) -> PySCFCalculation:
    """
    Factory function to create a PySCF calculation.

    Args:
        molecule: Molecule to calculate
        method: Calculation method
        basis: Basis set

    Returns:
        PySCFCalculation: Initialized calculation object
    """
    return PySCFCalculation(molecule=molecule, method=method, basis=basis)
