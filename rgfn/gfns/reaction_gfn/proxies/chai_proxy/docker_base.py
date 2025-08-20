import abc
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DockingScoringResult:
    """Dataclass to hold the results of a docking calculation. Invalid PDB should be represented as an empty string
    and the corresponding score should be set to float('nan'). A valid PDB should be parsable with RDKit's
    Chem.MolFromPDBBlock(pdb, sanitize=True). The `complex_pdb_list` contains the PDBs of the complexes,
    which are the ligand and receptor bound together. The `ligand_pdb_list` contains the PDBs of the bound ligands
    only. The invalid scores should be set to float('nan').
    """

    ligand_pdb_list: List[str] | None = None
    complex_pdb_list: List[str] | None = None
    scores_list: List[float] | None = None

    def to_dict(self):
        return {
            "ligand_pdb_list": self.ligand_pdb_list,
            "complex_pdb_list": self.complex_pdb_list,
            "scores_list": self.scores_list,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            ligand_pdb_list=data.get("ligand_pdb_list", None),
            complex_pdb_list=data.get("complex_pdb_list", None),
            scores_list=data.get("scores_list", None),
        )

    def is_docked(self):
        if self.ligand_pdb_list:
            return any(pdb != "" for pdb in self.ligand_pdb_list)
        elif self.complex_pdb_list:
            return any(pdb != "" for pdb in self.complex_pdb_list)
        return False

    def is_scored(self):
        if self.scores_list:
            return any(score != float("nan") for score in self.scores_list)
        return False


class DockerBase(abc.ABC):
    """
    Abstract base class for docking implementations.
    """

    n_conformers: int = 1
    receptor_file_path: str
    receptor_center: Optional[List[float]] = None
    receptor_size: Optional[List[float]] = None
    receptor_pocket_residues: Optional[List[Tuple[str, int]]] = None
    receptor_pocket_distance: Optional[float] = None

    @abc.abstractmethod
    def get_important_params_values(self) -> List[Any]:
        """
        Returns a list of important parameters to be hashed when creating a cache folder for the docker. It should
        include parameters that affect the docking results (e.g. excluding caching folder etc.)
        """
        ...

    def prepare_docker(
        self,
        receptor_file_path: str,
        n_conformers: int,
        receptor_metadata_file_path: Optional[str] = None,
    ) -> None:
        """
        Prepares the docker for docking. This method should be called before docking any SMILES strings.

        Args:
            receptor_file_path (str): pdbqt file path of the receptor to dock against.
            n_conformers (int): The number of conformers to generate for each SMILES string.
            receptor_metadata_file_path (Optional[str]): Path to a JSON file containing metadata about the receptor
        """
        self.n_conformers = n_conformers
        self.receptor_file_path = receptor_file_path
        if receptor_metadata_file_path and os.path.exists(receptor_metadata_file_path):
            import json

            receptor_metadata = json.load(open(receptor_metadata_file_path, "r"))
            self.receptor_center = receptor_metadata.get("center", None)
            self.receptor_size = receptor_metadata.get("size", None)
            self.receptor_pocket_residues = receptor_metadata.get("pocket_residues", None)
            self.receptor_pocket_distance = receptor_metadata.get("pocket_distance", None)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the docker."""

    @property
    @abc.abstractmethod
    def returns_single_result(self) -> bool:
        """
        Returns True if the docker returns a single result for a given SMILES string, False otherwise.
        For example, this would be True if the docker operates on SMILES strings directly, rather than on conformers.
        """
        ...

    @abc.abstractmethod
    def dock_smiles(self, smiles_list: List[str]) -> List[DockingScoringResult]:
        """
        Docks a list of SMILES strings.

        Args:
            smiles_list: A list of length N of SMILES strings to dock.

        Returns:
            A list of length N containing the docking results.
        """
        ...
