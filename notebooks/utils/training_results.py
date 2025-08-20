import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem

from rgfn.gfns.reaction_gfn.api.data_structures import Molecule


@dataclass
class TrainingResults:
    model_name: str
    seed: int
    templates_name: str
    task_name: str
    threshold: float
    include_paths: bool = True
    max_paths: int = 320000
    fp_type: str = "morgan_3"
    use_cache: bool = True
    paths: List[str | Tuple[str, ...]] | None = None
    results_dir: Path = Path("results")

    base_dir: Path = field(init=False, repr=False)
    molecules: List[str] = field(init=False, repr=False)
    rewards: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.base_dir = (
            self.results_dir / self.templates_name / self.task_name / self.model_name / "training"
        )
        if self.include_paths:
            self.file_path = self.base_dir / f"paths_{self.seed}.csv"
        else:
            self.file_path = self.base_dir / f"molecules_{self.seed}.csv"
        if not self.file_path.exists():
            self.results_dir = Path("/Volumes/External Disk/RGFN/notebooks/results/")
            self.base_dir = (
                self.results_dir
                / self.templates_name
                / self.task_name
                / self.model_name
                / "training"
            )
            if self.include_paths:
                self.file_path = self.base_dir / f"paths_{self.seed}.csv"
            else:
                self.file_path = self.base_dir / f"molecules_{self.seed}.csv"
            if not self.file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")
        if self.use_cache:
            self.load()

    def _get_molecule_from_path(self, path: List[Any]) -> str:
        assert path[-1] == ""
        template = path[-2][0]
        previous_mol = Molecule(path[-3])
        reactants = [previous_mol.rdkit_mol, Molecule(path[-2][1]).rdkit_mol]
        rxn = AllChem.ReactionFromSmarts(template)

        products = rxn.RunReactants(reactants)
        if len(products) == 0:
            products = rxn.RunReactants(reactants[::-1])

        if len(products) == 0:
            return previous_mol.smiles

        product = products[0][0]
        return Molecule(product).smiles

    def load_heavy_stuff(self):
        if hasattr(self, "molecules"):
            return
        df = pd.read_csv(self.file_path)
        if self.include_paths:
            self.paths = df["path"].apply(eval).tolist()
            self.molecules = [p[-1] for p in self.paths]
            if len(set(self.molecules)) <= 1:
                self.molecules = [self._get_molecule_from_path(p) for p in self.paths]
        else:
            column = df.columns[-2]
            self.molecules = df[column].tolist()[: self.max_paths]
        if " proxy" in df.columns:
            self.rewards = np.array(df[" proxy"].tolist())[: self.max_paths]
        else:
            self.rewards = np.array(df["proxy"].tolist())[: self.max_paths]
        if self.task_name.lower() != "seh" and self.rewards.max() > 1.1:
            self.rewards = self.rewards / 8.0

    def cache_path(self):
        return self.base_dir / f"results_{self.seed}_{self.threshold}_{self.fp_type}.pkl"

    scaffolds: List[str] | None = None
    num_scaffolds: np.ndarray | None = None
    num_scaffold_modes: np.ndarray | None = None
    num_modes: np.ndarray | None = None
    path_costs: np.ndarray | None = None
    moving_average_path_costs: np.ndarray | None = None
    cheapest_scaffolds_path_costs: np.ndarray | None = None
    last_n_average_path_costs: np.ndarray | None = None
    molecule_to_cheapest_cost: dict | None = None

    def save(self):
        results_dict = {
            "scaffolds": self.scaffolds,
            "num_scaffolds": self.num_scaffolds,
            "num_scaffold_modes": self.num_scaffold_modes,
            "num_modes": self.num_modes,
            "path_costs": self.path_costs,
            "moving_average_path_costs": self.moving_average_path_costs,
            "cheapest_scaffolds_path_costs": self.cheapest_scaffolds_path_costs,
            "last_n_average_path_costs": self.last_n_average_path_costs,
            "molecule_to_cheapest_cost": self.molecule_to_cheapest_cost,
        }
        pickle.dump(results_dict, open(self.cache_path(), "wb"))

    def load(self):
        result_path = self.cache_path()
        if not result_path.exists():
            return
        results_dict = pickle.load(open(result_path, "rb"))
        for key, value in results_dict.items():
            setattr(self, key, value)


class TrainingResultsList:
    def __init__(self, results: List[TrainingResults]):
        self.results = results
        self.name = results[0].model_name

    def get_num_scaffolds_mean_std(self, n: int = -1):
        values = [
            result.num_scaffolds[n] for result in self.results if result.num_scaffolds is not None
        ]
        return np.mean(values), np.std(values)

    def get_num_scaffold_modes_mean_std(self):
        values = [
            result.num_scaffold_modes[-1]
            for result in self.results
            if result.num_scaffold_modes is not None
        ]
        return np.mean(values), np.std(values)

    def get_num_modes_mean_std(self, n: int = -1):
        values = [result.num_modes[n] for result in self.results if result.num_modes is not None]
        return np.mean(values), np.std(values)

    def get_average_path_cost_mean_std(self):
        values = [
            result.moving_average_path_costs[-1]
            for result in self.results
            if result.moving_average_path_costs is not None
        ]
        return np.mean(values), np.std(values)

    def get_cheapest_scaffolds_path_costs_mean_std(self):
        values = [
            result.cheapest_scaffolds_path_costs[-1]
            for result in self.results
            if result.cheapest_scaffolds_path_costs is not None
        ]
        return np.mean(values), np.std(values)

    def get_last_n_average_path_costs_mean_std(self):
        values = [
            result.last_n_average_path_costs[-1]
            for result in self.results
            if result.last_n_average_path_costs is not None
        ]
        return np.mean(values), np.std(values)
