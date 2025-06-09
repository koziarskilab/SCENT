import abc
import random
from typing import Any, Dict, List, Tuple

import gin
import numpy as np
import torch
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch import Tensor, nn

from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.gfns.reaction_gfn.api.data_structures import Cache, Molecule
from rgfn.gfns.reaction_gfn.policies.utils import one_hot
from rgfn.gfns.reaction_gfn.proxies.path_cost_proxy import PathCostProxy


class CostModelBase(nn.Module, abc.ABC, TrainingHooksMixin):
    def __init__(
        self,
        path_cost_proxy: PathCostProxy,
        use_dataset: str,
        mol_fingerprint_size: int,
        max_num_reactions: int,
        train_n_samples: int,
        valid_n_samples: int,
    ):
        super().__init__()
        assert use_dataset in ["none", "best", "override"]
        self.path_cost_proxy = path_cost_proxy
        self.use_dataset = use_dataset
        self.mol_fingerprint_size = mol_fingerprint_size
        self.max_num_reactions = max_num_reactions
        self._fingerprint_cache = Cache(max_size=50_000)
        self.device = "cpu"
        self.train_n_samples = train_n_samples
        self.valid_n_samples = valid_n_samples

    @abc.abstractmethod
    def train_one_epoch(self) -> Dict[str, Any]:
        ...

    @torch.no_grad()
    @abc.abstractmethod
    def valid_one_epoch(self) -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    def _predict_costs_with_model(
        self, molecules_num_reactions: List[Tuple[str | Molecule, int]]
    ) -> Tensor:
        ...

    @torch.no_grad()
    def predict_costs(self, molecules_num_reactions: List[Tuple[str | Molecule, int]]) -> Tensor:
        predicted_costs = self._predict_costs_with_model(molecules_num_reactions)

        dataset = self.path_cost_proxy.molecule_num_reaction_to_cost
        smiles_list = [
            item[0] if isinstance(item[0], str) else item[0].smiles
            for item in molecules_num_reactions
        ]
        smiles_num_reactions = list(zip(smiles_list, [item[1] for item in molecules_num_reactions]))
        in_dataset_mask = torch.tensor(
            [item in dataset for item in smiles_num_reactions],
            dtype=torch.bool,
            device=self.device,
        )
        dataset_costs = torch.tensor(
            [dataset[item] for item in smiles_num_reactions if item in dataset],
            dtype=predicted_costs.dtype,
            device=self.device,
        )

        if self.use_dataset == "best":
            predicted_costs[in_dataset_mask] = torch.min(
                predicted_costs[in_dataset_mask], dataset_costs
            )
        elif self.use_dataset == "override":
            predicted_costs[in_dataset_mask] = dataset_costs

        return predicted_costs

    def _get_fingerprint(self, mol_or_smiles: str | Molecule, num_reactions: int):
        smiles = mol_or_smiles if isinstance(mol_or_smiles, str) else mol_or_smiles.smiles
        item = (smiles, num_reactions)
        assert num_reactions > 0
        if item not in self._fingerprint_cache:
            rdkit_mol = (
                Molecule(mol_or_smiles).rdkit_mol
                if isinstance(mol_or_smiles, str)
                else mol_or_smiles.rdkit_mol
            )
            fp = GetMorganFingerprintAsBitVect(rdkit_mol, radius=2, nBits=self.mol_fingerprint_size)
            array = np.zeros((0,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, array)
            molecular_fingerprint = torch.tensor(array, dtype=torch.float32)
            num_reaction_fingerprints = torch.tensor(
                one_hot(num_reactions - 1, self.max_num_reactions)
            )
            fingerprint = torch.cat([molecular_fingerprint, num_reaction_fingerprints], dim=0)
            self._fingerprint_cache[item] = fingerprint
        return self._fingerprint_cache[item]


@gin.configurable()
class ConstantCostModel(CostModelBase):
    def __init__(
        self,
        path_cost_proxy: PathCostProxy,
        use_dataset: str,
        constant_cost: float | None = None,
    ):
        super().__init__(
            path_cost_proxy,
            use_dataset,
            mol_fingerprint_size=1,
            max_num_reactions=1,
            train_n_samples=0,
            valid_n_samples=0,
        )
        self.constant_cost = constant_cost or path_cost_proxy.get_fragment_costs_mean_std()[0]

    def train_one_epoch(self) -> Dict[str, Any]:
        return {}

    @torch.no_grad()
    def valid_one_epoch(self) -> Dict[str, Any]:
        return {}

    def _predict_costs_with_model(
        self, molecules_num_reactions: List[Tuple[str | Molecule, int]]
    ) -> Tensor:
        return torch.tensor([self.constant_cost] * len(molecules_num_reactions), device=self.device)


@gin.configurable()
class SimpleCostModel(CostModelBase):
    def __init__(
        self,
        path_cost_proxy: PathCostProxy,
        max_num_reactions: int,
        mol_fingerprint_size: int = 2048,
        use_dataset: str = "none",
        train_n_samples: int = 1024,
        train_n_iterations: int = 5,
        valid_n_samples: int = 1024,
        lr: float = 1e-2,
        hidden_dim: int = 128,
        n_recent_datapoints: int = 10_000,
        normalize_costs: bool = True,
        clamp_zero_costs: bool = False,
    ):
        super().__init__(
            path_cost_proxy,
            use_dataset,
            mol_fingerprint_size,
            max_num_reactions,
            train_n_samples,
            valid_n_samples,
        )
        self.clamp_zero_costs = clamp_zero_costs
        self.mlp_c = nn.Sequential(
            nn.Linear(self.mol_fingerprint_size + max_num_reactions, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.optimizer = torch.optim.Adam(self.mlp_c.parameters(), lr=lr)
        self.n_recent_datapoints = n_recent_datapoints
        self.train_n_iterations = train_n_iterations
        self.normalize_costs = normalize_costs
        (
            self.fragment_cost_mean,
            self.fragment_cost_std,
        ) = self.path_cost_proxy.get_fragment_costs_mean_std()

    def train_one_epoch(self) -> Dict[str, Any]:
        for _ in range(self.train_n_iterations):
            self.optimizer.zero_grad()
            loss, metrics = self._compute_loss_on_random_samples(self.train_n_samples)
            loss.backward()
            self.optimizer.step()
        return metrics

    @torch.no_grad()
    def valid_one_epoch(self) -> Dict[str, Any]:
        return self._compute_loss_on_random_samples(self.valid_n_samples)[1]

    def _compute_loss_on_random_samples(self, n_samples: int) -> Tuple[Tensor, Dict[str, Any]]:
        dataset = self.path_cost_proxy.molecule_num_reaction_to_cost
        n_recent_datapoints = (
            len(dataset) if self.n_recent_datapoints == -1 else self.n_recent_datapoints
        )
        positive_dataset = list(item for item in dataset.items() if item[1] != float("inf"))[
            -n_recent_datapoints:
        ]
        n_samples = min(n_samples, len(positive_dataset))
        samples = random.choices(positive_dataset, k=n_samples)
        molecules_num_reactions, costs = zip(*samples)
        costs = torch.tensor(costs).float().to(self.device)
        predicted_costs = self._predict_costs_with_model(molecules_num_reactions)
        mean_predicted_cost = predicted_costs.mean().item()
        if self.normalize_costs:
            predicted_costs = (predicted_costs - self.fragment_cost_mean) / self.fragment_cost_std
            costs = (costs - self.fragment_cost_mean) / self.fragment_cost_std

        loss = nn.functional.mse_loss(predicted_costs, costs)
        spearman_correlation = np.corrcoef(
            predicted_costs.cpu().detach().numpy(), costs.cpu().detach().numpy()
        )[0, 1]
        return loss, {
            "cost_spearman_correlation": spearman_correlation,
            "cost_loss": loss.item(),
            "mean_predicted_cost": mean_predicted_cost,
        }

    def _predict_costs_with_model(
        self, molecules_num_reactions: List[Tuple[str | Molecule, int]]
    ) -> Tensor:
        fingerprints = torch.stack(
            [
                self._get_fingerprint(*molecule_num_reactions)
                for molecule_num_reactions in molecules_num_reactions
            ]
        ).to(self.device)
        costs = self.mlp_c(fingerprints).squeeze(-1)
        if not self.normalize_costs:
            costs = costs**2
        else:
            costs = costs * self.fragment_cost_std + self.fragment_cost_mean
        if self.clamp_zero_costs:
            costs = torch.clamp(costs, min=0.0)
        return costs
