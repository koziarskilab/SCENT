import abc
import random
from typing import Any, Dict, List, Tuple

import gin
import numpy as np
import torch
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch import Tensor, nn
from torchmetrics.functional import auroc, average_precision, precision, recall

from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.gfns.reaction_gfn.api.data_structures import Cache, Molecule
from rgfn.gfns.reaction_gfn.policies.utils import one_hot
from rgfn.gfns.reaction_gfn.proxies.path_cost_proxy import PathCostProxy


class DecomposableModelBase(nn.Module, abc.ABC, TrainingHooksMixin):
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
    def _predict_is_decomposable_with_model(
        self, molecules_num_reactions: List[Tuple[str | Molecule, int]]
    ) -> Tensor:
        ...

    @torch.no_grad()
    def predict_is_decomposable(
        self, molecules_num_reactions: List[Tuple[str | Molecule, int]]
    ) -> Tensor:
        dataset = self.path_cost_proxy.molecule_num_reaction_to_cost
        in_dataset_mask = torch.tensor(
            [item in dataset for item in molecules_num_reactions],
            dtype=torch.bool,
            device=self.device,
        )
        dataset_costs = torch.tensor(
            [dataset[item] for item in molecules_num_reactions if item in dataset],
            dtype=torch.float32,
            device=self.device,
        )
        dataset_is_decomposable = dataset_costs != float("inf")

        predicted_is_decomposable = self._predict_is_decomposable_with_model(
            molecules_num_reactions
        )

        if self.use_dataset == "best":
            predicted_is_decomposable[in_dataset_mask][dataset_is_decomposable] = True
        elif self.use_dataset == "override":
            predicted_is_decomposable[in_dataset_mask] = dataset_costs
        return predicted_is_decomposable

    def _get_fingerprint(self, mol_or_smiles: str | Molecule, num_reactions: int):
        smiles = mol_or_smiles if isinstance(mol_or_smiles, str) else mol_or_smiles.smiles
        item = (smiles, num_reactions)
        assert num_reactions > 0, f"num_reactions should be positive, got {num_reactions}"
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
class BinaryDecomposableModel(DecomposableModelBase):
    def __init__(
        self,
        path_cost_proxy: PathCostProxy,
        max_num_reactions: int,
        use_dataset: str = "best",
        mol_fingerprint_size: int = 2048,
        train_n_samples: int = 1024,
        train_n_iterations: int = 5,
        valid_n_samples: int = 1024,
        negative_ratio: float = 0.2,
        lr: float = 5e-3,
        hidden_dim: int = 128,
        n_recent_datapoints: int = 10_000,
    ):
        super().__init__(
            path_cost_proxy,
            use_dataset,
            mol_fingerprint_size,
            max_num_reactions,
            train_n_samples,
            valid_n_samples,
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(self.mol_fingerprint_size + max_num_reactions, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.optimizer = torch.optim.Adam(self.mlp_c.parameters(), lr=lr)
        self.device = "cpu"
        self.negative_ratio = negative_ratio
        self.n_recent_datapoints = n_recent_datapoints
        self.train_n_iterations = train_n_iterations

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
        negative_dataset = self.path_cost_proxy.negative_molecule_num_reaction
        n_negative_samples = min(int(n_samples * self.negative_ratio), len(negative_dataset))
        negative_samples = (
            random.choices(list(negative_dataset.items()), k=n_negative_samples)
            if n_negative_samples > 0
            else []
        )

        dataset = self.path_cost_proxy.molecule_num_reaction_to_cost
        n_positive_samples = min(n_samples - n_negative_samples, len(dataset))
        n_recent_datapoints = (
            len(dataset) if self.n_recent_datapoints == -1 else self.n_recent_datapoints
        )
        positive_dataset = list(dataset.items())[-n_recent_datapoints:]
        positive_samples = (
            random.choices(positive_dataset, k=n_positive_samples) if n_positive_samples > 0 else []
        )

        samples = negative_samples + positive_samples
        molecules_num_reactions, costs = zip(*samples)
        is_decomposable = torch.tensor(
            [cost != float("inf") for cost in costs], dtype=torch.long, device=self.device
        )

        is_decomposable_predicted = self._predict_is_decomposable_with_model(
            molecules_num_reactions
        )
        loss = torch.nn.functional.binary_cross_entropy(
            is_decomposable_predicted, is_decomposable.float()
        )

        metrics = {
            "is_decomposable_acc": (is_decomposable_predicted > 0)
            .eq(is_decomposable > 0)
            .float()
            .mean()
            .item(),
            "is_decomposable_auc": auroc(
                is_decomposable_predicted, is_decomposable, task="binary"
            ).item(),
            "is_decomposable_auprc": average_precision(
                is_decomposable_predicted, is_decomposable, task="binary"
            ).item(),
            "is_decomposable_precision": precision(
                is_decomposable_predicted, is_decomposable, task="binary"
            ).item(),
            "is_decomposable_recall": recall(
                is_decomposable_predicted, is_decomposable, task="binary"
            ).item(),
            "is_decomposable_loss": loss.item(),
        }
        return loss, metrics

    def _predict_is_decomposable_with_model(
        self, molecules_num_reactions: List[Tuple[str | Molecule, int]]
    ) -> Tensor:
        fingerprints = torch.stack(
            [
                self._get_fingerprint(*molecule_num_reactions)
                for molecule_num_reactions in molecules_num_reactions
            ]
        ).to(self.device)
        logits = self.mlp_c(fingerprints).squeeze(-1)
        return torch.sigmoid(logits)
