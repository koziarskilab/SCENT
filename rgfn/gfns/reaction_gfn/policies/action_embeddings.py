import abc
import math
from typing import Any, Dict, List

import gin
import numpy as np
import torch
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch import Tensor, nn
from torch.nn import init

from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.api.trajectories import TrajectoriesContainer
from rgfn.gfns.reaction_gfn.api.data_structures import Molecule
from rgfn.gfns.reaction_gfn.api.reaction_data_factory import ReactionDataFactory
from rgfn.gfns.reaction_gfn.dynamic_library.reaction_dynamic_library import (
    DynamicLibrary,
)


class ActionEmbeddingBase(abc.ABC, nn.Module, TrainingHooksMixin):
    def __init__(self, data_factory: ReactionDataFactory, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.data_factory = data_factory
        self._cache: Tensor | None = None
        self.device = "cpu"

    def get_embeddings(self) -> Tensor:
        if self._cache is None:
            self._cache = self._get_embeddings()
        return self._cache

    def clear_cache(self):
        self._cache = None

    def on_start_sampling(self, iteration_idx: int, recursive: bool = True) -> Dict[str, Any]:
        self._cache = self._get_embeddings()
        return {}

    def on_end_sampling(
        self,
        iteration_idx: int,
        trajectories_container: TrajectoriesContainer,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        self._cache = None
        return {}

    @abc.abstractmethod
    def _get_embeddings(self) -> Tensor:
        pass

    def set_device(self, device: str, recursive: bool = True):
        if self._cache is not None:
            self._cache = self._cache.to(device)
        super().set_device(device, recursive=recursive)


@gin.configurable()
class ReactionsOneHotEmbedding(ActionEmbeddingBase):
    def __init__(self, data_factory: ReactionDataFactory, hidden_dim: int = 64):
        super().__init__(data_factory, hidden_dim)
        self.weights = nn.Parameter(
            torch.empty(len(data_factory.get_anchored_reactions()) + 1, hidden_dim),
            requires_grad=True,
        )
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def _get_embeddings(self) -> Tensor:
        return self.weights


@gin.configurable()
class FragmentOneHotEmbedding(ActionEmbeddingBase):
    def __init__(
        self,
        data_factory: ReactionDataFactory,
        hidden_dim: int = 64,
        dynamic_library: DynamicLibrary | None = None,
    ):
        super().__init__(data_factory, hidden_dim)
        self.initial_fragments = len(data_factory.get_fragments())
        self.current_fragments = self.initial_fragments

        max_n_fragments = self.initial_fragments
        if dynamic_library is not None:
            max_n_fragments += dynamic_library.max_num_additional_fragments
        self.weights = nn.Parameter(torch.empty(max_n_fragments, hidden_dim), requires_grad=True)
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def _get_embeddings(self) -> Tensor:
        return self.weights[: self.current_fragments]

    def on_update_fragments_library(
        self,
        iteration_idx: int,
        fragments: List[Molecule],
        costs: List[float],
        recursive: bool = True,
    ) -> Dict[str, Any]:
        self.clear_cache()
        self.current_fragments = self.initial_fragments + len(fragments)
        return {}


@gin.configurable()
class FragmentFingerprintEmbedding(ActionEmbeddingBase):
    def __init__(
        self,
        data_factory: ReactionDataFactory,
        fingerprint_list: List[str],
        hidden_dim: int = 64,
        one_hot_weight: float = 0.5,
        linear_embedding: bool = True,
        dynamic_library: DynamicLibrary | None = None,
    ):
        super().__init__(data_factory, hidden_dim)
        self.n_initial_fragments = len(data_factory.get_fragments())
        self.fingerprint_list = fingerprint_list
        self.one_hot_weight = one_hot_weight

        self.one_hot_embeddings = FragmentOneHotEmbedding(
            data_factory, hidden_dim, dynamic_library=dynamic_library
        )
        self.all_fingerprints = self._get_fingerprints(data_factory.get_fragments())
        if linear_embedding:
            self.fp_embedding = nn.Linear(self.all_fingerprints.shape[-1], hidden_dim)
        else:
            self.fp_embedding = nn.Sequential(
                nn.Linear(self.all_fingerprints.shape[-1], hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def on_update_fragments_library(
        self,
        iteration_idx: int,
        fragments: List[Molecule],
        costs: List[float],
        recursive: bool = True,
    ) -> Dict[str, Any]:
        self.clear_cache()
        self.one_hot_embeddings.on_update_fragments_library(
            iteration_idx, fragments, costs, recursive=recursive
        )
        n_new_fragments = self.n_initial_fragments + len(fragments) - len(self.all_fingerprints)
        fragments_to_add = fragments[-n_new_fragments:]
        if len(fragments_to_add) > 0:
            new_fingerprints = self._get_fingerprints(fragments_to_add).to(self.device)
            self.all_fingerprints = torch.cat([self.all_fingerprints, new_fingerprints], dim=0)
        return {}

    def _get_fingerprints(self, fragments: List[Molecule]) -> Tensor:
        fps_list = []
        for molecule in fragments:
            mol = molecule.rdkit_mol
            for fp_type in self.fingerprint_list:
                fps = []
                if fp_type == "maccs":
                    fp = MACCSkeys.GenMACCSKeys(mol)
                    array = np.zeros((0,), dtype=np.float32)
                    DataStructs.ConvertToNumpyArray(fp, array)
                    fps.append(array)
                elif fp_type == "ecfp":
                    fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=20480)
                    array = np.zeros((0,), dtype=np.float32)
                    DataStructs.ConvertToNumpyArray(fp, array)
                    fps.append(array)
            fps = np.concatenate(fps, axis=0)
            fps_list.append(fps)
        fps_numpy = np.stack(fps_list, axis=0)
        return torch.tensor(fps_numpy).float()

    def _get_embeddings(self) -> Tensor:
        fingerprints = self.fp_embedding(self.all_fingerprints)
        if self.one_hot_weight > 0:
            return (
                1 - self.one_hot_weight
            ) * fingerprints + self.one_hot_weight * self.one_hot_embeddings.get_embeddings()
        return fingerprints

    def set_device(self, device: str, recursive: bool = True):
        self.all_fingerprints = self.all_fingerprints.to(device)
        super().set_device(device, recursive=recursive)
