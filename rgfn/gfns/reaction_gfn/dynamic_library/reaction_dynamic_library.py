import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import gin
import numpy as np
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import BulkTanimotoSimilarity

from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.api.trajectories import TrajectoriesContainer
from rgfn.gfns.reaction_gfn.api.data_structures import Molecule
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState0,
    ReactionStateA,
    ReactionStateTerminal,
)
from rgfn.gfns.reaction_gfn.api.reaction_data_factory import ReactionDataFactory
from rgfn.gfns.reaction_gfn.proxies.path_cost_proxy import PathCostProxy


@gin.configurable()
class DynamicLibrary(TrainingHooksMixin):
    def __init__(
        self,
        data_factory: ReactionDataFactory,
        path_cost_proxy: PathCostProxy,
        max_num_reactions: int = 2,
        every_n_iterations: int = 500,
        num_additions: int = 3,
        n_new_fragments: int = 200,
        pattern_matched_threshold: int = 3,
        use_forward_only: bool = True,
        cost_threshold: float = float("inf"),
        criterion: str = "mean_reward",
        similarity_threshold: float = 1.0,
        similarity_to_all: bool = False,
    ):
        self.path_cost_proxy = path_cost_proxy
        self.criterion = criterion
        self.similarity_threshold = similarity_threshold
        self.initial_smiles_set = set(x.smiles for x in data_factory.get_fragments())
        self.smiles_to_mean_reward: Dict[str, float] = defaultdict(lambda: 0.0)
        self.smiles_to_count: Dict[str, int] = defaultdict(lambda: 0)
        self.smiles_to_min_num_reactions: Dict[str, int] = defaultdict(lambda: 100000000)
        self.smiles_to_fp: Dict[str, Any] = defaultdict(lambda: None)
        self.smiles_to_cost: Dict[str, float] = defaultdict(lambda: float("inf"))

        self.chosen_smiles: List[str] = []
        self.chosen_smiles_costs: List[float] = []
        self.max_num_reactions = max_num_reactions

        self.n_iterations_schedule = [every_n_iterations * i for i in range(1, num_additions + 1)]
        self.n_new_fragments = n_new_fragments
        self.max_num_additional_fragments = len(self.n_iterations_schedule) * n_new_fragments
        self.pattern_matched_threshold = pattern_matched_threshold
        self.all_patterns = set()
        for reaction in data_factory.get_reactions():
            for pattern in reaction.left_side_patterns:
                self.all_patterns.add(pattern)

        self.use_forward_only = use_forward_only
        self.cost_threshold = cost_threshold
        self.similarity_threshold = similarity_threshold
        if similarity_to_all:
            self.initial_fps_list = [self._get_fp(smiles) for smiles in self.initial_smiles_set]
        else:
            self.initial_fps_list = []
        self.chosen_smiles_fps = []

    def state_dict(self) -> Dict[str, Any]:
        return {
            "initial_smiles_set": list(self.initial_smiles_set),
            "smiles_to_mean_reward": self.smiles_to_mean_reward,
            "smiles_to_count": self.smiles_to_count,
            "smiles_to_min_num_reactions": self.smiles_to_min_num_reactions,
            "chosen_smiles": self.chosen_smiles,
            "chosen_smiles_costs": self.chosen_smiles_costs,
        }

    def on_end_sampling(
        self,
        iteration_idx: int,
        trajectories_container: TrajectoriesContainer,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        if iteration_idx > self.n_iterations_schedule[-1]:
            return {}
        if self.use_forward_only:
            trajectories = trajectories_container.forward_trajectories
        else:
            trajectories = trajectories_container.get_all_non_backward_trajectories()
        valid_trajectories_mask = [
            isinstance(state, ReactionState0) for state in trajectories.get_source_states_flat()
        ]
        trajectories = trajectories.masked_select(valid_trajectories_mask)
        reward_values = trajectories.get_reward_outputs().reward
        state_groups = trajectories.get_all_states_grouped()
        for state_list, reward in zip(state_groups, reward_values):
            reward = reward.item()
            for state in state_list:
                if (
                    isinstance(state, (ReactionStateA, ReactionStateTerminal))
                    and state.molecule.smiles not in self.initial_smiles_set
                ):
                    smiles = state.molecule.smiles
                    n = self.smiles_to_count[smiles]
                    current_reward = self.smiles_to_mean_reward[smiles]
                    new_reward = current_reward * (n / (n + 1)) + reward / (n + 1)
                    self.smiles_to_mean_reward[smiles] = new_reward
                    self.smiles_to_count[smiles] += 1
                    self.smiles_to_min_num_reactions[smiles] = min(
                        self.smiles_to_min_num_reactions[smiles], state.num_reactions
                    )
                    current_cost = self.path_cost_proxy.molecule_num_reaction_to_cost[
                        (smiles, state.num_reactions)
                    ]
                    self.smiles_to_cost[smiles] = min(self.smiles_to_cost[smiles], current_cost)
        return {}

    def is_ready(self, i: int) -> bool:
        return i in self.n_iterations_schedule

    def _is_molecule_useful(self, smiles: str) -> bool:
        matched_count = 0
        mol = Molecule(smiles)
        for pattern in self.all_patterns:
            if mol.rdkit_mol.HasSubstructMatch(pattern.rdkit_pattern):
                matched_count += 1
                if matched_count >= self.pattern_matched_threshold:
                    return True
        return False

    def _is_molecule_different(self, smiles: str) -> bool:
        if self.similarity_threshold == 1.0:
            return True
        fp = self._get_fp(smiles)
        similarities = BulkTanimotoSimilarity(fp, self.initial_fps_list + self.chosen_smiles_fps)
        return all(similarity < self.similarity_threshold for similarity in similarities)

    def _get_fp(self, smiles: str):
        rdkit_mol = Molecule(smiles).rdkit_mol
        return GetMorganFingerprintAsBitVect(rdkit_mol, radius=2, nBits=1024)

    def retrieve_all_additional_fragments(
        self,
    ) -> Tuple[List[Molecule], List[float], Dict[str, Any]]:
        if self.criterion == "mean_reward":
            sorted_smiles = [
                x[0]
                for x in sorted(
                    self.smiles_to_mean_reward.items(), key=lambda x: x[1], reverse=True
                )
            ]
        elif self.criterion == "count":
            sorted_smiles = [
                x[0] for x in sorted(self.smiles_to_count.items(), key=lambda x: x[1], reverse=True)
            ]
        elif self.criterion == "uniform":
            sorted_smiles = list(self.smiles_to_mean_reward.keys())
            random.shuffle(sorted_smiles)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

        chosen_smiles_set = set(self.chosen_smiles)

        new_smiles = []
        new_costs = []
        for smiles in sorted_smiles:
            if (
                smiles not in chosen_smiles_set
                and smiles not in self.initial_smiles_set
                and self.smiles_to_min_num_reactions[smiles] <= self.max_num_reactions
                and self._is_molecule_useful(smiles)
                and self.smiles_to_cost[smiles] <= self.cost_threshold
                and self._is_molecule_different(smiles)
            ):
                new_smiles.append(smiles)
                new_costs.append(self.smiles_to_cost[smiles])
                self.chosen_smiles_fps.append(self._get_fp(smiles))
                if len(new_smiles) == self.n_new_fragments:
                    break

        new_chosen_rewards = [self.smiles_to_mean_reward[smiles] for smiles in new_smiles]
        new_chosen_counts = [self.smiles_to_count[smiles] for smiles in new_smiles]
        new_chosen_min_num_reactions = [
            self.smiles_to_min_num_reactions[smiles] for smiles in new_smiles
        ]
        new_chosen_costs = [self.smiles_to_cost[smiles] for smiles in new_smiles]
        metrics = {
            "mean_reward": np.mean(new_chosen_rewards),
            "mean_visited_count": np.mean(new_chosen_counts),
            "mean_min_num_reactions": np.mean(new_chosen_min_num_reactions),
            "mean_cost": np.mean(new_chosen_costs),
            "num_added_fragments": len(new_smiles),
        }

        self.chosen_smiles.extend(new_smiles)
        self.chosen_smiles_costs.extend(new_costs)

        molecules = [
            Molecule(
                smiles,
                idx=len(self.initial_smiles_set) + i,
                num_reactions=self.smiles_to_min_num_reactions[smiles],
            )
            for i, smiles in enumerate(self.chosen_smiles)
        ]
        return molecules, self.chosen_smiles_costs, metrics
