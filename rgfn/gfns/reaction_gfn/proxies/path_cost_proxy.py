from typing import Any, Dict, List, Tuple

import gin
import numpy as np

from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.api.trajectories import Trajectories, TrajectoriesContainer
from rgfn.gfns.reaction_gfn.api.data_structures import AnchoredReaction, Cache, Molecule
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionAction0,
    ReactionAction0Invalid,
    ReactionActionC,
    ReactionState0Invalid,
)
from rgfn.gfns.reaction_gfn.api.reaction_data_factory import ReactionDataFactory


@gin.configurable()
class PathCostProxy(TrainingHooksMixin):
    def __init__(
        self,
        data_factory: ReactionDataFactory,
        yield_value: float | None = None,
    ):
        self.data_factory = data_factory
        self.fragment_to_cost = data_factory.get_fragment_to_cost()
        if not self.fragment_to_cost:
            print("No costs for fragments provided. Setting them to 0")
            self.fragment_to_cost = {fragment: 0.0 for fragment in data_factory.get_fragments()}
        self.fragment_smiles_to_cost = {
            fragment.smiles: cost for fragment, cost in self.fragment_to_cost.items()
        }
        self.reaction_to_yield = data_factory.get_reaction_to_yield()
        self.anchor_to_reaction = data_factory.get_anchor_to_reaction_map()
        assert len(self.reaction_to_yield) > 0 or yield_value is not None

        self.molecule_num_reaction_to_cost = Cache(max_size=1_000_000)
        self.negative_molecule_num_reaction = Cache(max_size=50_000)
        self.n_recent_updates = 0
        self.yield_value = yield_value
        print("Cost mean and variance", self.get_fragment_costs_mean_std())
        print("Cost max", self.get_fragment_costs_max())

    def get_fragment_costs_mean_std(self) -> Tuple[float, float]:
        costs = list(self.fragment_to_cost.values())
        return np.mean(costs), np.std(costs)

    def get_fragment_costs_max(self) -> float:
        return max(self.fragment_to_cost.values())

    def _compute_costs(self, trajectories: Trajectories) -> List[float]:
        path_costs = []
        for actions, states in zip(
            trajectories.get_all_actions_grouped(),
            trajectories.get_all_states_grouped(),
        ):
            current_cost = (
                self.get_action_cost(actions[0])
                if not isinstance(states[0], ReactionState0Invalid)
                else float("inf")
            )
            for action, state in zip(actions, states[1:]):
                if isinstance(action, ReactionAction0Invalid):
                    if state.num_reactions == 0:
                        raise ValueError(f"States with num_reactions == 0, {state}")
                    item = (state.molecule.smiles, state.num_reactions)
                    self.negative_molecule_num_reaction[item] = float("inf")
                    self.molecule_num_reaction_to_cost[item] = float("inf")
                elif isinstance(action, ReactionActionC):
                    if state.num_reactions == 0:
                        raise ValueError(f"States with num_reactions == 0, {state}")
                    fragment_cost = self.get_action_cost(action)
                    yield_value = self.compute_yield(action)
                    current_cost = (current_cost + fragment_cost) * yield_value**-1
                    item = (state.molecule.smiles, state.num_reactions)
                    previous_cost = self.molecule_num_reaction_to_cost[item] or float("inf")
                    cost = min(previous_cost, current_cost)
                    if previous_cost > current_cost:
                        self.molecule_num_reaction_to_cost[item] = current_cost
                        self.n_recent_updates += 1
                    if cost == float("inf"):
                        self.negative_molecule_num_reaction[item] = float("inf")
                    else:
                        self.negative_molecule_num_reaction.pop(item)

            path_costs.append(current_cost)
        return path_costs

    def assign_costs(self, trajectories_container: TrajectoriesContainer) -> Dict[str, Any]:
        if trajectories_container.forward_trajectories is not None:
            costs = self._compute_costs(trajectories_container.forward_trajectories)
            trajectories_container.forward_trajectories.set_costs(costs)
        if trajectories_container.replay_trajectories is not None:
            costs = self._compute_costs(trajectories_container.replay_trajectories)
            trajectories_container.replay_trajectories.set_costs(costs)
        if trajectories_container.backward_trajectories is not None:
            costs = self._compute_costs(trajectories_container.backward_trajectories)
            trajectories_container.backward_trajectories.set_costs(costs)
        return {}

    def compute_yield(self, action: ReactionActionC) -> float:
        if self.yield_value is not None:
            return self.yield_value
        reaction = self.anchor_to_reaction[action.input_reaction]
        return self.reaction_to_yield[reaction]

    def compute_yield_raw(
        self, input_smiles_list: List[str], output_smiles: str, reaction: str
    ) -> float:
        if self.yield_value is not None:
            return self.yield_value
        anchored_reaction = AnchoredReaction(reaction, 0, 0)
        reaction = self.anchor_to_reaction[anchored_reaction]
        return self.reaction_to_yield[reaction]

    def get_fragment_cost(self, fragment: Molecule | str) -> float:
        if isinstance(fragment, str):
            return self.fragment_smiles_to_cost[fragment]
        else:
            return self.fragment_to_cost[fragment]

    def get_action_cost(self, action: ReactionActionC | ReactionAction0) -> float:
        if isinstance(action, ReactionAction0):
            return self.fragment_to_cost[action.fragment]
        else:
            return sum(self.fragment_to_cost[fragment] for fragment in action.input_fragments)

    def on_update_fragments_library(
        self,
        iteration_idx: int,
        fragments: List[Molecule],
        costs: List[float],
        recursive: bool = True,
    ) -> Dict[str, Any]:
        for fragment, cost in zip(fragments, costs):
            self.fragment_to_cost[fragment] = cost
        return {}
