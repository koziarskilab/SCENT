from typing import Any, Callable, Dict, Iterator, List, Type

import gin
import torch
from torch import Tensor
from torch.nn import Parameter

from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.api.trajectories import TrajectoriesContainer
from rgfn.api.type_variables import TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionAction,
    ReactionActionC,
    ReactionActionSpace,
    ReactionActionSpace0,
    ReactionActionSpace0Invalid,
    ReactionActionSpace0orCBackward,
    ReactionActionSpaceA,
    ReactionActionSpaceB,
    ReactionActionSpaceEarlyTerminate,
    ReactionState,
    ReactionStateC,
)
from rgfn.gfns.reaction_gfn.policies.guidance_models.cost_models import CostModelBase
from rgfn.gfns.reaction_gfn.policies.utils import OrderedSet, to_dense_embeddings
from rgfn.gfns.reaction_gfn.proxies.path_cost_proxy import PathCostProxy
from rgfn.shared.policies.few_phase_policy import FewPhasePolicyBase, TSharedEmbeddings
from rgfn.shared.policies.uniform_policy import TIndexedActionSpace


@gin.configurable()
class CostGuidedBackwardPolicy(
    FewPhasePolicyBase[ReactionState, ReactionActionSpace, ReactionAction, None],
):
    def __init__(
        self,
        path_cost_proxy: PathCostProxy,
        cost_prediction_model: CostModelBase | None,
        temperature: float = 5.0,
        valid_every_n_iterations: int = 10,
        normalize_costs: bool = True,
    ):
        super().__init__()
        self.path_cost_proxy = path_cost_proxy
        self.cost_prediction_model = cost_prediction_model
        self.temperature = temperature
        self.valid_every_n_iterations = valid_every_n_iterations
        self.inf_cost_value = 1e5

        self._action_space_type_to_forward_fn = {
            ReactionActionSpace0: self._forward_deterministic,
            ReactionActionSpace0Invalid: self._forward_deterministic,
            ReactionActionSpaceA: self._forward_deterministic,
            ReactionActionSpaceB: self._forward_deterministic,
            ReactionActionSpace0orCBackward: self._forward_c,
            ReactionActionSpaceEarlyTerminate: self._forward_deterministic,
        }
        if normalize_costs:
            mean_fragment_cost = path_cost_proxy.get_fragment_costs_mean_std()[1]
            if mean_fragment_cost == 0.0:
                print("WARNING: mean fragment cost is 0. Setting normalization temperature to 1.0")
                self.normalization_temperature = 1.0
            else:
                self.normalization_temperature = (
                    1 / path_cost_proxy.get_fragment_costs_mean_std()[1] if normalize_costs else 1.0
                )
        else:
            self.normalization_temperature = 1.0

    @property
    def hook_objects(self) -> List["TrainingHooksMixin"]:
        return [self.cost_prediction_model] if self.cost_prediction_model is not None else []

    @property
    def action_space_to_forward_fn(
        self,
    ) -> Dict[
        Type[TIndexedActionSpace],
        Callable[[List[TState], List[TIndexedActionSpace], TSharedEmbeddings], Tensor],
    ]:
        return self._action_space_type_to_forward_fn

    @torch.no_grad()
    def _forward_c(
        self,
        states: List[ReactionStateC],
        action_spaces: List[ReactionActionSpace0orCBackward],
        shared_embeddings: None,
    ) -> Tensor:
        all_molecule_num_reactions = OrderedSet()
        for state, action_space in zip(states, action_spaces):
            if isinstance(action_space, ReactionActionSpace0orCBackward):
                for action in action_space.possible_actions:
                    if isinstance(action, ReactionActionC) and state.num_reactions > 1:
                        all_molecule_num_reactions.add(
                            (action.input_molecule, state.num_reactions - 1)
                        )
        if len(all_molecule_num_reactions) > 0:
            if self.cost_prediction_model is not None:
                predicted_costs = self.cost_prediction_model.predict_costs(
                    list(all_molecule_num_reactions)
                )
            else:
                predicted_costs = torch.zeros(len(all_molecule_num_reactions), dtype=torch.float32)
            molecule_num_reactions_to_cost = {
                molecule_num_reactions: cost.item()
                for molecule_num_reactions, cost in zip(all_molecule_num_reactions, predicted_costs)
            }
        else:
            molecule_num_reactions_to_cost = {}

        total_costs_list = []
        for state, action_space in zip(states, action_spaces):
            total_costs = []
            for action in action_space.possible_actions:
                if isinstance(action, ReactionActionC) and state.num_reactions > 1:
                    item = (action.input_molecule, state.num_reactions - 1)
                    previous_molecule_cost = min(
                        molecule_num_reactions_to_cost[item], self.inf_cost_value
                    )
                else:  # it's a fragment
                    previous_fragment = (
                        action.input_molecule
                        if isinstance(action, ReactionActionC)
                        else action.fragment
                    )
                    previous_molecule_cost = self.path_cost_proxy.get_fragment_cost(
                        previous_fragment
                    )
                if isinstance(action, ReactionActionC):
                    cost = self.path_cost_proxy.get_action_cost(action)
                    yield_value = self.path_cost_proxy.compute_yield(action)
                    cost = (cost + previous_molecule_cost) * (yield_value**-1)
                else:
                    cost = previous_molecule_cost
                total_costs.append(cost)
            total_costs_list.append(total_costs)

        total_cost_list_flat = [cost for costs in total_costs_list for cost in costs]
        cost_tensor = torch.tensor(total_cost_list_flat).float().to(self.device)
        logits = -cost_tensor * self.temperature * self.normalization_temperature
        logits, _ = to_dense_embeddings(
            logits, [len(total_costs) for total_costs in total_costs_list], fill_value=float("-inf")
        )
        return logits

    def _forward_deterministic(
        self,
        states: List[ReactionState],
        action_spaces: List[ReactionActionSpace],
        shared_embeddings: None,
    ) -> Tensor:
        assert len(states) == len(action_spaces)
        max_action_idx = max(
            action_space.get_possible_actions_indices()[0] for action_space in action_spaces
        )
        logits_list = []
        for action_space in action_spaces:
            logits = [-float("inf")] * (max_action_idx + 1)
            logits[action_space.get_possible_actions_indices()[0]] = 0
            logits_list.append(logits)
        return torch.tensor(logits_list).float().to(self.device)

    def get_shared_embeddings(
        self, states: List[ReactionState], action_spaces: List[ReactionActionSpace]
    ) -> None:
        return None

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        yield from []

    def compute_states_log_flow(self, states: List[ReactionState]) -> Tensor:
        raise NotImplementedError()

    def on_start_computing_objective(
        self,
        iteration_idx: int,
        trajectories_container: TrajectoriesContainer,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        if self.cost_prediction_model is None:
            return {}

        metrics = self.cost_prediction_model.train_one_epoch()

        if iteration_idx % self.valid_every_n_iterations == 0:
            with torch.no_grad():
                valid_metrics = self.cost_prediction_model.valid_one_epoch()
                valid_metrics = {f"valid_{key}": value for key, value in valid_metrics.items()}
                metrics.update(valid_metrics)
        return metrics
