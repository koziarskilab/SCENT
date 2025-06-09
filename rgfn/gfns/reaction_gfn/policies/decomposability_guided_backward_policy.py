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
from rgfn.gfns.reaction_gfn.policies.guidance_models.decomposable_models import (
    DecomposableModelBase,
)
from rgfn.gfns.reaction_gfn.policies.utils import OrderedSet, to_dense_embeddings
from rgfn.gfns.reaction_gfn.proxies.path_cost_proxy import PathCostProxy
from rgfn.shared.policies.few_phase_policy import FewPhasePolicyBase, TSharedEmbeddings
from rgfn.shared.policies.uniform_policy import TIndexedActionSpace


@gin.configurable()
class DecomposabilityGuidedBackwardPolicy(
    FewPhasePolicyBase[ReactionState, ReactionActionSpace, ReactionAction, None],
):
    def __init__(
        self,
        path_cost_proxy: PathCostProxy,
        decomposable_prediction_model: DecomposableModelBase,
        temperature: float = 5.0,
        valid_every_n_iterations: int = 10,
    ):
        super().__init__()
        self.path_cost_proxy = path_cost_proxy
        self.decomposable_prediction_model = decomposable_prediction_model
        self.temperature = temperature
        self.valid_every_n_iterations = valid_every_n_iterations

        self._action_space_type_to_forward_fn = {
            ReactionActionSpace0: self._forward_deterministic,
            ReactionActionSpace0Invalid: self._forward_deterministic,
            ReactionActionSpaceA: self._forward_deterministic,
            ReactionActionSpaceB: self._forward_deterministic,
            ReactionActionSpace0orCBackward: self._forward_c,
            ReactionActionSpaceEarlyTerminate: self._forward_deterministic,
        }

    @property
    def hook_objects(self) -> List["TrainingHooksMixin"]:
        return [self.decomposable_prediction_model]

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
            predicted_is_decomposable = self.decomposable_prediction_model.predict_is_decomposable(
                list(all_molecule_num_reactions)
            )
            molecule_num_reactions_to_is_decomposable = {
                molecule_num_reactions: is_decomposable.item()
                for molecule_num_reactions, is_decomposable in zip(
                    all_molecule_num_reactions, predicted_is_decomposable
                )
            }
        else:
            molecule_num_reactions_to_is_decomposable = {}

        is_decomposable_flat = []
        for state, action_space in zip(states, action_spaces):
            for action in action_space.possible_actions:
                if isinstance(action, ReactionActionC) and state.num_reactions > 1:
                    item = (action.input_molecule, state.num_reactions - 1)
                    is_decomposable = molecule_num_reactions_to_is_decomposable[item]
                else:
                    is_decomposable = 1.0
                is_decomposable_flat.append(is_decomposable)

        is_decomposable_tensor = torch.tensor(is_decomposable_flat).float().to(self.device)
        logits = is_decomposable_tensor * self.temperature
        logits, _ = to_dense_embeddings(
            logits, [len(action_space) for action_space in action_spaces], fill_value=float("-inf")
        )
        return logits

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
        metrics = self.decomposable_prediction_model.train_one_epoch()

        if iteration_idx % self.valid_every_n_iterations == 0:
            with torch.no_grad():
                valid_metrics = self.decomposable_prediction_model.valid_one_epoch()
                valid_metrics = {f"valid_{key}": value for key, value in valid_metrics.items()}
                metrics.update(valid_metrics)
        return metrics
