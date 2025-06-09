from typing import Callable, Dict, Iterator, List, Sequence, Type

import gin
import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Parameter

from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.api.type_variables import TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionAction,
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
from rgfn.gfns.reaction_gfn.policies.cost_biased_backward_policy import (
    CostGuidedBackwardPolicy,
)
from rgfn.gfns.reaction_gfn.policies.decomposability_guided_backward_policy import (
    DecomposabilityGuidedBackwardPolicy,
)
from rgfn.shared.policies.few_phase_policy import FewPhasePolicyBase, TSharedEmbeddings
from rgfn.shared.policies.uniform_policy import TIndexedActionSpace


@gin.configurable()
class JointlyGuidedBackwardPolicy(
    FewPhasePolicyBase[ReactionState, ReactionActionSpace, ReactionAction, None],
):
    def __init__(
        self,
        policies: List[DecomposabilityGuidedBackwardPolicy | CostGuidedBackwardPolicy],
    ):
        super().__init__()
        self.policies = policies

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
        return self.policies

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
        logits_list = [
            policy._forward_c(states, action_spaces, shared_embeddings) for policy in self.policies
        ]
        logprobs_list = [torch.nn.functional.log_softmax(logits, dim=-1) for logits in logits_list]
        logprobs = torch.stack(logprobs_list, dim=0)
        logprobs = logprobs.sum(dim=0)
        logprobs = logprobs - torch.logsumexp(logprobs, dim=-1, keepdim=True)
        return logprobs

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
        logits = torch.tensor(logits_list).float().to(self.device)
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def _sample_actions_from_logits(
        self, logits: Tensor, action_spaces: List[ReactionActionSpace]
    ) -> List[ReactionAction]:
        """
        A helper function to sample actions from the log probabilities.

        Args:
            logits: logits of the shape (N, max_num_actions). Those are actually logprobs
            action_spaces: the list of action spaces of the length N.

        Returns:
            the list of sampled actions.
        """
        probs = torch.exp(logits)
        action_indices = Categorical(probs=probs).sample()
        return [
            action_space.get_action_at_idx(idx.item())
            for action_space, idx in zip(action_spaces, action_indices)
        ]

    def _select_actions_log_probs(
        self,
        logits: Tensor,
        action_spaces: Sequence[ReactionActionSpace],
        actions: Sequence[ReactionAction],
    ) -> Tensor:
        """
        A helper function to select the log probabilities of the actions.

        Args:
            logits: logits of the shape (N, max_num_actions). Those are actually logprobs
            action_spaces: the list of action spaces of the length N.
            actions: the list of chosen actions of the length N.

        Returns:
            the log probabilities of the chosen actions of the shape (N,).
        """
        action_indices = [
            action_space.get_idx_of_action(action)  # type: ignore
            for action_space, action in zip(action_spaces, actions)
        ]
        max_num_actions = logits.shape[1]
        action_indices = [
            idx * max_num_actions + action_idx for idx, action_idx in enumerate(action_indices)
        ]
        action_tensor_indices = torch.tensor(action_indices).long().to(self.device)
        log_probs = torch.index_select(logits.view(-1), index=action_tensor_indices, dim=0)
        return log_probs

    def get_shared_embeddings(
        self, states: List[ReactionState], action_spaces: List[ReactionActionSpace]
    ) -> None:
        return None

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        yield from []

    def compute_states_log_flow(self, states: List[ReactionState]) -> Tensor:
        raise NotImplementedError()
