import math
import random
from copy import deepcopy
from enum import Enum
from itertools import compress
from typing import Callable, Dict, Generic, Iterator, List, Tuple

import gin
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import DataStructs
from rdkit.Chem import AllChem
from torch.distributions import Categorical
from tqdm import tqdm

from rgfn.api.env_base import EnvBase
from rgfn.api.policy_base import PolicyBase
from rgfn.api.reward import Reward
from rgfn.api.sampler_base import SamplerBase
from rgfn.api.trajectories import Trajectories
from rgfn.api.type_variables import TAction, TState
from rgfn.shared.policies.few_phase_policy import FewPhasePolicyBase
from rgfn.shared.policies.uniform_policy import TIndexedActionSpace, UniformPolicy


@gin.constants_from_enum
class SelectionStrategy(Enum):
    RANDOM = "random"
    BOLTZMANN = "boltzmann"
    TOPK = "topk"
    TOPP = "topp"
    # -----------
    BEAM = "beam"
    MCTS = "mcts"
    GA = "genetic_algorithm"
    ENUM = "enumerate"


@gin.constants_from_enum
class BacktrackStrategy(Enum):
    SIMPLE = "simple"
    LSD = "lsd"
    ASMS = "asms"


class BeamSearchCandidate(Generic[TState]):
    def __init__(self, state: TState, initial_score: float = 0.0):
        self.states: List[TState] = [state]
        self.score = initial_score
        self.normalized_score = initial_score

    def add_state(self, state: TState, score: float):
        self.states.append(state)
        self.score += score
        self.normalized_score = self.score / len(self.states)

    def __repr__(self):
        return f"BeamSearchCandidate(states={self.states}, score={self.score})"

    def __len__(self):
        return len(self.states)

    def __lt__(self, other: "BeamSearchCandidate"):
        return self.normalized_score < other.normalized_score

    def __le__(self, other: "BeamSearchCandidate"):
        return self.normalized_score <= other.normalized_score

    def __gt__(self, other: "BeamSearchCandidate"):
        return self.normalized_score > other.normalized_score

    def __ge__(self, other: "BeamSearchCandidate"):
        return self.normalized_score >= other.normalized_score


def mean_tanimoto_similarity(states: List[TState]):
    from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateTerminal

    terminal_states = [s for s in states if isinstance(s, ReactionStateTerminal)]
    mols = [s.molecule.rdkit_mol for s in terminal_states]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]

    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            similarities.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    if len(similarities) == 0:
        return 1.0

    return sum(similarities) / len(similarities)


@gin.configurable()
class ParallelSampler(
    SamplerBase[TState, TIndexedActionSpace, TAction], Generic[TState, TIndexedActionSpace, TAction]
):
    """
    A sampler that samples trajectories from the environment using a policy. Then, it samples
    a mixture around each end product of the original trajectories to compute a batched reward
    for the original trajectory. The final state consists only of the original trajectory, but
    the reward is computed using the sampled batch around it, potentially making it stochastic.
    """

    def __init__(
        self,
        policy: PolicyBase[TState, TIndexedActionSpace, TAction],
        env: EnvBase[TState, TIndexedActionSpace, TAction],
        mixture_reward: Reward[TState],
        max_mixture_size: int = 12,
        n_backtrack_steps: int = 2,
        backtrack_strategy=BacktrackStrategy.LSD,
        selection_strategy=SelectionStrategy.ENUM,
        temperature: float = 1.0,
        top_p: float = 0.7,
        top_k: float = 0.3,
    ):
        super().__init__(policy, env, None)
        self.mixture_reward = mixture_reward
        self.max_mixture_size = max_mixture_size
        self.n_backtrack_steps = n_backtrack_steps
        self.backtrack_strategy = backtrack_strategy
        self.strategy = selection_strategy
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self.forward_from_source_to_mixture: Dict[
            SelectionStrategy, Callable[..., List[TState]]
        ] = {
            SelectionStrategy.RANDOM: self.randomly_sample_from_sources,
            SelectionStrategy.BOLTZMANN: self.boltzmann_sample_from_sources,
            SelectionStrategy.BEAM: self.beam_search_from_source,
            SelectionStrategy.TOPK: self.topk_sample_from_sources,
            SelectionStrategy.TOPP: self.topp_sample_from_sources,
            SelectionStrategy.ENUM: self.enumerate_from_source,
        }

    def sample_trajectories(
        self, n_trajectories: int
    ) -> Trajectories[TState, TIndexedActionSpace, TAction]:
        """
        Sample n_trajectories from the environment using the policy.
        Args:
            n_trajectories: the number of trajectories to sample.
        Returns:
            the sampled trajectories.
        """
        source_states = self.env.sample_source_states(n_trajectories)
        return self.sample_trajectories_from_sources(source_states)

    @torch.no_grad()
    def sample_trajectories_from_sources(
        self, source_states: List[TState]
    ) -> Trajectories[TState, TIndexedActionSpace, TAction]:
        trajectories = SamplerBase.sample_trajectories_from_sources(self, source_states)
        backtracked_states, _ = self.backtrack_trajectories(trajectories)
        mixture_states = [
            self.forward_from_source_to_mixture[self.strategy](state)
            for state in tqdm(backtracked_states)
        ]
        print(f"Maximum final mixture size: {np.nanmax([len(ms) for ms in mixture_states])}")
        print(f"Average final mixture size: {np.mean([len(ms) for ms in mixture_states])}")
        reward_outputs = self.mixture_reward.compute_reward_output(mixture_states)
        trajectories.set_reward_outputs(reward_outputs)
        return trajectories

    def backtrack_trajectories(
        self,
        trajectories: Trajectories[TState, TIndexedActionSpace, TAction],
    ) -> Tuple[List[TState], Trajectories[TState, TIndexedActionSpace, TAction]]:
        from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateA

        if self.backtrack_strategy == BacktrackStrategy.SIMPLE:
            backtrack_idx = [
                max(len(x) - self.n_backtrack_steps, 0) for x in trajectories._states_list
            ]
            backtracked_trajectories = trajectories.backtrack_to_state_idx(backtrack_idx)

        elif self.backtrack_strategy == BacktrackStrategy.LSD:
            branching_state_idxs = [
                [i + 1 for i, s in enumerate(states) if isinstance(s, ReactionStateA)]
                for states in trajectories._states_list
            ]
            backtrack_idx = [x[-2] if len(x) >= 2 else x[-1] for x in branching_state_idxs]
            backtracked_trajectories = trajectories.backtrack_to_state_idx(backtrack_idx)

        elif self.backtrack_strategy == BacktrackStrategy.ASMS:
            raise NotImplementedError("ASMS backtracking strategy is not implemented yet.")

        else:
            raise ValueError(f"Unknown backtracking strategy: {self.backtrack_strategy}")

        branching_states = trajectories.get_last_states_flat()
        return branching_states, backtracked_trajectories

    def enumerate_from_source(self, source_state: TState) -> List[TState]:
        """Randomly enumerates possible states from the source state, up to n terminal states"""
        non_terminal_candidates = [BeamSearchCandidate(state=source_state)]
        terminal_candidates: List[BeamSearchCandidate] = []

        while True:
            current_states = [c.states[-1] for c in non_terminal_candidates]
            terminal_mask = self.env.get_terminal_mask(current_states)
            terminal_candidates.extend(compress(non_terminal_candidates, terminal_mask))

            non_terminal_mask = [not is_terminal for is_terminal in terminal_mask]
            non_terminal_candidates = list(compress(non_terminal_candidates, non_terminal_mask))
            non_terminal_candidates = non_terminal_candidates[: self.max_mixture_size]

            non_terminal_states = [c.states[-1] for c in non_terminal_candidates]
            if len(non_terminal_states) == 0 or len(terminal_candidates) >= self.max_mixture_size:
                break

            forward_action_spaces = self.env.get_forward_action_spaces(non_terminal_states)
            new_candidate_states: List[BeamSearchCandidate] = []

            for i in range(len(non_terminal_states)):
                possible_actions_indices = forward_action_spaces[i].get_possible_actions_indices()
                random.shuffle(possible_actions_indices)
                width_per_candidate = math.ceil(self.max_mixture_size / len(non_terminal_states))
                top_actions = [
                    forward_action_spaces[i].get_action_at_idx(idx)
                    for idx in possible_actions_indices[:width_per_candidate]
                ]
                source_states = [non_terminal_states[i]] * len(top_actions)
                new_states = self.env.apply_forward_actions(source_states, top_actions)

                for new_state in new_states:
                    new_candidate = deepcopy(non_terminal_candidates[i])
                    new_candidate.add_state(new_state, 1)
                    new_candidate_states.append(new_candidate)

            random.shuffle(new_candidate_states)
            non_terminal_candidates = new_candidate_states

        terminal_candidates = terminal_candidates[: self.max_mixture_size]
        return [c.states[-1] for c in terminal_candidates]

    def randomly_sample_from_sources(self, source_state: TState) -> List[TState]:
        """Randomly samples up to n terminal states from the source state"""
        source_states = [source_state] * self.max_mixture_size
        trajectories = self._base_sample_trajectories_from_sources(source_states, UniformPolicy())
        final_states = list(set(trajectories.get_last_states_flat()))
        return final_states

    def boltzmann_sample_from_sources(self, source_state: TState) -> List[TState]:
        """Uses boltzman sampling to sample up to n terminal states from a source state"""
        policy = self._get_forward_policy()
        source_states = [source_state] * self.max_mixture_size
        trajectories = self._base_sample_trajectories_from_sources(source_states, policy)
        final_states = list(set(trajectories.get_last_states_flat()))
        return final_states

    def topk_sample_from_sources(self, source_state: TState) -> List[TState]:
        """Uses top-k sampling to sample up to n terminal states from a source state"""
        policy = self._get_forward_policy()
        source_states = [source_state] * self.max_mixture_size
        trajectories = self._base_sample_trajectories_from_sources(source_states, policy)
        final_states = list(set(trajectories.get_last_states_flat()))
        return final_states

    def topp_sample_from_sources(self, source_state: TState) -> List[TState]:
        """Uses top-p sampling to sample up to n terminal states from a source state"""
        policy = self._get_forward_policy()
        source_states = [source_state] * self.max_mixture_size
        trajectories = self._base_sample_trajectories_from_sources(source_states, policy)
        final_states = list(set(trajectories.get_last_states_flat()))
        return final_states

    def beam_search_from_source(self, source_state: TState) -> List[TState]:
        """Uses beam search to sample up to n terminal states from a source state."""
        policy = self._get_forward_policy()
        assert isinstance(policy, FewPhasePolicyBase)
        non_terminal_candidates: List[BeamSearchCandidate] = [
            BeamSearchCandidate(state=source_state)
        ]
        terminal_candidates: List[BeamSearchCandidate] = []

        while True:
            current_states = [c.states[-1] for c in non_terminal_candidates]
            terminal_mask = self.env.get_terminal_mask(current_states)
            terminal_candidates.extend(compress(non_terminal_candidates, terminal_mask))

            non_terminal_mask = [not is_terminal for is_terminal in terminal_mask]
            non_terminal_candidates = list(compress(non_terminal_candidates, non_terminal_mask))
            non_terminal_candidates = non_terminal_candidates[: self.max_mixture_size]

            non_terminal_states = [c.states[-1] for c in non_terminal_candidates]
            if len(non_terminal_states) == 0 or len(terminal_candidates) >= self.max_mixture_size:
                break

            forward_action_spaces = self.env.get_forward_action_spaces(non_terminal_states)
            list_of_logits = policy._get_action_logits(non_terminal_states, forward_action_spaces)

            assert len(non_terminal_states) == len(list_of_logits)
            assert len(forward_action_spaces) == len(list_of_logits)

            new_candidate_states: List[BeamSearchCandidate] = []
            for i in range(len(non_terminal_states)):
                probs = F.softmax(list_of_logits[i], dim=-1)
                top_probs, top_ids = probs.topk(min(self.max_mixture_size, len(probs)), dim=-1)
                top_actions = [
                    forward_action_spaces[i].get_action_at_idx(int(idx.item()))
                    for idx, prob in zip(top_ids, top_probs)
                    if prob > 0.0
                ]
                log_top_probs = torch.log(top_probs[top_probs > 0.0]).detach().cpu().numpy()
                source_states = [non_terminal_states[i]] * len(top_actions)
                new_states = self.env.apply_forward_actions(source_states, top_actions)

                for new_state, log_prob in zip(new_states, log_top_probs):
                    new_candidate = deepcopy(non_terminal_candidates[i])
                    new_candidate.add_state(new_state, log_prob)
                    new_candidate_states.append(new_candidate)

            # Sort the new candidate states by their normalized score
            new_candidate_states.sort(reverse=True)
            non_terminal_candidates = new_candidate_states

        # Sort the terminal candidates by their normalized score
        terminal_candidates.sort(reverse=True)
        terminal_candidates = terminal_candidates[: self.max_mixture_size]
        return [c.states[-1] for c in terminal_candidates]

    def get_trajectories_iterator(
        self, n_total_trajectories: int, batch_size: int
    ) -> Iterator[Trajectories[TState, TIndexedActionSpace, TAction]]:
        """
        Get an iterator that samples trajectories from the environment. It can be used to sampled trajectories in
            batched manner.
        Args:
            n_total_trajectories: total number of trajectories to sample. If set to -1, the sampler should iterate over
                all source states (used in `SequentialSampler`).
            batch_size: the size of the batch. If -1, the batch size is equal to the number of n_total_trajectories.

        Returns:
            an iterator that samples trajectories.
        """
        batch_size = n_total_trajectories if batch_size == -1 else batch_size
        batches_sizes = [batch_size] * (n_total_trajectories // batch_size)
        if n_total_trajectories % batch_size:
            batches_sizes.append(n_total_trajectories % batch_size)
        for batch_size in batches_sizes:
            yield self.sample_trajectories(batch_size)

    def _get_forward_policy(self):
        from rgfn.gfns.reaction_gfn.policies import (
            ReactionForwardPolicyExploitationPenalty,
            ReactionForwardPolicyWithRND,
        )
        from rgfn.shared.policies.exploratory_policy import ExploratoryPolicy

        policy = self.policy
        if isinstance(self.policy, ExploratoryPolicy):
            policy = self.policy.first_policy
        if isinstance(self.policy, ReactionForwardPolicyExploitationPenalty):
            policy = self.policy.reaction_forward_policy
        if isinstance(self.policy, ReactionForwardPolicyWithRND):
            policy = self.policy.reaction_forward_policy

        setattr(policy, "_sample_actions_from_logits", self._sample_actions_from_logits)
        return policy

    def _sample_actions_from_logits(
        self, logits: torch.Tensor, action_spaces: List[TIndexedActionSpace]
    ) -> List[TAction]:
        """
        A helper function to sample actions from the log probabilities.

        Args:
            logits: logits of the shape (N, max_num_actions)
            action_spaces: the list of action spaces of the length N.

        Returns:
            the list of sampled actions.
        """
        match self.strategy:
            case SelectionStrategy.RANDOM:
                action_indices = [
                    random.choice(action_space.get_possible_actions_indices())
                    for action_space in action_spaces
                ]

            case SelectionStrategy.TOPK:
                k = max(1, math.ceil(self.top_k * logits.shape[-1]))
                top_k_indices = torch.topk(logits, k=k, dim=1).indices
                top_k_mask = torch.zeros_like(logits, dtype=torch.bool)
                top_k_mask = top_k_mask.scatter(1, top_k_indices, True)
                logits[~top_k_mask] = float("-inf")
                probs = torch.softmax(logits / self.temperature, dim=1)
                action_indices = Categorical(probs=probs).sample()

            case SelectionStrategy.TOPP:
                probs = torch.softmax(logits / self.temperature, dim=1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=1)
                mask = cumulative_probs <= self.top_p
                mask[:, 0] = True  # Ensure at least one action is selected
                masked_probs = sorted_probs * mask.float()
                masked_probs /= masked_probs.sum(dim=1, keepdim=True)  # Normalize
                action_indices = Categorical(probs=masked_probs).sample()
                action_indices = sorted_indices.gather(1, action_indices.unsqueeze(1)).squeeze(1)

            case SelectionStrategy.BOLTZMANN | _:
                probs = torch.softmax(logits / self.temperature, dim=1)
                action_indices = Categorical(probs=probs).sample()

        return [
            action_space.get_action_at_idx(idx.item())
            for action_space, idx in zip(action_spaces, action_indices)
        ]

    def _base_sample_trajectories_from_sources(self, source_states, policy: PolicyBase):
        original_policy = self.policy
        self.policy = policy
        trajectories = SamplerBase.sample_trajectories_from_sources(self, source_states)
        self.policy = original_policy
        return trajectories
