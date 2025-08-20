from typing import Generic, Iterator, Optional

import gin
import numpy as np

from rgfn.api.env_base import EnvBase
from rgfn.api.policy_base import PolicyBase
from rgfn.api.reward import Reward
from rgfn.api.sampler_base import SamplerBase
from rgfn.api.trajectories import Trajectories
from rgfn.api.type_variables import TAction, TActionSpace, TState


@gin.configurable()
class LocalSearchSampler(
    SamplerBase[TState, TActionSpace, TAction], Generic[TState, TActionSpace, TAction]
):
    """
    A sampler that samples trajectories from the environment using a policy, then performs local search
    to refine the trajectories according to Local Search GFlowNets (Kim et al., 2024). This creates a bunch
    of partial trajectories that are then used to train the model.
    """

    def __init__(
        self,
        fw_policy: PolicyBase[TState, TActionSpace, TAction],
        bw_policy: PolicyBase[TState, TActionSpace, TAction],
        env: EnvBase[TState, TActionSpace, TAction],
        reward: Reward[TState] | None,
        n_revisions: int,
        n_bcktrk_steps: int,
        proxy_thresh: Optional[float] = None,
    ):
        super().__init__(fw_policy, env, reward)
        self.fw_policy = fw_policy
        self.bw_policy = bw_policy
        self.fw_env = env
        self.bw_env = env.reversed()
        self.n_revisions = n_revisions
        self.n_bcktrk_steps = n_bcktrk_steps
        self.proxy_thresh = proxy_thresh
        assert n_revisions > 0, "n_revisions must be greater than 0"
        assert n_bcktrk_steps > 0, "n_bcktrk_steps must be greater than 0"

    def sample_trajectories(
        self, n_trajectories: int
    ) -> Trajectories[TState, TActionSpace, TAction]:
        """
        Sample n_trajectories from the environment using the policy.
        Args:
            n_trajectories: the number of trajectories to sample.
        Returns:
            the sampled trajectories.
        """
        assert (
            n_trajectories % (self.n_revisions + 1) == 0
        ), f"n_trajectories={n_trajectories} must be divisible by (n_revisions+1)={self.n_revisions+1}"

        n_independent_samples = n_trajectories // (self.n_revisions + 1)
        source_states = self.env.sample_source_states(n_independent_samples)

        trajectories = self.sample_trajectories_from_sources(source_states)
        all_trajectories = [trajectories]

        last_states = trajectories.get_last_states_flat()
        terminal_mask = self.env.get_terminal_mask(last_states)
        trajectories = trajectories.masked_select(terminal_mask)

        if self.proxy_thresh is not None:
            proxy_rewards = trajectories.get_reward_outputs().proxy
            proxy_mask = proxy_rewards > self.proxy_thresh
            trajectories = trajectories.masked_select(proxy_mask)

        if len(trajectories) == 0:
            n_remaining_trajectories = n_trajectories - len(all_trajectories[0])
            source_states = self.env.sample_source_states(n_remaining_trajectories)
            remaining_trajectories = self.sample_trajectories_from_sources(source_states)
            all_trajectories.append(remaining_trajectories)
            final_trajectories = Trajectories.from_trajectories(all_trajectories)

            print(f"Rewards too low, sampling {n_trajectories} regular trajectories.")
            print(f"Number of final trajectories: {len(final_trajectories)}")
            print(f"Number of unique final states: {number_unique_final_states}")

            return final_trajectories

        update_success_rates = []

        for _ in range(self.n_revisions):
            proposal_trajectories = self.backforth_sample(trajectories)
            assert len(proposal_trajectories) == len(trajectories)
            n_updates = 0

            proposal_rewards = proposal_trajectories.get_reward_outputs().proxy
            original_rewards = trajectories.get_reward_outputs().proxy

            proposal_last_states = proposal_trajectories.get_last_states_flat()
            proposal_terminal_mask = self.env.get_terminal_mask(proposal_last_states)

            for idx in range(len(proposal_trajectories)):
                if not proposal_terminal_mask[idx]:
                    continue
                if proposal_rewards[idx] > original_rewards[idx]:
                    trajectories[idx] = proposal_trajectories[idx]
                    n_updates += 1
                update_success_rates.append(n_updates / len(proposal_trajectories))

            all_trajectories.append(proposal_trajectories)

        final_trajectories = Trajectories.from_trajectories(all_trajectories)
        final_states = final_trajectories.get_last_states_flat()
        number_unique_final_states = len(set(final_states))

        print(f"Update success rate: {np.mean(update_success_rates):.2f}")
        print(f"Number of final trajectories: {len(final_trajectories)}")
        print(f"Number of unique final states: {number_unique_final_states}")

        return final_trajectories

    def backforth_sample(self, trajectories: Trajectories) -> Trajectories:
        """
        Backtracks n_bcktrk_steps using the backward policy, then samples forward using the forward policy.
        Args:
            trajectories: the trajectories to backtrack and sample forward.
        Returns:
            the refined trajectories.
        """
        terminal_states = trajectories.get_last_states_flat()

        # Use the backwards policy to backtrack n_bcktrk_steps
        self.policy = self.bw_policy
        self.env = self.bw_env
        bw_trajectories = self.sample_k_steps_from_sources(terminal_states, self.n_bcktrk_steps)

        # Use the forward policy to sample forward n_bcktrk_steps from partial states
        self.policy = self.fw_policy
        self.env = self.fw_env
        partial_states = bw_trajectories.get_source_states_flat()
        fw_trajectories = self.sample_trajectories_from_sources(partial_states)

        return fw_trajectories

    def get_trajectories_iterator(
        self, n_total_trajectories: int, batch_size: int
    ) -> Iterator[Trajectories[TState, TActionSpace, TAction]]:
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
