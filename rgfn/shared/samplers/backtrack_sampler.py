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
class BacktrackSampler(
    SamplerBase[TState, TActionSpace, TAction], Generic[TState, TActionSpace, TAction]
):
    """
    A sampler that samples trajectories from the environment using a policy, then backtracks K steps along the
    original trajectory to branch out and find new trajectories via "local search". These new trajectories are
    then added to the replay buffer and used to train the model. Each backtracked trajectory is treated as if
    it were independently sampled from the forward policy to train the model
    """

    def __init__(
        self,
        policy: PolicyBase[TState, TActionSpace, TAction],
        env: EnvBase[TState, TActionSpace, TAction],
        reward: Reward[TState] | None,
        n_revisions: int,
        n_bcktrk_steps: int,
        proxy_thresh: Optional[float] = None,
    ):
        super().__init__(policy, env, reward)
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
        n_independent_samples = n_trajectories
        source_states = self.env.sample_source_states(n_independent_samples)

        trajectories = self.sample_trajectories_from_sources(source_states)
        all_trajectories = [trajectories]

        last_states = trajectories.get_last_states_flat()
        terminal_mask = self.env.get_terminal_mask(last_states)
        trajectories = trajectories.masked_select(terminal_mask)

        if self.proxy_thresh is not None:
            proxy_rewards = trajectories.get_reward_outputs().proxy
            proxy_mask = (proxy_rewards > self.proxy_thresh).detach().cpu().tolist()
            trajectories = trajectories.masked_select(proxy_mask)

        if len(trajectories) == 0:
            final_trajectories = Trajectories.from_trajectories(all_trajectories)
            final_states = final_trajectories.get_last_states_flat()
            number_unique_final_states = len(set(final_states))

            print(f"Rewards too low, sampled {n_trajectories} regular trajectories.")
            print(f"Number of final trajectories: {len(final_trajectories)}")
            print(f"Number of unique final states: {number_unique_final_states}")
            return final_trajectories

        update_success_rates = []
        original_rewards = trajectories.get_reward_outputs().proxy

        # Backtrack n_bcktrk_steps along the current trajectories
        trajectory_lengths = trajectories.trajectory_lengths()
        bcktrk_indices = [
            np.clip(length - (self.n_bcktrk_steps + 1), 0, length - 1)
            for length in trajectory_lengths
        ]
        _ = trajectories.backtrack_to_state_idx(bcktrk_indices)

        for _ in range(self.n_revisions):
            # Use the forward policy to sample forward n_bcktrk_steps from partial states
            partial_states = trajectories.get_last_states_flat()
            proposal_trajectories = self.sample_trajectories_from_sources(partial_states)
            assert len(proposal_trajectories) == len(trajectories)

            proposal_rewards = proposal_trajectories.get_reward_outputs().proxy
            should_update = proposal_rewards > original_rewards
            original_rewards[should_update] = proposal_rewards[should_update]
            update_success_rates.append(should_update.float().mean().cpu().item())

            all_trajectories.append(proposal_trajectories)

        final_trajectories = Trajectories.from_trajectories(all_trajectories)
        final_states = final_trajectories.get_last_states_flat()
        number_unique_final_states = len(set(final_states))

        print(f"Update success rate: {np.mean(update_success_rates):.2f}")
        print(f"Number of final trajectories: {len(final_trajectories)}")
        print(f"Number of unique final states: {number_unique_final_states}")

        return final_trajectories

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
