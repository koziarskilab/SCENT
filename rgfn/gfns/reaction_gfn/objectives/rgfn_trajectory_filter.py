import gin
import torch

from rgfn.api.trajectories import Trajectories
from rgfn.api.trajectory_filter_base import TrajectoryFilterBase
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionAction,
    ReactionActionSpace,
    ReactionActionSpace0orCBackward,
    ReactionState,
    ReactionState0,
)


@gin.configurable()
class RGFNTrajectoryFilter(
    TrajectoryFilterBase[ReactionState, ReactionActionSpace, ReactionAction]
):
    def filter_trajectories(
        self, trajectories: Trajectories[ReactionState, ReactionActionSpace, ReactionAction]
    ) -> Trajectories[ReactionState, ReactionActionSpace, ReactionAction]:
        source_states = trajectories.get_source_states_flat()
        valid_source_mask = torch.tensor(
            [isinstance(state, ReactionState0) for state in source_states], dtype=torch.bool
        )
        other_valid_mask = []
        for action_list, backward_action_spaces_list in zip(
            trajectories._actions_list, trajectories._backward_action_spaces_list
        ):
            valid = True
            for action, backward_action_space in zip(action_list, backward_action_spaces_list):
                if (
                    isinstance(backward_action_space, ReactionActionSpace0orCBackward)
                    and action not in backward_action_space.possible_actions
                ):
                    valid = False
                    break
            other_valid_mask.append(valid)

        valid_source_mask = valid_source_mask & torch.tensor(other_valid_mask, dtype=torch.bool)
        return trajectories.masked_select(valid_source_mask)
