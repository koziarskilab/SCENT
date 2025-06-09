import abc
from typing import Generic

from rgfn.api.trajectories import Trajectories, TrajectoriesContainer
from rgfn.api.type_variables import TAction, TActionSpace, TState


class TrajectoryFilterBase(Generic[TState, TActionSpace, TAction], abc.ABC):
    @abc.abstractmethod
    def filter_trajectories(
        self, trajectories: Trajectories[TState, TActionSpace, TAction]
    ) -> Trajectories[TState, TActionSpace, TAction]:
        raise NotImplementedError

    def __call__(
        self,
        trajectories_or_container: Trajectories[TState, TActionSpace, TAction]
        | TrajectoriesContainer[TState, TActionSpace, TAction],
    ) -> (
        Trajectories[TState, TActionSpace, TAction]
        | TrajectoriesContainer[TState, TActionSpace, TAction]
    ):
        if isinstance(trajectories_or_container, TrajectoriesContainer):
            if trajectories_or_container.forward_trajectories is not None:
                trajectories_or_container.forward_trajectories = self.filter_trajectories(
                    trajectories_or_container.forward_trajectories
                )
            if trajectories_or_container.backward_trajectories is not None:
                trajectories_or_container.backward_trajectories = self.filter_trajectories(
                    trajectories_or_container.backward_trajectories
                )
            if trajectories_or_container.replay_trajectories is not None:
                trajectories_or_container.replay_trajectories = self.filter_trajectories(
                    trajectories_or_container.replay_trajectories
                )
        else:
            trajectories_or_container = self.filter_trajectories(trajectories_or_container)
        return trajectories_or_container


class IdentityTrajectoryFilter(TrajectoryFilterBase[TState, TActionSpace, TAction]):
    def filter_trajectories(
        self, trajectories: Trajectories[TState, TActionSpace, TAction]
    ) -> Trajectories[TState, TActionSpace, TAction]:
        return trajectories
