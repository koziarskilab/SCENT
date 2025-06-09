from typing import Any, Dict, List

from torch import nn

from rgfn.api.trajectories import TrajectoriesContainer
from rgfn.gfns.reaction_gfn.api.data_structures import Molecule


class TrainingHooksMixin:
    """
    A mixin class for training hooks. It provides simple and efficient way to add hooks to the child classes that
    are then used in the Trainer. The hooks are called recursively so that a hook from a unique object is called only
    once. The recursion is defined by the `hook_objects` property that returns the list of underlying objects that
    will be entered recursively.
    """

    @property
    def hook_objects(self) -> List["TrainingHooksMixin"]:
        """
        The property should return the list of underlying objects that will be used in the recursive hook calls.
        """
        return []

    def _gather_all_recursive_hook_objects_dict(self) -> Dict[int, "TrainingHooksMixin"]:
        """
        Returns a dictionary containing all the recursive hook objects. The keys are the unique ids of the hook objects
        and the values are the hook objects themselves.
        """
        hook_unique_object_dict = {}
        for hook_object in self.hook_objects:
            hook_unique_object_dict[id(hook_object)] = hook_object
            hook_unique_object_dict.update(hook_object._gather_all_recursive_hook_objects_dict())
        return hook_unique_object_dict

    def _gather_all_recursive_hook_objects(self) -> List["TrainingHooksMixin"]:
        """
        Returns a list containing all the recursive hook objects. The objects are unique.
        """
        return list(self._gather_all_recursive_hook_objects_dict().values())

    def set_device(self, device: str, recursive: bool = True):
        """
        Sets the device for the hook object and all the recursive hook objects.

        Args:
            device: The device to set.
            recursive: Whether to set the device on all the recursive hook objects.
        """
        if hasattr(self, "device"):
            self.device = device
        if isinstance(self, nn.Module):
            self.to(device)
        if recursive:
            for hook_object in self._gather_all_recursive_hook_objects():
                hook_object.set_device(device, recursive=False)

    def on_start_sampling(self, iteration_idx: int, recursive: bool = True) -> Dict[str, Any]:
        """
        Hook called at the start of the sampling phase of the training loop.

        Args:
            iteration_idx: The current iteration index.
            recursive: Whether to call the hook on all the recursive hook objects.

        Returns:
            A dictionary containing the metrics returned by the hook. The metrics are aggregated across all the hooks
            and logged by the Trainer. The dictionary may be empty.
        """
        update_outputs = {}
        if recursive:
            for hook_object in self._gather_all_recursive_hook_objects():
                update_outputs |= hook_object.on_start_sampling(iteration_idx, recursive=False)
        return update_outputs

    def on_end_sampling(
        self,
        iteration_idx: int,
        trajectories_container: TrajectoriesContainer,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        Hook called at the end of the sampling phase of the training loop.

        Args:
            iteration_idx: The current iteration index.
            trajectories_container: The trajectories collected during the sampling phase.
            recursive: Whether to call the hook on all the recursive hook objects.
        Returns:
            A dictionary containing the metrics returned by the hook. The metrics are aggregated across all the hooks
            and logged by the Trainer. The dictionary may be empty.
        """
        update_outputs = {}
        if recursive:
            for hook_object in self._gather_all_recursive_hook_objects():
                update_outputs |= hook_object.on_end_sampling(
                    iteration_idx, trajectories_container, recursive=False
                )
        return update_outputs

    def on_start_computing_objective(
        self,
        iteration_idx: int,
        trajectories_container: TrajectoriesContainer,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        Hook called at the start of the computing objective phase of the training loop.

        Args:
            iteration_idx: The current iteration index.
            trajectories_container: The trajectories collected during the sampling phase.
            recursive: Whether to call the hook on all the recursive hook objects.
        Returns:
            A dictionary containing the metrics returned by the hook. The metrics are aggregated across all the hooks
            and logged by the Trainer. The dictionary may be empty.
        """
        update_outputs = {}
        if recursive:
            for hook_object in self._gather_all_recursive_hook_objects():
                update_outputs |= hook_object.on_start_computing_objective(
                    iteration_idx, trajectories_container, recursive=False
                )
        return update_outputs

    def on_end_computing_objective(
        self,
        iteration_idx: int,
        trajectories_container: TrajectoriesContainer,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        Hook called at the end of the computing objective phase of the training loop.

        Args:
            iteration_idx: The current iteration index.
            trajectories_container: The trajectories collected during the sampling phase.
            recursive: Whether to call the hook on all the recursive hook objects.
        Returns:
            A dictionary containing the metrics returned by the hook. The metrics are aggregated across all the hooks
            and logged by the Trainer. The dictionary may be empty.
        """
        update_outputs = {}
        if recursive:
            for hook_object in self._gather_all_recursive_hook_objects():
                update_outputs |= hook_object.on_end_computing_objective(
                    iteration_idx, trajectories_container, recursive=False
                )
        return update_outputs

    def on_update_fragments_library(
        self,
        iteration_idx: int,
        fragments: List[Molecule],
        costs: List[float],
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        Hook called when the fragments library is updated.

        Args:
            fragments: The updated fragments library.

        Returns:
            A dictionary containing the metrics returned by the hook. The metrics are aggregated across all the hooks
            and logged by the Trainer. The dictionary may be empty.
        """
        update_outputs = {}
        if recursive:
            for hook_object in self._gather_all_recursive_hook_objects():
                update_outputs |= hook_object.on_update_fragments_library(
                    iteration_idx, fragments, costs, recursive=False
                )
        return update_outputs
