from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Sequence

from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.api.trajectories import TrajectoriesContainer


class MetricsBase(ABC, TrainingHooksMixin):
    """
    The base class for metrics used in Trainer.
    """

    @abstractmethod
    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, Any]:
        ...

    def collect_files(self) -> List[Path | str]:
        return []


class MetricsList(MetricsBase):
    def __init__(self, metrics: Sequence[MetricsBase]):
        self.metrics = metrics

    @property
    def hook_objects(self) -> List["TrainingHooksMixin"]:
        """
        The property should return the list of underlying objects that will be used in the recursive hook calls.
        """
        return list(self.metrics)

    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, Any]:
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.compute_metrics(trajectories_container))
        return metrics

    def collect_files(self) -> List[Path | str]:
        file_paths = []
        for metric in self.metrics:
            file_paths.extend(metric.collect_files())
        return file_paths
