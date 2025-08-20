from abc import ABC, abstractmethod
from typing import Hashable, List, TypeVar

import botorch.models.model
import gin
import torch
from botorch.posteriors import Posterior
from psalm.api.acquisition import AcquisitionFn

from rgfn.api.proxy_base import ProxyBase

THashableState = TypeVar("THashableState", bound=Hashable)


class BayesianModel(botorch.models.model.Model, ABC):
    """A Bayesian model that implements a posterior method"""

    device: str = "cpu"
    num_outputs: int = 1

    @abstractmethod
    def _compute_X_from_states(self, states: List[THashableState]) -> torch.Tensor:
        """Convert a list of states to a tensor of design points"""
        raise NotImplementedError("Subclasses must implement this method.")

    def predict(self, states: List[THashableState]) -> Posterior:
        """Forward pass to compute the posterior for a list of states."""
        return self.posterior(self._compute_X_from_states(states))


@gin.configurable()
class BayesianProxy(ProxyBase[THashableState], ABC):
    """A proxy that uses a Bayesian model and an acquisition function to compute the
    output for a list of states.
    """

    def __init__(
        self,
        model: BayesianModel,
        acquisition_factory: AcquisitionFn,
        best_f: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.acquisition_fn = acquisition_factory.get_acq_function(model=model, best_f=best_f)

    def _compute_acquisition_values(self, states: List[THashableState]) -> List[float]:
        choices = self.model._compute_X_from_states(states)  # (b, q, d)
        acq_values: torch.Tensor = self.acquisition_fn(choices)
        return acq_values.detach().cpu().tolist()
