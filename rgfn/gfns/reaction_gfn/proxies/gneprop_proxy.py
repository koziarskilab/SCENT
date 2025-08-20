import abc
import sys
from typing import List

import gin
import numpy as np
import numpy.typing as npt
import torch
from botorch.models.ensemble import EnsembleModel
from psalm.api.acquisition import AcquisitionFn
from tqdm import tqdm
from wurlitzer import pipes

from rgfn import ROOT_DIR
from rgfn.api.type_variables import TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.shared.proxies.bayesian_proxy import BayesianModel, BayesianProxy
from rgfn.shared.proxies.cached_proxy import CachedProxyBase

GNEPROP_PATH = ROOT_DIR / "external" / "gneprop"


@gin.configurable()
class GNEpropBayesianModel(EnsembleModel, BayesianModel):
    """Bayesian wrapper for GNEprop model."""

    def __init__(
        self,
        checkpoint_path: str,
        batch_size: int = 128,
        mc_iteration: int = 25,
        dropout_p: float = 0.2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        sys.path.append(str(GNEPROP_PATH))
        from gneprop.rewards import load_dropout_model

        self.mc_iteration = mc_iteration
        self.dropout_p = dropout_p
        self._num_outputs = 1
        self.batch_size = batch_size
        self.model = load_dropout_model(
            checkpoint_path, p=self.dropout_p, mc_iteration=self.mc_iteration
        )
        self.device = self.model.device

    def _compute_X_from_states(self, states: List[ReactionState]) -> npt.NDArray:
        smiles = [state.molecule.smiles for state in states]
        return np.array(smiles, dtype=object).reshape(-1, 1, 1)

    def forward(self, X: np.ndarray) -> torch.Tensor:
        from gneprop.data import MolDatasetOD
        from gneprop.gneprop_pyg import convert_to_dataloader

        bs, q, _ = X.shape
        smiles = X.flatten().tolist()

        dataset = MolDatasetOD(smiles)
        dataloader = convert_to_dataloader(dataset, batch_size=self.batch_size, num_workers=0)

        with pipes():
            preds = []
            for batch in tqdm(dataloader):
                self.model.dropout.train()
                batch = batch.to(device=self.device)
                batch.edge_attr = batch.edge_attr.to(self.model.dtype)
                with torch.no_grad():
                    pred = [
                        self.model.dropout(self.model.model(batch).squeeze(-1)).unsqueeze(
                            0
                        )  # (1, mbs)
                        for _ in range(self.mc_iteration)
                    ]
                pred = torch.cat(pred, dim=0)  # (mc_iteration, mbs)
                preds.append(pred)

        preds = torch.cat(preds, dim=1)  # (mc_iteration, bs)
        preds = preds.T.reshape(bs, self.mc_iteration, q, self._num_outputs)
        return preds  # (bs, mc_iteration, q, num_outputs)


class GNEpropProxy(BayesianProxy, CachedProxyBase[ReactionState], abc.ABC):
    def __init__(
        self,
        model: GNEpropBayesianModel,
        acquisition_factory: AcquisitionFn,
        best_f: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            acquisition_factory=acquisition_factory,
            best_f=best_f,
            *args,
            **kwargs,
        )
        self.device = model.device
        self.cache = {ReactionStateEarlyTerminal(None): 0.0}

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    @torch.no_grad()
    def _compute_gneprop_output(self, states: List[TState]) -> List[float]:
        return self._compute_acquisition_values(states)
