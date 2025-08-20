from typing import List, cast

import gin
import numpy as np
import torch
from psalm.acquisition.functions import UpperConfidenceBound
from psalm.data.featurizers import MorganFeaturizer
from psalm.models.gp import ApproximateTanimotoGP

from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.shared.proxies.bayesian_proxy import BayesianModel, BayesianProxy
from rgfn.shared.proxies.cached_proxy import CachedProxyBase


class ClppBayesianModel(ApproximateTanimotoGP, BayesianModel):
    """A Bayesian model loaded from a checkpoint"""

    def _compute_X_from_states(self, states: List[ReactionState]) -> np.ndarray:
        smiles = [state.molecule.smiles for state in states]
        return np.array(smiles, dtype=object).reshape(-1, 1, 1)

    def posterior(self, X: np.ndarray, **kwargs):
        from psalm.data.dataset import MoleculePool

        bs, q, _ = X.shape
        smiles = X.flatten().tolist()
        pool: MoleculePool = MoleculePool.from_smiles(smiles, featurizer=MorganFeaturizer())
        X_tensor = torch.as_tensor(pool.X, dtype=torch.float64, device=self.device)
        X_tensor = X_tensor.reshape(bs, q, -1)
        return super().posterior(X_tensor, **kwargs)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, **kwargs) -> "ClppBayesianModel":
        checkpoint = torch.load(checkpoint_path, **kwargs)
        model = ApproximateTanimotoGP(inducing_points=checkpoint["inducing_points"])
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint["likelihood_state_dict"] and model.likelihood:
            model.likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
        model.__class__ = cls
        return cast("ClppBayesianModel", model)


@gin.configurable()
class ClppBayesianProxy(BayesianProxy, CachedProxyBase[ReactionState]):
    def __init__(
        self,
        model: ClppBayesianModel,
        acquisition_factory=UpperConfidenceBound(beta=0.0),
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            acquisition_factory=acquisition_factory,
            *args,
            **kwargs,
        )
        self.device = model.device
        self.cache = {ReactionStateEarlyTerminal(None): -float(16)}

    @property
    def is_non_negative(self) -> bool:
        return False

    @property
    def higher_is_better(self) -> bool:
        return True

    def _compute_proxy_output(self, states: List[ReactionState]) -> List[float]:
        return self._compute_acquisition_values(states)


if __name__ == "__main__":
    import timeit

    from rgfn.gfns.reaction_gfn.api.data_structures import Molecule
    from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateTerminal

    ckpt_path = "external/bayesian/clp_proxy_vgp.pt"
    model = ClppBayesianModel.load_from_checkpoint(ckpt_path, map_location="cpu")
    acquisition_factory = UpperConfidenceBound(beta=1.0)
    proxy = ClppBayesianProxy(model=model, acquisition_factory=acquisition_factory)

    smiles_list = [
        "CC(C)(C)OC(=O)N1CCN(CC1)C(=O)C(CC2CCCO2)Nc2cc(N)cc(N)c2",
        "CN1CCN(CC1)C(=O)C(CC2CCCO2)Nc2ccc(F)cc2",
    ]
    states: List[ReactionState] = [
        ReactionStateTerminal(molecule=Molecule(x), num_reactions=1) for x in smiles_list
    ]
    start_time = timeit.default_timer()
    result = proxy.compute_proxy_output(states)
    end_time = timeit.default_timer()
    print(f"Time taken to dock {len(smiles_list)} SMILES: {end_time - start_time:.2f} seconds")

    for smiles, res in zip(smiles_list, result.value):
        print(f"SMILES: {smiles}")
        print(f"Score: {res}")
        print()
