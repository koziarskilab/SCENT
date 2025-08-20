from typing import List

import gin
import numpy as np
from psalm.acquisition import UpperConfidenceBound

from rgfn.api.type_variables import TState
from rgfn.gfns.reaction_gfn.proxies.gneprop_proxy import (
    GNEpropBayesianModel,
    GNEpropProxy,
)


@gin.configurable()
class SenoProxy(GNEpropProxy):
    def __init__(
        self,
        model: GNEpropBayesianModel,
        acquisition_factory=UpperConfidenceBound(beta=1.0),
        best_f: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model, acquisition_factory=acquisition_factory, best_f=best_f, *args, **kwargs
        )

    def _compute_proxy_output(self, states: List[TState]) -> List[float]:
        output = np.array(self._compute_gneprop_output(states), dtype=np.float32)
        return output.clip(1e-6, 1.0).tolist()


# Example usage
if __name__ == "__main__":
    import timeit

    from psalm.acquisition.functions import UpperConfidenceBound

    from rgfn.gfns.reaction_gfn.api.data_structures import Molecule
    from rgfn.gfns.reaction_gfn.api.reaction_api import (
        ReactionState,
        ReactionStateTerminal,
    )

    model = GNEpropBayesianModel(
        checkpoint_path="external/gneprop/models/seno.ckpt",
        batch_size=64,
    )
    acquisition_factory = UpperConfidenceBound(beta=1.0)
    proxy = SenoProxy(model=model, acquisition_factory=acquisition_factory, best_f=0.0)

    smiles_list = [
        "CC(C)(C)OC(=O)N1CCN(CC1)C(=O)C(CC2CCCO2)Nc2cc(N)cc(N)c2",
        # "CN1CCN(CC1)C(=O)C(CC2CCCO2)Nc2ccc(F)cc2",
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
