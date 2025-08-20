import abc
from typing import Dict, List

import gin
from rdkit import Chem
from rdkit.Chem.QED import qed

from rgfn.gfns.reaction_gfn.api.data_structures import Molecule
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
    ReactionStateTerminal,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase
from rgfn.shared.proxies.server_proxy import ServerProxy


@gin.configurable()
class QEDProxy(
    ServerProxy[ReactionState],
    CachedProxyBase[ReactionState],
    abc.ABC,
):
    def __init__(self, use_server=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_server = use_server
        self.cache = {ReactionStateEarlyTerminal(None): 0.0}

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def __call__(self, smiles_list: List[str]) -> List[Dict[str, float]] | List[float]:
        return [qed(Chem.MolFromSmiles(smi)) for smi in smiles_list]

    def _compute_proxy_output(self, states: List[ReactionState]):
        if self.use_server:
            raw_proxy = self.distributed_compute_proxy_output(states)
        else:
            raw_proxy = self.__call__([state.molecule.smiles for state in states])
        return raw_proxy


if __name__ == "__main__":
    proxy = QEDProxy(use_server=False)
    smiles = ["CCO", "CCN", "C1=CC=CC=C1"]
    states: List[ReactionState] = [
        ReactionStateTerminal(molecule=Molecule(x), num_reactions=1) for x in smiles
    ]
    result = proxy.compute_proxy_output(states)

    for smi, qed_value in zip(smiles, result.value):
        print(f"SMILES: {smi}, QED: {qed_value:.4f}")
