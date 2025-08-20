import abc
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List

from rgfn.api.proxy_base import ProxyBase
from rgfn.api.type_variables import TState


class ServerProxy(ProxyBase[TState], abc.ABC):
    def __init__(
        self,
        batch_size_per_node: int = 16,  # Should be rgfn batch size / number of nodes
        registry_file: Path = Path(__file__).parent / "server_registry.json",
        *args,
        **kwargs,
    ):
        from .server.client import ProxyClient

        super().__init__(*args, **kwargs)
        self.registry_file = registry_file
        self.batch_size_per_node = batch_size_per_node
        self.client = ProxyClient(registry_file=str(registry_file), batch_size=batch_size_per_node)

    @abstractmethod
    def __call__(self, smiles_list: List[str]) -> List[Dict[str, float]] | List[float]:
        raise NotImplementedError("Subclasses should implement this method.")

    def distributed_compute_proxy_output(
        self, states: List[TState]
    ) -> List[Dict[str, float]] | List[float]:
        smiles_list = [state.molecule.smiles for state in states]
        results = self.client.query(smiles_list)

        if len(results) != len(smiles_list):
            raise ValueError(f"Expected {len(smiles_list)} results, but got {len(results)}.")
        if not all(isinstance(r, (float, int)) for r in results):
            raise TypeError("All results must be numeric (float or int).")

        return results
