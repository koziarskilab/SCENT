import abc
from typing import Dict, List

from rgfn.api.proxy_base import ProxyBase
from rgfn.api.type_variables import TState


class NormalisedProxy(ProxyBase[TState], abc.ABC):
    """Abstract base class for proxies that compute a normalised output value
    using min-max scaling to produce an output between 0 and 1.
    """

    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = 1.0,
        clip_at_min: bool = True,
        clip_at_max: bool = False,
        *args,
        **kwargs,
    ):
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.clip_at_min = clip_at_min
        self.clip_at_max = clip_at_max
        super().__init__(*args, **kwargs)

    def _apply(self, value: float) -> float:
        norm_value = (value - self.min_value) / (self.max_value - self.min_value)
        if self.clip_at_min:
            norm_value = max(0.0, norm_value)
        if self.clip_at_max:
            norm_value = min(1.0, norm_value)
        return norm_value

    def min_max_normalise(
        self, proxy_output: List[Dict[str, float]] | List[float]
    ) -> List[Dict[str, float]] | List[float]:
        """Normalises the proxy output using min-max scaling."""
        if isinstance(proxy_output[0], dict):
            return [
                {k: self._apply(v) if k == "value" else v for k, v in item.items()}  # type: ignore
                for item in proxy_output
            ]
        elif isinstance(proxy_output[0], float):
            return [self._apply(x) for x in proxy_output]  # type: ignore
        else:
            raise TypeError("Proxy output must be a list of dicts or a list of floats/ints.")
