from datetime import datetime
from typing import Any, List

import gin

from rgfn.api.env_base import EnvBase
from rgfn.api.type_variables import TAction, TActionSpace, TState


@gin.configurable()
def get_time_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@gin.configurable()
def get_str(format: str, values: List[Any]) -> str:
    return format.format(*values)


@gin.configurable()
def reverse(
    env: EnvBase[TState, TActionSpace, TAction],
) -> EnvBase[TState, TActionSpace, TAction]:
    return env.reversed()


# Taken from:https://github.com/google/gin-config/issues/154
def gin_config_to_readable_dictionary(gin_config: dict):
    """
    Parses the gin configuration to a dictionary. Useful for logging to e.g. W&B
    :param gin_config: the gin's config dictionary. Can be obtained by gin.config._OPERATIVE_CONFIG
    :return: the parsed (mainly: cleaned) dictionary
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data[".".join([name, k])] = v

    return data
