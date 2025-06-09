from pathlib import Path

import pytest
from training.training_test_helpers import helper__test_training__runs_properly


@pytest.mark.parametrize(
    "config_override_str",
    [
        "",
        "include 'configs/policies/rgfn_cost_guided.gin'",
        "include 'configs/policies/rgfn_cost_guided.gin'\ninclude 'configs/envs/dynamic_library/dynamic_library.gin'",
    ],
)
def test__rgfn__trains_properly(config_override_str: str, tmp_path: Path):
    helper__test_training__runs_properly("configs/rgfn_test.gin", config_override_str, tmp_path)
