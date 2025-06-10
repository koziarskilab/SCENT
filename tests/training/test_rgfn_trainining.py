from pathlib import Path

import pytest
from training.training_test_helpers import helper__test_training__runs_properly


@pytest.mark.parametrize(
    "config_path",
    ["configs/rgfn_test.gin", "configs/scent_test.gin"],
)
def test__rgfn__trains_properly(config_path: str, tmp_path: Path):
    helper__test_training__runs_properly(config_path, "", tmp_path)
