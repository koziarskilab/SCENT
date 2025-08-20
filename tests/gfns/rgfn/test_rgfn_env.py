from gfns.helpers.env_test_helpers import (
    helper__test_env__backward_forward_consistency,
    helper__test_env__forward_backward_consistency,
)

from .fixtures import *


@pytest.mark.parametrize("n_trajectories", [10])
@pytest.mark.parametrize("seed", list(range(1000)))
def test__rgfn_env__forward_backward_consistency(
    rgfn_env: ReactionEnv, n_trajectories: int, seed: int
):
    helper__test_env__forward_backward_consistency(
        rgfn_env, n_trajectories=n_trajectories, seed=seed
    )


@pytest.mark.parametrize("n_trajectories", [10])
@pytest.mark.parametrize("seed", list(range(10)))
def test__rgfn_env__backward_forward_consistency(
    rgfn_env: ReactionEnv, n_trajectories: int, seed: int
):
    helper__test_env__backward_forward_consistency(
        rgfn_env, n_trajectories=n_trajectories, sample_directly_from_reversed_env=False, seed=seed
    )
