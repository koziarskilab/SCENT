from pathlib import Path

import pytest

from rgfn import CostGuidedBackwardPolicy
from rgfn.gfns.reaction_gfn import ReactionDataFactory, ReactionEnv
from rgfn.gfns.reaction_gfn.policies.action_embeddings import FragmentOneHotEmbedding
from rgfn.gfns.reaction_gfn.policies.decomposability_guided_backward_policy import (
    DecomposabilityGuidedBackwardPolicy,
)
from rgfn.gfns.reaction_gfn.policies.guidance_models.cost_models import SimpleCostModel
from rgfn.gfns.reaction_gfn.policies.guidance_models.decomposable_models import (
    BinaryDecomposableModel,
)
from rgfn.gfns.reaction_gfn.policies.jointly_biased_backward_policy import (
    JointlyGuidedBackwardPolicy,
)
from rgfn.gfns.reaction_gfn.policies.reaction_backward_policy import (
    ReactionBackwardPolicy,
)
from rgfn.gfns.reaction_gfn.policies.reaction_forward_policy import (
    ReactionForwardPolicy,
)
from rgfn.gfns.reaction_gfn.proxies.path_cost_proxy import PathCostProxy


@pytest.fixture(scope="module")
def reaction_path() -> Path:
    return Path(__file__).parent / "../../../data/rgfn/templates.txt"


@pytest.fixture(scope="module")
def fragment_path() -> Path:
    return Path(__file__).parent / "../../../data/rgfn/fragments.txt"


@pytest.fixture(scope="module")
def rgfn_data_factory(reaction_path: Path, fragment_path: Path) -> ReactionDataFactory:
    return ReactionDataFactory(
        reaction_path=reaction_path,
        fragment_path=fragment_path,
    )


@pytest.fixture(scope="module")
def rgfn_env(rgfn_data_factory: ReactionDataFactory) -> ReactionEnv:
    return ReactionEnv(data_factory=rgfn_data_factory, max_num_reactions=2)


@pytest.fixture(scope="module")
def rgfn_one_hot_action_embedding_fn(
    rgfn_data_factory: ReactionDataFactory,
) -> FragmentOneHotEmbedding:
    return FragmentOneHotEmbedding(
        data_factory=rgfn_data_factory,
    )


@pytest.fixture(scope="module")
def rgfn_forward_policy(
    rgfn_data_factory: ReactionDataFactory,
    rgfn_one_hot_action_embedding_fn: FragmentOneHotEmbedding,
) -> ReactionForwardPolicy:
    return ReactionForwardPolicy(
        data_factory=rgfn_data_factory, action_embedding_fn=rgfn_one_hot_action_embedding_fn
    )


@pytest.fixture(scope="module")
def rgfn_backward_policy(
    rgfn_data_factory: ReactionDataFactory,
    rgfn_one_hot_action_embedding_fn: FragmentOneHotEmbedding,
) -> ReactionBackwardPolicy:
    return ReactionBackwardPolicy(
        data_factory=rgfn_data_factory,
    )


@pytest.fixture(scope="module")
def path_cost_proxy(rgfn_data_factory: ReactionDataFactory) -> PathCostProxy:
    return PathCostProxy(
        data_factory=rgfn_data_factory,
        yield_value=0.75,
    )


@pytest.fixture(scope="module")
def simple_cost_model(path_cost_proxy: PathCostProxy) -> SimpleCostModel:
    return SimpleCostModel(path_cost_proxy=path_cost_proxy, max_num_reactions=4)


@pytest.fixture(scope="module")
def binary_decomposable_model(path_cost_proxy: PathCostProxy) -> BinaryDecomposableModel:
    return BinaryDecomposableModel(path_cost_proxy=path_cost_proxy, max_num_reactions=4)


@pytest.fixture(scope="module")
def rgfn_decomposability_guided_backward_policy(
    path_cost_proxy: PathCostProxy, binary_decomposable_model: BinaryDecomposableModel
) -> DecomposabilityGuidedBackwardPolicy:
    return DecomposabilityGuidedBackwardPolicy(
        path_cost_proxy=path_cost_proxy, decomposable_prediction_model=binary_decomposable_model
    )


@pytest.fixture(scope="module")
def rgfn_cost_guided_backward_policy(
    path_cost_proxy: PathCostProxy, simple_cost_model: SimpleCostModel
) -> CostGuidedBackwardPolicy:
    return CostGuidedBackwardPolicy(
        path_cost_proxy=path_cost_proxy,
        cost_prediction_model=simple_cost_model,
    )


@pytest.fixture(scope="module")
def rgfn_jointly_guided_backward_policy(
    rgfn_cost_guided_backward_policy: CostGuidedBackwardPolicy,
    rgfn_decomposability_guided_backward_policy: DecomposabilityGuidedBackwardPolicy,
) -> JointlyGuidedBackwardPolicy:
    return JointlyGuidedBackwardPolicy(
        policies=[rgfn_cost_guided_backward_policy, rgfn_decomposability_guided_backward_policy]
    )
