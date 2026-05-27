import pytest


@pytest.fixture
def default_glm_config():
    """Default GLM configuration for testing.

    The import is deferred so this conftest itself does not require ``ttnn``
    at collection time — that lets ``pytest tests/auto/`` run without the
    TTNN runtime installed (see ``tests/auto/conftest.py``).
    """
    from tt_symbiote.integrations.ttnn_moe import Glm4MoeConfig

    return Glm4MoeConfig(
        hidden_size=2048,
        intermediate_size=10240,
        moe_intermediate_size=1536,
        num_local_experts=64,
        num_experts_per_tok=4,
        n_shared_experts=1,
        routed_scaling_factor=1.8,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
    )
