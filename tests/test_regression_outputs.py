from __future__ import annotations

from open_match_lca.regression.topk_factor_mixture import topk_factor_mixture


def test_topk_factor_mixture_output_shape() -> None:
    output = topk_factor_mixture([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert set(output.keys()) == {
        "prediction",
        "prob_max",
        "factor_mean",
        "factor_std",
        "factor_min",
        "factor_max",
    }
    assert output["factor_min"] == 1.0
