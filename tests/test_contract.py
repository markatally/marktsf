import pytest
import torch

from spectre.data.contract import CovariateSpec, WindowSample, collate_samples


def _sample(L=8, H=4, c_endo=1, c_p=3, c_f=2, c_s=0):
    return WindowSample(
        x_endo=torch.randn(L, c_endo),
        x_past_cov=torch.randn(L, c_p),
        x_known_past=torch.randn(L, c_f),
        x_known_fut=torch.randn(H, c_f),
        static=torch.randn(c_s),
        y=torch.randn(H, c_endo),
    )


def test_valid_sample_constructs():
    s = _sample()
    assert s.lookback == 8 and s.horizon == 4 and s.c_endo == 1


def test_multivariate_endo_constructs():
    s = _sample(c_endo=7, c_p=0)
    assert s.c_endo == 7 and s.y.shape == (4, 7)


def test_endo_label_channel_mismatch_raises():
    with pytest.raises(ValueError, match="C_endo"):
        WindowSample(
            x_endo=torch.randn(8, 3),
            x_past_cov=torch.randn(8, 0),
            x_known_past=torch.randn(8, 2),
            x_known_fut=torch.randn(4, 2),
            static=torch.randn(0),
            y=torch.randn(4, 2),  # wrong C_endo (2 != 3)
        )


def test_known_future_horizon_mismatch_raises():
    with pytest.raises(ValueError, match="x_known_fut"):
        WindowSample(
            x_endo=torch.randn(8, 1),
            x_past_cov=torch.randn(8, 3),
            x_known_past=torch.randn(8, 2),
            x_known_fut=torch.randn(5, 2),  # wrong H
            static=torch.randn(0),
            y=torch.randn(4, 1),
        )


def test_collate_shapes():
    batch = collate_samples([_sample(c_endo=2) for _ in range(5)])
    assert batch["x_endo"].shape == (5, 8, 2)
    assert batch["x_past_cov"].shape == (5, 8, 3)
    assert batch["x_known_fut"].shape == (5, 4, 2)
    assert batch["y"].shape == (5, 4, 2)
    assert len(batch["meta"]) == 5


def test_collate_empty_raises():
    with pytest.raises(ValueError):
        collate_samples([])


def test_covariate_spec_counts_and_target():
    spec = CovariateSpec(
        endogenous=("OT",), past_only=("a", "b"), known_future=("c",)
    )
    assert (spec.c_endo, spec.c_p, spec.c_f, spec.c_s) == (1, 2, 1, 0)
    assert spec.target == "OT"
    assert spec.is_miso is True


def test_covariate_spec_requires_endogenous():
    with pytest.raises(ValueError, match="endogenous"):
        CovariateSpec(endogenous=())
