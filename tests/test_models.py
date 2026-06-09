import pytest
import torch

from spectre.data.contract import WindowSample, collate_samples
from spectre.models.layers.revin import RevIN
from spectre.models.registry import MODEL_REGISTRY, build_model


def _batch(B=4, L=96, H=24, c_endo=7, c_p=0, c_f=2):
    samples = [
        WindowSample(
            x_endo=torch.randn(L, c_endo),
            x_past_cov=torch.randn(L, c_p),
            x_known_past=torch.randn(L, c_f),
            x_known_fut=torch.randn(H, c_f),
            static=torch.randn(0),
            y=torch.randn(H, c_endo),
        )
        for _ in range(B)
    ]
    return collate_samples(samples)


# ---------------- RevIN ----------------

def test_revin_invertible_no_affine():
    rev = RevIN(num_features=3, affine=False)
    x = torch.randn(5, 96, 3) * 7 + 3
    x_d = rev(rev(x, "norm"), "denorm")
    assert torch.allclose(x, x_d, atol=1e-4)


def test_revin_invertible_identity_affine():
    rev = RevIN(num_features=3, affine=True)  # weight=1, bias=0 at init
    x = torch.randn(5, 96, 3) * 2 - 1
    x_d = rev(rev(x, "norm"), "denorm")
    assert torch.allclose(x, x_d, atol=1e-4)


def test_revin_normalizes_to_zero_mean():
    rev = RevIN(num_features=2, affine=False)
    x = torch.randn(8, 50, 2) * 5 + 10
    x_n = rev(x, "norm")
    assert torch.allclose(x_n.mean(dim=1), torch.zeros(8, 2), atol=1e-4)


# ---------------- baselines (multivariate) ----------------

@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_baseline_forward_shapes(name):
    B, L, H, C = 4, 96, 24, 7
    model = build_model(name, lookback=L, horizon=H, c_endo=C)
    out = model(_batch(B, L, H, c_endo=C))
    assert out["y_hat"].shape == (B, H, C)
    assert out.get("x_hat") is None


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_baseline_backward(name):
    B, L, H, C = 4, 96, 24, 7
    model = build_model(name, lookback=L, horizon=H, c_endo=C)
    out = model(_batch(B, L, H, c_endo=C))
    out["y_hat"].pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_build_model_unknown_raises():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("nope", lookback=96, horizon=24, c_endo=7)
