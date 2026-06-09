import numpy as np

from spectre.data.contract import CovariateSpec
from spectre.data.windowing import DatasetBundle, WindowDataset, make_splits


def _bundle(T=1000, c_p=3, c_f=2):
    spec = CovariateSpec(
        endogenous=("y",),
        past_only=tuple(f"p{i}" for i in range(c_p)),
        known_future=tuple(f"k{i}" for i in range(c_f)),
    )
    return DatasetBundle(
        # endo is [T, 1]; values = arange so alignment is checkable.
        endo=np.arange(T, dtype=np.float64).reshape(T, 1),
        past_cov=np.random.randn(T, c_p),
        known_cov=np.random.randn(T, c_f),
        static=np.zeros(0),
        spec=spec,
        name="synthetic",
    )


def _label_indices(starts, L, H):
    idx = set()
    for s in starts:
        idx.update(range(s + L, s + L + H))
    return idx


def test_label_regions_partition_at_boundaries():
    # No FUTURE label leaks across a split boundary (the real leakage rule).
    T, L, H = 2000, 96, 24
    sp = make_splits(T, L, H, ratios=(0.7, 0.1, 0.2))
    n_train, n_val = sp.bounds["train"][1], sp.bounds["val"][1]

    # train labels strictly before n_train; val labels in [n_train, n_val); etc.
    assert (sp.train + L + H).max() <= n_train
    assert sp.val.size and (sp.val + L).min() >= n_train
    assert (sp.val + L + H).max() <= n_val
    assert sp.test.size and (sp.test + L).min() >= n_val


def test_no_label_overlap_between_splits():
    T, L, H = 2000, 64, 16
    sp = make_splits(T, L, H)
    tr = _label_indices(sp.train, L, H)
    va = _label_indices(sp.val, L, H)
    te = _label_indices(sp.test, L, H)
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)


def test_lookback_may_cross_boundary():
    # val windows are allowed to read history from the train segment.
    T, L, H = 2000, 96, 24
    sp = make_splits(T, L, H)
    n_train = sp.bounds["train"][1]
    assert sp.val.min() < n_train  # earliest val start reaches back into train


def test_embargo_purges_train_tail():
    T, L, H, emb = 2000, 96, 24, 50
    base = make_splits(T, L, H, embargo=0)
    purged = make_splits(T, L, H, embargo=emb)
    n_train = base.bounds["train"][1]
    assert purged.train.size < base.train.size
    assert (purged.train + L + H).max() <= n_train - emb


def test_dataset_item_alignment():
    b = _bundle(T=500)
    L, H = 96, 24
    sp = make_splits(500, L, H)
    ds = WindowDataset(b, sp.train, L, H)
    s = ds[0]
    start = int(sp.train[0])
    # endo future label must equal the raw series slice (endo is arange)
    assert s.y[0, 0].item() == start + L
    assert s.x_endo[0, 0].item() == start
    assert s.x_known_fut.shape == (H, b.spec.c_f)


def test_ratios_must_sum_to_one():
    import pytest

    with pytest.raises(ValueError):
        make_splits(100, 8, 4, ratios=(0.6, 0.1, 0.2))
