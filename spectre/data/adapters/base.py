"""Dataset adapter interface.

An adapter's only job: map a raw dataset into a normalized ``DatasetBundle``
honoring the unified contract. Adapters apply train-only standardization given
the train split boundary so no statistic leaks from val/test.
"""

from __future__ import annotations

from typing import Protocol

from ..windowing import DatasetBundle

__all__ = ["DatasetAdapter"]


class DatasetAdapter(Protocol):
    """Maps raw files to a (optionally standardized) ``DatasetBundle``.

    Implemented structurally (Protocol) — concrete adapters need not inherit;
    they only need a matching ``load`` signature. This keeps adapters decoupled
    from this module and trivially mockable in tests.
    """

    def load(self, train_end: int | None = None) -> DatasetBundle:
        """Return a bundle. If ``train_end`` is given, standardize using only
        ``[:train_end]`` statistics for the target and past covariates."""
        ...
