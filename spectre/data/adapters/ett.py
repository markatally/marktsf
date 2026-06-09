"""ETT adapter — supports both the symmetric and asymmetric framings.

ETT columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT.

Two modes (PROPOSAL.md §6.1):
- ``mode="multivariate"`` : all 7 value columns are endogenous (input+predicted).
  This matches the canonical iTransformer/PatchTST benchmark tables and is what
  P1 reproduction is evaluated against.
- ``mode="miso"`` : OT is the single endogenous target; the 6 load columns
  become past-only covariates. This is the SPECTRE method's asymmetric setting.

Both modes attach calendar features as known-future covariates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..features.calendar import CALENDAR_COLUMNS, calendar_features
from ..normalization import Standardizer
from ..windowing import CovariateSpec, DatasetBundle

__all__ = ["ETTAdapter"]

# All numeric value columns, in canonical order; OT is the designated target.
_VALUE_COLS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
_TARGET = "OT"
_LOAD_COLS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL")


class ETTAdapter:
    def __init__(self, csv_path: str | Path, mode: str = "multivariate") -> None:
        self.csv_path = Path(csv_path)
        # Fail at construction if the file is missing (clear, early error).
        if not self.csv_path.exists():
            raise FileNotFoundError(f"ETT csv not found: {self.csv_path}")
        if mode not in ("multivariate", "miso"):
            raise ValueError(f"mode must be 'multivariate' or 'miso', got {mode!r}")
        self.mode = mode
        # Build the channel-role spec for the chosen mode.
        if mode == "multivariate":
            # Every value column is endogenous; no exogenous past covariates.
            self.spec = CovariateSpec(
                endogenous=_VALUE_COLS,
                past_only=(),
                known_future=CALENDAR_COLUMNS,
            )
        else:  # miso
            # OT is the lone endogenous target; load columns are covariates.
            self.spec = CovariateSpec(
                endogenous=(_TARGET,),
                past_only=_LOAD_COLS,
                known_future=CALENDAR_COLUMNS,
            )

    def load(self, train_end: int | None = None) -> DatasetBundle:
        df = pd.read_csv(self.csv_path)
        # Validate the schema up front so a wrong/renamed file fails clearly.
        missing = {*_VALUE_COLS, "date"} - set(df.columns)
        if missing:
            raise ValueError(f"ETT csv missing columns: {sorted(missing)}")

        # Pull the endogenous + (optionally) covariate blocks per mode.
        endo = df[list(self.spec.endogenous)].to_numpy(dtype=np.float64)   # [T, C_endo]
        if self.spec.c_p:
            past = df[list(self.spec.past_only)].to_numpy(dtype=np.float64)  # [T, C_p]
        else:
            past = np.zeros((len(df), 0), dtype=np.float64)
        # Calendar covariates are already centered to [-0.5, 0.5] → no scaling.
        known = calendar_features(df["date"])                              # [T, 4]

        # Standardize endogenous + past covariates using TRAIN-ONLY statistics.
        # `train_end` is passed by the runner; when None we return raw values
        # (used for a quick length peek before the split is known).
        if train_end is not None:
            endo = Standardizer.fit(endo, train_end).transform(endo)
            if past.shape[1]:
                past = Standardizer.fit(past, train_end).transform(past)

        # Package everything into the bundle the windower understands.
        return DatasetBundle(
            endo=endo,
            past_cov=past,
            known_cov=known,
            static=np.zeros(0, dtype=np.float64),  # ETT has no static covariates
            spec=self.spec,
            name=f"{self.csv_path.stem}-{self.mode}",  # e.g. "ETTh1-multivariate"
        )
