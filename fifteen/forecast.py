"""Day-ahead probabilistic load forecasts for peak-shaving control.

One global gradient-boosted model is trained across many sites on *scale-free* targets
(load / the site's trailing 28-day mean), so a new site with a few weeks of interval data
gets a usable forecast on day one. Quantile loss gives a distribution, not a point: the
controller decides how much tail risk to plan for.

Every feature for day d is computed from days <= d-1. `assert_no_lookahead` checks that by
perturbing the future and confirming the features do not move.
"""

from __future__ import annotations

from dataclasses import dataclass

import holidays as hol
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

SLOTS = 96
WARMUP_DAYS = 28

FEATURES = [
    "slot", "dow", "month", "holiday", "holiday_prev",
    "y1", "y7", "y14", "y1_win", "y7_win", "same_dow_mean4", "same_dow_max4",
    "y1_max", "y1_mean", "y7d_max", "y1_last", "y7_dmax",
]


def to_days(kw: np.ndarray, start: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    days = kw.size // SLOTS
    return kw[: days * SLOTS].reshape(days, SLOTS).astype(np.float64), pd.date_range(start, periods=days, freq="D")


def day_features(X: np.ndarray, dates: pd.DatetimeIndex, country: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Features for every (day, slot) with day >= WARMUP_DAYS. Returns (F [n_days*96, n_feat], scale [n_days])."""
    nd = X.shape[0]
    hdays = set(hol.country_holidays(country, years=sorted(set(dates.year))).keys()) if country else set()
    is_hol = np.array([d.date() in hdays for d in dates], dtype=float)
    daily_mean = X.mean(1)
    scale = np.full(nd, np.nan)
    cs = np.concatenate([[0.0], np.cumsum(daily_mean)])
    for d in range(WARMUP_DAYS, nd):
        scale[d] = (cs[d] - cs[d - WARMUP_DAYS]) / WARMUP_DAYS
    scale = np.where(scale > 1e-6, scale, np.nan)

    rows = []
    slot = np.arange(SLOTS)
    for d in range(WARMUP_DAYS, nd):
        s = scale[d]
        y1, y7, y14 = X[d - 1] / s, X[d - 7] / s, X[d - 14] / s
        same = np.stack([X[d - 7 * k] for k in range(1, 5)]) / s
        win = lambda v: np.convolve(np.pad(v, 2, mode="edge"), np.ones(5) / 5, mode="valid")  # noqa: E731
        f = np.column_stack([
            slot,
            np.full(SLOTS, dates[d].dayofweek),
            np.full(SLOTS, dates[d].month),
            np.full(SLOTS, is_hol[d]),
            np.full(SLOTS, is_hol[d - 1]),
            y1, y7, y14, win(y1), win(y7), same.mean(0), same.max(0),
            np.full(SLOTS, y1.max()), np.full(SLOTS, y1.mean()),
            np.full(SLOTS, (X[d - 7:d] / s).max()),
            np.full(SLOTS, y1[-1]),
            np.full(SLOTS, y7.max()),
        ])
        rows.append(f)
    F = np.vstack(rows) if rows else np.empty((0, len(FEATURES)))
    return F, scale


def assert_no_lookahead(X: np.ndarray, dates: pd.DatetimeIndex, country: str | None, d: int) -> None:
    """Features of day d must be unchanged if day d and everything after it is scrambled."""
    F0, _ = day_features(X, dates, country)
    Xp = X.copy()
    Xp[d:] = np.random.default_rng(0).permutation(Xp[d:].ravel()).reshape(Xp[d:].shape) * 3.0
    F1, _ = day_features(Xp, dates, country)
    i0 = (d - WARMUP_DAYS) * SLOTS
    if not np.allclose(F0[i0:i0 + SLOTS], F1[i0:i0 + SLOTS], equal_nan=True):
        raise AssertionError(f"features for day {d} depend on day >= {d}")


@dataclass
class QuantileForecaster:
    quantiles: tuple[float, ...] = (0.5, 0.8, 0.95)
    max_iter: int = 300
    models: dict | None = None

    def fit(self, F: np.ndarray, y: np.ndarray, seed: int = 0) -> "QuantileForecaster":
        ok = np.isfinite(y) & np.isfinite(F).all(1)
        F, y = F[ok], y[ok]
        self.models = {}
        for q in self.quantiles:
            m = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=self.max_iter, learning_rate=0.08,
                max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
                categorical_features=[1], random_state=seed,
            )
            self.models[q] = m.fit(F, y)
        return self

    def predict(self, F: np.ndarray) -> dict[float, np.ndarray]:
        out = {q: m.predict(F) for q, m in self.models.items()}
        # enforce monotone quantiles (crossing is possible with independent models)
        qs = sorted(out)
        for a, b in zip(qs, qs[1:]):
            out[b] = np.maximum(out[b], out[a])
        return out


def seasonal_naive(X: np.ndarray) -> np.ndarray:
    """Benchmark: same slot, same weekday last week."""
    out = np.full_like(X, np.nan)
    out[7:] = X[:-7]
    return out


def pinball(y, f, q):
    d = y - f
    return np.nanmean(np.maximum(q * d, (q - 1) * d))
