import numpy as np
import pytest

from fifteen.battery import POWERBLOCK as B
from fifteen.control import monthly_peaks, run_mpc


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("bias", [0.4, 1.0, 2.5])
def test_mpc_never_worse_than_no_battery(seed, bias):
    """Even with a badly wrong forecast, the controller may not raise any month's billed demand."""
    rng = np.random.default_rng(seed)
    n = 96 * 60
    tod = (np.arange(n) % 96) / 96
    x = 400 + 300 * np.clip(np.sin(np.pi * (tod - 0.3) / 0.5), 0, None) + rng.normal(0, 40, n)
    for _ in range(20):
        i = rng.integers(0, n - 12)
        x[i:i + rng.integers(1, 12)] += rng.uniform(50, 500)
    x = np.clip(x, 0, None)
    fc = x * bias * (1 + rng.normal(0, 0.3, n))  # deliberately terrible forecast
    month = (np.arange(n) // (96 * 30)).astype(np.int64)
    w = np.where((np.arange(n) % 96 >= 32) & (np.arange(n) % 96 < 84), 1.0, 1 / 0.45)
    net, soc = run_mpc(x, fc, month, w, B.power_kw, B.energy_kwh, B.eta_charge, B.eta_discharge, B.energy_kwh, 0.25, 0.0, 1.0, 4, 0.9)
    assert (monthly_peaks(net, month, w) <= monthly_peaks(x, month, w) + 1e-9).all()
