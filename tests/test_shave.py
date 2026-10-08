import numpy as np
import pytest

from fifteen.audit import check_dispatch, lp_month
from fifteen.battery import POWERBLOCK, Battery
from fifteen.shave import hold, plan_month


def synthetic_month(seed, n=96 * 7, base=400.0, spikes=6):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    tod = (t % 96) / 96
    load = base + 250 * np.clip(np.sin(np.pi * (tod - 0.25) / 0.5), 0, None) + rng.normal(0, 25, n)
    for _ in range(spikes):
        i = rng.integers(0, n - 8)
        load[i:i + rng.integers(1, 8)] += rng.uniform(100, 350)
    return np.clip(load, 0, None)


@pytest.mark.parametrize("seed", range(6))
def test_single_window_matches_lp(seed):
    load = synthetic_month(seed)
    masks = np.ones((1, load.size), bool)
    rates = np.array([20.0])
    plan = plan_month(load, masks, rates, POWERBLOCK, s0=POWERBLOCK.energy_kwh)
    lp = lp_month(load, masks, rates, POWERBLOCK, s0=POWERBLOCK.energy_kwh)
    assert plan.thresholds[0] == pytest.approx(lp["thresholds"][0], abs=0.05)


@pytest.mark.parametrize("seed", range(4))
def test_two_windows_match_lp(seed):
    load = synthetic_month(seed + 10)
    n = load.size
    tod = np.arange(n) % 96
    onpeak = (tod >= 32) & (tod < 84)
    masks = np.vstack([np.ones(n, bool), onpeak])
    rates = np.array([8.0, 14.0])
    plan = plan_month(load, masks, rates, POWERBLOCK, s0=POWERBLOCK.energy_kwh * 0.5)
    lp = lp_month(load, masks, rates, POWERBLOCK, s0=POWERBLOCK.energy_kwh * 0.5)
    assert plan.cost == pytest.approx(lp["cost"], rel=2e-4, abs=0.5)


@pytest.mark.parametrize("seed", range(4))
def test_greedy_trace_is_physical(seed):
    load = synthetic_month(seed + 20)
    masks = np.ones((1, load.size), bool)
    plan = plan_month(load, masks, np.array([1.0]), POWERBLOCK, s0=POWERBLOCK.energy_kwh)
    net, soc, worst, _ = hold(load, plan.tau, POWERBLOCK, s0=POWERBLOCK.energy_kwh)
    assert worst < 1e-6
    assert net.max() <= plan.thresholds[0] + 1e-6
    assert check_dispatch(load, net, soc, POWERBLOCK, s0=POWERBLOCK.energy_kwh) == []


def test_power_limit_binds_for_short_spike():
    load = np.full(96, 300.0)
    load[50] = 900.0  # one 15-minute spike, far above what 250 kW can cover
    plan = plan_month(load, np.ones((1, 96), bool), np.array([1.0]), POWERBLOCK, s0=POWERBLOCK.energy_kwh)
    assert plan.thresholds[0] == pytest.approx(650.0, abs=1e-3)


def test_energy_limit_binds_for_long_plateau():
    load = np.full(96, 200.0)
    load[40:72] = 500.0  # 8 hours at +300 kW: energy, not power, is the constraint
    bat = Battery()
    plan = plan_month(load, np.ones((1, 96), bool), np.array([1.0]), bat, s0=bat.energy_kwh)
    shave = 500.0 - plan.thresholds[0]
    # energy needed for the plateau = shave * 8 h / eta_d  must equal usable energy
    assert shave * 8 / bat.eta_discharge == pytest.approx(bat.energy_kwh, rel=1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_offpeak_discount_matches_lp(seed):
    """Eversource-style billing demand: max(peak-period kW, 0.45 x off-peak kW)."""
    load = synthetic_month(seed + 30)
    n = load.size
    tod = np.arange(n) % 96
    on = (tod >= 36) & (tod < 72) & ((np.arange(n) // 96) % 7 < 5)
    w = np.where(on, 1.0, 1 / 0.45)[None, :]
    plan = plan_month(load, None, np.array([33.95]), POWERBLOCK, s0=POWERBLOCK.energy_kwh, weights=w)
    lp = lp_month(load, None, np.array([33.95]), POWERBLOCK, s0=POWERBLOCK.energy_kwh, weights=w)
    assert plan.thresholds[0] == pytest.approx(lp["thresholds"][0], abs=0.05)
    net, _, worst, _ = hold(load, plan.tau, POWERBLOCK, s0=POWERBLOCK.energy_kwh)
    billed = max(net[on].max(), 0.45 * net[~on].max())
    assert billed == pytest.approx(plan.thresholds[0], abs=1e-4)
