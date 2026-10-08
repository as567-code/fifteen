"""Battery model.

Defaults follow Powertown's published Powerblock spec sheet (powertownusa.com/product,
read 2026-10-08): 250 kW rated AC power, 522 kWh energy capacity, LFP, 97.5% PCS efficiency.

Two numbers are NOT published and are therefore explicit assumptions:
  * usable depth of discharge (90% -> 470 kWh usable), typical for LFP with a 5-95% SOC window
  * DC round-trip efficiency of the cells (~95%), giving ~90% AC round-trip efficiency
Every downstream result can be re-run with different values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Battery:
    power_kw: float = 250.0
    nameplate_kwh: float = 522.0
    usable_fraction: float = 0.90
    pcs_efficiency: float = 0.975
    dc_round_trip: float = 0.95

    @property
    def energy_kwh(self) -> float:
        return self.nameplate_kwh * self.usable_fraction

    @property
    def round_trip(self) -> float:
        return self.pcs_efficiency**2 * self.dc_round_trip

    @property
    def eta_charge(self) -> float:
        return math.sqrt(self.round_trip)

    @property
    def eta_discharge(self) -> float:
        return math.sqrt(self.round_trip)

    def scaled(self, n_blocks: float) -> "Battery":
        """n Powerblocks in parallel."""
        return replace(self, power_kw=self.power_kw * n_blocks, nameplate_kwh=self.nameplate_kwh * n_blocks)


POWERBLOCK = Battery()
