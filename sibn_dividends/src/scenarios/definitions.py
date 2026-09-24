"""Loads config/scenarios/*.yaml and applies a scenario's mean-shifts on top
of the historically-calibrated (Brent, USDRUB) random walk before running
Monte Carlo. See calibration.py and montecarlo.py for what's estimated from
data vs. scenario-defined.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import yaml

from src.config import ROOT_DIR
from src.scenarios.calibration import CalibrationResult

SCENARIOS_DIR = ROOT_DIR / "config" / "scenarios"


@dataclass
class Scenario:
    name: str
    description: str
    discount_min: float
    discount_base: float
    discount_max: float
    brent_spot_override_usd: float | None
    fx_shift_multiplier: float
    refining_volume_shock: float


def load_scenario(name: str) -> Scenario:
    path = SCENARIOS_DIR / f"{name}.yaml"
    if not path.exists():
        available = [p.stem for p in SCENARIOS_DIR.glob("*.yaml")]
        raise FileNotFoundError(f"Unknown scenario {name!r}. Available: {available}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Scenario(
        name=data["name"],
        description=data["description"],
        discount_min=data["discount_usd_per_bbl"]["min"],
        discount_base=data["discount_usd_per_bbl"]["base"],
        discount_max=data["discount_usd_per_bbl"]["max"],
        brent_spot_override_usd=data.get("brent_spot_override_usd"),
        fx_shift_multiplier=data.get("fx_shift_multiplier", 1.0),
        refining_volume_shock=data.get("netback", {}).get("refining_volume_shock", 0.0),
    )


def list_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIOS_DIR.glob("*.yaml"))


def apply_scenario_to_calibration(
    calib: CalibrationResult, scenario: Scenario
) -> CalibrationResult:
    """Returns a copy of `calib` with the scenario's spot-level overrides
    applied. Volatility and correlation (the actually-estimated quantities)
    are left untouched — only the starting level of the random walk shifts."""
    brent_spot = (
        scenario.brent_spot_override_usd
        if scenario.brent_spot_override_usd is not None
        else calib.brent_spot
    )
    usdrub_spot = calib.usdrub_spot * scenario.fx_shift_multiplier
    return dataclasses.replace(calib, brent_spot=brent_spot, usdrub_spot=usdrub_spot)
