"""
Modelos dinamicos genericos para HVAC y refrigerador.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class DeviceConfig:
    """Configuracion fisica simplificada de un dispositivo controlado."""

    name: str
    display_name: str
    control_display_name: str
    temperature_display_name: str
    target_temperature: float
    comfort_range: float
    initial_temperature: float
    min_temperature: float
    max_temperature: float
    ambient_coupling: float
    occupancy_gain: float
    solar_gain: float
    usage_gain: float
    control_gain: float
    max_power_kw: float
    standby_kw: float
    cop: float


def get_hvac_config() -> DeviceConfig:
    """Configuracion calibrada para climatizacion residencial."""
    return DeviceConfig(
        name="hvac",
        display_name="HVAC (Climatizacion)",
        control_display_name="Nivel de climatizacion",
        temperature_display_name="Temperatura interior",
        target_temperature=22.0,
        comfort_range=2.0,
        initial_temperature=25.5,
        min_temperature=14.0,
        max_temperature=45.0,
        ambient_coupling=0.055,
        occupancy_gain=0.12,
        solar_gain=0.008,
        usage_gain=0.0,
        control_gain=1.85,
        max_power_kw=3.5,
        standby_kw=0.12,
        cop=3.2,
    )


def get_refrigerator_config() -> DeviceConfig:
    """Configuracion para refrigerador domestico."""
    return DeviceConfig(
        name="refrigerador",
        display_name="Refrigerador",
        control_display_name="Nivel de enfriamiento",
        temperature_display_name="Temperatura interna del refrigerador",
        target_temperature=4.0,
        comfort_range=1.5,
        initial_temperature=5.5,
        min_temperature=-3.0,
        max_temperature=18.0,
        ambient_coupling=0.060,
        occupancy_gain=0.0,
        solar_gain=0.0,
        usage_gain=0.85,
        control_gain=4.60,
        max_power_kw=0.22,
        standby_kw=0.02,
        cop=2.3,
    )


AVAILABLE_DEVICES = {
    "hvac": get_hvac_config,
    "refrigerador": get_refrigerator_config,
}


class ControlledDevice:
    """Dispositivo termico controlado por una salida difusa en [0, 100]."""

    def __init__(self, config: DeviceConfig, dt: float = 1.0):
        self.config = config
        self.dt = dt
        self.reset()

    def reset(self, initial_temp: Optional[float] = None):
        self.temperature = initial_temp if initial_temp is not None else self.config.initial_temperature
        self.power_level = 0.0
        self.consumption_kw = 0.0

    def step(
        self,
        ambient_temperature: float,
        occupancy: float,
        solar_radiation: float,
        usage_load: float = 0.0,
        control_level: Optional[float] = None,
    ) -> Dict[str, float]:
        cfg = self.config
        if control_level is None:
            control_level = usage_load
            usage_load = 0.0
        ctrl_norm = float(np.clip(control_level / 100.0, 0.0, 1.0))
        self.power_level = ctrl_norm

        delta_ambient = cfg.ambient_coupling * self.dt * (ambient_temperature - self.temperature)
        delta_occupancy = cfg.occupancy_gain * self.dt * occupancy
        delta_solar = cfg.solar_gain * self.dt * (solar_radiation / 1000.0) * 10.0
        delta_usage = cfg.usage_gain * self.dt * usage_load
        delta_control = cfg.control_gain * self.dt * ctrl_norm

        self.temperature += delta_ambient + delta_occupancy + delta_solar + delta_usage - delta_control
        self.temperature = float(np.clip(self.temperature, cfg.min_temperature, cfg.max_temperature))

        if ctrl_norm > 0.01:
            self.consumption_kw = cfg.max_power_kw * ctrl_norm / max(cfg.cop, 0.1) + cfg.standby_kw
        else:
            self.consumption_kw = 0.0

        return self.get_state()

    def get_state(self) -> Dict[str, float]:
        return {
            "device_temperature": round(self.temperature, 3),
            "device_power_level": round(self.power_level, 4),
            "device_consumption_kw": round(self.consumption_kw, 4),
        }

    def get_temp_error(self) -> float:
        return self.temperature - self.config.target_temperature

    @property
    def display_name(self) -> str:
        return self.config.display_name

    @property
    def name(self) -> str:
        return self.config.name
