"""
Generacion de perfiles ambientales y de uso.
"""

from typing import Dict

import numpy as np

from app.config import EnvironmentConfig, SimulationConfig


class EnvironmentProfile:
    """Genera perfiles temporales reproducibles para la simulacion."""

    def __init__(self, sim_config: SimulationConfig, env_config: EnvironmentConfig):
        self.sim_config = sim_config
        self.env_config = env_config
        self.rng = np.random.RandomState(sim_config.random_seed)
        self.num_steps = sim_config.num_steps
        self.dt = sim_config.time_step_hours
        self.time_hours = np.arange(self.num_steps) * self.dt

        self.temperature_outdoor = self._generate_outdoor_temperature()
        self.humidity = self._generate_humidity()
        self.solar_radiation = self._generate_solar_radiation()
        self.occupancy = self._generate_occupancy()
        self.tariff = self._generate_tariff()
        self.tariff_normalized = self._normalize_tariff()
        self.base_consumption = self._generate_base_consumption()
        self.room_temperature = self._generate_room_temperature_proxy()
        self.door_openings = self._generate_door_openings()
        self.fridge_load = self._generate_fridge_load()

    def _generate_outdoor_temperature(self) -> np.ndarray:
        cfg = self.env_config
        hour_of_day = self.time_hours % 24
        phase = 2 * np.pi * (hour_of_day - cfg.temp_peak_hour + 6) / 24
        base = cfg.temp_mean + cfg.temp_amplitude * np.sin(phase)
        noise = self.rng.normal(0, cfg.temp_noise_std, self.num_steps)
        kernel_size = max(1, int(3 / self.dt))
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            noise = np.convolve(noise, kernel, mode="same")
        return base + noise

    def _generate_humidity(self) -> np.ndarray:
        cfg = self.env_config
        hour_of_day = self.time_hours % 24
        phase = 2 * np.pi * (hour_of_day - cfg.temp_peak_hour + 6) / 24
        base = cfg.humidity_mean - cfg.humidity_amplitude * np.sin(phase)
        noise = self.rng.normal(0, cfg.humidity_noise_std, self.num_steps)
        return np.clip(base + noise, 0.05, 0.99)

    def _generate_solar_radiation(self) -> np.ndarray:
        cfg = self.env_config
        hour_of_day = self.time_hours % 24
        duration = cfg.sunset_hour - cfg.sunrise_hour
        radiation = np.zeros(self.num_steps)
        for idx, hour in enumerate(hour_of_day):
            if cfg.sunrise_hour <= hour <= cfg.sunset_hour:
                t_norm = (hour - cfg.sunrise_hour) / duration
                radiation[idx] = cfg.solar_max * np.sin(np.pi * t_norm) ** 2
        cloud_factor = 1.0 + self.rng.uniform(-0.15, 0.10, self.num_steps)
        return radiation * np.clip(cloud_factor, 0.0, None)

    def _generate_occupancy(self) -> np.ndarray:
        cfg = self.env_config
        max_occ = cfg.max_occupants
        hour_of_day = self.time_hours % 24
        occupancy = np.zeros(self.num_steps)

        for idx, hour in enumerate(hour_of_day):
            if 0 <= hour < 6:
                base = max_occ * 0.95
            elif 6 <= hour < 8:
                base = max_occ * 0.75
            elif 8 <= hour < 13:
                base = max_occ * 0.15
            elif 13 <= hour < 15:
                base = max_occ * 0.40
            elif 15 <= hour < 18:
                base = max_occ * 0.20
            elif 18 <= hour < 22:
                base = max_occ * 0.85
            else:
                base = max_occ * 0.90
            occupancy[idx] = np.clip(base + self.rng.choice([-0.3, 0.0, 0.3]), 0, max_occ)
        return np.round(occupancy, 1)

    def _generate_tariff(self) -> np.ndarray:
        cfg = self.env_config
        hour_of_day = self.time_hours % 24
        tariff = np.zeros(self.num_steps)
        for idx, hour in enumerate(hour_of_day):
            if 0 <= hour < 8:
                tariff[idx] = cfg.tariff_off_peak
            elif 8 <= hour < 10:
                tariff[idx] = cfg.tariff_mid_peak
            elif 10 <= hour < 14:
                tariff[idx] = cfg.tariff_on_peak
            elif 14 <= hour < 18:
                tariff[idx] = cfg.tariff_mid_peak
            elif 18 <= hour < 22:
                tariff[idx] = cfg.tariff_on_peak
            else:
                tariff[idx] = cfg.tariff_mid_peak
        return tariff

    def _normalize_tariff(self) -> np.ndarray:
        low = self.env_config.tariff_off_peak
        high = self.env_config.tariff_on_peak
        if abs(high - low) < 1e-9:
            return np.zeros(self.num_steps)
        return (self.tariff - low) / (high - low)

    def _generate_base_consumption(self) -> np.ndarray:
        cfg = self.env_config
        hour_of_day = self.time_hours % 24
        c_min = cfg.base_consumption_min
        c_range = cfg.base_consumption_max - cfg.base_consumption_min
        consumption = np.zeros(self.num_steps)

        for idx, hour in enumerate(hour_of_day):
            if 0 <= hour < 6:
                factor = 0.08
            elif 6 <= hour < 9:
                factor = 0.60
            elif 9 <= hour < 13:
                factor = 0.25
            elif 13 <= hour < 15:
                factor = 0.55
            elif 15 <= hour < 18:
                factor = 0.20
            elif 18 <= hour < 22:
                factor = 0.95
            else:
                factor = 0.22
            consumption[idx] = c_min + c_range * np.clip(factor + self.rng.uniform(-0.05, 0.05), 0, 1)
        return consumption

    def _generate_room_temperature_proxy(self) -> np.ndarray:
        baseline = 22.5 + 0.18 * (self.temperature_outdoor - 22.5)
        occupancy_gain = 0.15 * self.occupancy
        return np.clip(baseline + occupancy_gain, 18.0, 32.0)

    def _generate_door_openings(self) -> np.ndarray:
        hour_of_day = self.time_hours % 24
        openings = np.zeros(self.num_steps)
        for idx, hour in enumerate(hour_of_day):
            if 6 <= hour < 9 or 12 <= hour < 15 or 18 <= hour < 22:
                base = 0.70
            elif 0 <= hour < 5:
                base = 0.10
            else:
                base = 0.25
            noise = self.rng.uniform(-0.10, 0.10)
            openings[idx] = np.clip(base + noise, 0.0, 1.0)
        return openings

    def _generate_fridge_load(self) -> np.ndarray:
        hour_of_day = self.time_hours % 24
        load = np.zeros(self.num_steps)
        for idx, hour in enumerate(hour_of_day):
            if 10 <= hour < 14:
                base = 0.75
            elif 18 <= hour < 21:
                base = 0.85
            elif 0 <= hour < 6:
                base = 0.25
            else:
                base = 0.45
            load[idx] = np.clip(base + self.rng.uniform(-0.08, 0.08), 0.0, 1.0)
        return load

    def get_state_at(self, step: int) -> Dict[str, float]:
        return {
            "time_hours": self.time_hours[step],
            "hour_of_day": self.time_hours[step] % 24,
            "temperature_outdoor": self.temperature_outdoor[step],
            "room_temperature": self.room_temperature[step],
            "humidity": self.humidity[step],
            "solar_radiation": self.solar_radiation[step],
            "occupancy": self.occupancy[step],
            "tariff": self.tariff[step],
            "tariff_normalized": self.tariff_normalized[step],
            "base_consumption": self.base_consumption[step],
            "door_openings": self.door_openings[step],
            "load_level": self.fridge_load[step],
        }

    def get_scenario_name(self) -> str:
        return (
            f"{self.sim_config.scenario_type}_"
            f"{self.sim_config.horizon_hours}h_"
            f"seed{self.sim_config.random_seed}"
        )
