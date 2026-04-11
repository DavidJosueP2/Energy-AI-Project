"""
Motor de simulacion temporal multi-dispositivo.
"""

from dataclasses import replace
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config import AppConfig
from app.simulation.devices import ControlledDevice, build_device_definition, build_runtime_dynamics_config
from app.simulation.environment import EnvironmentProfile
from app.simulation.scenario_generator import get_scenario_configs


class SimulationResult:
    """Encapsula los resultados completos de una simulacion."""

    def __init__(self, data: pd.DataFrame, config: AppConfig, label: str = "base"):
        self.data = data
        self.config = config
        self.label = label

    def to_csv(self, filepath: str):
        self.data.to_csv(filepath, index=False, float_format="%.4f")

    @property
    def num_steps(self) -> int:
        return len(self.data)

    @property
    def duration_hours(self) -> float:
        return float(self.data["time_hours"].iloc[-1]) if not self.data.empty else 0.0


class Simulator:
    """Ejecuta una simulacion temporal usando el controlador difuso activo.

    El flujo matematico por paso es:

    1. leer el entorno en el instante k;
    2. construir las entradas crisp del controlador difuso;
    3. obtener un nivel de control u(k) en [0, 100];
    4. actualizar la temperatura del dispositivo con el modelo dinamico;
    5. calcular consumo, costo y confort;
    6. registrar el estado en una fila del DataFrame.
    """

    def __init__(self, config: AppConfig):
        self.config = config

    def run(
        self,
        controller_fn: Callable[[Dict[str, float]], float],
        label: str = "base",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SimulationResult:
        cfg = self.config
        sim_cfg = cfg.simulation
        sim_c, env_c = get_scenario_configs(
            scenario_type=sim_cfg.scenario_type,
            horizon_hours=sim_cfg.horizon_hours,
            seed=sim_cfg.random_seed,
            target_temp=sim_cfg.target_temperature,
        )
        sim_c.time_step_hours = sim_cfg.time_step_hours
        environment = EnvironmentProfile(sim_c, env_c)

        definition = build_device_definition(sim_cfg.device_key)
        dynamics_config = build_runtime_dynamics_config(sim_cfg.device_key, self.config.house)
        device = ControlledDevice(
            definition=definition,
            target_temperature=sim_cfg.target_temperature,
            comfort_range=sim_cfg.comfort_range,
            dt=sim_cfg.time_step_hours,
            dynamics_config=dynamics_config,
        )
        target_temp = device.target_temperature
        comfort_range = device.comfort_range

        records: List[Dict] = []
        cumulative_cost = 0.0
        cumulative_energy = 0.0

        for step in range(sim_c.num_steps):
            env_state = environment.get_state_at(step)
            # Entradas crisp que luego seran fuzzificadas por el controlador.
            controller_inputs = self._build_controller_inputs(device, env_state)

            try:
                control_level = float(np.clip(controller_fn(controller_inputs), 0.0, 100.0))
            except Exception:
                control_level = 0.0

            device_state = device.step(
                ambient_temperature=self._resolve_ambient_temperature(sim_cfg.device_key, env_state),
                occupancy=env_state["occupancy"] if sim_cfg.device_key == "hvac" else 0.0,
                solar_radiation=env_state["solar_radiation"] if sim_cfg.device_key == "hvac" else 0.0,
                usage_load=env_state["door_openings"] + 0.35 * env_state["load_level"] if sim_cfg.device_key == "refrigerador" else 0.0,
                control_level=control_level,
            )

            # Balance electrico horario:
            #   P_total = P_base + P_dispositivo
            total_consumption = env_state["base_consumption"] + device_state["device_consumption_kw"]
            # Costo horario:
            #   costo_step = P_total * tarifa * Delta_t
            step_cost = total_consumption * env_state["tariff"] * sim_cfg.time_step_hours
            cumulative_cost += step_cost
            cumulative_energy += total_consumption * sim_cfg.time_step_hours

            # El confort se evalua respecto a una banda alrededor del setpoint.
            temp_deviation = abs(device_state["device_temperature"] - target_temp)
            comfort_index = self._compute_comfort_index(temp_deviation, comfort_range)

            record = {
                "step": step,
                "time_hours": env_state["time_hours"],
                "hour_of_day": env_state["hour_of_day"],
                "device_key": sim_cfg.device_key,
                "device_display_name": device.display_name,
                "ambient_temperature": self._resolve_ambient_temperature(sim_cfg.device_key, env_state),
                "temperature_outdoor": env_state["temperature_outdoor"],
                "room_temperature": env_state["room_temperature"],
                "humidity": env_state["humidity"],
                "occupancy": env_state["occupancy"],
                "solar_radiation": env_state["solar_radiation"],
                "door_openings": env_state["door_openings"],
                "load_level": env_state["load_level"],
                "tariff": env_state["tariff"],
                "tariff_normalized": env_state["tariff_normalized"],
                "base_consumption_kw": env_state["base_consumption"],
                "device_temperature": device_state["device_temperature"],
                "target_temperature": target_temp,
                "comfort_range": comfort_range,
                "temp_error": device.get_temp_error(),
                "control_level": control_level,
                "control_mode": device_state.get("control_mode", "idle"),
                "device_power_level": device_state["device_power_level"],
                "device_consumption_kw": device_state["device_consumption_kw"],
                "total_consumption_kw": total_consumption,
                "step_cost": step_cost,
                "cumulative_cost": cumulative_cost,
                "cumulative_energy_kwh": cumulative_energy,
                "comfort_index": comfort_index,
                "temp_deviation": temp_deviation,
                # Aliases de compatibilidad con la capa existente
                "temperature_indoor": device_state["device_temperature"],
                "hvac_level": control_level,
                "hvac_power_level": device_state["device_power_level"],
                "hvac_consumption_kw": device_state["device_consumption_kw"],
                "consumption_normalized": min(env_state["base_consumption"] / max(env_c.base_consumption_max, 0.1), 1.0),
            }
            records.append(record)

            if progress_callback and step % max(1, sim_c.num_steps // 50) == 0:
                progress_callback(step, sim_c.num_steps)

        return SimulationResult(pd.DataFrame(records), cfg, label)

    def _build_controller_inputs(self, device: ControlledDevice, env_state: Dict[str, float]) -> Dict[str, float]:
        """Construye las entradas crisp del sistema difuso.

        Refrigerador:
            x = [T_interna, aperturas, carga, tarifa_norm]

        HVAC:
            x = [|T_interior - T_objetivo|, humedad, ocupacion, tarifa_norm]

        En HVAC se conserva tambien ``raw_temp_error`` para trazabilidad y para
        ajustar la sensibilidad segun ``comfort_range`` dentro del controlador.
        """
        if self.config.simulation.device_key == "refrigerador":
            return {
                "device_temperature": device.temperature,
                "door_openings": env_state["door_openings"],
                "load_level": env_state["load_level"],
                "tariff_normalized": env_state["tariff_normalized"],
                "tariff": env_state["tariff_normalized"],
            }

        raw_temp_error = device.get_temp_error()
        return {
            "temp_error": abs(raw_temp_error),
            "raw_temp_error": raw_temp_error,
            "comfort_range": device.comfort_range,
            "humidity": env_state["humidity"],
            "occupancy": env_state["occupancy"],
            "tariff_normalized": env_state["tariff_normalized"],
            "tariff": env_state["tariff_normalized"],
            "temperature_indoor": device.temperature,
            "temperature_outdoor": env_state["temperature_outdoor"],
            "target_temperature": device.target_temperature,
        }

    def _resolve_ambient_temperature(self, device_key: str, env_state: Dict[str, float]) -> float:
        """Selecciona la temperatura ambiente efectiva del dispositivo."""
        if device_key == "refrigerador":
            return float(env_state["room_temperature"])
        return float(env_state["temperature_outdoor"])

    @staticmethod
    def _compute_comfort_index(temp_deviation: float, comfort_range: float) -> float:
        """Calcula un indice de confort simple en [0, 1].

        - Dentro de la banda de confort: 1.0
        - Fuera de la banda: decae linealmente con la desviacion adicional
        """
        if temp_deviation <= comfort_range:
            return 1.0
        return max(0.0, 1.0 - (temp_deviation - comfort_range) / 5.0)


def run_baseline_simulation(
    config: AppConfig,
    controller_fn: Callable,
    progress_callback: Optional[Callable] = None,
) -> SimulationResult:
    return Simulator(config).run(controller_fn, label="base", progress_callback=progress_callback)


def run_optimized_simulation(
    config: AppConfig,
    controller_fn: Callable,
    progress_callback: Optional[Callable] = None,
) -> SimulationResult:
    return Simulator(config).run(controller_fn, label="optimizado", progress_callback=progress_callback)
