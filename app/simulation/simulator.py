# ==============================================================================
# simulator.py - Motor de simulación principal
# ==============================================================================
"""
Orquesta la simulación completa de la vivienda inteligente.
Integra el modelo térmico, el ambiente y el controlador difuso para
ejecutar la simulación paso a paso y registrar todos los resultados.
"""

import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict, List

from app.config import AppConfig
from app.simulation.environment import EnvironmentProfile
from app.simulation.house_model import HouseModel
from app.simulation.scenario_generator import get_scenario_configs


class SimulationResult:
    """Encapsula los resultados completos de una simulación."""

    def __init__(self, data: pd.DataFrame, config: AppConfig, label: str = "base"):
        self.data = data
        self.config = config
        self.label = label

    def to_csv(self, filepath: str):
        """Exporta resultados a CSV."""
        self.data.to_csv(filepath, index=False, float_format='%.4f')

    @property
    def num_steps(self) -> int:
        return len(self.data)

    @property
    def duration_hours(self) -> float:
        return self.data['time_hours'].iloc[-1]


class Simulator:
    """
    Motor de simulación que ejecuta el bucle principal.
    
    Flujo por cada paso temporal:
    1. Obtener estado ambiental (temperatura, radiación, ocupación, tarifa)
    2. Preparar entradas para el controlador difuso
    3. Evaluar controlador → obtener nivel de climatización
    4. Aplicar al modelo térmico → actualizar temperatura interior
    5. Calcular consumos y costos
    6. Registrar todos los datos
    """

    def __init__(self, config: AppConfig):
        self.config = config

    def run(self,
            controller_fn: Callable[[Dict[str, float]], float],
            label: str = "base",
            progress_callback: Optional[Callable[[int, int], None]] = None
            ) -> SimulationResult:
        """
        Ejecuta la simulación completa.
        
        Args:
            controller_fn: Función que recibe un dict con las variables de entrada
                          y retorna el nivel de climatización [0, 100].
            label: Etiqueta para identificar esta simulación.
            progress_callback: Función opcional (step, total) para reportar progreso.
            
        Returns:
            SimulationResult con todos los datos registrados.
        """
        cfg = self.config
        sim_cfg = cfg.simulation
        
        # Generar escenario usando la configuración
        sim_c, env_c = get_scenario_configs(
            scenario_type=sim_cfg.scenario_type,
            horizon_hours=sim_cfg.horizon_hours,
            seed=sim_cfg.random_seed,
            target_temp=sim_cfg.target_temperature,
        )
        # Sobrescribir con configuración completa
        sim_c.time_step_hours = sim_cfg.time_step_hours
        
        # Crear perfil ambiental
        environment = EnvironmentProfile(sim_c, env_c)
        
        # Crear modelo de vivienda
        house = HouseModel(cfg.house, dt=sim_cfg.time_step_hours)
        house.reset()
        
        # Registros de simulación
        records: List[Dict] = []
        
        # Acumuladores
        cumulative_cost = 0.0
        cumulative_energy = 0.0
        
        num_steps = sim_c.num_steps
        target_temp = sim_cfg.target_temperature
        comfort_range = sim_cfg.comfort_range
        
        for step in range(num_steps):
            # 1. Estado ambiental
            env_state = environment.get_state_at(step)
            
            # 2. Preparar entradas del controlador
            temp_error = house.temperature_indoor - target_temp
            
            # Normalizar consumo base respecto al máximo del escenario
            max_base = env_c.base_consumption_max
            consumption_normalized = min(env_state['base_consumption'] / max_base, 1.0) if max_base > 0 else 0.0
            
            controller_inputs = {
                'temp_error': temp_error,
                'temperature_indoor': house.temperature_indoor,
                'temperature_outdoor': env_state['temperature_outdoor'],
                'occupancy': env_state['occupancy'],
                'tariff_normalized': env_state['tariff_normalized'],
                'consumption_normalized': consumption_normalized,
                'target_temperature': target_temp,
            }
            
            # 3. Evaluar controlador
            try:
                hvac_level = controller_fn(controller_inputs)
                hvac_level = float(np.clip(hvac_level, 0.0, 100.0))
            except Exception:
                hvac_level = 0.0
            
            # 4. Actualizar modelo térmico
            house_state = house.step(
                temp_outdoor=env_state['temperature_outdoor'],
                occupancy=env_state['occupancy'],
                solar_radiation=env_state['solar_radiation'],
                hvac_level=hvac_level,
            )
            
            # 5. Calcular consumos y costos
            total_consumption = env_state['base_consumption'] + house_state['hvac_consumption_kw']
            step_cost = total_consumption * env_state['tariff'] * sim_cfg.time_step_hours
            cumulative_cost += step_cost
            cumulative_energy += total_consumption * sim_cfg.time_step_hours
            
            # Indicador de confort: 1.0 si está en rango, decrece fuera
            temp_deviation = abs(house_state['temperature_indoor'] - target_temp)
            if temp_deviation <= comfort_range:
                comfort_index = 1.0
            else:
                comfort_index = max(0.0, 1.0 - (temp_deviation - comfort_range) / 5.0)
            
            # 6. Registrar
            record = {
                'step': step,
                'time_hours': env_state['time_hours'],
                'hour_of_day': env_state['hour_of_day'],
                'temperature_outdoor': env_state['temperature_outdoor'],
                'temperature_indoor': house_state['temperature_indoor'],
                'temp_error': house_state['temperature_indoor'] - target_temp,
                'occupancy': env_state['occupancy'],
                'solar_radiation': env_state['solar_radiation'],
                'tariff': env_state['tariff'],
                'tariff_normalized': env_state['tariff_normalized'],
                'base_consumption_kw': env_state['base_consumption'],
                'hvac_level': hvac_level,
                'hvac_power_level': house_state['hvac_power_level'],
                'hvac_consumption_kw': house_state['hvac_consumption_kw'],
                'total_consumption_kw': total_consumption,
                'step_cost': step_cost,
                'cumulative_cost': cumulative_cost,
                'cumulative_energy_kwh': cumulative_energy,
                'comfort_index': comfort_index,
                'temp_deviation': temp_deviation,
            }
            records.append(record)
            
            # Reportar progreso
            if progress_callback and step % max(1, num_steps // 50) == 0:
                progress_callback(step, num_steps)
        
        # Crear DataFrame
        df = pd.DataFrame(records)
        
        return SimulationResult(df, self.config, label)


def run_baseline_simulation(config: AppConfig,
                            controller_fn: Callable,
                            progress_callback: Optional[Callable] = None
                            ) -> SimulationResult:
    """Función de conveniencia para ejecutar simulación base."""
    simulator = Simulator(config)
    return simulator.run(controller_fn, label="base", progress_callback=progress_callback)


def run_optimized_simulation(config: AppConfig,
                              controller_fn: Callable,
                              progress_callback: Optional[Callable] = None
                              ) -> SimulationResult:
    """Función de conveniencia para ejecutar simulación optimizada."""
    simulator = Simulator(config)
    return simulator.run(controller_fn, label="optimizado", progress_callback=progress_callback)
