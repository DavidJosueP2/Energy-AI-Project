# ==============================================================================
# scenario_generator.py - Generador de escenarios predefinidos
# ==============================================================================
"""
Genera configuraciones de escenarios ambientales para diferentes estaciones
y condiciones. Cada escenario ajusta los parámetros del ambiente para
representar condiciones realistas.
"""

from app.config import EnvironmentConfig, SimulationConfig
from typing import Tuple


def get_scenario_configs(scenario_type: str, 
                         horizon_hours: int = 72,
                         seed: int = 42,
                         target_temp: float = 22.0) -> Tuple[SimulationConfig, EnvironmentConfig]:
    """
    Genera configuraciones para un escenario específico.
    
    Args:
        scenario_type: Tipo de escenario ('verano', 'invierno', 'primavera', 'mixto').
        horizon_hours: Duración de la simulación en horas.
        seed: Semilla para reproducibilidad.
        target_temp: Temperatura objetivo de confort.
        
    Returns:
        Tupla (SimulationConfig, EnvironmentConfig) configurados para el escenario.
    """
    sim_config = SimulationConfig(
        horizon_hours=horizon_hours,
        random_seed=seed,
        target_temperature=target_temp,
        scenario_type=scenario_type,
    )
    
    if scenario_type == 'verano':
        env_config = _summer_scenario()
    elif scenario_type == 'invierno':
        env_config = _winter_scenario()
    elif scenario_type == 'primavera':
        env_config = _spring_scenario()
    elif scenario_type == 'mixto':
        env_config = _mixed_scenario()
    else:
        # Escenario por defecto: verano
        env_config = _summer_scenario()
    
    return sim_config, env_config


def _summer_scenario() -> EnvironmentConfig:
    """
    Escenario de verano caluroso.
    Temperaturas altas, fuerte radiación solar, tarifas punta elevadas.
    """
    return EnvironmentConfig(
        temp_mean=32.0,
        temp_amplitude=8.0,
        temp_peak_hour=15.0,
        temp_noise_std=0.9,
        humidity_mean=0.65,
        humidity_amplitude=0.15,
        humidity_noise_std=0.04,
        solar_max=950.0,
        sunrise_hour=6.0,
        sunset_hour=20.0,
        max_occupants=4,
        tariff_off_peak=0.07,
        tariff_mid_peak=0.14,
        tariff_on_peak=0.30,
        base_consumption_min=0.4,
        base_consumption_max=1.9,
    )


def _winter_scenario() -> EnvironmentConfig:
    """
    Escenario de invierno frío.
    Temperaturas bajas, poca radiación solar, noches largas.
    Nota: el sistema HVAC en este modelo refrigera; en invierno el controlador
    debería reducir al mínimo su acción (la casa necesitaría calefacción).
    """
    return EnvironmentConfig(
        temp_mean=12.0,
        temp_amplitude=5.0,
        temp_peak_hour=14.0,
        temp_noise_std=1.0,
        humidity_mean=0.70,
        humidity_amplitude=0.10,
        humidity_noise_std=0.05,
        solar_max=450.0,
        sunrise_hour=7.5,
        sunset_hour=17.5,
        max_occupants=4,
        tariff_off_peak=0.09,
        tariff_mid_peak=0.16,
        tariff_on_peak=0.26,
        base_consumption_min=0.5,
        base_consumption_max=2.0,
    )


def _spring_scenario() -> EnvironmentConfig:
    """
    Escenario de primavera templada.
    Temperaturas moderadas, radiación media. Clima más benigno.
    """
    return EnvironmentConfig(
        temp_mean=24.0,
        temp_amplitude=6.0,
        temp_peak_hour=15.0,
        temp_noise_std=0.7,
        humidity_mean=0.50,
        humidity_amplitude=0.18,
        humidity_noise_std=0.04,
        solar_max=700.0,
        sunrise_hour=6.5,
        sunset_hour=19.0,
        max_occupants=4,
        tariff_off_peak=0.08,
        tariff_mid_peak=0.15,
        tariff_on_peak=0.28,
        base_consumption_min=0.3,
        base_consumption_max=1.6,
    )


def _mixed_scenario() -> EnvironmentConfig:
    """
    Escenario mixto con condiciones variables.
    Simula transición estacional con temperaturas cambiantes.
    """
    return EnvironmentConfig(
        temp_mean=27.0,
        temp_amplitude=9.0,
        temp_peak_hour=15.5,
        temp_noise_std=1.2,
        humidity_mean=0.55,
        humidity_amplitude=0.22,
        humidity_noise_std=0.06,
        solar_max=750.0,
        sunrise_hour=6.5,
        sunset_hour=19.5,
        max_occupants=4,
        tariff_off_peak=0.08,
        tariff_mid_peak=0.15,
        tariff_on_peak=0.28,
        base_consumption_min=0.4,
        base_consumption_max=1.8,
    )


AVAILABLE_SCENARIOS = ['verano', 'invierno', 'primavera', 'mixto']
