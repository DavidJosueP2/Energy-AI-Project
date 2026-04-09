# ==============================================================================
# metrics.py - Cálculo de métricas de evaluación
# ==============================================================================
"""
Calcula métricas de desempeño para evaluar la calidad del controlador:
- Consumo energético total (kWh)
- Costo total ($)
- Tiempo en zona de confort (%)
- Pico de demanda máxima (kW)
- Suavidad/variabilidad del control
- Score global de fitness multiobjetivo

Las métricas se normalizan para permitir comparación justa entre
controladores y para su uso en la función de fitness del GA.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from app.config import MetricsConfig, SimulationConfig


@dataclass
class PerformanceMetrics:
    """Métricas completas de desempeño de una simulación."""
    
    # Energía
    total_energy_kwh: float = 0.0
    hvac_energy_kwh: float = 0.0
    base_energy_kwh: float = 0.0
    avg_consumption_kw: float = 0.0
    
    # Costo
    total_cost: float = 0.0
    avg_hourly_cost: float = 0.0
    
    # Demanda
    peak_demand_kw: float = 0.0
    avg_demand_kw: float = 0.0
    
    # Confort
    comfort_percentage: float = 0.0  # % del tiempo en zona de confort
    avg_comfort_index: float = 0.0
    avg_temp_deviation: float = 0.0
    max_temp_deviation: float = 0.0
    
    # Control
    avg_hvac_level: float = 0.0
    control_variability: float = 0.0  # desviación estándar de cambios en HVAC
    hvac_active_percentage: float = 0.0  # % del tiempo con HVAC encendido
    
    # Score global
    fitness_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convierte métricas a diccionario."""
        return {
            'Energía Total (kWh)': round(self.total_energy_kwh, 2),
            'Energía HVAC (kWh)': round(self.hvac_energy_kwh, 2),
            'Energía Base (kWh)': round(self.base_energy_kwh, 2),
            'Consumo Promedio (kW)': round(self.avg_consumption_kw, 3),
            'Costo Total ($)': round(self.total_cost, 2),
            'Costo Promedio/Hora ($)': round(self.avg_hourly_cost, 4),
            'Pico Demanda (kW)': round(self.peak_demand_kw, 3),
            'Demanda Promedio (kW)': round(self.avg_demand_kw, 3),
            'Confort (%)': round(self.comfort_percentage, 1),
            'Índice Confort Promedio': round(self.avg_comfort_index, 3),
            'Desviación Temp Promedio (°C)': round(self.avg_temp_deviation, 2),
            'Desviación Temp Máxima (°C)': round(self.max_temp_deviation, 2),
            'Nivel HVAC Promedio (%)': round(self.avg_hvac_level, 1),
            'Variabilidad Control': round(self.control_variability, 3),
            'HVAC Activo (%)': round(self.hvac_active_percentage, 1),
            'Fitness Score': round(self.fitness_score, 4),
        }


def calculate_metrics(df: pd.DataFrame, 
                       sim_config: SimulationConfig,
                       metrics_config: MetricsConfig) -> PerformanceMetrics:
    """
    Calcula todas las métricas de desempeño a partir de los datos de simulación.
    
    Args:
        df: DataFrame con resultados de simulación.
        sim_config: Configuración de simulación.
        metrics_config: Configuración de métricas y pesos.
        
    Returns:
        PerformanceMetrics con todos los indicadores calculados.
    """
    dt = sim_config.time_step_hours
    m = PerformanceMetrics()
    
    if df.empty:
        return m
    
    # === MÉTRICAS DE ENERGÍA ===
    m.total_energy_kwh = float(df['total_consumption_kw'].sum() * dt)
    m.hvac_energy_kwh = float(df['hvac_consumption_kw'].sum() * dt)
    m.base_energy_kwh = float(df['base_consumption_kw'].sum() * dt)
    m.avg_consumption_kw = float(df['total_consumption_kw'].mean())
    
    # === MÉTRICAS DE COSTO ===
    m.total_cost = float(df['step_cost'].sum())
    duration_hours = sim_config.horizon_hours
    m.avg_hourly_cost = m.total_cost / max(duration_hours, 1)
    
    # === MÉTRICAS DE DEMANDA ===
    m.peak_demand_kw = float(df['total_consumption_kw'].max())
    m.avg_demand_kw = float(df['total_consumption_kw'].mean())
    
    # === MÉTRICAS DE CONFORT ===
    target_temp = sim_config.target_temperature
    comfort_range = sim_config.comfort_range
    comfort_min = target_temp - comfort_range
    comfort_max = target_temp + comfort_range
    
    in_comfort = ((df['temperature_indoor'] >= comfort_min) & 
                  (df['temperature_indoor'] <= comfort_max))
    m.comfort_percentage = float(in_comfort.sum() / len(df) * 100)
    m.avg_comfort_index = float(df['comfort_index'].mean())
    m.avg_temp_deviation = float(df['temp_deviation'].mean())
    m.max_temp_deviation = float(df['temp_deviation'].max())
    
    # === MÉTRICAS DE CONTROL ===
    m.avg_hvac_level = float(df['hvac_level'].mean())
    
    # Variabilidad: desviación estándar de los cambios en el nivel HVAC
    hvac_changes = df['hvac_level'].diff().dropna()
    m.control_variability = float(hvac_changes.std()) if len(hvac_changes) > 0 else 0.0
    
    # Porcentaje del tiempo con HVAC activo (nivel > 5%)
    m.hvac_active_percentage = float((df['hvac_level'] > 5.0).sum() / len(df) * 100)
    
    # === FITNESS SCORE ===
    m.fitness_score = calculate_fitness(df, sim_config, metrics_config)
    
    return m


def calculate_fitness(df: pd.DataFrame,
                       sim_config: SimulationConfig,
                       metrics_config: MetricsConfig) -> float:
    """
    Calcula el score de fitness multiobjetivo.
    
    La función combina múltiples objetivos normalizados:
    
    fitness = w1·confort_norm + w2·(1 - energia_norm) + w3·(1 - costo_norm)
              - w4·pico_norm - w5·variabilidad_norm
    
    Normalización:
    - confort_norm: promedio del índice de confort [0, 1]
    - energia_norm: energía HVAC relativa al máximo teórico
    - costo_norm: costo relativo al máximo teórico
    - pico_norm: pico de demanda relativo a la capacidad máxima
    - variabilidad_norm: variabilidad del control normalizada
    
    Args:
        df: DataFrame con resultados de simulación.
        sim_config: Configuración de simulación.
        metrics_config: Configuración con pesos de fitness.
        
    Returns:
        Score de fitness (mayor es mejor).
    """
    if df.empty:
        return 0.0
    
    dt = sim_config.time_step_hours
    mcfg = metrics_config
    
    # --- Componente de confort ---
    # Promedio del índice de confort [0, 1]
    comfort_norm = float(df['comfort_index'].mean())
    
    # --- Componente de energía ---
    # Normalizar respecto al consumo máximo teórico (HVAC al 100% todo el tiempo)
    # Esto permite comparar diferentes duraciones de simulación
    max_hvac_energy = 3.5 / 3.2 * sim_config.horizon_hours  # kWh a potencia máxima
    hvac_energy = float(df['hvac_consumption_kw'].sum() * dt)
    energy_norm = min(hvac_energy / max(max_hvac_energy, 0.1), 1.0)
    
    # --- Componente de costo ---
    max_cost = (3.5 / 3.2 + 2.0) * 0.30 * sim_config.horizon_hours  # Escenario peor
    total_cost = float(df['step_cost'].sum())
    cost_norm = min(total_cost / max(max_cost, 0.1), 1.0)
    
    # --- Componente de pico de demanda ---
    peak = float(df['total_consumption_kw'].max())
    max_possible_peak = 3.5 / 3.2 + 2.0 + 0.15  # HVAC max + base max + standby
    peak_norm = min(peak / max(max_possible_peak, 0.1), 1.0)
    
    # --- Componente de variabilidad ---
    hvac_changes = df['hvac_level'].diff().dropna().abs()
    variability = float(hvac_changes.mean()) if len(hvac_changes) > 0 else 0.0
    # Normalizar: un cambio medio de 50% sería variabilidad = 1.0
    variability_norm = min(variability / 50.0, 1.0)
    
    # --- Combinar con pesos ---
    fitness = (
        mcfg.weight_comfort * comfort_norm
        + mcfg.weight_energy * (1.0 - energy_norm)
        + mcfg.weight_cost * (1.0 - cost_norm)
        - mcfg.weight_peak * peak_norm
        - mcfg.weight_variability * variability_norm
    )
    
    return float(fitness)


def compare_metrics(base: PerformanceMetrics, 
                     optimized: PerformanceMetrics) -> Dict[str, Dict[str, float]]:
    """
    Compara métricas entre controlador base y optimizado.
    
    Returns:
        Diccionario con valores base, optimizado y mejora porcentual.
    """
    comparison = {}
    base_dict = base.to_dict()
    opt_dict = optimized.to_dict()
    
    for key in base_dict:
        base_val = base_dict[key]
        opt_val = opt_dict[key]
        
        # Calcular mejora porcentual
        if base_val != 0:
            improvement = ((opt_val - base_val) / abs(base_val)) * 100
        else:
            improvement = 0.0
        
        comparison[key] = {
            'base': base_val,
            'optimizado': opt_val,
            'cambio_%': round(improvement, 2),
        }
    
    return comparison
