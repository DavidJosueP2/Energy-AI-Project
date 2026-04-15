from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from app.config import MetricsConfig, SimulationConfig
from app.simulation.devices import build_device_definition


HIGHER_IS_BETTER_METRICS = {
    "Confort (%)",
    "Indice de Confort Promedio",
    "Performance Score",
}


@dataclass
class PerformanceMetrics:
    """Metricas sinteticas comparables entre dispositivo base y optimizado."""

    total_energy_kwh: float = 0.0
    device_energy_kwh: float = 0.0
    base_energy_kwh: float = 0.0
    avg_consumption_kw: float = 0.0
    total_cost: float = 0.0
    avg_hourly_cost: float = 0.0
    peak_demand_kw: float = 0.0
    avg_demand_kw: float = 0.0
    comfort_percentage: float = 0.0
    avg_comfort_index: float = 0.0
    avg_temp_deviation: float = 0.0
    max_temp_deviation: float = 0.0
    avg_control_level: float = 0.0
    control_variability: float = 0.0
    control_active_percentage: float = 0.0
    performance_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "Energia Total (kWh)": round(self.total_energy_kwh, 2),
            "Energia del Dispositivo (kWh)": round(self.device_energy_kwh, 2),
            "Energia Base (kWh)": round(self.base_energy_kwh, 2),
            "Consumo Promedio (kW)": round(self.avg_consumption_kw, 3),
            "Costo Total ($)": round(self.total_cost, 2),
            "Costo Promedio/Hora ($)": round(self.avg_hourly_cost, 4),
            "Pico Demanda (kW)": round(self.peak_demand_kw, 3),
            "Demanda Promedio (kW)": round(self.avg_demand_kw, 3),
            "Confort (%)": round(self.comfort_percentage, 1),
            "Indice de Confort Promedio": round(self.avg_comfort_index, 3),
            "Desviacion Promedio (C)": round(self.avg_temp_deviation, 2),
            "Desviacion Maxima (C)": round(self.max_temp_deviation, 2),
            "Nivel de Control Promedio (%)": round(self.avg_control_level, 1),
            "Variabilidad del Control": round(self.control_variability, 3),
            "Control Activo (%)": round(self.control_active_percentage, 1),
            "Performance Score": round(self.performance_score, 4),
        }


def calculate_metrics(df: pd.DataFrame, sim_config: SimulationConfig, metrics_config: MetricsConfig) -> PerformanceMetrics:
    metrics = PerformanceMetrics()
    if df.empty:
        return metrics

    dt = sim_config.time_step_hours
    metrics.total_energy_kwh = float(df["total_consumption_kw"].sum() * dt)
    metrics.device_energy_kwh = float(df["device_consumption_kw"].sum() * dt)
    metrics.base_energy_kwh = float(df["base_consumption_kw"].sum() * dt)
    metrics.avg_consumption_kw = float(df["total_consumption_kw"].mean())
    metrics.total_cost = float(df["step_cost"].sum())
    metrics.avg_hourly_cost = metrics.total_cost / max(sim_config.horizon_hours, 1)
    metrics.peak_demand_kw = float(df["total_consumption_kw"].max())
    metrics.avg_demand_kw = float(df["total_consumption_kw"].mean())

    target_temp = float(df["target_temperature"].iloc[0])
    comfort_range = _resolve_comfort_range(df, sim_config)
    in_comfort = (
        (df["device_temperature"] >= target_temp - comfort_range)
        & (df["device_temperature"] <= target_temp + comfort_range)
    )
    metrics.comfort_percentage = float(in_comfort.sum() / len(df) * 100)
    metrics.avg_comfort_index = float(df["comfort_index"].mean())
    metrics.avg_temp_deviation = float(df["temp_deviation"].mean())
    metrics.max_temp_deviation = float(df["temp_deviation"].max())

    metrics.avg_control_level = float(df["control_level"].mean())
    control_changes = df["control_level"].diff().dropna()
    metrics.control_variability = float(control_changes.std()) if len(control_changes) > 0 else 0.0
    metrics.control_active_percentage = float((df["control_level"] > 5.0).sum() / len(df) * 100)
    metrics.performance_score = calculate_performance_score(df, sim_config, metrics_config)
    return metrics


def calculate_performance_score(df: pd.DataFrame, sim_config: SimulationConfig, metrics_config: MetricsConfig) -> float:
    if df.empty:
        return 0.0

    target_temp = float(df["target_temperature"].iloc[0])
    comfort_range = _resolve_comfort_range(df, sim_config)
    in_comfort = (
        (df["device_temperature"] >= target_temp - comfort_range)
        & (df["device_temperature"] <= target_temp + comfort_range)
    )
    comfort_band_norm = float(in_comfort.mean())
    comfort_quality_norm = float(df["comfort_index"].mean())
    comfort_norm = 0.7 * comfort_band_norm + 0.3 * comfort_quality_norm
    dt = sim_config.time_step_hours
    definition = build_device_definition(sim_config.device_key)
    device_max_power = float(definition.dynamics.max_power_kw)
    device_cop = float(definition.dynamics.cop)
    base_peak = float(df["base_consumption_kw"].max()) if "base_consumption_kw" in df else 2.0

    max_device_energy = device_max_power / max(device_cop, 0.1) * sim_config.horizon_hours
    device_energy = float(df["device_consumption_kw"].sum() * dt)
    energy_norm = min(device_energy / max(max_device_energy, 0.1), 1.0)

    worst_case_cost = (device_max_power / max(device_cop, 0.1) + base_peak) * 0.30 * sim_config.horizon_hours
    total_cost = float(df["step_cost"].sum())
    cost_norm = min(total_cost / max(worst_case_cost, 0.1), 1.0)

    peak_norm = min(float(df["total_consumption_kw"].max()) / max(device_max_power / max(device_cop, 0.1) + base_peak, 0.1), 1.0)
    variability = float(df["control_level"].diff().dropna().abs().mean()) if len(df) > 1 else 0.0
    variability_norm = min(variability / 50.0, 1.0)
    avg_dev_norm = min(float(df["temp_deviation"].mean()) / max(comfort_range * 2.5, 0.1), 1.0)
    max_dev_norm = min(float(df["temp_deviation"].max()) / max(comfort_range * 4.0, 0.1), 1.0)

    score = (
        metrics_config.weight_comfort * comfort_norm
        + metrics_config.weight_energy * (1.0 - energy_norm)
        + metrics_config.weight_cost * (1.0 - cost_norm)
        - metrics_config.weight_peak * peak_norm
        - metrics_config.weight_variability * variability_norm
        - 0.08 * avg_dev_norm
        - 0.05 * max_dev_norm
    )
    return float(score)


def compare_metrics(base: PerformanceMetrics, optimized: PerformanceMetrics) -> Dict[str, Dict[str, float]]:
    comparison: Dict[str, Dict[str, float]] = {}
    base_dict = base.to_dict()
    optimized_dict = optimized.to_dict()
    for key, base_value in base_dict.items():
        optimized_value = optimized_dict[key]
        if base_value != 0:
            raw_change = ((optimized_value - base_value) / abs(base_value)) * 100
        else:
            raw_change = 0.0
        benefit_change = raw_change if is_higher_better_metric(key) else -raw_change
        comparison[key] = {
            "base": base_value,
            "optimizado": optimized_value,
            "cambio_%": round(raw_change, 2),
            "mejora_%": round(benefit_change, 2),
        }
    return comparison


def _resolve_comfort_range(df: pd.DataFrame, sim_config: SimulationConfig) -> float:
    if "comfort_range" in df.columns and not df.empty:
        return float(df["comfort_range"].iloc[0])
    return sim_config.comfort_range


def is_higher_better_metric(metric_name: str) -> bool:
    return metric_name in HIGHER_IS_BETTER_METRICS
