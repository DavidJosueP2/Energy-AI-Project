# ==============================================================================
# dashboard.py - Dashboard multiplot para visualización rápida
# ==============================================================================
"""
Genera dashboards compuestos con múltiples gráficos en una sola figura.
Útil para obtener una vista general rápida de los resultados.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np

from app.visualization.plots import apply_style, COLORS
from app.simulation.metrics import PerformanceMetrics
import pandas as pd
from typing import Optional


def create_simulation_dashboard(df: pd.DataFrame,
                                 target_temp: float = 22.0,
                                 comfort_range: float = 2.0,
                                 title: str = "Dashboard de Simulación") -> Figure:
    """
    Crea un dashboard completo con 6 paneles para una simulación.
    
    Paneles:
    1. Temperaturas (interior/exterior)
    2. Nivel HVAC
    3. Consumo eléctrico
    4. Ocupación y tarifa
    5. Costo acumulado
    6. Índice de confort
    """
    apply_style()
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98, color='#eee')
    
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    time = df['time_hours']
    
    # 1. Temperaturas
    ax = fig.add_subplot(gs[0, 0])
    ax.axhspan(target_temp - comfort_range, target_temp + comfort_range,
               alpha=0.12, color=COLORS['green'])
    ax.plot(time, df['temperature_outdoor'], color=COLORS['warm'],
            linewidth=1.2, alpha=0.7, label='Exterior')
    ax.plot(time, df['temperature_indoor'], color=COLORS['cool'],
            linewidth=1.8, label='Interior')
    ax.axhline(target_temp, color=COLORS['green'], linewidth=0.8,
               linestyle='--', alpha=0.5)
    ax.set_ylabel('Temp (°C)')
    ax.set_title('Temperaturas', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)
    
    # 2. Nivel HVAC
    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(time, 0, df['hvac_level'], alpha=0.4, color=COLORS['accent'])
    ax.plot(time, df['hvac_level'], color=COLORS['accent'], linewidth=1.2)
    ax.set_ylabel('HVAC (%)')
    ax.set_title('Control HVAC', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2)
    
    # 3. Consumo
    ax = fig.add_subplot(gs[1, 0])
    ax.fill_between(time, 0, df['base_consumption_kw'],
                    alpha=0.5, color=COLORS['secondary'], label='Base')
    ax.fill_between(time, df['base_consumption_kw'], df['total_consumption_kw'],
                    alpha=0.5, color=COLORS['primary'], label='HVAC')
    ax.set_ylabel('Consumo (kW)')
    ax.set_title('Consumo Eléctrico', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    
    # 4. Ocupación y tarifa
    ax = fig.add_subplot(gs[1, 1])
    ax2 = ax.twinx()
    ax.fill_between(time, 0, df['occupancy'], alpha=0.4, color=COLORS['purple'])
    ax.plot(time, df['occupancy'], color=COLORS['purple'], linewidth=1.2, label='Ocupación')
    ax2.plot(time, df['tariff'], color=COLORS['orange'], linewidth=1.5,
             drawstyle='steps-mid', label='Tarifa')
    ax.set_ylabel('Ocupantes', color=COLORS['purple'])
    ax2.set_ylabel('$/kWh', color=COLORS['orange'])
    ax.set_title('Ocupación y Tarifa', fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # 5. Costo acumulado
    ax = fig.add_subplot(gs[2, 0])
    ax.fill_between(time, 0, df['cumulative_cost'], alpha=0.3, color=COLORS['warm'])
    ax.plot(time, df['cumulative_cost'], color=COLORS['warm'], linewidth=2)
    ax.set_xlabel('Tiempo (horas)')
    ax.set_ylabel('Costo ($)')
    ax.set_title('Costo Acumulado', fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # 6. Confort
    ax = fig.add_subplot(gs[2, 1])
    ax.fill_between(time, 0, df['comfort_index'], alpha=0.4, color=COLORS['green'])
    ax.plot(time, df['comfort_index'], color=COLORS['green'], linewidth=1.5)
    avg_comfort = df['comfort_index'].mean()
    ax.axhline(avg_comfort, color=COLORS['red'], linewidth=1, linestyle='--',
               label=f'Promedio: {avg_comfort:.2f}')
    ax.set_xlabel('Tiempo (horas)')
    ax.set_ylabel('Índice')
    ax.set_title('Índice de Confort', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    
    return fig


def create_comparison_dashboard(df_base: pd.DataFrame,
                                 df_opt: pd.DataFrame,
                                 base_metrics: PerformanceMetrics,
                                 opt_metrics: PerformanceMetrics,
                                 target_temp: float = 22.0,
                                 comfort_range: float = 2.0) -> Figure:
    """Dashboard comparativo base vs optimizado."""
    apply_style()
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Comparación: Controlador Base vs Optimizado',
                 fontsize=16, fontweight='bold', y=0.99, color='#eee')
    
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    time = df_base['time_hours']
    
    # 1. Temperaturas comparadas
    ax = fig.add_subplot(gs[0, 0])
    ax.axhspan(target_temp - comfort_range, target_temp + comfort_range,
               alpha=0.10, color=COLORS['green'])
    ax.plot(time, df_base['temperature_indoor'], color=COLORS['base'],
            linewidth=1.5, alpha=0.8, label='T. Interior (Base)')
    ax.plot(time, df_opt['temperature_indoor'], color=COLORS['optimized'],
            linewidth=1.5, alpha=0.8, label='T. Interior (Opt)')
    if 'temperature_outdoor' in df_base.columns:
        ax.plot(time, df_base['temperature_outdoor'], color='#f5a623',
                linewidth=2.0, alpha=0.85, linestyle='--', label='T. Exterior')
    ax.set_ylabel('T. Interior (°C)')
    ax.set_title('Temperatura Interior', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    
    # 2. HVAC comparado
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(time, df_base['hvac_level'], color=COLORS['base'],
            linewidth=1.2, alpha=0.7, label='Base')
    ax.plot(time, df_opt['hvac_level'], color=COLORS['optimized'],
            linewidth=1.2, alpha=0.7, label='Optimizado')
    ax.set_ylabel('HVAC (%)')
    ax.set_title('Nivel de Climatización', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    
    # 3. Consumo comparado
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(time, df_base['total_consumption_kw'], color=COLORS['base'],
            linewidth=1.5, label='Base')
    ax.plot(time, df_opt['total_consumption_kw'], color=COLORS['optimized'],
            linewidth=1.5, label='Optimizado')
    ax.set_ylabel('Consumo (kW)')
    ax.set_title('Consumo Total', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    
    # 4. Costo comparado
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(time, df_base['cumulative_cost'], color=COLORS['base'],
            linewidth=2, label=f"Base: ${base_metrics.total_cost:.2f}")
    ax.plot(time, df_opt['cumulative_cost'], color=COLORS['optimized'],
            linewidth=2, label=f"Opt: ${opt_metrics.total_cost:.2f}")
    ax.set_ylabel('Costo ($)')
    ax.set_title('Costo Acumulado', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # 5. Confort comparado
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(time, df_base['comfort_index'], color=COLORS['base'],
            linewidth=1.5, alpha=0.7, label=f"Base: {base_metrics.comfort_percentage:.0f}%")
    ax.plot(time, df_opt['comfort_index'], color=COLORS['optimized'],
            linewidth=1.5, alpha=0.7, label=f"Opt: {opt_metrics.comfort_percentage:.0f}%")
    ax.set_xlabel('Tiempo (horas)')
    ax.set_ylabel('Índice Confort')
    ax.set_title('Confort Térmico', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    
    # 6. Barras de métricas clave
    ax = fig.add_subplot(gs[2, 1])
    labels = ['Energía\n(kWh)', 'Costo\n($)', 'Confort\n(%)', 'Pico\n(kW)']
    base_vals = [base_metrics.total_energy_kwh, base_metrics.total_cost,
                 base_metrics.comfort_percentage, base_metrics.peak_demand_kw]
    opt_vals = [opt_metrics.total_energy_kwh, opt_metrics.total_cost,
                opt_metrics.comfort_percentage, opt_metrics.peak_demand_kw]
    
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, base_vals, width, label='Base',
           color=COLORS['base'], alpha=0.8)
    ax.bar(x + width/2, opt_vals, width, label='Optimizado',
           color=COLORS['optimized'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title('Métricas Clave', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.2)
    
    return fig
