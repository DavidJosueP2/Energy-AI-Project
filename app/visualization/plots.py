# ==============================================================================
# plots.py - Generación de gráficos con matplotlib
# ==============================================================================
"""
Genera todos los gráficos requeridos para el análisis del sistema:
1. Temperatura interior vs exterior
2. Nivel de climatización
3. Consumo total por hora
4. Costo acumulado
5. Confort térmico
6. Evolución del fitness del GA
7. Comparación base vs optimizado
8. Distribución de consumo
9. Ocupación y tarifa
10. Radar de métricas comparativas
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo por defecto
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Dict, Tuple

from app.simulation.metrics import PerformanceMetrics


# ---- Estilo global de gráficos ----
STYLE_CONFIG = {
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eee',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'text.color': '#eee',
    'xtick.color': '#ccc',
    'ytick.color': '#ccc',
    'grid.color': '#333',
    'grid.alpha': 0.4,
    'legend.facecolor': '#16213e',
    'legend.edgecolor': '#444',
    'legend.fontsize': 9,
    'font.family': 'sans-serif',
}

COLORS = {
    'primary': '#e94560',
    'secondary': '#0f3460',
    'accent': '#53d8fb',
    'warm': '#f5a623',
    'cool': '#4fc3f7',
    'green': '#66bb6a',
    'red': '#ef5350',
    'purple': '#ab47bc',
    'orange': '#ff7043',
    'comfort_zone': '#66bb6a33',
    'base': '#e94560',
    'optimized': '#53d8fb',
}


def apply_style():
    """Aplica estilo oscuro profesional a matplotlib."""
    plt.rcParams.update(STYLE_CONFIG)


def plot_temperatures(df: pd.DataFrame, target_temp: float = 22.0,
                      comfort_range: float = 2.0) -> Figure:
    """Gráfico 1: Temperatura interior vs exterior con zona de confort."""
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    
    time = df['time_hours']
    
    # Zona de confort
    ax.axhspan(target_temp - comfort_range, target_temp + comfort_range,
               alpha=0.15, color=COLORS['green'], label='Zona de confort')
    ax.axhline(target_temp, color=COLORS['green'], linewidth=1,
               linestyle='--', alpha=0.6, label=f'Objetivo ({target_temp}°C)')
    
    # Temperaturas
    ax.plot(time, df['temperature_outdoor'], color=COLORS['warm'],
            linewidth=1.5, alpha=0.8, label='T. Exterior')
    ax.plot(time, df['temperature_indoor'], color=COLORS['cool'],
            linewidth=2, label='T. Interior')
    
    ax.set_xlabel('Tiempo (horas)')
    ax.set_ylabel('Temperatura (°C)')
    ax.set_title('Evolución de Temperaturas', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_humidity(df: pd.DataFrame) -> Figure:
    """Gráfico de Humedad simulada."""
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    
    time = df['time_hours']
    
    if 'humidity' in df.columns:
        # Humidity is 0 to 1, plot as percentage
        humidity_pct = df['humidity'] * 100
        ax.fill_between(time, 0, humidity_pct, alpha=0.3, color=COLORS['cool'])
        ax.plot(time, humidity_pct, color=COLORS['cool'], linewidth=2, label='Humedad')
    
    ax.set_xlabel('Tiempo (horas)')
    ax.set_ylabel('Humedad (%)')
    ax.set_title('Evolución de Humedad Ambiental', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_hvac_level(df: pd.DataFrame) -> Figure:
    """Gráfico 2: Nivel de climatización a lo largo del tiempo."""
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    
    time = df['time_hours']
    ax.fill_between(time, 0, df['hvac_level'], alpha=0.4, color=COLORS['accent'])
    ax.plot(time, df['hvac_level'], color=COLORS['accent'], linewidth=1.5)
    
    ax.set_xlabel('Tiempo (horas)')
    ax.set_ylabel('Nivel HVAC (%)')
    ax.set_title('Señal de Control del Climatizador', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_consumption(df: pd.DataFrame) -> Figure:
    """Gráfico 3: Consumo total por hora."""
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    
    time = df['time_hours']
    ax.fill_between(time, 0, df['base_consumption_kw'],
                    alpha=0.5, color=COLORS['secondary'], label='Consumo base')
    ax.fill_between(time, df['base_consumption_kw'], df['total_consumption_kw'],
                    alpha=0.5, color=COLORS['primary'], label='Consumo HVAC')
    ax.plot(time, df['total_consumption_kw'], color=COLORS['red'],
            linewidth=1.5, label='Total')
    
    ax.set_xlabel('Tiempo (horas)')
    ax.set_ylabel('Consumo (kW)')
    ax.set_title('Consumo Eléctrico', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cumulative_cost(df: pd.DataFrame) -> Figure:
    """Gráfico 4: Costo acumulado."""
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    
    time = df['time_hours']
    ax.fill_between(time, 0, df['cumulative_cost'], alpha=0.3, color=COLORS['warm'])
    ax.plot(time, df['cumulative_cost'], color=COLORS['warm'], linewidth=2)
    
    final_cost = df['cumulative_cost'].iloc[-1]
    ax.annotate(f'${final_cost:.2f}', 
                xy=(time.iloc[-1], final_cost),
                xytext=(-60, 15), textcoords='offset points',
                fontsize=12, fontweight='bold', color=COLORS['warm'],
                arrowprops=dict(arrowstyle='->', color=COLORS['warm']))
    
    ax.set_xlabel('Tiempo (horas)')
    ax.set_ylabel('Costo Acumulado ($)')
    ax.set_title('Costo Eléctrico Acumulado', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_comfort(df: pd.DataFrame, target_temp: float = 22.0) -> Figure:
    """Gráfico 5: Desviación de confort térmico."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1])
    
    time = df['time_hours']
    
    # Panel superior: desviación de temperatura
    deviation = df['temperature_indoor'] - target_temp
    colors_arr = np.where(deviation > 0, COLORS['red'], COLORS['cool'])
    ax1.bar(time, deviation, width=0.8, color=[COLORS['red'] if d > 0 else COLORS['cool'] 
            for d in deviation], alpha=0.6)
    ax1.axhline(0, color=COLORS['green'], linewidth=1.5, linestyle='-')
    ax1.axhline(2, color='#666', linewidth=0.8, linestyle='--', alpha=0.5)
    ax1.axhline(-2, color='#666', linewidth=0.8, linestyle='--', alpha=0.5)
    ax1.set_ylabel('Desviación (°C)')
    ax1.set_title('Desviación Respecto al Objetivo Térmico', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel inferior: índice de confort
    ax2.fill_between(time, 0, df['comfort_index'], alpha=0.4, color=COLORS['green'])
    ax2.plot(time, df['comfort_index'], color=COLORS['green'], linewidth=1.5)
    ax2.set_xlabel('Tiempo (horas)')
    ax2.set_ylabel('Índice de Confort')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_ga_evolution(fitness_history: List[float],
                       avg_history: Optional[List[float]] = None) -> Figure:
    """Gráfico 6: Evolución del fitness por generación del GA."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    
    generations = list(range(len(fitness_history)))
    
    ax.plot(generations, fitness_history, color=COLORS['primary'],
            linewidth=2.5, marker='o', markersize=4, label='Mejor fitness')
    
    if avg_history:
        ax.plot(generations, avg_history, color=COLORS['accent'],
                linewidth=1.5, linestyle='--', alpha=0.7, label='Fitness promedio')
        ax.fill_between(generations, avg_history, fitness_history,
                        alpha=0.15, color=COLORS['primary'])
    
    ax.set_xlabel('Generación')
    ax.set_ylabel('Fitness')
    ax.set_title('Evolución del Algoritmo Genético', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_comparison(df_base: pd.DataFrame,
                     df_opt: pd.DataFrame,
                     target_temp: float = 22.0,
                     comfort_range: float = 2.0) -> Figure:
    """Gráfico 7: Comparación base vs optimizado (multiplot)."""
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    time = df_base['time_hours']
    
    # Panel 1: Temperaturas
    ax = axes[0]
    ax.axhspan(target_temp - comfort_range, target_temp + comfort_range,
               alpha=0.12, color=COLORS['green'])
    ax.plot(time, df_base['temperature_indoor'], color=COLORS['base'],
            linewidth=1.8, label='T. Interior (Base)', alpha=0.85)
    ax.plot(time, df_opt['temperature_indoor'], color=COLORS['optimized'],
            linewidth=1.8, label='T. Interior (Optimizado)', alpha=0.85)
    ax.plot(time, df_base['temperature_outdoor'], color='#f5a623',
            linewidth=2.0, alpha=0.85, linestyle='--', label='T. Exterior')
    ax.set_ylabel('Temperatura (°C)')
    ax.set_title('Comparación Base vs Optimizado', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: HVAC Level
    ax = axes[1]
    ax.fill_between(time, 0, df_base['hvac_level'], alpha=0.3, color=COLORS['base'])
    ax.fill_between(time, 0, df_opt['hvac_level'], alpha=0.3, color=COLORS['optimized'])
    ax.plot(time, df_base['hvac_level'], color=COLORS['base'],
            linewidth=1.5, label='HVAC Base')
    ax.plot(time, df_opt['hvac_level'], color=COLORS['optimized'],
            linewidth=1.5, label='HVAC Optimizado')
    ax.set_ylabel('Nivel HVAC (%)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Costo acumulado
    ax = axes[2]
    ax.plot(time, df_base['cumulative_cost'], color=COLORS['base'],
            linewidth=2, label=f"Base: ${df_base['cumulative_cost'].iloc[-1]:.2f}")
    ax.plot(time, df_opt['cumulative_cost'], color=COLORS['optimized'],
            linewidth=2, label=f"Opt: ${df_opt['cumulative_cost'].iloc[-1]:.2f}")
    ax.set_xlabel('Tiempo (horas)')
    ax.set_ylabel('Costo Acumulado ($)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_occupancy_tariff(df: pd.DataFrame) -> Figure:
    """Gráfico 9: Ocupación y tarifa a lo largo del tiempo."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    
    time = df['time_hours']
    
    # Ocupación
    ax1.fill_between(time, 0, df['occupancy'], alpha=0.5, color=COLORS['purple'])
    ax1.plot(time, df['occupancy'], color=COLORS['purple'], linewidth=1.5)
    ax1.set_ylabel('Ocupantes')
    ax1.set_title('Ocupación y Tarifa Eléctrica', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Tarifa
    ax2.fill_between(time, 0, df['tariff'], alpha=0.5, color=COLORS['orange'],
                     step='mid')
    ax2.plot(time, df['tariff'], color=COLORS['orange'], linewidth=1.5,
             drawstyle='steps-mid')
    ax2.set_xlabel('Tiempo (horas)')
    ax2.set_ylabel('Tarifa ($/kWh)')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_metrics_comparison_bars(base_metrics: PerformanceMetrics,
                                  opt_metrics: PerformanceMetrics) -> Figure:
    """Gráfico 10: Barras comparativas de métricas clave."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = ['Energía\nTotal\n(kWh)', 'Costo\nTotal\n($)', 'Confort\n(%)',
              'Pico\nDemanda\n(kW)', 'Nivel HVAC\nPromedio\n(%)', 'Fitness\nScore']
    
    base_vals = [
        base_metrics.total_energy_kwh,
        base_metrics.total_cost,
        base_metrics.comfort_percentage,
        base_metrics.peak_demand_kw,
        base_metrics.avg_hvac_level,
        base_metrics.fitness_score * 100,
    ]
    opt_vals = [
        opt_metrics.total_energy_kwh,
        opt_metrics.total_cost,
        opt_metrics.comfort_percentage,
        opt_metrics.peak_demand_kw,
        opt_metrics.avg_hvac_level,
        opt_metrics.fitness_score * 100,
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, base_vals, width, label='Base',
                   color=COLORS['base'], alpha=0.8, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, opt_vals, width, label='Optimizado',
                   color=COLORS['optimized'], alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Etiquetas de valor
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}',
                ha='center', va='bottom', fontsize=8, color='#ccc')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}',
                ha='center', va='bottom', fontsize=8, color='#ccc')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title('Comparación de Métricas: Base vs Optimizado',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_consumption_distribution(df: pd.DataFrame) -> Figure:
    """Gráfico 8: Distribución estadística del consumo."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histograma
    ax1.hist(df['total_consumption_kw'], bins=30, color=COLORS['accent'],
             alpha=0.7, edgecolor='white', linewidth=0.5)
    ax1.axvline(df['total_consumption_kw'].mean(), color=COLORS['red'],
                linewidth=2, linestyle='--', label=f"Media: {df['total_consumption_kw'].mean():.2f} kW")
    ax1.set_xlabel('Consumo (kW)')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución del Consumo Total', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot del consumo por franja horaria
    df_temp = df.copy()
    df_temp['franja'] = pd.cut(df_temp['hour_of_day'], 
                                bins=[0, 8, 14, 22, 24],
                                labels=['Valle\n(0-8h)', 'Llano AM\n(8-14h)',
                                       'Punta\n(14-22h)', 'Llano PM\n(22-24h)'],
                                include_lowest=True)
    
    groups = []
    labels_box = []
    for name in ['Valle\n(0-8h)', 'Llano AM\n(8-14h)', 'Punta\n(14-22h)', 'Llano PM\n(22-24h)']:
        subset = df_temp[df_temp['franja'] == name]['total_consumption_kw']
        if len(subset) > 0:
            groups.append(subset.values)
            labels_box.append(name)
    
    if groups:
        bp = ax2.boxplot(groups, labels=labels_box, patch_artist=True,
                         boxprops=dict(facecolor=COLORS['secondary'], alpha=0.7),
                         medianprops=dict(color=COLORS['warm'], linewidth=2),
                         whiskerprops=dict(color='#888'),
                         capprops=dict(color='#888'),
                         flierprops=dict(markerfacecolor=COLORS['red'], markersize=4))
    
    ax2.set_ylabel('Consumo (kW)')
    ax2.set_title('Consumo por Franja Horaria', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, filepath: str, dpi: int = 150):
    """Guarda una figura en disco."""
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
